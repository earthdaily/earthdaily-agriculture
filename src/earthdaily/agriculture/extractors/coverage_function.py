#  Standard Library
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

#  Third-Party Libraries
import requests
from tqdm import tqdm

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.api_utils import (
    export_results,
    filter_entities,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
)
from earthdaily.agriculture.core.api_utils import validate_historical_years as validate_historical_seasons
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator


def require_coverage_params(func):
    """Decorator to ensure coverage parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "coverage_params"):
            raise RuntimeError("❌ No coverage parameters found. Call setup_coverage_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class CoverageExtractor(BaseExtractor):
    """
    Extracts satellite image coverage analytics for agricultural entities.

    Retrieves available satellite imagery metadata (coverage percentage, sensor, date, mask)
    for a given geometry and date range. Supports duplicate filtering across sensors and
    multi-sensor coverage analysis.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_coverage.ipynb

    Args (setup_coverage_parameters):
        vegetation_index (str): Vegetation index type ('NDVI', 'EVI', etc.). Default: 'NDVI'
        start_date (str): Start date in YYYY-MM-DD format. Default: '2025-01-01'
        end_date (str): End date in YYYY-MM-DD format. Default: None — **no upper bound**,
            i.e. every image from start_date onward is returned (the query becomes
            image.date=$gte:start_date). It is *not* defaulted to today; set it explicitly
            to bound the catalogue query.
        clear_cover_min (int): Minimum clear sky coverage percentage (0-100). Default: 90
        clear_cover_max (int): Maximum clear sky coverage percentage (0-100). Default: 100
        use_specific_date (bool): Restrict to exact date. Default: False
        filter (str): Filtering mode - 'none', 'duplicate' or 'crop_coverage'. Default: 'none'
            'duplicate' collapses near-simultaneous multi-sensor captures (see delay).
            'crop_coverage' expands the query per entity across prior seasons and
            **requires historical_seasons**; the per-entity window is computed from the
            entity's start_date/end_date month-day applied to each historical year.
        delay (int): Max days between multi-sensor captures for duplicate filter. Default: 3
        mask (str): Cloud mask type — 'Auto', 'Native', 'ACM', 'ML', 'MLCirrus' or 'All'
            (case-insensitive). 'All' returns one row per image *per mask*, so the same image
            appears several times and the 'mask' output column tells them apart — use it to
            compare masks. Not compatible with filter='duplicate'. Default: 'auto'
        recalibration (bool): Apply cross-sensor recalibration to coverage metrics. Default: False
        historical_seasons (list[int]): Prior season years used when filter='crop_coverage'
            (required for that filter, e.g. [2024, 2023, 2022]). Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid
            re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required)

    Output columns:
        image_id, coverage_percent, date, mask, sensor, spatial_resolution
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.coverage_params = None

        # Specific API endpoint for coverage
        self.map_products_url = agro_urls["map_products_url"][self.env]

    def get_new_token(self):
        """
        Implements token refresh logic for CoverageExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_coverage_parameters(
        self,
        vegetation_index="NDVI",
        start_date="2025-01-01",
        end_date=None,
        clear_cover_min=90,
        clear_cover_max=100,
        use_specific_date=False,
        filter="none",
        delay=3,
        mask="auto",
        partial_frequency=50,
        recalibration=False,
        historical_seasons=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure params for imagery catalog extraction for an entity.

        When filter='crop_coverage', historical_seasons must be provided.
        The extraction period is computed per entity from start_date/end_date fields
        across all historical years (smallest year start to largest year end).
        """
        valid_indexes = {"NDVI", "EVI", "CVI", "CVIN", "GNDVI", "LAI", "NDWI", "NDMI", "S2REP"}
        if vegetation_index not in valid_indexes:
            raise ValueError(f"Invalid vegetation_index '{vegetation_index}'. Must be one of {valid_indexes}")

        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD")

        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD")

        if not (0 <= clear_cover_min <= 100):
            raise ValueError(f"Invalid clear_cover_min '{clear_cover_min}'. Must be between 0 and 100.")

        valid_filters = {"none", "duplicate", "crop_coverage"}
        if filter not in valid_filters:
            raise ValueError(f"Invalid filter selected '{filter}'. Must be one of {valid_filters}")

        if delay < 0:
            raise ValueError(f"Invalid delay '{delay}'. Must be non-negative.")

        # CoverageMaskType — 'All' returns one row per (image, mask) so masks can be compared
        # side by side via the 'mask' output column. Case-insensitive, normalized to API casing.
        valid_masks = {
            "native": "Native",
            "acm": "ACM",
            "auto": "Auto",
            "ml": "ML",
            "mlcirrus": "MLCirrus",
            "all": "All",
        }
        if not isinstance(mask, str) or mask.lower() not in valid_masks:
            raise ValueError(f"Invalid mask '{mask}'. Must be one of {sorted(valid_masks.values())}")
        mask = valid_masks[mask.lower()]

        # 'All' repeats each image once per available mask. The duplicate filter pairs images
        # across different sensors and sums their coverage, so those repeats would inflate the
        # sums and select pairs that do not exist as a single acquisition.
        if mask == "All" and filter == "duplicate":
            raise ValueError(
                "mask='All' cannot be combined with filter='duplicate': 'All' returns one row per "
                "(image, mask), which double-counts coverage when pairing images. Use a single mask "
                "(e.g. 'Auto') for duplicate filtering, or filter='none' to compare masks."
            )

        # Historical seasons validation
        historical_seasons = validate_historical_seasons(historical_seasons)

        if filter == "crop_coverage" and historical_seasons is None:
            raise ValueError(
                "historical_seasons is required when filter='crop_coverage'. "
                "Provide a list of years (e.g., [2024, 2023, 2022])."
            )

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # ✅ store in self.coverage_params
        self.coverage_params = {
            "vegetation_index": vegetation_index,
            "start_date": start_date,
            "end_date": end_date,
            "clear_cover_min": clear_cover_min,
            "clear_cover_max": clear_cover_max,
            "use_specific_date": use_specific_date,
            "filter": filter,
            "delay": delay,
            "mask": mask,
            "partial_frequency": partial_frequency,
            "recalibration": recalibration,
            "historical_seasons": historical_seasons,
        }

        # Enable/disable cache if explicitly provided
        self.apply_cache_setting(use_cache)

        # Configure cache key columns for coverage results
        # Include mask because the same entity+image can have different coverage status per mask type
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "image_id", "mask"]

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        print("🌍 Coverage parameters configured:")
        for k, v in self.coverage_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_coverage_params
    def get_satellite_coverage_by_geometry(self, entity_data: dict):
        """
        Query the coverage API to get imagery metadata for a given entity geometry.
        If recalibration is enabled in coverage_params, uses the recalibration endpoint
        and includes crop in the payload.

        Args:
            entity_data (dict): Must contain a 'geometry' key in WKT format.
                If recalibration=True, must also contain a 'crop' key.

        Returns:
            dict: API response with imagery metadata
        """
        params = self.coverage_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")
        recalibration = params.get("recalibration", False)

        # Step 1: Geometry validation
        if not self.has_entity_field(entity_data, "geometry"):
            error_msg = f"entity_data for {entity_id} must include a 'geometry' field in WKT format."
            log.error(error_msg)
            raise ValueError(error_msg)

        geometry = self.get_entity_value(entity_data, "geometry")
        try:
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except ValueError as e:
            log.error(f"Invalid geometry for entity {entity_id}: {e}")
            raise

        # Step 2: API URL
        url = f"{self.map_products_url}/season-fields/catalog-imagery"
        if recalibration:
            url += "/recalibration"

        if params["use_specific_date"]:
            date_param = f"image.date={params['start_date']}"
            log.debug(f"Using specific date: {params['start_date']}")
        else:
            # Entity-level date overrides (used by crop_coverage filter mode)
            start_date = entity_data.get("_coverage_start_date") or params.get("start_date")
            end_date = entity_data.get("_coverage_end_date") or params.get("end_date")

            if start_date and end_date:
                date_param = f"image.date=$between:{start_date}|{end_date}"
                log.debug(f"Using date range: {start_date} to {end_date}")
            elif start_date:
                date_param = f"image.date=$gte:{start_date}"
                log.debug(f"Using date range from: {start_date}")
            elif end_date:
                date_param = f"image.date=$lte:{end_date}"
                log.debug(f"Using date range up to: {end_date}")
            else:
                date_param = ""
                log.warning("No date filter applied")

        query_params = [
            f"Maps.Type={params['vegetation_index']}",
            f"coveragePercent=$gte:{params['clear_cover_min']}",
            f"coveragePercent=$lte:{params['clear_cover_max']}",
            "$fields=coveragePercent,image.id,image.spatialResolution,image.date,image.sensor,mask",
            "$limit=10000",
            date_param,
            f"mask={params['mask']}",
        ]

        # --- Full URL ---
        full_url = f"{url}?{'&'.join(query_params)}"
        log.debug(f"API URL: {url}")

        # --- Headers ---
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # --- Payload ---
        payload = {"seasonFields": [{"geometry": geometry}]}
        if recalibration:
            crop = self.get_entity_value(entity_data, "crop")
            payload["seasonFields"][0]["crop"] = crop

        # Step 3: Request data
        log.debug(f"API URL: {full_url}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(full_url, headers=headers, json=payload, timeout=60)  # type: ignore[arg-type]
            response.raise_for_status()
            json_response = response.json()
            log.debug(f"API response: {json_response}")
            return json_response
        except requests.exceptions.Timeout as e:
            log.error(f"Timeout requesting coverage for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Unexpected error for entity {entity_id}: {e}")
            raise

    def get_satellite_coverage_by_geometry_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_satellite_coverage_by_geometry().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain a 'geometry' key in WKT format.

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "seasonfield_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response_json = self.get_satellite_coverage_by_geometry(entity_data)
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}

    @require_coverage_params
    def format_coverage_json(self, response_coverage_json, entity_id="unknown"):
        """
        Process API response to extract image ID, coverage percent, formatted date,
        sensor, spatial resolution, and mask information.

        Args:
            response_coverage_json (str | list | dict): API response (stringified JSON, parsed list, or dict).
            entity_id (str): Entity ID for logging purposes.

        Returns:
            pd.DataFrame: Cleaned DataFrame of imagery coverage.
        """
        params = self.coverage_params
        log = self.get_contextualized_logger("FORMAT")

        #  Step 1: Normalize response into a list of records
        if isinstance(response_coverage_json, str):
            log.debug(f"Parsing string JSON for entity {entity_id}")
            data = json.loads(response_coverage_json)

        elif isinstance(response_coverage_json, dict):
            if "items" in response_coverage_json:
                data = response_coverage_json["items"]
                log.debug(f"Found {len(data)} items for entity {entity_id}")
            elif "results" in response_coverage_json:
                data = response_coverage_json["results"]
                log.debug(f"Found {len(data)} results for entity {entity_id}")
            else:
                log.warning(f"Unknown dict structure for entity {entity_id}, treating as empty")
                data = []

        elif isinstance(response_coverage_json, list):
            data = response_coverage_json
            log.debug(f"Processing list with {len(data)} items for entity {entity_id}")

        else:
            error_msg = f"Unsupported response type: {type(response_coverage_json)}"
            log.error(error_msg)
            raise TypeError(error_msg)

        empty = self.validate_api_response(data, entity_id, "coverage")
        if empty is not None:
            return empty

        # Step 2: Parse into catalog list
        log.debug(f"Parsing {len(data)} coverage records for entity {entity_id}")
        catalog = []

        for record in data:
            # Parse ISO datetime
            iso_date = record["image"]["date"]
            parsed_date = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
            formatted_date = parsed_date.strftime("%Y-%m-%d")  # standardized format

            row = {
                "image_id": record["image"]["id"],
                "coverage_percent": record["coveragePercent"],
                "date": formatted_date,  # string for readability
                "date_obj": parsed_date,  # datetime for filtering
                "mask": record.get("mask", None),
            }

            # Sensor field
            if "sensor" in record["image"]:
                row["sensor"] = record["image"]["sensor"]
            else:
                image_id = record["image"]["id"]
                if "sentinel-2" in image_id.lower():
                    row["sensor"] = "SENTINEL_2"
                elif "landsat" in image_id.lower():
                    if "LC09" in image_id:
                        row["sensor"] = "LANDSAT_9"
                    elif "LC08" in image_id:
                        row["sensor"] = "LANDSAT_8"
                    else:
                        row["sensor"] = "LANDSAT"
                else:
                    row["sensor"] = "UNKNOWN"

            # Spatial resolution if available
            if "spatialResolution" in record["image"]:
                row["spatial_resolution"] = record["image"]["spatialResolution"]

            catalog.append(row)

        df = pd.DataFrame(catalog)

        # Strict coverage_percent filtering
        if not df.empty:
            df = df[
                (df["coverage_percent"] >= params["clear_cover_min"])
                & (df["coverage_percent"] <= params["clear_cover_max"])
            ]

        log.debug(f"Created DataFrame with {len(df)} rows for entity {entity_id}")

        # Apply duplicate/delay filtering if requested
        if params.get("filter") == "duplicate":
            log.info(f"Applying duplicate filter for entity {entity_id}")
            original_count = len(df)
            df = self._filter_duplicates(df)
            log.debug(f"Filtering reduced {original_count} -> {len(df)} images")

        # Drop helper column after filtering
        if "date_obj" in df.columns:
            df = df.drop("date_obj", axis=1)

        return df

    @require_coverage_params
    def _filter_duplicates(self, df):
        """
        Filtering function using vectorized operations and smart grouping.
        """
        params = self.coverage_params
        delay = params.get("delay", 0)
        log = self.get_contextualized_logger("FILTER")

        if df.empty or len(df) < 2:
            log.debug(f"Skipping filter: DataFrame has {len(df)} rows")
            return df if len(df) < 2 else pd.DataFrame([])

        log.debug(f"Starting duplicate filter with delay={delay} days on {len(df)} images")

        df = df.copy().sort_values("date_obj").reset_index(drop=True)

        # Convert to numpy for faster operations
        dates = df["date_obj"].values
        sensors = df["sensor"].values
        coverage = df["coverage_percent"].values

        # Find all valid pairs using vectorized operations
        valid_pairs = []
        n = len(df)

        # Create boolean masks for different sensors
        for i in range(n):
            # Get indices of rows after current row with different sensors
            future_mask = np.arange(n) > i
            sensor_mask = sensors != sensors[i]
            combined_mask = future_mask & sensor_mask

            if not np.any(combined_mask):
                continue

            # Calculate date differences for valid candidates
            future_indices = np.where(combined_mask)[0]
            date_diffs = np.abs((dates[future_indices] - dates[i]).astype("timedelta64[D]").astype(int))

            # Filter by delay
            valid_future = future_indices[date_diffs <= delay]

            # Calculate combined coverage for all valid pairs
            for j in valid_future:
                combined_cov = coverage[i] + coverage[j]
                valid_pairs.append((combined_cov, i, j))

        if not valid_pairs:
            log.warning("No valid pairs found after filtering")
            return pd.DataFrame([])

        log.debug(f"Found {len(valid_pairs)} valid pairs")

        # Sort by combined coverage (descending)
        valid_pairs.sort(key=lambda x: -x[0])

        # Greedy selection
        used_indices = set()
        selected_pairs = []

        for cov, i, j in valid_pairs:
            if i not in used_indices and j not in used_indices:
                selected_pairs.extend([i, j])
                used_indices.update([i, j])

        log.info(f"Selected {len(selected_pairs)} images from {n} after duplicate filtering")
        return df.iloc[selected_pairs].reset_index(drop=True)

    def _compute_crop_coverage_dates(self, row):
        """
        Compute the expanded date range for crop_coverage filter mode.

        Uses the entity's start_date/end_date month-day combined with historical_seasons
        years to build the widest extraction window: from the start of the smallest
        historical year to the end of the largest historical year.

        Falls back to coverage_params start_date/end_date if the entity does not
        have these fields.

        Args:
            row (dict): Entity data, optionally containing start_date and end_date

        Returns:
            tuple: (expanded_start_date, expanded_end_date) as 'YYYY-MM-DD' strings

        Raises:
            ValueError: If start_date/end_date not found in entity data nor in params
        """
        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("CROP_COVERAGE")

        # Per-entity historical_seasons override, fall back to global params
        entity_hs = (
            self.get_entity_value(row, "historical_seasons")
            if self.has_entity_field(row, "historical_seasons")
            else None
        )
        historical_seasons = (
            validate_historical_seasons(entity_hs)
            if entity_hs is not None
            else self.coverage_params["historical_seasons"]
        )
        if entity_hs is not None:
            log.debug(f"Entity {entity_id}: using per-entity historical_seasons: {historical_seasons}")

        # Get entity start/end dates, fall back to params if not in entity data
        start_date_str = self.get_entity_value(row, "start_date")
        end_date_str = self.get_entity_value(row, "end_date")

        if not start_date_str:
            start_date_str = self.coverage_params.get("start_date")
            log.debug(f"Entity {entity_id}: no start_date in entity, falling back to params: {start_date_str}")

        if not end_date_str:
            end_date_str = self.coverage_params.get("end_date")
            log.debug(f"Entity {entity_id}: no end_date in entity, falling back to params: {end_date_str}")

        if not start_date_str or not end_date_str:
            raise ValueError(
                f"Entity {entity_id} must have 'start_date' and 'end_date' "
                f"either in entity data or in coverage parameters when using filter='crop_coverage'."
            )

        # Parse dates (handle both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats)
        start_dt = datetime.strptime(str(start_date_str)[:10], "%Y-%m-%d")
        end_dt = datetime.strptime(str(end_date_str)[:10], "%Y-%m-%d")

        min_year = min(historical_seasons)
        max_year = max(historical_seasons)

        # Build expanded dates using month-day from entity dates + historical years
        expanded_start = start_dt.replace(year=min_year).strftime("%Y-%m-%d")
        expanded_end = end_dt.replace(year=max_year).strftime("%Y-%m-%d")

        log.info(
            f"Entity {entity_id}: crop_coverage date range "
            f"{expanded_start} -> {expanded_end} "
            f"(historical_seasons={historical_seasons})"
        )

        return expanded_start, expanded_end

    def _filter_crop_coverage_periods(self, df, row):
        """
        Post-filter coverage results to keep only images within the entity's
        start_date/end_date window for each historical year.

        The API call uses the widest date range (min year start → max year end),
        so this method narrows results to the actual crop season period replicated
        across each historical year.

        Args:
            df (pd.DataFrame): Coverage DataFrame with a 'date' column (YYYY-MM-DD strings)
            row (dict): Entity data used to resolve start_date/end_date

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("CROP_COVERAGE")

        # Per-entity historical_seasons override, fall back to global params
        entity_hs = (
            self.get_entity_value(row, "historical_seasons")
            if self.has_entity_field(row, "historical_seasons")
            else None
        )
        historical_seasons = (
            validate_historical_seasons(entity_hs)
            if entity_hs is not None
            else self.coverage_params["historical_seasons"]
        )

        # Resolve start/end dates with same fallback logic as _compute_crop_coverage_dates
        start_date_str = self.get_entity_value(row, "start_date")
        end_date_str = self.get_entity_value(row, "end_date")
        if not start_date_str:
            start_date_str = self.coverage_params.get("start_date")
        if not end_date_str:
            end_date_str = self.coverage_params.get("end_date")

        start_dt = datetime.strptime(str(start_date_str)[:10], "%Y-%m-%d")
        end_dt = datetime.strptime(str(end_date_str)[:10], "%Y-%m-%d")

        # Build a mask: keep rows whose date falls within [start_mm_dd, end_mm_dd] for any historical year
        image_dates = pd.to_datetime(df["date"])
        mask = pd.Series(False, index=df.index)

        for year in historical_seasons:
            year_start = start_dt.replace(year=year)
            year_end = end_dt.replace(year=year)
            mask = mask | ((image_dates >= year_start) & (image_dates <= year_end))

        original_count = len(df)
        filtered_df = df[mask].reset_index(drop=True)

        log.info(
            f"Entity {entity_id}: crop_coverage period filter "
            f"{original_count} -> {len(filtered_df)} images "
            f"(keeping {start_dt.strftime('%m-%d')} to {end_dt.strftime('%m-%d')} "
            f"for years {historical_seasons})"
        )

        return filtered_df

    @requires_token
    @require_coverage_params
    @cache_single_entity("coverage_params")
    def process_single_entity_coverage(self, row, params=None):
        """
        Process coverage extraction for a single entity with retry logic.

        Args:
            row (dict): Entity data containing id and geometry
            params (dict, optional): Coverage parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.coverage_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Step 1: Validate geometry
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except Exception as e:
            log.error(f"Validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 1b: For crop_coverage filter, embed expanded dates into the row
        # so get_satellite_coverage_by_geometry picks them up (thread-safe)
        if params.get("filter") == "crop_coverage":
            try:
                expanded_start, expanded_end = self._compute_crop_coverage_dates(row)
                row = dict(row)  # ensure mutable copy
                row["_coverage_start_date"] = expanded_start
                row["_coverage_end_date"] = expanded_end
            except Exception as e:
                log.error(f"crop_coverage date computation failed for entity {entity_id}: {e}")
                return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 2: Call coverage API by geometry
        def _call_api():
            return self.get_satellite_coverage_by_geometry(row)

        # Step 3: Call API with retry
        try:
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            coverage_df = self.format_coverage_json(raw_json, entity_id=entity_id)

            # Step 3b: Post-filter to keep only dates within crop season per historical year
            if params.get("filter") == "crop_coverage" and coverage_df is not None and not coverage_df.empty:
                coverage_df = self._filter_crop_coverage_periods(coverage_df, row)

            # Step 4: Package response
            if coverage_df is None or coverage_df.empty:
                log.warning(f"No imagery found for entity {entity_id}")
                return {"data": None, "error": {"message": "No imagery found", "entity_id": entity_id}}
            else:
                return {"data": normalize_with_metadata(row, coverage_df), "error": None}

        except Exception as e:
            log.error(f"Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_coverage_params
    def process_entity_coverage_bulk_parallel(
        self,
        entity_list,
        params=None,
        max_workers=5,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="coverage",
        flm_params=None,
        recalibration=False,
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity coverage using threads + progress bar,
        with optional fail-safe retry, filter capabilities, and caching.

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain 'id' and 'geometry'.
            params (dict, optional): Coverage parameters override.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final CSV results.
            partial_frequency (int, optional): How often to save partial results.
            fail_safe (bool, optional): If True, per-entity failures are captured to
                ``failed_ids`` instead of aborting the run. Error tolerance only -- it does
                NOT change which entities are processed. To reprocess just the IDs a
                previous run recorded, set ``retry_failed_only`` (see BaseExtractor).
            filter_column (str, optional): Column name to filter entities by.
            filter_value (any, optional): Value to filter in the filter_column.
            filter_type (str, optional): 'exclude' to skip rows with filter_value, 'include' to process only rows with filter_value. Defaults to 'exclude'.
            merge_existing (str, optional): Merge strategy - 'auto', 'preserve', or 'mark'. If None, uses instance default.
            skip_export (bool, optional): If True, skip final export (useful when chaining extractions). Default: False.
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "coverage".
            flm_params (dict, optional): FLM parameters. If provided, sets recalibration=True and stores on self.
            recalibration (bool, optional): If True, uses recalibration API endpoint. Default: False.
            use_cache (bool, optional): Override instance-level cache setting. Default: None (uses self.use_cache).

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics.
                  When cache is active, also includes cache_hit and cache_miss counts.
        """
        # Route through cache wrapper if caching is enabled
        cache_enabled = use_cache if use_cache is not None else self.use_cache
        if cache_enabled and self.cache_key_columns is not None:
            return self._bulk_with_cache(
                entity_list=entity_list,
                bulk_method=self._process_entity_coverage_bulk_parallel_inner,
                params=self.coverage_params,
                use_cache=True,
                params_kw=params,
                max_workers=max_workers,
                output_path=output_path,
                partial_frequency=partial_frequency,
                fail_safe=fail_safe,
                filter_column=filter_column,
                filter_value=filter_value,
                filter_type=filter_type,
                merge_existing=merge_existing,
                skip_export=skip_export,
                prefix=prefix,
                flm_params=flm_params,
                recalibration=recalibration,
                generate_report=generate_report,
                report_options=report_options,
            )

        return self._process_entity_coverage_bulk_parallel_inner(
            entity_list=entity_list,
            params_kw=params,
            max_workers=max_workers,
            output_path=output_path,
            partial_frequency=partial_frequency,
            fail_safe=fail_safe,
            filter_column=filter_column,
            filter_value=filter_value,
            filter_type=filter_type,
            merge_existing=merge_existing,
            skip_export=skip_export,
            prefix=prefix,
            flm_params=flm_params,
            recalibration=recalibration,
            generate_report=generate_report,
            report_options=report_options,
        )

    @requires_token
    @require_coverage_params
    def _process_entity_coverage_bulk_parallel_inner(
        self,
        entity_list,
        params_kw=None,
        max_workers=5,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="coverage",
        flm_params=None,
        recalibration=False,
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        if flm_params is not None:
            flm_params["recalibration"] = True
            self.flm_params = flm_params
            recalibration = True

        # Apply recalibration flag to coverage_params so the API method picks it up
        if recalibration:
            self.coverage_params["recalibration"] = True

        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk coverage extraction: {prefix}")
        log.info("=" * 60)
        log.info(f"Total input entities: {len(entity_list)}")
        log.info(f"Max workers: {max_workers}")
        log.info(f"Partial frequency: {partial_frequency}")
        log.info(f"Fail-safe mode: {fail_safe}")
        log.info(f"Recalibration mode: {recalibration}")

        # Use instance default if not specified
        if merge_existing is None:
            merge_existing = self.merge_existing

        # Validate merge_existing value
        valid_modes = ["auto", "preserve", "mark"]
        if merge_existing not in valid_modes:
            error_msg = f"Invalid merge_existing='{merge_existing}'. Choose from: {valid_modes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(f"Merge strategy: {merge_existing}")

        # Apply filter to skip certain entities
        log.debug("Applying entity filters...")
        filtered_entity_list, skipped_entities, skip_count = filter_entities(
            entity_list, filter_column, filter_value, filter_type
        )

        if skip_count > 0:
            log.info(f"Filtered out {skip_count} entities by {filter_column}={filter_value} ({filter_type})")

        # Resume mode is explicit (self.retry_failed_only) and never implied
        # by fail_safe -- see BaseExtractor._resolve_retry_entity_list.
        filtered_entity_list = self._resolve_retry_entity_list(
            filtered_entity_list, prefix, fail_safe=fail_safe, log=log
        )

        all_rows = []
        global_errors = []
        successful_calculations = 0
        total_calculations = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        single_entity_method = self.process_single_entity_coverage

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(single_entity_method, row.to_dict(), params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🔍 Processing {prefix.title()}", unit="entity") as pbar:
                for future in as_completed(future_to_id):
                    total_calculations += 1
                    entity_id = future_to_id[future]

                    try:
                        result = future.result()
                        df = result.get("data")
                        error = result.get("error")

                        if df is not None and not df.empty:
                            successful_calculations += 1
                            all_rows.append(df)
                            buffer_rows.append(df)
                            log.debug(f"Entity {entity_id}: Success")
                        else:
                            failed_ids.append(entity_id)
                            log.debug(f"Entity {entity_id}: No data returned")

                        if error:
                            error_record = {"entity_id": entity_id, **error}
                            global_errors.append(error_record)
                            buffer_errors.append(error_record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)
                            log.warning(f"Entity {entity_id}: Error - {error.get('message', 'Unknown error')}")

                    except Exception as e:
                        error_record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(error_record)
                        buffer_errors.append(error_record)
                        failed_ids.append(entity_id)
                        log.error(f"Thread error for entity {entity_id}: {e}", exc_info=True)

                    # Partial export
                    if self.partial_path and partial_frequency > 0 and total_calculations % partial_frequency == 0:
                        log.debug(f"Saving partial results at {total_calculations} entities")
                        partial_df = pd.concat(buffer_rows, ignore_index=True) if buffer_rows else pd.DataFrame()
                        export_results(
                            results_df=partial_df,
                            errors=buffer_errors,
                            output_path=self.partial_path,
                            prefix=prefix,
                            partial=True,
                            verbose=False,
                        )
                        log.info(f"💾 Partial results saved: {len(buffer_rows)} rows")
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        log.info("=" * 60)
        log.success(f"⏱️ Total processing time: {elapsed_time:.2f} seconds")
        log.success(f"✅ Successful calculations: {successful_calculations}/{total_calculations}")
        log.info(f"❌ Failed calculations: {total_calculations - successful_calculations}/{total_calculations}")
        log.info("=" * 60)

        print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Successful calculations: {successful_calculations}/{total_calculations}")

        # Concatenate all DataFrames
        log.debug("Concatenating all results...")
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        log.info(f"Results DataFrame shape: {results_df.shape}")

        # Merge back skipped entities to maintain all input rows in output
        log.debug("Merging with skipped entities...")
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df, skipped_entities_df=skipped_entities, merge_mode=merge_existing, verbose=True
        )

        # Finalize extraction: export, store failed IDs, cleanup partials
        log.info("Finalizing extraction...")
        results_df, finalization_status = self._finalize_extraction(
            results_df=results_df,
            global_errors=global_errors,
            failed_ids=failed_ids,
            output_path=output_path,
            prefix=prefix,
            skip_export=skip_export,
            verbose=True,
            generate_report=generate_report,
            report_options=report_options,
            extraction_stats={
                "total_entities": len(entity_list),
                "total_calculations": total_calculations,
                "successful": successful_calculations,
                "failed": total_calculations - successful_calculations,
                "elapsed_seconds": elapsed_time,
                "parameters": getattr(self, "coverage_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk coverage extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }

    # Backward compatibility aliases
    def get_satellite_coverage_by_geometry_recalibration(self, entity_data: dict):
        self.coverage_params["recalibration"] = True
        return self.get_satellite_coverage_by_geometry(entity_data)

    def get_satellite_coverage_by_geometry_safe_recalibration(self, entity_data: dict) -> dict:
        self.coverage_params["recalibration"] = True
        return self.get_satellite_coverage_by_geometry_safe(entity_data)

    def process_single_entity_coverage_recalibration(self, row, params=None):
        self.coverage_params["recalibration"] = True
        return self.process_single_entity_coverage(row, params)

    def process_entity_coverage_bulk_parallel_recalibration(self, *args, **kwargs):
        kwargs["recalibration"] = True
        return self.process_entity_coverage_bulk_parallel(*args, **kwargs)
