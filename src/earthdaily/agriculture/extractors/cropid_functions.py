# processor_emergence_functions.py - Emergence Functions
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps

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
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

available_mask_types = {"InSeason", "EndSeason", "PreSeason"}
available_modes = {"history", "year", "historical_season", "full_history"}


def require_cropid_params(func):
    """Decorator to ensure crop id parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "cropid_params") or self.cropid_params is None:
            error_msg = "❌ No crop id parameters found. Call setup_cropid_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class cropidExtractor(BaseExtractor):
    """
    Extracts crop identification analytics for agricultural entities.

    AI-powered crop type classification using multi-temporal satellite imagery. Supports
    end-of-season and in-season mask types, historical crop rotation analysis, and
    configurable confidence thresholds.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_cropid.ipynb

    Args (setup_cropid_parameters):
        begin_year (int): Start year for historical analysis. Default: 2020
        end_year (int): End year for analysis. Default: 2025
        mask_type (str): Mask type ('EndSeason', 'InSeason', 'PreSeason'). Default: 'EndSeason'
        limit_nb_crop (int): Max number of crop candidates per field. Default: 1
        crop_mask_percent (int): Minimum crop mask percentage. Default: 50
        mode (str): Output mode — 'history' (one row per (year, crop)),
            'year' (filter to entity crop, long-form),
            'historical_season' (comma-separated string of matching years),
            'full_history' (wide-form, one row per entity, one column per year
            in begin_year..end_year filled with the top crop code per year).
            Default: 'history'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop (required for 'year' and 'historical_season' modes)

    Output columns:
        Depend on ``mode``:
        - 'history'           : entity_id, year, eda_crop_code, crop_name, raw_crop_name, crop_mask_percent
        - 'year'              : entity_id, year, eda_crop_code
        - 'historical_season' : entity_id, historical_season (comma-separated years)
        - 'full_history'      : entity_id + one column per year in [begin_year, end_year],
                                filled with the top crop code (highest cropMaskPercent) per year
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.cropid_params = None

        # Get resource URL
        self.cropid_url = agro_urls["crop_id_url"][self.env]

        self.logger.debug(f"Crop ID URL configured: {self.cropid_url}")
        self.logger.info(f"🌾 cropidExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for cropidExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for cropidExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_cropid_parameters(
        self,
        begin_year=2020,
        end_year=2025,
        mask_type="EndSeason",
        limit_nb_crop=1,
        crop_mask_percent=50,
        mode="history",
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for Crop ID extraction.

        Args:
            begin_year (int): Starting year (YYYY format, e.g., 2020)
            end_year (int): Ending year (YYYY format, e.g., 2025)
            mask_type (str): Type of crop mask. Options: 'InSeason', 'EndSeason', 'PreSeason'
            limit_nb_crop (int): Maximum number of crops to return per field
            crop_mask_percent (int): Minimum percentage of field covered by crop (0-100)
            mode (str): Processing mode — one of:
                - 'history'           : all crops/years (long-form, one row per (year, crop))
                - 'year'              : filtered to entity crop (long-form)
                - 'historical_season' : comma-separated string of matching years
                - 'full_history'      : wide-form pivot, one row per entity with
                                         one column per year (begin_year..end_year)
                                         filled with the top crop code per year
            partial_frequency (int): How often to save partial results during bulk processing

        Returns:
            None
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring crop ID parameters...")

        # Mask type validation
        if mask_type not in available_mask_types:
            error_msg = f"Invalid mask_type '{mask_type}'. Choose from: {available_mask_types}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Year validation
        current_year = datetime.now().year
        if not (2000 <= begin_year <= current_year + 1):
            error_msg = f"Invalid begin_year={begin_year}. Must be between 2000 and {current_year + 1}."
            log.error(error_msg)
            raise ValueError(error_msg)

        if not (2000 <= end_year <= current_year + 1):
            error_msg = f"Invalid end_year={end_year}. Must be between 2000 and {current_year + 1}."
            log.error(error_msg)
            raise ValueError(error_msg)

        if end_year < begin_year:
            error_msg = f"end_year ({end_year}) cannot be earlier than begin_year ({begin_year})."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Limit number of crops validation
        if not isinstance(limit_nb_crop, int) or limit_nb_crop < 1:
            error_msg = f"Invalid limit_nb_crop={limit_nb_crop}. Must be a positive integer."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Crop mask percent validation
        if not (0 <= crop_mask_percent <= 100):
            error_msg = f"Invalid crop_mask_percent={crop_mask_percent}. Must be between 0 and 100."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Mode validation
        if mode not in available_modes:
            error_msg = f"Invalid mode '{mode}'. Choose from: {available_modes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Partial frequency validation
        if not isinstance(partial_frequency, int) or partial_frequency < 0:
            error_msg = f"Invalid partial_frequency={partial_frequency}. Must be a non-negative integer."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.apply_cache_setting(use_cache)

        # Store parameters
        self.cropid_params = {
            "begin_year": begin_year,
            "end_year": end_year,
            "mask_type": mask_type,
            "limit_nb_crop": limit_nb_crop,
            "crop_mask_percent": crop_mask_percent,
            "mode": mode,
            "partial_frequency": partial_frequency,
        }

        # Configure cache key columns for cropid results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "year"]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("✅ Crop ID parameters configured successfully")
        log.debug(f"Parameters: {self.cropid_params}")

        print("🌾 Crop ID parameters configured:")
        for k, v in self.cropid_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_cropid_params
    def get_cropid_api(self, entity_data: dict):
        """
        Request crop id for an entity

        Args:
            entity_data (dict): entity data (id, geometry, crop)

        Returns:
            dict: JSON response from the crop ID API
        """
        params = self.cropid_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

        # Step 1: Input validation
        log.debug(f"Validating entity {entity_id}")

        if not self.has_entity_field(entity_data, "id"):
            error_msg = "❌ entity_data must include an 'id' field."
            log.error(error_msg)
            raise ValueError(error_msg)

        # ==== Geometry validation ====
        if not self.has_entity_field(entity_data, "geometry"):
            error_msg = f"❌ entity_data for {entity_id} must include a 'geometry' field in WKT format."
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
        url = f"{self.cropid_url}"
        full_url = url
        log.debug(f"API URL: {full_url}")

        # --- Payload ---
        payload = {
            "geometryWkt": geometry,
            "filters": {
                "BeginYear": params["begin_year"],
                "EndYear": params["end_year"],
                "Products": [params["mask_type"]],
                "LimitNbCrop": params["limit_nb_crop"],
                "CropMaskPercentMin": params["crop_mask_percent"],
            },
        }

        log.debug(
            f"Requesting crop ID with filters: {params['begin_year']}-{params['end_year']}, mask={params['mask_type']}"
        )

        # Step 3: Request data
        log.info(f"Requesting crop ID data for entity {entity_id}")
        log.debug(f"API URL: {full_url}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(
                full_url,
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,  # type: ignore[arg-type]
                timeout=60,
            )
            response.raise_for_status()
            json_response = response.json()
            log.success(f"✅ Crop ID data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting crop ID for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_cropid_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_cropid_api().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain a 'geometry' key in WKT format.

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "entity_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response_json = self.get_cropid_api(entity_data)
            log.debug(f"Successful safe API call for entity {entity_id}")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}

    @require_cropid_params
    def format_cropid_json(self, response_json, entity_data: dict = None):
        """
        Format crop ID API response into a DataFrame.

        Args:
            response_json (dict): JSON response from get_cropid_api
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame: DataFrame with columns: entity_id (optional), year, crop_name,
                        raw_crop_name, crop_mask_percent, eda_crop_code
        """
        log = self.get_contextualized_logger("FORMAT")
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )

        log.debug(f"Formatting crop ID data for entity {entity_id} (mode: history)")

        rows = []

        # Extract results by year
        results_by_year = response_json.get("resultsByYear", {})
        log.debug(f"Processing {len(results_by_year)} years of crop data for entity {entity_id}")

        for year, year_data in results_by_year.items():
            crops = year_data.get("crops", [])

            for crop in crops:
                row = {
                    "year": int(year),
                    "eda_crop_code": crop.get("edaCropCode"),
                    "crop_name": crop.get("cropName"),
                    "raw_crop_name": crop.get("rawCropName"),
                    "crop_mask_percent": crop.get("cropMaskPercent"),
                }

                # Add entity_id if entity_data provided
                if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
                    row["entity_id"] = self.get_entity_value(entity_data, "id")

                rows.append(row)

        # Create DataFrame
        cropid_df = pd.DataFrame(rows)

        # Reorder columns if entity_id exists
        if "entity_id" in cropid_df.columns:
            cropid_df = cropid_df[
                ["entity_id", "year", "eda_crop_code", "crop_name", "raw_crop_name", "crop_mask_percent"]
            ]
        else:
            cropid_df = cropid_df[["year", "eda_crop_code", "crop_name", "raw_crop_name", "crop_mask_percent"]]

        # Sort by year
        cropid_df = cropid_df.sort_values("year").reset_index(drop=True)

        log.debug(f"Created crop ID DataFrame with {len(cropid_df)} rows for entity {entity_id}")

        return cropid_df

    @staticmethod
    def _matching_years(response_json: dict, entity_crop_code: str) -> list[int]:
        """
        Return the sorted list of years where the API response reports the given
        ``edaCropCode``. Deduplicated when a year lists the same crop multiple times.

        Pure function — no logging, no entity validation. The public format methods
        wrap it with their own validation + log lines.
        """
        return sorted(
            {
                int(year)
                for year, ydata in response_json.get("resultsByYear", {}).items()
                for crop in ydata.get("crops", [])
                if crop.get("edaCropCode") == entity_crop_code
            }
        )

    @require_cropid_params
    def format_cropid_year_json(self, response_json, entity_data: dict):
        """
        Format crop ID API response and filter to only years matching the entity's crop.

        Args:
            response_json (dict): JSON response from get_cropid_api
            entity_data (dict): Original entity data with 'crop' as string (e.g., {'crop': 'SOYBEANS'})

        Returns:
            pd.DataFrame: DataFrame with entity_id, year, eda_crop_code for matching crops only
        """
        log = self.get_contextualized_logger("FORMAT")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        log.debug(f"Formatting crop ID data for entity {entity_id} (mode: year)")

        if not self.has_entity_field(entity_data, "crop"):
            error_msg = "❌ entity_data must include a 'crop' field for filtering"
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_crop_code = self.get_entity_value(entity_data, "crop")

        if not entity_crop_code:
            error_msg = "❌ crop field is empty in entity_data"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Filtering for crop: {entity_crop_code}")

        years = self._matching_years(response_json, entity_crop_code)

        if not years:
            log.warning(f"No years found matching crop: {entity_crop_code} for entity {entity_id}")
            print(f"⚠️ No years found matching crop: {entity_crop_code}")
            return pd.DataFrame(columns=["entity_id", "year", "eda_crop_code"])

        df = pd.DataFrame(
            [
                {
                    "entity_id": self.get_entity_value(entity_data, "id"),
                    "year": y,
                    "eda_crop_code": entity_crop_code,
                }
                for y in years
            ]
        )
        log.debug(f"Created filtered crop ID DataFrame with {len(df)} rows for entity {entity_id}")
        return df

    @require_cropid_params
    def format_cropid_historical_season_json(self, response_json, entity_data: dict) -> str:
        """
        Format crop ID API response to return a comma-separated string of years
        where the crop matches the entity's crop.

        Args:
            response_json (dict): JSON response from get_cropid_api
            entity_data (dict): Entity data with 'crop' as string (e.g., {'crop': 'SOYBEANS'})

        Returns:
            str: Comma-separated years matching the entity crop (e.g., "2020,2022,2024,2025")
        """
        log = self.get_contextualized_logger("FORMAT")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        log.debug(f"Formatting crop ID data for entity {entity_id} (mode: historical_season)")

        if not self.has_entity_field(entity_data, "crop"):
            error_msg = "❌ entity_data must include a 'crop' field for filtering"
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_crop_code = self.get_entity_value(entity_data, "crop")

        if not entity_crop_code:
            error_msg = "❌ crop field is empty in entity_data"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Filtering for crop: {entity_crop_code}")

        years = self._matching_years(response_json, entity_crop_code)

        if not years:
            log.warning(f"No years found matching crop: {entity_crop_code} for entity {entity_id}")
            return ""

        result = ",".join(str(y) for y in years)
        log.debug(f"Historical seasons for entity {entity_id}: {result}")
        return result

    @require_cropid_params
    def format_cropid_full_history_json(self, response_json, entity_data: dict):
        """
        Format crop ID API response into a wide-form, single-row DataFrame:
        one column per year in ``[begin_year, end_year]``, filled with the top
        crop code (highest ``cropMaskPercent``) for that year — or ``NaN`` if
        the year is absent from the response.

        Year columns are string-typed (e.g. ``"2020"``) so the wide frame plays
        nicely with parquet and bulk concat across entities. When multiple crops
        are reported for one year (``limit_nb_crop > 1`` upstream), only the top
        crop by ``cropMaskPercent`` is kept; tie-break is the API's natural
        ordering.

        Args:
            response_json (dict): JSON response from get_cropid_api
            entity_data (dict): Entity data — must include 'id'

        Returns:
            pd.DataFrame: 1-row DataFrame with columns ``entity_id`` + one
                column per year between ``begin_year`` and ``end_year`` inclusive.
        """
        log = self.get_contextualized_logger("FORMAT")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        log.debug(f"Formatting crop ID data for entity {entity_id} (mode: full_history)")

        begin_year = self.cropid_params["begin_year"]
        end_year = self.cropid_params["end_year"]
        year_columns = [str(y) for y in range(begin_year, end_year + 1)]

        # Top crop per year, by cropMaskPercent. None for years without data
        # so reindexing later turns them into NaN automatically.
        top_crop_per_year: dict[str, str | None] = {}
        for year, year_data in response_json.get("resultsByYear", {}).items():
            crops = year_data.get("crops", []) or []
            if not crops:
                continue
            top_crop = max(crops, key=lambda c: c.get("cropMaskPercent", 0))
            top_crop_per_year[str(year)] = top_crop.get("edaCropCode")

        row: dict[str, object] = {"entity_id": entity_id}
        for col in year_columns:
            row[col] = top_crop_per_year.get(col, pd.NA)

        df = pd.DataFrame([row], columns=["entity_id", *year_columns])
        log.debug(
            f"Created full-history DataFrame for entity {entity_id}: "
            f"{len(year_columns)} year columns, "
            f"{sum(1 for c in year_columns if pd.notna(row[c]))} populated"
        )
        return df

    @requires_token
    @require_cropid_params
    @cache_single_entity("cropid_params")
    def process_single_entity_cropid(self, row, params=None):
        """
        Process a single entity for crop ID extraction.

        Args:
            row (dict): Entity data with 'id', 'geometry', and 'crop'
            params (dict, optional): Override cropid_params

        Returns:
            dict: Contains 'data' (DataFrame) and 'error' (dict or None)
        """
        if params is None:
            params = self.cropid_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Step 1: validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except Exception as e:
            log.error(f"Validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Prepare entity_data for API call
        entity_data = {
            self.get_mapped_column("id"): entity_id,
            self.get_mapped_column("geometry"): geometry,
            self.get_mapped_column("crop"): self.get_entity_value(row, "crop"),
        }

        # Step 2: define API call wrapper
        def _call_api():
            return self.get_cropid_api(entity_data)

        try:
            # Step 3: Call API with retry logic
            log.info(f"Calling crop ID API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Step 4: Format response based on mode
            mode = params.get("mode", "history")  # Default to history
            log.debug(f"Processing with mode: {mode}")

            if mode == "history":
                # Use format_cropid_json for all years/crops
                cropid_df = self.format_cropid_json(raw_json, entity_data)
            elif mode == "year":
                # Use format_cropid_year_json for filtered years matching entity crop
                cropid_df = self.format_cropid_year_json(raw_json, entity_data)
            elif mode == "historical_season":
                # Return string of years matching entity crop
                historical_years = self.format_cropid_historical_season_json(raw_json, entity_data)
                if not historical_years:
                    log.warning(f"No historical seasons found for entity {entity_id}")
                    return {"data": None, "error": {"message": "No historical seasons found", "entity_id": entity_id}}
                cropid_df = pd.DataFrame(
                    [{"entity_id": self.get_entity_value(entity_data, "id"), "historical_season": historical_years}]
                )
            elif mode == "full_history":
                # Wide-form: one row per entity, one column per year in window.
                cropid_df = self.format_cropid_full_history_json(raw_json, entity_data)
            else:
                error_msg = f"Invalid mode '{mode}'. Must be one of: {sorted(available_modes)}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Step 5: Check if results exist
            if cropid_df is None or cropid_df.empty:
                log.warning(f"No crop ID results found for entity {entity_id}")
                return {"data": None, "error": {"message": "No crop ID results found", "entity_id": entity_id}}
            else:
                log.success(f"✅ Successfully processed entity {entity_id}: {len(cropid_df)} crop records")
                return {"data": normalize_with_metadata(row, cropid_df), "error": None}

        except Exception as e:
            # Step 6: Handle error cleanly
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_cropid_params
    def process_cropid_bulk_extraction_parallel(
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
        prefix="cropid",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of cropid requests using threads + progress bar,
        with optional fail-safe retry, partial save capabilities, and caching.

        Args:
            entity_list (pd.DataFrame): Entities to process (must contain 'id', 'geometry', and 'crop').
            params (dict, optional): Override monitoring parameters.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final results and error logs.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "cropid".
            use_cache (bool, optional): Override instance-level cache setting. Default: None (uses self.use_cache).

        Returns:
            dict:
                {
                    "results_df": pd.DataFrame,
                    "global_errors": list[dict],
                    "total_entities": int,
                    "total_calculations": int,
                    "successful_calculations": int,
                    "failed_calculations": int,
                    "failed_ids": list[str]
                }
        """
        # Route through cache wrapper if caching is enabled
        cache_enabled = use_cache if use_cache is not None else self.use_cache
        if cache_enabled and self.cache_key_columns is not None:
            return self._bulk_with_cache(
                entity_list=entity_list,
                bulk_method=self._process_cropid_bulk_extraction_parallel_inner,
                params=self.cropid_params,
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
                generate_report=generate_report,
                report_options=report_options,
            )

        return self._process_cropid_bulk_extraction_parallel_inner(
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
            generate_report=generate_report,
            report_options=report_options,
        )

    @requires_token
    @require_cropid_params
    def _process_cropid_bulk_extraction_parallel_inner(
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
        prefix="cropid",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk crop ID extraction: {prefix}")
        log.info("=" * 60)
        log.info(f"Total input entities: {len(entity_list)}")
        log.info(f"Max workers: {max_workers}")
        log.info(f"Partial frequency: {partial_frequency}")
        log.info(f"Fail-safe mode: {fail_safe}")

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

        log.info(f"Processing {len(filtered_entity_list)} entities in parallel...")
        print(f"🔄 Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_cropid, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🌾 Processing {prefix.title()}", unit="entity") as pbar:
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
                "parameters": getattr(self, "cropid_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk crop ID extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
