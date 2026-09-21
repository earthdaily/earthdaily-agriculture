# processor_harvest_functions.py - Harvest Functions
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

import pandas as pd

#  Third-Party Libraries
import requests
from tqdm import tqdm

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.api_utils import (
    average_dates_mmdd,
    export_results,
    filter_entities,
    normalize_with_metadata,
    parse_matching_seasons,
    retry_with_backoff_no_retry_on_400,
    safe_parse_date,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

available_type = {"INSEASON_HARVEST", "HISTORICAL_HARVEST", "HARVEST_READINESS"}
available_crops = {"CORN", "SECOND CORN", "SOYBEANS", "SUGARCANE", "OTHERS"}


def require_harvest_params(func):
    """Decorator to ensure harvest parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "harvest_params") or self.harvest_params is None:
            error_msg = "❌ No harvest parameters found. Call setup_harvest_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class HarvestExtractor(BaseExtractor):
    """
    Extracts harvest detection analytics for agricultural entities.

    Identifies harvest dates and monitors harvest progress using satellite-based temporal
    analysis. Supports in-season, historical and harvest-readiness detection.

    Documentation: https://docs.earthdaily.com/agro/library/Harvest_Detection/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_harvest.ipynb

    Args (setup_harvest_parameters):
        harvest_type (str): Detection type - 'INSEASON_HARVEST', 'HISTORICAL_HARVEST' or
            'HARVEST_READINESS'. Default: 'INSEASON_HARVEST'
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        year (int): Target year. Default: 2025
        data_source (str): Data source ('LR', 'MR'). Default: 'LR'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    For HISTORICAL_HARVEST, a per-field ``historical_seasons`` input (via column_mapping) lists
    the calendar years a field actually grew the crop, as a comma-separated string
    ("2020,2022,2024") or a list of ints. When supplied, harvest_year_N is kept only for those
    years (others nulled) and a separate ``avg_harvest_matching_years`` column holds the average
    recomputed over the retained years (MM-DD). ``historical_harvest_average`` (the raw API
    average over all 5 seasons) is left untouched.

    Entity fields (via column_mapping):
        id, geometry, crop (required); historical_seasons (optional — HISTORICAL_HARVEST only).
        sowing_date is NOT read: the season window comes from setup.

    Output columns:
        Varies by harvest_type.
        INSEASON_HARVEST: entity_id, harvest_date, harvest_status
        HISTORICAL_HARVEST: entity_id, harvest_year_1..5, historical_harvest_average,
            avg_harvest_matching_years (MM-DD)
        HARVEST_READINESS: entity_id, harvest_readiness_date, is_ready
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.harvest_params = None

        # Get resource URL
        self.harvest_url = agro_urls["harvest_url"][self.env]

        self.logger.debug(f"Harvest URL configured: {self.harvest_url}")
        self.logger.info(f"🌾 HarvestExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for HarvestExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for HarvestExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    @staticmethod
    def validate_crop(crop: str, available_crops: set):
        """
        Validate and normalize the crop type for harvest requests.

        Args:
            crop (str): The crop name to validate.
            available_crops (list): List of accepted crop names (expected in uppercase).

        Raises:
            ValueError: If the crop is missing, invalid, or not a string.

        Returns:
            str: The validated crop name, normalized to uppercase.
        """
        if not crop:
            raise ValueError("❌ Missing crop value. The 'crop' field is required.")

        if not isinstance(crop, str):
            raise ValueError(f"❌ Invalid crop type '{type(crop)}'. Must be a string.")

        # Normalize to uppercase
        normalized_crop = crop.strip().upper()

        # Validate against allowed list
        allowed = [c.upper() for c in available_crops]
        if normalized_crop not in allowed:
            raise ValueError(f"❌ Invalid crop '{crop}'. Choose from: {available_crops}")
        return normalized_crop

    def setup_harvest_parameters(
        self,
        harvest_type="INSEASON_HARVEST",
        season_duration=120,
        season_start_day=1,
        season_start_month=4,
        year=2025,
        data_source="LR",
        publish_af=False,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for harvest extraction

        Args:
            harvest_type (str): One of 'INSEASON_HARVEST' (harvest date + status for the
                current season), 'HISTORICAL_HARVEST' (harvest date for each of the last 5
                seasons) or 'HARVEST_READINESS' (readiness date + is_ready flag)
            season_duration (int): Crop cycle duration in days
            season_start_day (int): Crop cycle start day (1-31)
            season_start_month (int): Crop cycle start month (1-12)
            year (int): Year to analyze
            data_source (str): Imagery type - 'LR' (low resolution) or 'MR' (medium resolution)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring harvest parameters...")

        # harvest type validation
        if harvest_type not in available_type:
            error_msg = f"Invalid harvest type '{harvest_type}'. Choose from: {available_type}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Month validation
        if not (1 <= season_start_month <= 12):
            error_msg = f"Invalid season_start_month={season_start_month}. Must be between 1 and 12."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Day validation
        if not (1 <= season_start_day <= 31):
            error_msg = f"Invalid season_start_day={season_start_day}. Must be between 1 and 31."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Data source validation
        available_sources = {"LR", "MR"}
        if data_source not in available_sources:
            error_msg = f"Invalid data_source '{data_source}'. Choose from: {available_sources}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Publish AF validation
        if not isinstance(publish_af, bool):
            error_msg = f"Invalid publish_af={publish_af}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.apply_cache_setting(use_cache)

        self.harvest_params = {
            "harvest_type": harvest_type,
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "year": year,
            "data_source": data_source,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
        }

        # Configure cache key columns for harvest results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("✅ Harvest parameters configured successfully")
        log.debug(f"Parameters: {self.harvest_params}")

        print("🌍 harvest parameters configured:")
        for k, v in self.harvest_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_harvest_params
    def get_harvest_api(self, entity_data: dict):
        """
        Request harvest for an entity

        Args:
            entity_data (dict): entity data (id, geometry, crop)
        """
        log = self.get_contextualized_logger("API")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        params = self.harvest_params

        # Step 1: Input validation
        log.debug(f"Validating entity {entity_id}")

        if not self.has_entity_field(entity_data, "id"):
            error_msg = "❌ entity_data must include a 'id' field."
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

        # ==== Crop validation ====
        try:
            crop = self.validate_crop(self.get_entity_value(entity_data, "crop"), available_crops)
            log.debug(f"Crop validated for entity {entity_id}: {crop}")
        except ValueError as e:
            log.error(f"Crop validation failed for entity {entity_id}: {e}")
            raise

        # Step 2: API URL construction
        url = f"{self.harvest_url}/launch"

        # --- Params dict ---
        params_list = [
            f"harvestType={params['harvest_type']}",
            f"seasonDuration={params['season_duration']}",
            f"seasonStartDay={params['season_start_day']}",
            f"seasonStartMonth={params['season_start_month']}",
            f"year={params['year']}",
            f"dataSource={params['data_source']}",
            f"crop={crop}",
        ]
        # --- Full URL ---
        full_url = f"{url}?{'&'.join(params_list)}"
        log.debug(f"API URL: {full_url}")

        # --- Payload ---
        payload = {"geometry": geometry}
        # Add id to payload only if publish_af is True
        if params.get("publish_af", False):
            payload["id"] = f"SeasonField:{self.get_entity_value(entity_data, 'id')}@LEGACY_ID_NA"
            log.debug("Including entity ID in payload (publish_af=True)")

        # Step 3: Request data
        log.info(f"Requesting harvest data for entity {entity_id}")
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
                data=json.dumps(payload),
                timeout=60,
            )
            response.raise_for_status()
            json_response = response.json()
            log.success(f"✅ Harvest data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting harvest for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_harvest_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_harvest_api().
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
            response_json = self.get_harvest_api(entity_data)
            log.debug(f"Successful safe API call for entity {entity_id}")
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            # HTTPError from requests includes response object
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}

    @require_harvest_params
    def format_harvest_json(self, response_harvest_json, params=None, historical_seasons=None):
        """
        Normalize Harvest API response into a clean pandas DataFrame.
        Supports multiple harvest types:
        - INSEASON_HARVEST
        - HISTORICAL_HARVEST
        - HARVEST_READINESS

        Args:
            response_harvest_json (dict): API response JSON with keys 'id' and 'data'.
            params (dict, optional): Harvest parameters (defaults to self.harvest_params).
            historical_seasons (str | list | None, optional): HISTORICAL_HARVEST only — the
                calendar years a field actually grew the target crop, as a comma-separated
                string ("2020,2022,2024") or a list of ints. When supplied, harvest_year_N is
                kept only for the retained years (others set to None) and a separate
                ``avg_harvest_matching_years`` column holds the average recomputed over the
                retained years (MM-DD). When None (default), no filtering is applied and the
                new column is null. ``historical_harvest_average`` is left untouched in all cases.

        Returns:
            pd.DataFrame: Flat normalized harvest data.
                        Columns depend on harvest_type.
        """
        if params is None:
            params = self.harvest_params

        harvest_type = params.get("harvest_type", "").upper()
        log = self.get_contextualized_logger("FORMAT")

        if not isinstance(response_harvest_json, dict):
            error_msg = "❌ response_harvest_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = response_harvest_json.get("id", None)
        data = response_harvest_json.get("data", None)

        if data is None:
            log.warning(f"No 'data' key found in response for entity {entity_id}")
            print(f"⚠️ No 'data' key found in response for entity {entity_id}")
            return pd.DataFrame()

        rows = []

        # ============================================================
        # 🌾 1️⃣ INSEASON HARVEST
        # ============================================================
        if harvest_type == "INSEASON_HARVEST":
            if isinstance(data, dict):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "harvest_date": safe_parse_date(data.get("HarvestDate")),
                        "harvest_status": data.get("HarvestStatus"),
                    }
                )
                log.debug(f"Formatted INSEASON_HARVEST for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for INSEASON_HARVEST: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for INSEASON_HARVEST: {type(data)}")

        # ============================================================
        # 📊 2️⃣ HISTORICAL HARVEST
        # ============================================================
        elif harvest_type == "HISTORICAL_HARVEST":
            if isinstance(data, dict):
                # harvest_year_N maps to calendar year (year - N) for N = 1..5.
                # year may arrive as a str (e.g. '2025') from setup — coerce for arithmetic.
                target_year = params.get("year")
                try:
                    target_year = int(target_year) if target_year is not None else None
                except (ValueError, TypeError):
                    target_year = None
                seasons = parse_matching_seasons(historical_seasons)

                # Parse each season's harvest date, then keep only the matching years.
                harvest_by_year = {}
                for n in range(1, 6):
                    parsed = safe_parse_date(data.get(f"harvest_year_{n}"))
                    if seasons is not None:
                        calendar_year = target_year - n if target_year is not None else None
                        if calendar_year is None or calendar_year not in seasons:
                            parsed = None
                    harvest_by_year[n] = parsed

                record = {
                    "entity_id": entity_id,
                    "harvest_year_1": harvest_by_year[1],
                    "harvest_year_2": harvest_by_year[2],
                    "harvest_year_3": harvest_by_year[3],
                    "harvest_year_4": harvest_by_year[4],
                    "harvest_year_5": harvest_by_year[5],
                    # Raw API average over all 5 seasons — left untouched by the season filter.
                    "historical_harvest_average": data.get("historical_harvest_average"),
                }

                # Separate column: average recomputed over the retained (matching) seasons.
                # Null when no season set was supplied (no filtering) or none were retained.
                if seasons is None:
                    record["avg_harvest_matching_years"] = None
                else:
                    record["avg_harvest_matching_years"] = average_dates_mmdd([harvest_by_year[n] for n in range(1, 6)])

                rows.append(record)
                log.debug(f"Formatted HISTORICAL_HARVEST for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for HISTORICAL_HARVEST: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for HISTORICAL_HARVEST: {type(data)}")

        # ============================================================
        # ⏰ 3️⃣ HARVEST READINESS
        # ============================================================
        elif harvest_type == "HARVEST_READINESS":
            if isinstance(data, dict):
                readiness_date = data.get("date")
                # Check for placeholder date (0001-01-01 means not ready)
                parsed_date = safe_parse_date(readiness_date)

                rows.append(
                    {
                        "entity_id": entity_id,
                        "harvest_readiness_date": parsed_date,
                        "is_ready": parsed_date is not None and readiness_date != "0001-01-01",
                    }
                )
                log.debug(f"Formatted HARVEST_READINESS for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for HARVEST_READINESS: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for HARVEST_READINESS: {type(data)}")

        else:
            log.error(f"Unknown harvest_type: {harvest_type}")
            print(f"⚠️ Unknown harvest_type: {harvest_type}")
            return pd.DataFrame()

        # Convert to df
        df = pd.DataFrame(rows)
        log.debug(f"Created DataFrame with {len(df)} rows for entity {entity_id}")

        return df

    @requires_token
    @require_harvest_params
    @cache_single_entity("harvest_params")
    def process_single_entity_harvest(self, row, params=None):
        """
        Process harvest extraction for a single entity with retry logic.

        Args:
            row (dict): Entity data containing id, geometry, and crop
            params (dict, optional): Harvest parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.harvest_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Step 1: validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")

            crop = self.validate_crop(self.get_entity_value(row, "crop"), available_crops)
            log.debug(f"Crop validated for entity {entity_id}: {crop}")
        except Exception as e:
            log.error(f"Validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 2: define API call wrapper
        def _call_api():
            return self.get_harvest_api(
                {
                    self.get_mapped_column("id"): entity_id,
                    self.get_mapped_column("crop"): crop,
                    self.get_mapped_column("geometry"): geometry,
                }
            )

        try:
            log.info(f"Calling harvest API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Per-field matching seasons (HISTORICAL_HARVEST only) flow in via column_mapping,
            # exactly like crop; absent column -> None -> no filtering.
            historical_seasons = self.get_entity_value(row, "historical_seasons", default=None)

            # Step 4: package response and errors
            harvest_df = self.format_harvest_json(raw_json, historical_seasons=historical_seasons)

            if harvest_df is None or harvest_df.empty:
                log.warning(f"No harvest results found for entity {entity_id}")
                return {"data": None, "error": {"message": "No harvest results found", "entity_id": entity_id}}
            else:
                log.success(f"✅ Successfully processed entity {entity_id}")
                return {"data": normalize_with_metadata(row, harvest_df), "error": None}

        except Exception as e:
            # ✅ Step 6: Handle error cleanly
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_harvest_params
    def process_harvest_bulk_extraction_parallel(
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
        prefix="harvest",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of harvest requests using threads + progress bar,
        with optional fail-safe retry, filter capabilities, and caching.

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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "harvest".
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
                bulk_method=self._process_harvest_bulk_extraction_parallel_inner,
                params=self.harvest_params,
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

        return self._process_harvest_bulk_extraction_parallel_inner(
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
    @require_harvest_params
    def _process_harvest_bulk_extraction_parallel_inner(
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
        prefix="harvest",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk harvest extraction: {prefix}")
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
                executor.submit(self.process_single_entity_harvest, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🌿 Processing {prefix.title()}", unit="entity") as pbar:
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
                "parameters": getattr(self, "harvest_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk harvest extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
