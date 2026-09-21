# processor_emergence_functions.py - Emergence Functions
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

available_type = {"INSEASON", "HISTORICAL", "DELAY"}
available_crops = {"CORN", "SECOND CORN", "SOYBEANS", "SUGARCANE", "COTTON", "OTHERS"}


def require_emergence_params(func):
    """Decorator to ensure emergence parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "emergence_params") or self.emergence_params is None:
            error_msg = "❌ No emergence parameters found. Call setup_emergence_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class EmergenceExtractor(BaseExtractor):
    """
    Detects crop emergence dates from imagery derived vegetation index time series.

    For each field (id + geometry + crop) returns an estimated emergence date
    with a confidence score and status. Supports INSEASON, HISTORICAL and DELAY
    detection modes and LR / MR data sources. The season window comes from
    setup (``season_*``, ``year``), never from a per-entity sowing date.

    Documentation: https://docs.earthdaily.com/agro/library/Emergence/
    Notebook:      EDAgriculture_Emergence_Processor_Function_Dev.ipynb
    API endpoint:  {agro_urls["emergence_url"][env]}/... (env-dependent)

    Args (setup_emergence_parameters):
        emergence_type (str): Detection type ('INSEASON', 'HISTORICAL', 'DELAY'). Default: 'INSEASON'
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        year (int): Target year. Default: 2025
        data_source (str): Data source ('LR', 'MR'). Default: 'LR'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    For HISTORICAL, a per-field ``historical_seasons`` input (via column_mapping) lists the
    calendar years a field actually grew the crop, as a comma-separated string
    ("2020,2022,2024") or a list of ints. When supplied, emergence_year_N is kept only for
    those years (others nulled) and a separate ``avg_emergence_matching_years`` column holds
    the average recomputed over the retained years (MM-DD). ``historical_average_emergence``
    (the raw API average over all 5 seasons) is left untouched.

    Entity fields (via column_mapping):
        id, geometry, crop (required); historical_seasons (optional — HISTORICAL only).
        sowing_date is NOT read: the season window comes from setup.

    Output columns:
        entity_id, emergence_date, confidence, status, emergence_year_1..5, historical_average_emergence, avg_emergence_matching_years (MM-DD)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.emergence_params = None

        # Get resource URL
        self.emergence_url = agro_urls["emergence_url"][self.env]

        self.logger.debug(f"Emergence URL configured: {self.emergence_url}")
        self.logger.info(f"🌱 EmergenceExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for EmergenceExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for EmergenceExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    @staticmethod
    def validate_crop(crop: str, available_crops: set):
        """
        Validate and normalize the crop type for emergence requests.

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

    def setup_emergence_parameters(
        self,
        emergence_type="INSEASON",
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
        Configure parameters for Emergence extraction

        Args:
            emergence_type (str): Selection between INSEASON, HISTORICAL or DELAY
            season_duration (int): Crop cycle duration in days
            season_start_day (int): Crop cycle start day (1-31)
            season_start_month (int): Crop cycle start month (1-12)
            year (int): Year to analyze
            data_source (str): Imagery type - 'LR' (low resolution) or 'MR' (medium resolution)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring emergence parameters...")

        # Emergence type validation
        if emergence_type not in available_type:
            error_msg = f"Invalid emergence type '{emergence_type}'. Choose from: {available_type}"
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

        self.emergence_params = {
            "emergence_type": emergence_type,
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "year": year,
            "data_source": data_source,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
        }

        # Configure cache key columns for emergence results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("✅ Emergence parameters configured successfully")
        log.debug(f"Parameters: {self.emergence_params}")

        print("🌍 Emergence parameters configured:")
        for k, v in self.emergence_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_emergence_params
    def get_emergence_api(self, entity_data: dict):
        """
        Request Emergence for an entity

        Args:
            entity_data (dict): entity data (id, geometry, crop)
        """
        params = self.emergence_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

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
            geometry = validate_wkt(geometry)  # validates and converts shapely objects to WKT
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
        url = f"{self.emergence_url}/launch"

        # --- Params dict ---
        params_list = [
            f"emergenceType={params['emergence_type']}",
            f"seasonDuration={params['season_duration']}",
            f"seasonStartDay={params['season_start_day']}",
            f"seasonStartMonth={params['season_start_month']}",
            f"year={params['year']}",
            f"dataSource={params['data_source']}",
            f"crop={crop}",
        ]
        # --- Full URL ---
        full_url = f"{url}?{'&'.join(params_list)}"
        log.debug(f"API URL: {url}")

        # --- Payload ---
        payload = {"geometry": geometry}
        # Add id to payload only if publish_af is True
        if params.get("publish_af", False):
            payload["id"] = f"SeasonField:{self.get_entity_value(entity_data, 'id')}@LEGACY_ID_NA"
            log.debug("Including entity ID in payload (publish_af=True)")

        # Step 3: Request data
        log.info(f"Requesting emergence data for entity {entity_id}")
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
            log.success(f"✅ Emergence data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting emergence for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_emergence_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_emergence_api().
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
            response_json = self.get_emergence_api(entity_data)
            log.debug(f"Successful safe API call for entity {entity_id}")
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

    @require_emergence_params
    def format_emergence_json(self, response_emergence_json, params=None, historical_seasons=None):
        """
        Normalize Emergence API response into a clean pandas DataFrame.
        Supports multiple emergence types:
        - INSEASON
        - HISTORICAL
        - DELAY

        Args:
            response_emergence_json (dict): API response JSON with keys 'id' and 'data'.
            params (dict, optional): Emergence parameters (defaults to self.emergence_params).
            historical_seasons (str | list | None, optional): HISTORICAL only — the calendar
                years a field actually grew the target crop, as a comma-separated string
                ("2020,2022,2024") or a list of ints. When supplied, emergence_year_N is kept
                only for the retained years (others set to None) and a separate
                ``avg_emergence_matching_years`` column holds the average recomputed over the
                retained years. When None (default), no filtering is applied and the new column
                is null. ``historical_average_emergence`` is left untouched in all cases.

        Returns:
            pd.DataFrame: Flat normalized emergence data.
                        Columns depend on emergence_type.
        """
        if params is None:
            params = self.emergence_params

        emergence_type = params.get("emergence_type", "").upper()
        log = self.get_contextualized_logger("FORMAT")

        if not isinstance(response_emergence_json, dict):
            error_msg = "❌ response_emergence_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = response_emergence_json.get("id", None)
        data = response_emergence_json.get("data", None)

        if data is None:
            log.warning(f"No 'data' key found in response for entity {entity_id}")
            print(f"⚠️ No 'data' key found in response for entity {entity_id}")
            return pd.DataFrame()

        log.debug(f"Formatting emergence data for entity {entity_id}, type: {emergence_type}")
        rows = []

        # ============================================================
        # 🌿 1️⃣ INSEASON EMERGENCE
        # ============================================================
        if emergence_type == "INSEASON":
            if isinstance(data, dict):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "emergence_date": safe_parse_date(data.get("EmergenceDate")),
                        "emergence_status": data.get("EmergenceStatus"),
                        "confirmation_status": data.get("ConfirmationStatus"),
                    }
                )
                log.debug(f"Formatted INSEASON emergence for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for INSEASON emergence: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for INSEASON emergence: {type(data)}")

        # ============================================================
        # 🌾 2️⃣ HISTORICAL EMERGENCE
        # ============================================================
        elif emergence_type == "HISTORICAL":
            if isinstance(data, dict):
                # emergence_year_N maps to calendar year (year - N) for N = 1..5.
                # year may arrive as a str (e.g. '2025') from setup — coerce for arithmetic.
                target_year = params.get("year")
                try:
                    target_year = int(target_year) if target_year is not None else None
                except (ValueError, TypeError):
                    target_year = None
                seasons = parse_matching_seasons(historical_seasons)

                # Parse each season's emergence date, then keep only the matching years.
                emergence_by_year = {}
                for n in range(1, 6):
                    parsed = safe_parse_date(data.get(f"Emergence_year-{n}"))
                    if seasons is not None:
                        calendar_year = target_year - n if target_year is not None else None
                        if calendar_year is None or calendar_year not in seasons:
                            parsed = None
                    emergence_by_year[n] = parsed

                record = {
                    "entity_id": entity_id,
                    "emergence_year_1": emergence_by_year[1],
                    "emergence_year_2": emergence_by_year[2],
                    "emergence_year_3": emergence_by_year[3],
                    "emergence_year_4": emergence_by_year[4],
                    "emergence_year_5": emergence_by_year[5],
                    # Raw API average over all 5 seasons — left untouched by the season filter.
                    "historical_average_emergence": data.get("Hist_avg_emergence"),
                }

                # Separate column: average recomputed over the retained (matching) seasons.
                # Null when no season set was supplied (no filtering) or none were retained.
                if seasons is None:
                    record["avg_emergence_matching_years"] = None
                else:
                    record["avg_emergence_matching_years"] = average_dates_mmdd(
                        [emergence_by_year[n] for n in range(1, 6)]
                    )

                rows.append(record)
                log.debug(f"Formatted HISTORICAL emergence for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for HISTORICAL emergence: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for HISTORICAL emergence: {type(data)}")

        # ============================================================
        # ⏱️ 3️⃣ DELAY EMERGENCE
        # ============================================================
        elif emergence_type == "DELAY":
            if isinstance(data, dict):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "emergence_date": safe_parse_date(data.get("EmergenceDate")),
                        "average_emergence_date": data.get("AverageEmergenceDate"),
                        "emergence_delay": data.get("EmergenceDelay"),
                    }
                )
                log.debug(f"Formatted DELAY emergence for entity {entity_id}")
            else:
                log.warning(f"Unexpected 'data' format for DELAY emergence: {type(data)}")
                print(f"⚠️ Unexpected 'data' format for DELAY emergence: {type(data)}")

        else:
            log.error(f"Unknown emergence_type: {emergence_type}")
            print(f"⚠️ Unknown emergence_type: {emergence_type}")
            return pd.DataFrame()

        # Convert to df
        df = pd.DataFrame(rows)
        log.debug(f"Created DataFrame with {len(df)} rows for entity {entity_id}")

        return df

    @requires_token
    @require_emergence_params
    @cache_single_entity("emergence_params")
    def process_single_entity_emergence(self, row, params=None):
        """
        Process emergence extraction for a single entity with retry logic.

        Args:
            row (dict): Entity data containing id, geometry, and crop
            params (dict, optional): Emergence parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.emergence_params

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
            return self.get_emergence_api(
                {
                    self.get_mapped_column("id"): entity_id,
                    self.get_mapped_column("crop"): crop,
                    self.get_mapped_column("geometry"): geometry,
                }
            )

        try:
            log.info(f"Calling emergence API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            # (the API may not echo it back when publish_af is False)
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Per-field matching seasons (HISTORICAL only) flow in via column_mapping,
            # exactly like crop; absent column -> None -> no filtering.
            historical_seasons = self.get_entity_value(row, "historical_seasons", default=None)

            # Step 4: package response and errors
            emergence_df = self.format_emergence_json(raw_json, historical_seasons=historical_seasons)

            if emergence_df is None or emergence_df.empty:
                log.warning(f"No emergence results found for entity {entity_id}")
                return {"data": None, "error": {"message": "No emergence results found", "entity_id": entity_id}}
            else:
                log.success(f"✅ Successfully processed entity {entity_id}")
                return {"data": normalize_with_metadata(row, emergence_df), "error": None}

        except Exception as e:
            # ✅ Step 6: Handle error cleanly
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_emergence_params
    def process_emergence_bulk_extraction_parallel(
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
        prefix="emergence",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of emergence requests using threads + progress bar,
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "emergence".
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
                bulk_method=self._process_emergence_bulk_extraction_parallel_inner,
                params=self.emergence_params,
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

        return self._process_emergence_bulk_extraction_parallel_inner(
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
    @require_emergence_params
    def _process_emergence_bulk_extraction_parallel_inner(
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
        prefix="emergence",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk emergence extraction: {prefix}")
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
                executor.submit(self.process_single_entity_emergence, row, params): self.get_entity_value(row, "id")
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
                "parameters": getattr(self, "emergence_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk emergence extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
