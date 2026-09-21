# processor_plantedarea_functions.py - Planted area Functions
import json
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

available_modes = {"PLANTED_AREA", "CONTROL"}


def require_planted_params(func):
    """Decorator to ensure planted area parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "planted_params") or self.planted_params is None:
            error_msg = "❌ No planted area parameters found. Call setup_planted_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class PlantedExtractor(BaseExtractor):
    """
    Extracts planted area estimation analytics for agricultural entities.

    Estimates planted acreage using crop identification and emergence data.
    Validates planting status and computes planted area percentages.

    Documentation: https://docs.earthdaily.com/agro/library/Planted_Area/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_coverage.ipynb

    Args (setup_planted_parameters):
        processor_mode (str): Processing mode - 'PLANTED_AREA' (planted area and percentage)
            or 'CONTROL' (compares the field against control_threshold). Default: 'PLANTED_AREA'
        emergence_date (str): Default emergence date in YYYY-MM-DD. Default: None
            (entities may override via the ``emergence_date`` row column)
        threshold (int): Decision threshold in days. Default: 120
        control_threshold (int | float): Percentage threshold, CONTROL mode only. Sent to the
            API divided by 100, so the default 4 becomes controlThreshold=0.04. Default: 4
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required);
        emergence_date (optional per-entity override of params.emergence_date —
            wire the EmergenceExtractor output in via column_mapping);
        crop, sowing_date (optional)

    Output columns:
        Varies by processor_mode.
        PLANTED_AREA: entity_id, planted_area_m2, planted_percentage
        CONTROL: entity_id, difference, control_threshold, control_result
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.planted_params = None

        # Get resource URL
        self.planted_url = agro_urls["planted_urls"][self.env]

        self.logger.debug(f"Planted area URL configured: {self.planted_url}")
        self.logger.info(f"🌱 PlantedExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for PlantedExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for PlantedExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_planted_parameters(
        self,
        processor_mode="PLANTED_AREA",
        emergence_date=None,
        threshold=120,
        control_threshold=4,
        publish_af=False,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for planted area extraction

        Args:
            processor_mode (str): Selection between 'PLANTED_AREA' or 'CONTROL'
            emergence_date (str, optional): Default emergence date in YYYY-MM-DD.
                May be overridden per entity via the ``emergence_date`` row column
                (column_mapping-aware, so upstream Emergence output can be wired in).
                If omitted here, every row must provide its own.
            threshold (int): Days between images before emergence and second image (default: 120)
            control_threshold (float): Percentage threshold for CONTROL mode (default: 4, used as 0.04)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring planted area parameters...")

        # Processor mode validation
        if processor_mode not in available_modes:
            error_msg = f"Invalid processor_mode '{processor_mode}'. Choose from: {available_modes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Emergence date is optional at setup-time — can be supplied per entity.
        # If provided here, validate format eagerly.
        if emergence_date:
            try:
                datetime.strptime(emergence_date, "%Y-%m-%d")
                log.debug(f"Default emergence date validated: {emergence_date}")
            except ValueError:
                error_msg = f"Invalid emergence_date format '{emergence_date}'. Use YYYY-MM-DD"
                log.error(error_msg)
                raise ValueError(error_msg)
        else:
            log.info(
                "No default emergence_date set — each entity row must supply "
                "an 'emergence_date' column (column_mapping-aware)."
            )

        # Threshold validation
        if not isinstance(threshold, int) or threshold <= 0:
            error_msg = f"Invalid threshold={threshold}. Must be a positive integer."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Control threshold validation
        if not isinstance(control_threshold, (int, float)) or control_threshold < 0:
            error_msg = f"Invalid control_threshold={control_threshold}. Must be a non-negative number."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Publish AF validation
        if not isinstance(publish_af, bool):
            error_msg = f"Invalid publish_af={publish_af}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.apply_cache_setting(use_cache)

        self.planted_params = {
            "processor_mode": processor_mode,
            "emergence_date": emergence_date,
            "threshold": threshold,
            "control_threshold": control_threshold,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("✅ Planted area parameters configured successfully")
        log.debug(f"Parameters: {self.planted_params}")

        print("🌱 Planted area parameters configured:")
        for k, v in self.planted_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_planted_params
    def get_planted_api(self, entity_data: dict):
        """
        Request planted area calculation for an entity

        Args:
            entity_data (dict): entity data (id, geometry)
        """
        log = self.get_contextualized_logger("API")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        params = self.planted_params

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

        # ==== Emergence date resolution: row (via column_mapping) overrides params ====
        emergence_date = self.get_entity_value(entity_data, "emergence_date")
        if emergence_date is None or (isinstance(emergence_date, float) and pd.isna(emergence_date)):
            emergence_date = params.get("emergence_date")
            source = "params"
        else:
            source = "row"

        if not emergence_date:
            error_msg = (
                f"Entity {entity_id}: emergence_date missing. Provide it on the "
                "row (column_mapping-aware) or pass emergence_date to setup_planted_parameters()."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate the resolved emergence_date format (row values may arrive as datetime/Timestamp)
        if isinstance(emergence_date, pd.Timestamp):
            emergence_date = emergence_date.strftime("%Y-%m-%d")
        try:
            datetime.strptime(emergence_date, "%Y-%m-%d")
        except ValueError:
            error_msg = f"Entity {entity_id}: invalid emergence_date '{emergence_date}'. Use YYYY-MM-DD."
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Entity {entity_id}: using emergence_date={emergence_date} (from {source})")

        # Step 2: API URL construction
        url = f"{self.planted_url}/launch"

        # --- Params dict ---
        params_list = [
            f"processorMode={params['processor_mode']}",
            f"emergenceDate={emergence_date}",
            f"threshold={params['threshold']}",
        ]

        # Add controlThreshold only for CONTROL mode
        if params["processor_mode"] == "CONTROL":
            # Convert percentage to decimal (e.g., 4 -> 0.04)
            control_threshold_decimal = params["control_threshold"] / 100
            params_list.append(f"controlThreshold={control_threshold_decimal}")
            log.debug(f"Using control threshold: {control_threshold_decimal}")

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
        log.info(f"Requesting planted area data for entity {entity_id}")
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
            log.success(f"✅ Planted area data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting planted area for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_planted_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_planted_api().
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
            response_json = self.get_planted_api(entity_data)
            log.debug(f"Successful safe API call for entity {entity_id}")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            # HTTPError from requests includes response object
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}

    @require_planted_params
    def format_planted_json(self, response_planted_json, params=None):
        """
        Normalize Planted Area API response into a clean pandas DataFrame.
        Supports two processor modes:
        - PLANTED_AREA: Returns planted area in m² and percentage
        - CONTROL: Returns difference, control threshold, and result boolean

        Args:
            response_planted_json (dict): API response JSON.
            params (dict, optional): Planted area parameters (defaults to self.planted_params).

        Returns:
            pd.DataFrame: Flat normalized planted area data.
                        Columns depend on processor_mode.
        """
        if params is None:
            params = self.planted_params

        processor_mode = params.get("processor_mode", "").upper()
        log = self.get_contextualized_logger("FORMAT")

        if not isinstance(response_planted_json, dict):
            error_msg = "❌ response_planted_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        # For planted area, the response doesn't include entity_id directly
        # We'll need to track it separately in the processing pipeline
        entity_id = response_planted_json.get("id", None)

        rows = []

        # ============================================================
        # 🌱 1️⃣ PLANTED_AREA MODE
        # ============================================================
        if processor_mode == "PLANTED_AREA":
            if "planted_area" in response_planted_json and "planted_percentage" in response_planted_json:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "planted_area_m2": response_planted_json.get("planted_area"),
                        "planted_percentage": response_planted_json.get("planted_percentage"),
                    }
                )
                log.debug(f"Formatted PLANTED_AREA for entity {entity_id}")
            else:
                log.warning(f"Missing expected keys in PLANTED_AREA response for entity {entity_id}")
                print("⚠️ Missing expected keys in PLANTED_AREA response")

        # ============================================================
        # 🔍 2️⃣ CONTROL MODE
        # ============================================================
        elif processor_mode == "CONTROL":
            if "difference" in response_planted_json and "control_threshold" in response_planted_json:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "difference": response_planted_json.get("difference"),
                        "control_threshold": response_planted_json.get("control_threshold"),
                        "control_result": response_planted_json.get("result"),
                    }
                )
                log.debug(f"Formatted CONTROL for entity {entity_id}")
            else:
                log.warning(f"Missing expected keys in CONTROL response for entity {entity_id}")
                print("⚠️ Missing expected keys in CONTROL response")

        else:
            log.error(f"Unknown processor_mode: {processor_mode}")
            print(f"⚠️ Unknown processor_mode: {processor_mode}")
            return pd.DataFrame()

        # Convert to df
        df = pd.DataFrame(rows)
        log.debug(f"Created DataFrame with {len(df)} rows for entity {entity_id}")

        return df

    @requires_token
    @require_planted_params
    @cache_single_entity("planted_params")
    def process_single_entity_planted(self, row, params=None):
        """
        Process planted area extraction for a single entity with retry logic.

        Args:
            row (dict): Entity data containing id and geometry
            params (dict, optional): Planted area parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.planted_params

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

        # Step 2: define API call wrapper
        # Pass the full row so column_mapping-aware fields (e.g. emergence_date) survive.
        def _call_api():
            return self.get_planted_api(row)

        try:
            log.info(f"Calling planted area API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Add entity_id to response for formatting
            raw_json["id"] = entity_id

            # Step 4: package response and errors
            planted_df = self.format_planted_json(raw_json)

            if planted_df is None or planted_df.empty:
                log.warning(f"No planted area results found for entity {entity_id}")
                return {"data": None, "error": {"message": "No planted area results found", "entity_id": entity_id}}
            else:
                log.success(f"✅ Successfully processed entity {entity_id}")
                return {"data": normalize_with_metadata(row, planted_df), "error": None}

        except Exception as e:
            # ✅ Step 6: Handle error cleanly
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_planted_params
    def process_planted_bulk_extraction_parallel(
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
        prefix="planted",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of planted area requests using threads + progress bar,
        with optional fail-safe retry and partial save capabilities.

        Args:
            entity_list (pd.DataFrame): Entities to process (must contain 'id' and 'geometry').
            params (dict, optional): Override planted area parameters.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "planted".
            use_cache (bool, optional): Enable/disable caching for this call.

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
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_planted_bulk_extraction_parallel_inner,
                entity_list=entity_list,
                params=self.planted_params,
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
        return self._process_planted_bulk_extraction_parallel_inner(
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
    @require_planted_params
    def _process_planted_bulk_extraction_parallel_inner(
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
        prefix="planted",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of planted area requests using threads + progress bar,
        with optional fail-safe retry and partial save capabilities.
        """
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk planted area extraction: {prefix}")
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
                executor.submit(self.process_single_entity_planted, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🌱 Processing {prefix.title()}", unit="entity") as pbar:
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
                "parameters": getattr(self, "planted_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk planted area extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
