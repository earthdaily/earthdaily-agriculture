# processor_greenness_functions.py - Greenness Detection Functions
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
    safe_parse_date,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

available_crops = {"CORN", "SECOND CORN", "SOYBEANS", "SUGARCANE", "COTTON", "OTHERS"}
available_data_sources = {"LR", "MR"}


def require_greenness_params(func):
    """Decorator to ensure greenness parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "greenness_params") or self.greenness_params is None:
            error_msg = "No greenness parameters found. Call setup_greenness_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class GreennessExtractor(BaseExtractor):
    """
    Extracts crop greenness (vegetation vigor) analytics for agricultural entities.

    Monitors crop canopy development and vegetation vigor through greenness detection.
    Evaluates crop health status based on satellite-derived vegetation indices.

    Documentation: https://docs.earthdaily.com/agro/library/greenness/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_greenness.ipynb

    Args (setup_greenness_parameters):
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        year (int): Target year. Default: 2025
        sowing_date (str): Sowing date in YYYY-MM-DD format. Default: '2025-04-01'
        data_source (str): Data source ('LR', 'MR'). Default: 'LR'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, sowing_date (optional)

    Output columns:
        entity_id, greenness_date, confidence, status, + detection metadata
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.greenness_params = None

        # Get resource URL
        self.greenness_url = agro_urls["greenness_urls"][self.env]

        self.logger.debug(f"Greenness URL configured: {self.greenness_url}")
        self.logger.info(f"GreennessExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for GreennessExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for GreennessExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    @staticmethod
    def validate_crop(crop: str, available_crops: set):
        """
        Validate and normalize the crop type for greenness requests.

        Args:
            crop (str): The crop name to validate.
            available_crops (set): Set of accepted crop names (expected in uppercase).

        Raises:
            ValueError: If the crop is missing, invalid, or not a string.

        Returns:
            str: The validated crop name, normalized to uppercase.
        """
        if not crop:
            raise ValueError("Missing crop value. The 'crop' field is required.")

        if not isinstance(crop, str):
            raise ValueError(f"Invalid crop type '{type(crop)}'. Must be a string.")

        normalized_crop = crop.strip().upper()

        allowed = {c.upper() for c in available_crops}
        if normalized_crop not in allowed:
            raise ValueError(f"Invalid crop '{crop}'. Choose from: {available_crops}")
        return normalized_crop

    def setup_greenness_parameters(
        self,
        season_duration=120,
        season_start_day=1,
        season_start_month=4,
        year=2025,
        sowing_date="2025-04-01",
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
        Configure parameters for greenness detection extraction.

        Args:
            season_duration (int): Crop cycle duration in days
            season_start_day (int): Crop cycle start day (1-31)
            season_start_month (int): Crop cycle start month (1-12)
            year (int): Year to analyze
            sowing_date (str): Sowing date in YYYY-MM-DD format
            data_source (str): Imagery type - 'LR' (low resolution) or 'MR' (medium resolution)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring greenness parameters...")

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
        if data_source not in available_data_sources:
            error_msg = f"Invalid data_source '{data_source}'. Choose from: {available_data_sources}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Sowing date validation
        if not sowing_date or not isinstance(sowing_date, str):
            error_msg = "Invalid sowing_date. Must be a non-empty string in YYYY-MM-DD format."
            log.error(error_msg)
            raise ValueError(error_msg)

        try:
            datetime.strptime(sowing_date, "%Y-%m-%d")
        except ValueError:
            error_msg = f"Invalid sowing_date format '{sowing_date}'. Expected YYYY-MM-DD."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Publish AF validation
        if not isinstance(publish_af, bool):
            error_msg = f"Invalid publish_af={publish_af}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.apply_cache_setting(use_cache)

        self.greenness_params = {
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "year": year,
            "sowing_date": sowing_date,
            "data_source": data_source,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
        }

        # Configure cache key columns for greenness results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("Greenness parameters configured successfully")
        log.debug(f"Parameters: {self.greenness_params}")

        print("Greenness parameters configured:")
        for k, v in self.greenness_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_greenness_params
    def get_greenness_api(self, entity_data: dict):
        """
        Request greenness detection for an entity.

        Args:
            entity_data (dict): entity data (id, geometry, crop)
        """
        log = self.get_contextualized_logger("API")
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        params = self.greenness_params

        # Step 1: Input validation
        log.debug(f"Validating entity {entity_id}")

        if not self.has_entity_field(entity_data, "id"):
            error_msg = "entity_data must include a 'id' field."
            log.error(error_msg)
            raise ValueError(error_msg)

        # ==== Geometry validation ====
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

        # ==== Crop validation ====
        try:
            crop = self.validate_crop(self.get_entity_value(entity_data, "crop"), available_crops)
            log.debug(f"Crop validated for entity {entity_id}: {crop}")
        except ValueError as e:
            log.error(f"Crop validation failed for entity {entity_id}: {e}")
            raise

        # Step 2: API URL construction
        url = f"{self.greenness_url}/greenness-detection"

        # Use sowing_date from entity_data if available, otherwise from params
        sowing_date = self.get_entity_value(entity_data, "sowing_date", params["sowing_date"])

        # --- Query params ---
        query_params = {
            "season_duration": params["season_duration"],
            "season_start_day": params["season_start_day"],
            "season_start_month": params["season_start_month"],
            "year": params["year"],
            "sowing_date": sowing_date,
            "crop": crop,
            "data_source": params["data_source"],
        }

        log.debug(f"API URL: {url} with params: {query_params}")

        # --- Payload ---
        payload = {"geometry": geometry}
        # Add id to payload only if publish_af is True
        if params.get("publish_af", False):
            payload["id"] = f"SeasonField:{self.get_entity_value(entity_data, 'id')}@LEGACY_ID_NA"
            log.debug("Including entity ID in payload (publish_af=True)")

        # Step 3: Request data
        log.info(f"Requesting greenness data for entity {entity_id}")
        log.debug(f"API URL: {url}")
        log.debug(f"Query params: {query_params}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(
                url,
                params=query_params,
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
            log.success(f"Greenness data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"Timeout requesting greenness for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Unexpected error for entity {entity_id}: {e}")
            raise

    def get_greenness_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_greenness_api().
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
            response_json = self.get_greenness_api(entity_data)
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

    @require_greenness_params
    def format_greenness_json(self, response_greenness_json, params=None):
        """
        Normalize Greenness Detection API response into a clean pandas DataFrame.

        Args:
            response_greenness_json (dict): API response JSON.
            params (dict, optional): Greenness parameters (defaults to self.greenness_params).

        Returns:
            pd.DataFrame: Flat normalized greenness data.
        """
        if params is None:
            params = self.greenness_params

        log = self.get_contextualized_logger("FORMAT")

        if not isinstance(response_greenness_json, dict):
            error_msg = "response_greenness_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = response_greenness_json.get("id", None)
        data = response_greenness_json.get("data", None)

        if data is None:
            log.warning(f"No 'data' key found in response for entity {entity_id}")
            print(f"No 'data' key found in response for entity {entity_id}")
            return pd.DataFrame()

        rows = []

        if isinstance(data, dict):
            row = {"entity_id": entity_id}
            # Flatten all keys from the data dict
            for key, value in data.items():
                if isinstance(value, str):
                    parsed = safe_parse_date(value)
                    row[key] = parsed if parsed else value
                else:
                    row[key] = value
            rows.append(row)
            log.debug(f"Formatted greenness data for entity {entity_id}")
        elif isinstance(data, list):
            for item in data:
                row = {"entity_id": entity_id}
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, str):
                            parsed = safe_parse_date(value)
                            row[key] = parsed if parsed else value
                        else:
                            row[key] = value
                rows.append(row)
            log.debug(f"Formatted {len(rows)} greenness records for entity {entity_id}")
        else:
            log.warning(f"Unexpected 'data' format for greenness: {type(data)}")
            print(f"Unexpected 'data' format for greenness: {type(data)}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        log.debug(f"Created DataFrame with {len(df)} rows for entity {entity_id}")

        return df

    @requires_token
    @require_greenness_params
    @cache_single_entity("greenness_params")
    def process_single_entity_greenness(self, row, params=None):
        """
        Process greenness extraction for a single entity with retry logic.

        Args:
            row (dict): Entity data containing id, geometry, and crop
            params (dict, optional): Greenness parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.greenness_params

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
            entity_payload = {
                self.get_mapped_column("id"): entity_id,
                self.get_mapped_column("crop"): crop,
                self.get_mapped_column("geometry"): geometry,
            }
            # Pass sowing_date from row if available
            sowing = self.get_entity_value(row, "sowing_date")
            if sowing:
                entity_payload[self.get_mapped_column("sowing_date")] = sowing
            return self.get_greenness_api(entity_payload)

        try:
            log.info(f"Calling greenness API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Step 4: package response and errors
            greenness_df = self.format_greenness_json(raw_json)

            if greenness_df is None or greenness_df.empty:
                log.warning(f"No greenness results found for entity {entity_id}")
                return {"data": None, "error": {"message": "No greenness results found", "entity_id": entity_id}}
            else:
                log.success(f"Successfully processed entity {entity_id}")
                return {"data": normalize_with_metadata(row, greenness_df), "error": None}

        except Exception as e:
            log.error(f"Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_greenness_params
    def process_greenness_bulk_extraction_parallel(
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
        prefix="greenness",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of greenness requests using threads + progress bar,
        with optional fail-safe retry, partial save capabilities, and caching.

        Args:
            entity_list (pd.DataFrame): Entities to process (must contain 'id', 'geometry', and 'crop').
            params (dict, optional): Override greenness parameters.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "greenness".
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
                bulk_method=self._process_greenness_bulk_extraction_parallel_inner,
                params=self.greenness_params,
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

        return self._process_greenness_bulk_extraction_parallel_inner(
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
    @require_greenness_params
    def _process_greenness_bulk_extraction_parallel_inner(
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
        prefix="greenness",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk greenness extraction: {prefix}")
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
        print(f"Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_greenness, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"Processing {prefix.title()}", unit="entity") as pbar:
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
                        log.info(f"Partial results saved: {len(buffer_rows)} rows")
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        log.info("=" * 60)
        log.success(f"Total processing time: {elapsed_time:.2f} seconds")
        log.success(f"Successful calculations: {successful_calculations}/{total_calculations}")
        log.info(f"Failed calculations: {total_calculations - successful_calculations}/{total_calculations}")
        log.info("=" * 60)

        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        print(f"Successful calculations: {successful_calculations}/{total_calculations}")

        # Concatenate all DataFrames
        log.debug("Concatenating all results...")
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        log.info(f"Results DataFrame shape: {results_df.shape}")

        # Merge back skipped entities to maintain all input rows in output
        log.debug("Merging with skipped entities...")
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df, skipped_entities_df=skipped_entities, merge_mode=merge_existing, verbose=True
        )

        # Finalize extraction: export, store failed IDs, cleanup partials, report
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
                "parameters": getattr(self, "greenness_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("Bulk greenness extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
