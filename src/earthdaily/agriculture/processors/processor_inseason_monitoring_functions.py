# processor_inseason_monitoring_functions.py - In season monitoring Functions
import os
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

available_crops = {"CORN", "SECOND CORN", "SOYBEANS", "SUGARCANE", "COTTON", "OTHERS"}


def require_inseason_monitoring_params(func):
    """Decorator to ensure in season monitoring parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "inseason_monitoring_params"):
            self.logger.error("No inseason monitoring parameters found")
            raise RuntimeError(
                "❌ No inseason monitoring parameters found. Call setup_inseason_monitoring_parameters() first."
            )
        return func(self, *args, **kwargs)

    return wrapper


class InSeasonMonitoringExtractor(BaseExtractor):
    """
    Extracts in-season crop monitoring analytics for agricultural entities.

    Combines satellite and weather data for real-time crop performance monitoring.
    Runs against either resolution (LR or MR) with configurable season windows.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_InSeasonMonitoring.ipynb

    Args (setup_inseason_monitoring_parameters):
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        year (str): Target year. Default: '2025'
        data_source (str): Data source - 'LR' or 'MR'. A single source per run (the API
            takes one dataSource value); run twice to cover both. Default: 'LR'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, sowing_date (optional)

    Output columns:
        entity_id, + in-season monitoring metrics and alerts
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # ✅ Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.inseason_monitoring_params = None

        # Specific API endpoint for coverage
        self.inseason_monitoring_url = agro_urls["inseason_monitoring_url"][self.env]

        self.logger.info(f"🛰️ InSeasonMonitoringExtractor initialized for env: {self.env}")
        self.logger.debug(f"API endpoint: {self.inseason_monitoring_url}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for InSeasonExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.info("🔑 Refreshing API token for InSeasonMonitoringExtractor...")
        try:
            new_token, exp_time = EDAuthenticator.get_new_token(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.api_username,
                password=self.api_password,
                env=self.env,
            )
            self.logger.info("✅ Token refreshed successfully")
            self.logger.debug(f"New token expires at: {datetime.fromtimestamp(exp_time)}")
            return new_token, exp_time
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {str(e)}")
            raise

    @staticmethod
    def validate_crop(crop: str, available_crops: set):
        """
        Validate and normalize the crop type for in-season monitoring requests.

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

    def setup_inseason_monitoring_parameters(
        self,
        season_duration=120,
        season_start_day=1,
        season_start_month=4,
        year="2025",
        data_source="LR",
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for in-season monitoring extraction

        Args:
            season_duration (int): Crop cycle duration in days
            season_start_month (int): Crop cycle start month (1-12)
            season_start_day (int): Crop cycle start day (1-31)
            year (int): Year to analyze
            data_source (str): Imagery type used to build the time series — 'LR' (low
                resolution) or 'MR' (medium resolution). One source per run; the API takes
                a single dataSource value. Default: 'LR'
            partial_frequency (int): How often to save partial results
        """
        self.logger.info("⚙️ Setting up in-season monitoring parameters...")

        # Month validation
        if not (1 <= season_start_month <= 12):
            self.logger.error(f"Invalid season_start_month={season_start_month}")
            raise ValueError(f"Invalid season_start_month={season_start_month}. Must be between 1 and 12.")

        # day validation
        if not (1 <= season_start_day <= 31):
            self.logger.error(f"Invalid season_start_day={season_start_day}")
            raise ValueError(f"Invalid season_start_day={season_start_day}. Must be between 1 and 31.")

        # Data source validation — a single source per run: the API query carries one
        # dataSource value, so a list would serialise straight into the URL.
        available_sources = {"LR", "MR"}
        if isinstance(data_source, (list, tuple, set)):
            error_msg = (
                f"data_source must be a single source string, got {data_source!r}. "
                f"Choose one of {available_sources} and run once per source if you need both."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if data_source not in available_sources:
            self.logger.error(f"Invalid data_source '{data_source}'")
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from: {available_sources}")

        self.apply_cache_setting(use_cache)

        self.inseason_monitoring_params = {
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "year": year,
            "data_source": data_source,
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

        self.logger.info("✅ In-season monitoring parameters configured:")
        for k, v in self.inseason_monitoring_params.items():
            self.logger.debug(f"   {k}: {v}")

        print("🌍 In season monitoring parameters configured:")
        for k, v in self.inseason_monitoring_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_inseason_monitoring_params
    def get_inseason_monitoring_api(self, entity_data: dict):
        """
        Request In-Season Monitoring for an entity

        Args:
            entity_data (dict): enttiy data (id, geometry, crop)
        """
        entity_id = self.get_entity_value(entity_data, "id")
        self.logger.debug(f"Requesting in-season monitoring for entity: {entity_id}")

        params = self.inseason_monitoring_params

        # Step 1: Input validation

        # ==== Geometry validation====
        if not self.has_entity_field(entity_data, "geometry"):
            self.logger.error(f"Entity {entity_id}: Missing geometry field")
            raise ValueError("❌ entity_data must include a 'geometry' field in WKT format.")

        geometry = self.get_entity_value(entity_data, "geometry")
        try:
            geometry = validate_wkt(geometry)
            self.logger.debug(f"Entity {entity_id}: Geometry validated")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Invalid geometry - {str(e)}")
            raise

        # ==== Crop validation====
        try:
            crop = self.validate_crop(self.get_entity_value(entity_data, "crop"), available_crops)
            self.logger.debug(f"Entity {entity_id}: Crop validated as {crop}")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Crop validation failed - {str(e)}")
            raise

        # Step 2: API URL

        url = f"{self.inseason_monitoring_url}/launch"

        # --- Params dict ---
        params_list = [
            f"seasonDuration={params['season_duration']}",
            f"seasonStartDay={params['season_start_day']}",
            f"seasonStartMonth={params['season_start_month']}",
            f"year={params['year']}",
            f"dataSource={params['data_source']}",
            f"crop={crop}",
        ]
        # --- Full URL ---
        full_url = f"{url}?{'&'.join(params_list)}"
        self.logger.debug(f"Entity {entity_id}: API URL constructed")

        # --- Headers ---
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # --- Payload ---
        payload = {"id": self.get_entity_value(entity_data, "id"), "geometry": geometry}

        # Step 3: Request data
        try:
            self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
            self.logger.debug(f"Entity {entity_id}: Payload: {payload}")
            response = requests.post(full_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.info(f"Entity {entity_id}: API request successful")
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | url={full_url} | payload={payload}")
            raise
        except requests.exceptions.Timeout:
            self.logger.error(f"Entity {entity_id}: Request timeout after 60s | url={full_url} | payload={payload}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Unexpected error - {str(e)} | url={full_url} | payload={payload}")
            raise

    def get_inseason_monitoring_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_inseason_monitoring_api().
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
        entity_id = self.get_entity_value(entity_data, "id")
        try:
            response_json = self.get_inseason_monitoring_api(entity_data)
            self.logger.debug(f"Entity {entity_id}: Safe API call succeeded")
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            # HTTPError from requests includes response object
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.warning(f"Entity {entity_id}: HTTP {status} error")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "seasonfield_id": entity_id}
        except Exception as e:
            self.logger.warning(f"Entity {entity_id}: Error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "seasonfield_id": entity_id}

    @require_inseason_monitoring_params
    def format_inseason_json(self, response_monitoring_json):
        """
        Unpack In-Season Monitoring JSON into flat rows.

        Args:
            response_json (dict): API response JSON with keys 'id' and 'data'

        Returns:
            list of dict: flattened rows
        """
        entity_id = response_monitoring_json.get("id")
        self.logger.debug(f"Entity {entity_id}: Formatting JSON response...")

        rows = []

        for entry in response_monitoring_json.get("data", []):
            rows.append(
                {
                    "entity_id": entity_id,
                    "season": entry.get("Season"),
                    "date": entry.get("RequestDate"),
                    "vegetation_index": entry.get("VegetationIndexValue"),
                    "cumulative_index": entry.get("CumulativeVegetationIndex"),
                    "emergence_date": entry.get("EmergenceDate"),
                    "emergence_status": entry.get("EmergenceStatus"),
                    "days_since_emergence": entry.get("DaysSinceEmergence"),
                    "cumulative_vs_avg": entry.get("CumulativeVegetationComparedToAverage"),
                    "historical_avg_cumulative": entry.get("HistoricalAverageCumulativeVegetationIndex"),
                    "delta": entry.get("Delta"),
                }
            )

        df = pd.DataFrame(rows)
        self.logger.debug(f"Entity {entity_id}: Formatted {len(df)} data points")
        return df

    @requires_token
    @require_inseason_monitoring_params
    @cache_single_entity("inseason_monitoring_params")
    def process_single_entity_inseason_monitoring(self, row, params=None):
        if params is None:
            params = self.inseason_monitoring_params

        entity_id = self.get_entity_value(row, "id")
        self.logger.debug(f"Entity {entity_id}: Processing single entity...")

        # Step 1: validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            crop = self.validate_crop(self.get_entity_value(row, "crop"), available_crops)
            self.logger.debug(f"Entity {entity_id}: Input validation passed")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Validation failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 2: define API call wrapper
        def _call_api():
            return self.get_inseason_monitoring_api(
                {"id": self.get_entity_value(row, "id"), "crop": crop, "geometry": geometry}
            )

        try:
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Step 4: package response and errors
            inseason_df = self.format_inseason_json(raw_json)

            if inseason_df is None or inseason_df.empty:
                self.logger.warning(f"Entity {entity_id}: No in-season monitoring results found")
                return {
                    "data": None,
                    "error": {"message": "No in-season monitoring results found", "entity_id": entity_id},
                }
            else:
                self.logger.info(f"Entity {entity_id}: Successfully processed with {len(inseason_df)} records")
                return {"data": normalize_with_metadata(row, inseason_df), "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_inseason_monitoring_params
    def process_inseason_bulk_extraction_parallel(
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
        prefix="inseason",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of In-Season Monitoring (ISM) requests using threads + progress bar,
        with optional fail-safe retry and partial save capabilities.

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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "inseason".
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
                bulk_method=self._process_inseason_bulk_extraction_parallel_inner,
                entity_list=entity_list,
                params=self.inseason_monitoring_params,
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
        return self._process_inseason_bulk_extraction_parallel_inner(
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
    @require_inseason_monitoring_params
    def _process_inseason_bulk_extraction_parallel_inner(
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
        prefix="inseason",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of In-Season Monitoring (ISM) requests using threads + progress bar,
        with optional fail-safe retry and partial save capabilities.
        """
        params = params_kw
        self.logger.info(f"🚀 Starting bulk in-season monitoring extraction for {len(entity_list)} entities")
        self.logger.debug(
            f"Parameters: max_workers={max_workers}, partial_frequency={partial_frequency}, fail_safe={fail_safe}"
        )

        # Use instance default if not specified
        if merge_existing is None:
            merge_existing = self.merge_existing

        # Validate merge_existing value
        valid_modes = ["auto", "preserve", "mark"]
        if merge_existing not in valid_modes:
            self.logger.error(f"Invalid merge_existing='{merge_existing}'")
            raise ValueError(f"Invalid merge_existing='{merge_existing}'. Choose from: {valid_modes}")

        self.logger.debug(f"Merge mode: {merge_existing}")

        # Apply filter to skip certain entities
        filtered_entity_list, skipped_entities, skip_count = filter_entities(
            entity_list, filter_column, filter_value, filter_type
        )

        if skip_count > 0:
            self.logger.info(
                f"📊 Applied filter: {skip_count} entities skipped, {len(filtered_entity_list)} to process"
            )

        # Resume mode is explicit (self.retry_failed_only) and never implied
        # by fail_safe -- see BaseExtractor._resolve_retry_entity_list.
        filtered_entity_list = self._resolve_retry_entity_list(
            filtered_entity_list, prefix, fail_safe=fail_safe, log=self.logger
        )

        all_rows = []
        global_errors = []
        successful_calculations = 0
        total_calculations = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        print(f"🔄 Processing {prefix.upper()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(self.process_single_entity_inseason_monitoring, row, params): self.get_entity_value(
                    row, "id"
                )
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🌿 Processing {prefix.upper()}", unit="entity") as pbar:
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
                            self.logger.debug(f"Entity {entity_id}: Added to results buffer")
                        else:
                            failed_ids.append(entity_id)
                            self.logger.warning(f"Entity {entity_id}: No data returned")

                        if error:
                            error_record = {"entity_id": entity_id, **error}
                            global_errors.append(error_record)
                            buffer_errors.append(error_record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)
                            self.logger.warning(
                                f"Entity {entity_id}: Error recorded - {error.get('message', 'Unknown')}"
                            )

                    except Exception as e:
                        error_record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(error_record)
                        buffer_errors.append(error_record)
                        failed_ids.append(entity_id)
                        self.logger.error(f"Entity {entity_id}: Thread execution error - {str(e)}")

                    # Partial export
                    if self.partial_path and partial_frequency > 0 and total_calculations % partial_frequency == 0:
                        partial_df = pd.concat(buffer_rows, ignore_index=True) if buffer_rows else pd.DataFrame()
                        self.logger.info(f"💾 Saving partial results at {total_calculations} entities...")
                        export_results(
                            results_df=partial_df,
                            errors=buffer_errors,
                            output_path=self.partial_path,
                            prefix=prefix,
                            partial=True,
                            verbose=False,
                        )
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        self.logger.info(f"⏱️ Total processing time: {elapsed_time:.2f} seconds")
        self.logger.info(f"✅ Successful: {successful_calculations}/{total_calculations} entities")
        self.logger.info(f"❌ Failed: {len(failed_ids)} entities")

        print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Successful calculations: {successful_calculations}/{total_calculations}")

        # Concatenate all DataFrames
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        self.logger.info(f"📊 Final results: {len(results_df)} total records from {successful_calculations} entities")

        # Merge back skipped entities to maintain all input rows in output
        if skip_count > 0:
            self.logger.info(f"🔀 Merging {skip_count} skipped entities back into results...")
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df, skipped_entities_df=skipped_entities, merge_mode=merge_existing, verbose=True
        )

        # Finalize extraction: export, store failed IDs, cleanup partials
        self.logger.info("📝 Finalizing extraction (export and cleanup)...")
        results_df, _ = self._finalize_extraction(
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
                "parameters": getattr(self, "inseason_monitoring_params", {}),
                "entity_df": entity_list,
            },
        )

        summary = {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }

        self.logger.info("✅ Bulk extraction complete")
        return summary

    def save_inseason_reports(self, results_df, global_errors, output_path):
        """
        Save extraction In-Season Monitoring extraction report as CSV.
        """
        self.logger.info(f"💾 Saving in-season monitoring reports to {output_path}...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ results_df is already a DataFrame, so save directly
        if isinstance(results_df, pd.DataFrame):
            output_df = results_df
        else:
            output_df = pd.DataFrame(results_df)

        results_path = os.path.join(output_path, f"inseason_results_{timestamp}.csv")
        output_df.to_csv(results_path, index=False)
        self.logger.info(f"📄 Results saved: {results_path} ({len(output_df)} rows)")
        print(f"📄 Results saved to: {results_path}")

        # Save errors if present
        errors_path = None
        if global_errors:
            errors_df = pd.DataFrame(global_errors)
            errors_path = os.path.join(output_path, f"inseason_errors_{timestamp}.csv")
            errors_df.to_csv(errors_path, index=False)
            self.logger.info(f"📄 Errors saved: {errors_path} ({len(errors_df)} errors)")
            print(f"📄 Errors saved to: {errors_path}")
        else:
            self.logger.info("No errors to save")

        # ✅ Return file paths for easier use
        return {"results_path": results_path, "errors_path": errors_path}
