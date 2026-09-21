# processor_change_index_functions.py - Change Index Functions

#  Standard Library
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps

import pandas as pd

#  Third-Party Libraries
import requests
from tqdm.auto import tqdm

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

AVAILABLE_MAP_TYPES = {"NDVI", "EVI", "CVI", "GNDVI", "NDWI"}
AVAILABLE_COLLECTIONS = {"Sentinel-2", "Landsat"}


def require_change_index_params(func):
    """Decorator to ensure change index parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "change_index_params") or self.change_index_params is None:
            error_msg = "❌ No change index parameters found. Call setup_change_index_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class ChangeIndexExtractor(BaseExtractor):
    """
    Extracts change detection analytics between two vegetation-index images.

    Compares a reference image (given per entity) against the nearest prior image in a
    configurable look-back window to flag significant change. Useful for detecting
    rapid field changes (harvest, stress events, tillage, etc.).

    Documentation: https://docs.earthdaily.com/agro/library/Change_Index/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_ChangeIndex.ipynb

    Args (setup_change_index_parameters):
        map_type (str): Vegetation index to analyze. Default: 'NDVI'.
            Choices: NDVI, EVI, CVI, GNDVI, NDWI.
        collections (list[str]): Satellite collections to search. Default: ['Sentinel-2'].
            Choices: 'Sentinel-2', 'Landsat'.
        max_period_reference (int): Max days back from reference_date to find the reference image. Default: 7.
        max_period_previous (int): Max days before the reference image to look for a previous image. Default: 15.
        min_period_previous (int): Min days before the reference image to look for a previous image. Default: 5.
        same_sensor (bool): Require both images to come from the same sensor. Default: False.
        parameter_profile (str): API parameter profile name. Default: 'change_index_v1'.
        publish_af (bool): Whether to publish to AF (includes ``field_id`` in the API
            payload if True). Default: False.

    Entity fields (via column_mapping):
        id, geometry (required);
        reference_date (required per entity — YYYY-MM-DD reference image date);
        crop, sowing_date (optional)

    Output columns:
        entity_id, reference_date, status, + any change metrics returned by the API
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.change_index_params = None
        self.change_index_url = agro_urls["change_index_url"][self.env]

        self.logger.debug(f"Change Index URL configured: {self.change_index_url}")
        self.logger.info(f"🔍 ChangeIndexExtractor ready for {self.env} environment")

    def get_new_token(self):
        """Implements token refresh logic for ChangeIndexExtractor."""
        self.logger.debug("Getting new token for ChangeIndexExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------

    def setup_change_index_parameters(
        self,
        map_type="NDVI",
        collections=None,
        max_period_reference=7,
        max_period_previous=15,
        min_period_previous=5,
        same_sensor=False,
        parameter_profile="change_index_v1",
        partial_frequency=50,
        publish_af=False,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for change index extraction.

        Args:
            map_type (str): Vegetation index (NDVI, EVI, CVI, GNDVI, NDWI). Default: 'NDVI'.
            collections (list[str]): Satellite collections. Default: ['Sentinel-2'].
            max_period_reference (int): Max days back from reference_date for reference image. Default: 7.
            max_period_previous (int): Max days before reference image for previous image. Default: 15.
            min_period_previous (int): Min days before reference image for previous image. Default: 5.
            same_sensor (bool): Require same sensor for both images. Default: False.
            parameter_profile (str): API parameter profile name. Default: 'change_index_v1'.
            partial_frequency (int): How often to save partial results. Default: 50.
            publish_af (bool): Whether to publish to AF (includes ``field_id`` in
                payload if True). Default: False. Mirrors the flag used by the rest
                of the processor family (Baresoil, Covercrop, Emergence, Greenness,
                Harvest, Planted, score, InSeasonMonitoring).
            column_mapping (dict): Column name overrides (must include 'reference_date').
            output_mapping (dict): Output column renames.
            exclude_columns (list): Columns to exclude from output.
            output_columns (list): Whitelist of output columns.
            use_cache (bool): Whether to use caching.
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring change index parameters...")

        # Map type validation
        if map_type not in AVAILABLE_MAP_TYPES:
            error_msg = f"Invalid map_type '{map_type}'. Choose from: {AVAILABLE_MAP_TYPES}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Collections validation
        if collections is None:
            collections = ["Sentinel-2"]
        if not isinstance(collections, (list, tuple)) or not collections:
            error_msg = f"Invalid collections={collections}. Must be a non-empty list."
            log.error(error_msg)
            raise ValueError(error_msg)
        invalid = set(collections) - AVAILABLE_COLLECTIONS
        if invalid:
            error_msg = f"Invalid collections {invalid}. Choose from: {AVAILABLE_COLLECTIONS}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Period validation
        for name, val in (
            ("max_period_reference", max_period_reference),
            ("max_period_previous", max_period_previous),
            ("min_period_previous", min_period_previous),
        ):
            if not isinstance(val, int) or val < 0:
                error_msg = f"Invalid {name}={val}. Must be a non-negative integer."
                log.error(error_msg)
                raise ValueError(error_msg)
        if min_period_previous >= max_period_previous:
            error_msg = (
                f"min_period_previous ({min_period_previous}) must be strictly less "
                f"than max_period_previous ({max_period_previous})."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Same sensor validation
        if not isinstance(same_sensor, bool):
            error_msg = f"Invalid same_sensor={same_sensor}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        # publish_af validation (must be a bool, not a truthy int — mirrors the family).
        if not isinstance(publish_af, bool):
            error_msg = f"Invalid publish_af={publish_af}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.apply_cache_setting(use_cache)

        self.change_index_params = {
            "map_type": map_type,
            "collections": list(collections),
            "max_period_reference": max_period_reference,
            "max_period_previous": max_period_previous,
            "min_period_previous": min_period_previous,
            "same_sensor": same_sensor,
            "parameter_profile": parameter_profile,
            "partial_frequency": partial_frequency,
            "publish_af": publish_af,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "reference_date"]

        self.configure_output(
            output_mapping=output_mapping,
            exclude_columns=exclude_columns,
            output_columns=output_columns,
        )

        log.success("✅ Change index parameters configured successfully")
        log.debug(f"Parameters: {self.change_index_params}")

        print("🔍 Change index parameters configured:")
        for k, v in self.change_index_params.items():
            print(f"   {k}: {v}")

    # -------------------------------------------------------------------------
    # API CALL
    # -------------------------------------------------------------------------

    @requires_token
    @require_change_index_params
    def get_change_index_api(self, entity_data):
        """
        Call the Change Index API for a single entity.

        Args:
            entity_data (dict|Series): Entity data with 'id', 'geometry', 'reference_date'
                and optionally 'crop' and 'sowing_date'.

        Returns:
            dict: Parsed JSON response from the Change Index API.
        """
        params = self.change_index_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

        log.debug(f"Validating entity {entity_id}")

        # When publish_af is True the payload must carry field_id, so the entity
        # must have an id. When False, id is only used for logging/result stamping
        # and may fall back to "unknown" via get_entity_value.
        if params.get("publish_af", False) and not self.has_entity_field(entity_data, "id"):
            error_msg = "❌ entity_data must include an 'id' field when publish_af=True."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Geometry validation
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

        # Reference date — required per entity
        reference_date = self._get_reference_date(entity_data, entity_id)

        # Optional row fields
        crop = self.get_entity_value(entity_data, "crop") or "Unknown"
        sowing_date = self.get_entity_value(entity_data, "sowing_date") or ""

        # Build URL
        url = f"{self.change_index_url}/change-index-processor"
        query_params = [
            f"Parameter%20Profile={params['parameter_profile']}",
            f"Start%20Date={reference_date}",
            f"Maximum%20period%20for%20the%20reference%20image={params['max_period_reference']}",
            f"Maximum%20period%20for%20the%20previous%20image={params['max_period_previous']}",
            f"Minimum%20period%20for%20the%20previous%20image={params['min_period_previous']}",
            f"MapType={params['map_type']}",
        ]
        for collection in params["collections"]:
            query_params.append(f"Collections={collection}")
        query_params.append(f"Same%20Sensor={str(params['same_sensor'])}")
        full_url = f"{url}?{'&'.join(query_params)}"

        # Payload — `field_id` is only included when publish_af is True
        # (mirrors the rest of the processor family).
        payload = {
            "crop": crop,
            "feature": geometry,
            "sowing_date": sowing_date,
        }
        if params.get("publish_af", False):
            payload["field_id"] = str(entity_id)
            log.debug("Including field_id in payload (publish_af=True)")

        log.info(f"Requesting Change Index for entity {entity_id} (reference_date={reference_date})")
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
                timeout=120,
            )
            response.raise_for_status()
            json_response = response.json()
            log.success(f"✅ Change Index data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            # Stamp for downstream formatting
            json_response.setdefault("id", entity_id)
            json_response.setdefault("reference_date", reference_date)
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting Change Index for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_change_index_api_safe(self, entity_data):
        """
        Safe wrapper around get_change_index_api().

        Returns:
            dict: {"success": bool, "data": dict|None, "error": str|None, "entity_id": str}
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response = self.get_change_index_api(entity_data)
            return {"success": True, "data": response, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}
        except Exception as e:
            log.warning(f"Error for entity {entity_id}: {e}")
            return {"success": False, "data": None, "error": str(e), "entity_id": entity_id}

    # -------------------------------------------------------------------------
    # FORMAT
    # -------------------------------------------------------------------------

    @require_change_index_params
    def format_change_index_json(self, response_json, params=None):
        """
        Normalize a Change Index API response into a single-row DataFrame.

        The API returns metrics nested under a ``data`` object (SPAEFIndex,
        ChangeIndex, CurrentMap*/ReferenceMap* stats, date_ref, sensor_ref,
        date_current, sensor_current, ...). This method flattens those onto
        the row so each metric is its own column.

        Args:
            response_json (dict): API response JSON (augmented with 'id' and 'reference_date'
                by get_change_index_api).
            params (dict, optional): Override change index parameters.

        Returns:
            pd.DataFrame: Single-row DataFrame with entity_id, reference_date,
            status, and one column per metric in the nested ``data`` object.
        """
        log = self.get_contextualized_logger("FORMAT")

        if not isinstance(response_json, dict):
            error_msg = "❌ response_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = response_json.get("id")
        validation = self.validate_api_response(response_json, entity_id, "change_index")
        if isinstance(validation, pd.DataFrame):
            return validation

        row = {
            "entity_id": entity_id,
            "reference_date": response_json.get("reference_date"),
            "status": response_json.get("status"),
        }

        # Flatten the nested 'data' dict — the inner 'status' mirrors the outer one, skip it.
        data = response_json.get("data") or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "status":
                    continue
                row[k] = v
        else:
            log.warning(f"Entity {entity_id}: 'data' field is not a dict ({type(data).__name__})")

        # Promote any other unexpected top-level keys (future-proofing).
        reserved = {"id", "reference_date", "status", "data"}
        for k, v in response_json.items():
            if k in reserved or k in row:
                continue
            row[k] = v

        log.debug(f"Formatted change index row for entity {entity_id} with {len(row)} columns")
        return pd.DataFrame([row])

    # -------------------------------------------------------------------------
    # PROCESS SINGLE ENTITY
    # -------------------------------------------------------------------------

    @requires_token
    @require_change_index_params
    @cache_single_entity("change_index_params")
    def process_single_entity_change_index(self, row, params=None):
        """
        Process change index extraction for a single entity with retry logic.

        Args:
            row (dict|Series): Entity data with id, geometry, reference_date.
            params (dict, optional): Override parameters.

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.change_index_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Validate geometry early
        try:
            geometry = self.get_entity_value(row, "geometry")
            validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except Exception as e:
            log.error(f"Validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Validate reference_date early to surface a clean error without retries
        try:
            self._get_reference_date(row, entity_id)
        except ValueError as e:
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        def _call_api():
            return self.get_change_index_api(row)

        try:
            log.info(f"Calling Change Index API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(
                func=_call_api,
                max_retries=5,
                base_delay=1.0,
                max_delay=60.0,
            )

            change_df = self.format_change_index_json(raw_json)
            if change_df is None or change_df.empty:
                log.warning(f"No change index results found for entity {entity_id}")
                return {
                    "data": None,
                    "error": {"message": "No change index results found", "entity_id": entity_id},
                }

            log.success(f"✅ Successfully processed entity {entity_id}")
            return {
                "data": normalize_with_metadata(
                    row,
                    change_df,
                ),
                "error": None,
            }

        except Exception as e:
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    # -------------------------------------------------------------------------
    # BULK PROCESSING
    # -------------------------------------------------------------------------

    def process_change_index_bulk_extraction_parallel(
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
        prefix="change_index",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of Change Index extraction using threads + progress bar.

        Args:
            entity_list (pd.DataFrame): Entities to process (must contain 'id', 'geometry',
                and 'reference_date'; column names may be remapped via column_mapping).
            params (dict, optional): Override change index parameters.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final results and error logs.
            partial_frequency (int, optional): How often to save partial results.
            fail_safe (bool, optional): If True, per-entity failures are captured to
                ``failed_ids`` instead of aborting the run. Error tolerance only -- it does
                NOT change which entities are processed. To reprocess just the IDs a
                previous run recorded, set ``retry_failed_only`` (see BaseExtractor).
            filter_column (str, optional): Column name to filter entities by.
            filter_value (any, optional): Value to filter in the filter_column.
            filter_type (str, optional): 'exclude' or 'include'. Defaults to 'exclude'.
            merge_existing (str, optional): Merge strategy - 'auto', 'preserve', or 'mark'.
            skip_export (bool, optional): If True, skip final export. Default: False.
            prefix (str, optional): Prefix for output filenames. Default: "change_index".
            generate_report (bool, optional): Generate HTML report. Default: False.
            report_options (dict, optional): Report configuration options.
            use_cache (bool, optional): Enable/disable caching for this call.

        Returns:
            dict: results_df, global_errors, total_entities, total_calculations,
                  successful_calculations, failed_calculations, failed_ids
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_change_index_bulk_extraction_parallel_inner,
                entity_list=entity_list,
                params=self.change_index_params,
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
        return self._process_change_index_bulk_extraction_parallel_inner(
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
    @require_change_index_params
    def _process_change_index_bulk_extraction_parallel_inner(
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
        prefix="change_index",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing of Change Index requests."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk change index extraction: {prefix}")
        log.info("=" * 60)
        log.info(f"Total input entities: {len(entity_list)}")
        log.info(f"Max workers: {max_workers}")
        log.info(f"Partial frequency: {partial_frequency}")
        log.info(f"Fail-safe mode: {fail_safe}")

        if merge_existing is None:
            merge_existing = self.merge_existing

        valid_modes = ["auto", "preserve", "mark"]
        if merge_existing not in valid_modes:
            error_msg = f"Invalid merge_existing='{merge_existing}'. Choose from: {valid_modes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(f"Merge strategy: {merge_existing}")

        # Apply filter
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
        print(f"🔍 Processing {prefix.upper()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_change_index, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🔍 Processing {prefix.upper()}", unit="entity") as pbar:
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

        log.debug("Concatenating all results...")
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        log.info(f"Results DataFrame shape: {results_df.shape}")

        log.debug("Merging with skipped entities...")
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df,
            skipped_entities_df=skipped_entities,
            merge_mode=merge_existing,
            verbose=True,
        )

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
                "parameters": getattr(self, "change_index_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk change index extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _get_reference_date(self, entity_data, entity_id):
        """
        Resolve and validate the per-entity reference_date.

        reference_date is mandatory on the row (column_mapping-aware) — this
        processor no longer resolves it from a coverage lookup. Accepts strings
        in YYYY-MM-DD format or pd.Timestamp; returns YYYY-MM-DD.
        """
        value = self.get_entity_value(entity_data, "reference_date")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            error_msg = (
                f"Entity {entity_id}: reference_date missing. Add a 'reference_date' "
                "column (column_mapping-aware) to the entity row."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")

        if isinstance(value, str):
            # Trim a trailing time component if present (e.g. 2025-04-01T00:00:00)
            date_str = value.split("T")[0]
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                error_msg = f"Entity {entity_id}: invalid reference_date '{value}'. Use YYYY-MM-DD."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            return date_str

        error_msg = f"Entity {entity_id}: unsupported reference_date type {type(value).__name__}."
        self.logger.error(error_msg)
        raise ValueError(error_msg)
