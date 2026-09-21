import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

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
    safe_parse_date,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt

# Decorator to ensure token validity before API calls


def require_zarc_params(func):
    """Decorator to ensure ZARC parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "zarc_params", None):
            raise RuntimeError("❌ No ZARC parameters found. Call setup_zarc_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class ZARCExtractor(BaseExtractor):
    """
    Extracts ZARC (Zoneamento Agricola de Risco Climatico) analytics for agricultural entities.

    Brazilian Agricultural Climate Risk Zoning compliance service. Validates crop cycle
    parameters against ZARC risk zones and computes soil water balance metrics.

    Documentation: https://docs.earthdaily.com/agro/library/ZARC/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_ZARC.ipynb

    Args (setup_zarc_parameters):
        crop (str): Crop type code. Default: 'OTHERS'
        nb_days_sowing_emergence (int): Days between sowing and emergence. Default: 20
        soil_type (str): Soil type classification. Default: None
        cycle (str): Crop cycle duration. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, sowing_date (optional)

    Output columns:
        entity_id, crop, risk_level, sowing_date, emergence_date, + soil water balance metrics
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref: Optional[str] = None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)
        self.env = getattr(self, "env", "prod")  # default prod if not set
        self.zarc_url = agro_urls["zarc_url"][self.env]
        self.zarc_params: Optional[Dict[str, Any]] = None

    def setup_zarc_parameters(
        self,
        crop: str = "OTHERS",
        nb_days_sowing_emergence: int = 20,
        soil_type: Optional[str] = None,
        cycle: Optional[str] = None,
        partial_frequency: int = 50,
        column_mapping: dict = None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ) -> None:
        """
        Configure ZARC global parameters that apply to all entities.

        Args:
            crop: Default crop type (e.g., "SOYBEANS", "CORN")
            nb_days_sowing_emergence: Default days between sowing and emergence
            soil_type: Optional default soil type
            cycle: Optional default crop cycle
            partial_frequency: How often to save partial results during bulk processing
        """
        log = self.get_contextualized_logger("SETUP")  # ✅ Initialize logger
        log.info("Configuring ZARC parameters...")

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.apply_cache_setting(use_cache)

        self.zarc_params = {
            "crop": crop,
            "nb_days_sowing_emergence": nb_days_sowing_emergence,
            "soil_type": soil_type,
            "cycle": cycle,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        log.success("✅ ZARC parameters configured successfully")
        log.debug(f"Parameters: {self.zarc_params}")

        print("🌾 ZARC parameters configured:")
        for k, v in self.zarc_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_zarc_params
    def get_zarc_api(self, entity_data: dict) -> Dict[str, Any]:
        """
        Call ZARC API for a specific entity.

        Args:
            entity_data (dict): Must contain 'id', 'geometry', 'emergence_date'.
                            Can optionally override 'crop', 'soil_type', 'cycle', 'nb_days_sowing_emergence'.

        Returns:
            dict: Raw JSON response from API.
        """
        params = self.zarc_params
        assert params is not None  # narrowed by @require_zarc_params
        log = self.get_contextualized_logger("API")

        # Get entity-specific data (REQUIRED from entity_data)
        if not self.has_entity_field(entity_data, "id"):
            error_msg = "❌ entity_data must include a 'id' field."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        log.debug(f"Processing ZARC request for entity {entity_id}")

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
        # `emergence_date` is this extractor's one REQUIRED entity input, and it was
        # read with a plain dict lookup — so `column_mapping` never applied to the
        # only column a caller must supply. It is already in DEFAULT_COLUMN_MAPPING
        # and DATE_FIELDS, so routing it here also normalises ISO timestamps and
        # turns a NaN into None, which the missing-value check below reads correctly.
        raw_emergence_date = self.get_entity_value(entity_data, "emergence_date")
        if not raw_emergence_date:
            error_msg = f"Missing emergence_date for entity {entity_id}"
            log.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Validate and convert emergence_date to YYYY-MM-DD format
        emergence_date = safe_parse_date(raw_emergence_date)

        if not emergence_date:
            error_msg = f"Invalid or placeholder emergence_date for entity {entity_id}: {raw_emergence_date}"
            log.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Validate YYYY-MM-DD format
        try:
            datetime.strptime(emergence_date, "%Y-%m-%d")
            if raw_emergence_date != emergence_date:
                log.debug(
                    f"Entity {entity_id}: Converted emergence_date from '{raw_emergence_date}' to '{emergence_date}'"
                )
            else:
                log.debug(f"Entity {entity_id}: Emergence date validated: {emergence_date}")
        except ValueError:
            error_msg = f"Emergence date must be in YYYY-MM-DD format for entity {entity_id}. Got: {emergence_date}"
            log.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Validate geometry
        try:
            geometry = validate_wkt(geometry)
            log.debug(f"Entity {entity_id}: Geometry validated")
        except Exception as e:
            log.error(f"❌ Invalid geometry for entity {entity_id}: {e}")
            raise

        # Priority: entity_data values > setup_zarc_parameters defaults
        # This allows per-entity overrides while using global defaults as fallback
        crop = self.get_entity_value(entity_data, "crop", params["crop"])
        soil_type = self.get_entity_value(entity_data, "soil_type", params.get("soil_type"))
        cycle = self.get_entity_value(entity_data, "cycle", params.get("cycle"))
        nb_days = self.get_entity_value(entity_data, "nb_days_sowing_emergence", params["nb_days_sowing_emergence"])

        log.debug(
            f"Entity {entity_id} - Crop: {crop}, Emergence: {emergence_date}, Days: {nb_days}, Soil: {soil_type}, Cycle: {cycle}"
        )

        # Build querystring
        query_params = [
            f"crop={crop}",
            f"date_emergence={emergence_date}",
            f"nb_days_sowing_emergence={nb_days}",
        ]
        full_url = f"{self.zarc_url}/zarc?{'&'.join(query_params)}"

        log.debug(f"API URL: {full_url}")

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "id": entity_id,
            "geometry": geometry,
            "soil_type": soil_type,
            "cycle": cycle,
        }

        log.debug(f"Payload: id={entity_id}, soil_type={soil_type}, cycle={cycle}")

        # Request data
        log.info(f"Requesting ZARC data for entity {entity_id}")
        log.debug(f"API URL: {full_url}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(full_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            log.success(f"✅ ZARC data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting ZARC data for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status_code = e.response.status_code if e.response is not None else "unknown"
            response_text = e.response.text if e.response is not None else str(e)
            log.error(f"❌ HTTP error for entity {entity_id}: {status_code} - {response_text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    @staticmethod
    def format_zarc_json(response_json: Dict[str, Any]) -> pd.DataFrame:
        """Format ZARC API response into a pandas DataFrame."""
        if not isinstance(response_json, dict):
            raise ValueError("❌ Response must be a dictionary.")

        entity_id = response_json.get("id", None)
        data = response_json.get("data", None)
        if data is None:
            return pd.DataFrame()

        row = {
            "entity_id": entity_id,
            "emergence_date": data.get("emergence_date"),
            "sowing_date": data.get("sowing_date"),
            "zarc_start_date": data.get("zarc_start_date"),
            "zarc_end_date": data.get("zarc_end_date"),
            "status": data.get("status"),
        }
        return pd.DataFrame([row])

    # ---------- Safe wrapper ----------

    def get_zarc_api_safe(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe wrapper around get_zarc_api that returns a structured dict instead of raising exceptions.

        Args:
            entity_data (dict): Entity data containing 'id', 'geometry', 'emergence_date', etc.

        Returns:
            dict: {"success": bool, "data": dict or None, "error": str or None, "entity_id": str}
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response_json = self.get_zarc_api(entity_data)
            log.debug(f"Entity {entity_id}: API call successful")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "unknown"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"Entity {entity_id}: HTTP error - {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}
        except ValueError as e:
            # Validation errors (missing fields, invalid dates, etc.)
            error_msg = str(e)
            log.warning(f"Entity {entity_id}: Validation error - {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.error(f"Entity {entity_id}: Unexpected error - {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "entity_id": entity_id}

    # ---------- Single-entity processing with retry ----------
    @cache_single_entity("zarc_params")
    def process_single_entity_zarc(self, row, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process ZARC for a single entity with retry and normalization.
        Uses per-row values when present, otherwise falls back to values passed via `params`.

        Args:
            row (dict | pd.Series): Entity data containing id, geometry, emergence_date
            params (dict, optional): Fallback parameters for nb_days_sowing_emergence

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.zarc_params

        # Handle pandas Series and ensure JSON-serializable types
        if isinstance(row, pd.Series):
            row = row.to_dict()

        # Convert numpy/pandas types to native Python types
        row = {
            k: (
                None
                if pd.isna(v)
                else int(v)
                if isinstance(v, (np.integer, pd.Int64Dtype))
                else float(v)
                if isinstance(v, (np.floating, pd.Float64Dtype))
                else bool(v)
                if isinstance(v, (np.bool_, pd.BooleanDtype))
                else v.tolist()
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in row.items()
        }

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Validate geometry
        geometry = self.get_entity_value(row, "geometry")
        if not geometry:
            error_msg = f"Missing geometry for entity {entity_id}"
            log.error(f"❌ {error_msg}")
            return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

        try:
            geometry = validate_wkt(geometry)
            log.debug(f"Entity {entity_id}: Geometry validated")
        except Exception as e:
            error_msg = f"Invalid geometry: {e}"
            log.error(f"❌ Entity {entity_id}: {error_msg}")
            return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

        # Get emergence_date from row or fallback to params.
        # get_entity_value so column_mapping applies to this REQUIRED input.
        emergence_date = self.get_entity_value(row, "emergence_date")
        if not emergence_date and params:
            emergence_date = params.get("emergence_date")

        if not emergence_date:
            error_msg = "Missing 'emergence_date' (row or params)"
            log.error(f"❌ Entity {entity_id}: {error_msg}")
            return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

        # Get nb_days_sowing_emergence from row or fallback to params
        nb_days = self.get_entity_value(row, "nb_days_sowing_emergence")
        if nb_days is None and params:
            nb_days = params.get("nb_days_sowing_emergence")

        if nb_days is None:
            error_msg = "Missing 'nb_days_sowing_emergence' (row or params)"
            log.error(f"❌ Entity {entity_id}: {error_msg}")
            return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

        log.debug(f"Entity {entity_id}: emergence_date={emergence_date}, nb_days={nb_days}")

        # Prepare entity_data for API call
        entity_data = {
            self.get_mapped_column("id"): entity_id,
            self.get_mapped_column("geometry"): geometry,
            "emergence_date": emergence_date,
            self.get_mapped_column("crop"): self.get_entity_value(
                row, "crop"
            ),  # Will fallback to params in get_zarc_api
            # get_entity_value already handles dict vs Series — the isinstance dance
            # here was hand-rolling it, and skipped column_mapping while doing so.
            "soil_type": self.get_entity_value(row, "soil_type"),
            "cycle": self.get_entity_value(row, "cycle"),
            "nb_days_sowing_emergence": nb_days,
        }

        # Define API call wrapper
        def _call_api():
            return self.get_zarc_api(entity_data)

        # Call API with retry
        try:
            log.info(f"Calling ZARC API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(
                func=_call_api,
                max_retries=5,
                base_delay=1.0,
                max_delay=60.0,
            )

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Format response
            zarc_df = self.format_zarc_json(raw_json)

            # Check if we got data
            if zarc_df is None or zarc_df.empty:
                log.warning(f"No ZARC data found for entity {entity_id}")
                return {"data": None, "error": {"message": "No ZARC results found", "entity_id": entity_id}}

            log.debug(f"Entity {entity_id}: Formatted {len(zarc_df)} ZARC record(s)")

            normalized_df = normalize_with_metadata(
                row,
                zarc_df,
            )

            log.success(f"✅ Successfully processed entity {entity_id}: {len(normalized_df)} record(s)")

            return {
                "data": normalized_df,
                "error": None,
            }

        except Exception as e:
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    # ---------- Bulk parallel processing with filter/export ----------
    @require_zarc_params
    def process_zarc_bulk_extraction_parallel(
        self,
        entity_list: pd.DataFrame,
        max_workers: int = 5,
        output_path: Optional[str] = None,
        partial_frequency: int = 50,
        fail_safe: bool = False,
        filter_column: Optional[str] = None,
        filter_value: Optional[Any] = None,
        filter_type: str = "exclude",
        merge_existing: Optional[str] = None,
        skip_export: bool = False,
        prefix: str = "zarc",
        generate_report: bool = False,
        report_options: Optional[Dict[str, Any]] = None,
        use_cache=None,
    ) -> Dict[str, Any]:
        """
        Bulk processing for ZARC with parallel execution, filtering, and export capabilities.

        Args:
            entity_list: DataFrame with entities to process (must contain 'id', 'geometry', 'emergence_date')
            max_workers: Number of parallel threads
            output_path: Directory to save final CSV results
            partial_frequency: How often to save partial results
            fail_safe: If True, retries only previously failed entities
            filter_column: Column name to filter entities by
            filter_value: Value to filter in the filter_column
            filter_type: 'exclude' to skip rows with filter_value, 'include' to process only those
            merge_existing: Merge strategy - 'auto', 'preserve', or 'mark'
            skip_export: If True, skip final export (useful when chaining extractions)
            prefix: Prefix for output filenames and failed IDs files
            use_cache: If True, use caching for bulk extraction. Default: None (uses instance setting).

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics
        """
        if use_cache or (use_cache is None and getattr(self, "_cache_enabled", False)):
            return self._bulk_with_cache(
                bulk_method=self._process_zarc_bulk_extraction_parallel_inner,
                entity_list=entity_list,
                params=self.zarc_params,
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
        return self._process_zarc_bulk_extraction_parallel_inner(
            entity_list=entity_list,
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
    @require_zarc_params
    def _process_zarc_bulk_extraction_parallel_inner(
        self,
        entity_list: pd.DataFrame,
        params_kw=None,
        max_workers: int = 5,
        output_path: Optional[str] = None,
        partial_frequency: int = 50,
        fail_safe: bool = False,
        filter_column: Optional[str] = None,
        filter_value: Optional[Any] = None,
        filter_type: str = "exclude",
        merge_existing: Optional[str] = None,
        skip_export: bool = False,
        prefix: str = "zarc",
        generate_report: bool = False,
        report_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inner implementation of process_zarc_bulk_extraction_parallel."""
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk ZARC extraction: {prefix}")
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

        all_rows: List[pd.DataFrame] = []
        global_errors: List[Dict[str, Any]] = []
        successful_calculations = 0
        total_calculations = 0
        buffer_rows: List[pd.DataFrame] = []
        buffer_errors: List[Dict[str, Any]] = []
        failed_ids: List[str] = []

        log.info(f"Processing {len(filtered_entity_list)} entities in parallel...")
        print(f"🔄 Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_zarc, row.to_dict()): self.get_entity_value(row, "id")
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
                            log.debug(f"Entity {entity_id}: Success - {len(df)} record(s)")
                        else:
                            if entity_id not in failed_ids:
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
            new_results_df=results_df,
            skipped_entities_df=skipped_entities,
            merge_mode=merge_existing,
            verbose=True,
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
                "parameters": getattr(self, "zarc_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk ZARC extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
