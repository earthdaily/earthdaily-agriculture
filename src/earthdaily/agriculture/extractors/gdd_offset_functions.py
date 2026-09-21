# gdd_offset_functions.py
# Standard Library
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps

import pandas as pd

# Third-Party
import requests
from tqdm import tqdm

# Internal
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.api_utils import (
    export_results,
    filter_entities,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
)
from earthdaily.agriculture.core.base_extractor import DEFAULT_SPATIAL_PRECISION, BaseExtractor, requires_token
from earthdaily.agriculture.core.geometry import get_centroid_wkt, validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

# --- Valid values ---
VALID_PROVIDERS = {"GLOBAL1"}


def require_gdd_offset_params(func):
    """Decorator to ensure gdd_offset parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "gdd_offset_params") or self.gdd_offset_params is None:
            self.logger.error("No GDD offset parameters found")
            raise RuntimeError("❌ No GDD offset parameters found. Call setup_gdd_offset_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class GDDOffsetExtractor(BaseExtractor):
    """
    Extracts GDD-offset (Growing Degree Days threshold offset) analytics.

    Given a base date, a lower/upper temperature threshold and a list of GDD
    offsets, the API returns the forecast date at which each cumulative GDD
    offset is reached for the point location (field centroid). The output is
    a wide DataFrame: one row per entity with one column per requested
    offset (``offset_<N>`` → reached date).

    Endpoint: ``{weather_url}/analytics/gdd-offset``

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_GDDOffset.ipynb

    Args (setup_gdd_offset_parameters):
        provider (str): Weather data provider. Default: 'GLOBAL1'
        lower_threshold (float): Lower GDD temperature threshold. Default: 10.0
        upper_threshold (float): Upper GDD temperature threshold. Default: 30.0
        date (str): Base date in YYYY-MM-DD the offsets are measured from (maps to API Date). Default: None
        offsets (list[int]): Cumulative GDD offsets to resolve to reached dates. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid
            re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required)
        start_date → optional per-entity override for the API ``Date``
        offsets    → optional per-entity override for the GDD offsets list
                     (int, comma-separated string, or list)

    Output columns:
        entity_id, offset_<N> (one column per requested offset; value = date the
        offset is reached, YYYY-MM-DD)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.gdd_offset_params = None
        self.weather_url = agro_urls["weather_url"][self.env]

        self.logger.info(f"🌡️ GDDOffsetExtractor initialized for env: {self.env}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """Refresh the authentication token."""
        self.logger.debug("Refreshing authentication token...")
        return EDAuthenticator.get_new_token(env=self.env)

    # ----------------------------- Setup -----------------------------

    @requires_token
    def setup_gdd_offset_parameters(
        self,
        provider="GLOBAL1",
        lower_threshold=10.0,
        upper_threshold=30.0,
        date=None,
        offsets=None,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for GDD offset extraction.

        Args:
            provider (str): Weather data provider (default ``'GLOBAL1'``).
            lower_threshold (float): Lower GDD temperature threshold.
            upper_threshold (float): Upper GDD temperature threshold.
            date (str, optional): Base date (``YYYY-MM-DD``). Default ``None``
                — can be overwritten per entity by ``start_date``.
            offsets (int | list[int] | str, optional): GDD offsets to query.
                Default ``None`` — can be overwritten per entity by the
                ``offsets`` column (int, list, or comma-separated string).
            partial_frequency (int): How often to save partial bulk results.
            column_mapping (dict): Custom entity column mapping.
            output_mapping / exclude_columns / output_columns: Output formatting.
            use_cache (bool, optional): Per-call cache override.
        """
        # Provider validation
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider '{provider}'. Must be one of {VALID_PROVIDERS}.")

        # Threshold validation
        for label, val in (("lower_threshold", lower_threshold), ("upper_threshold", upper_threshold)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"{label} must be a number, got {type(val).__name__}.")
        if lower_threshold >= upper_threshold:
            raise ValueError(f"lower_threshold ({lower_threshold}) must be < upper_threshold ({upper_threshold}).")

        # Date validation (optional)
        if date is not None:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date '{date}'. Expected YYYY-MM-DD.")

        # Offsets normalisation (optional)
        offsets_norm = self._normalise_offsets(offsets) if offsets is not None else None

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.configure_output(
            output_mapping=output_mapping,
            exclude_columns=exclude_columns,
            output_columns=output_columns,
        )

        self.apply_cache_setting(use_cache)

        self.gdd_offset_params = {
            "provider": provider,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "date": date,
            "offsets": offsets_norm,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date", "offsets"]

        self.logger.info("GDD offset parameters configured successfully")
        print("🌡️ GDD offset parameters configured:")
        for k, v in self.gdd_offset_params.items():
            print(f"   {k}: {v}")

    @staticmethod
    def _normalise_offsets(offsets):
        """Coerce offsets (int | list | comma-string) into a sorted list of ints."""
        if offsets is None:
            return None
        if isinstance(offsets, (int,)) and not isinstance(offsets, bool):
            return [int(offsets)]
        if isinstance(offsets, str):
            parts = [p.strip() for p in offsets.split(",") if p.strip()]
            return [int(p) for p in parts]
        if isinstance(offsets, (list, tuple)):
            return [int(o) for o in offsets]
        raise ValueError(
            f"Invalid offsets type '{type(offsets).__name__}'. Expected int, list[int], or comma-separated string."
        )

    # ----------------------------- API call --------------------------

    @require_gdd_offset_params
    @requires_token
    def get_gdd_offset(self, entity_data):
        """
        Call the GDD-offset endpoint for one entity.

        Resolves ``Date`` and ``Offsets`` from the entity row first, then
        falls back to the extractor params.

        Args:
            entity_data (dict | pd.Series): Must contain ``id`` and
                ``geometry``. May carry ``start_date`` / ``date`` and
                ``offsets`` overrides.

        Returns:
            list[dict]: Raw API response (one element per offset).
        """
        params = self.gdd_offset_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        # Resolve date: entity's start_date > entity's 'date' column > params
        date = self.get_entity_value(entity_data, "start_date")
        if not date and self.has_entity_field(entity_data, "date"):
            date = self.get_entity_value(entity_data, "date")
        if not date:
            date = params.get("date")
        if not date:
            raise ValueError(f"Entity {entity_id}: no date provided (row start_date/date or params.date).")
        if isinstance(date, pd.Timestamp):
            date = date.strftime("%Y-%m-%d")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Entity {entity_id}: invalid date '{date}'. Expected YYYY-MM-DD.")

        # Resolve offsets: entity row > params
        offsets_raw = None
        if self.has_entity_field(entity_data, "offsets"):
            offsets_raw = self.get_entity_value(entity_data, "offsets")
        if offsets_raw is None or (isinstance(offsets_raw, float) and pd.isna(offsets_raw)):
            offsets_raw = params.get("offsets")
        if offsets_raw is None:
            raise ValueError(f"Entity {entity_id}: no offsets provided (row.offsets or params.offsets).")
        offsets = self._normalise_offsets(offsets_raw)
        if not offsets:
            raise ValueError(f"Entity {entity_id}: empty offsets list.")

        # Geometry → centroid POINT
        geometry = self.get_entity_value(entity_data, "geometry")
        geometry = validate_wkt(geometry)
        location_wkt = get_centroid_wkt(geometry)

        # Build URL + query params (requests handles URL encoding + repeated Offsets)
        url = f"{self.weather_url}/analytics/gdd-offset"
        query = [
            ("Date", date),
            ("Provider", params["provider"]),
            ("Location", location_wkt),
            ("LowerThreshold", params["lower_threshold"]),
            ("UpperThreshold", params["upper_threshold"]),
        ]
        for off in offsets:
            query.append(("Offsets", off))

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }

        self.logger.debug(f"Entity {entity_id}: GDD offset URL: {url}")
        self.logger.debug(f"Entity {entity_id}: GDD offset params: {query}")

        try:
            response = requests.get(url, headers=headers, params=query, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | url={url} | params={query}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: GDD offset API call failed - {e} | url={url} | params={query}")
            raise

    @require_gdd_offset_params
    @requires_token
    def get_gdd_offset_safe(self, entity_data) -> dict:
        """
        Safe wrapper around :meth:`get_gdd_offset`.

        Returns:
            dict: ``{success, data, error, entity_id}``.
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        try:
            data = self.get_gdd_offset(entity_data)
            return {"success": True, "data": data, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            return {
                "success": False,
                "data": None,
                "error": f"HTTP {status} - {text}",
                "entity_id": entity_id,
            }
        except ValueError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Validation error: {e}",
                "entity_id": entity_id,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "entity_id": entity_id,
            }

    # ----------------------------- Formatting ------------------------

    def format_gdd_offset_json(self, response_json, entity_data=None):
        """
        Format the API response into a **single-row wide DataFrame**.

        Columns:
            ``entity_id`` (if provided), ``offset_<N>`` for each offset
            returned (value = date at which offset is reached, ``YYYY-MM-DD``).

        Args:
            response_json (list[dict]): API response.
            entity_data (dict, optional): Original entity row (for id column).

        Returns:
            pd.DataFrame: single-row wide frame, or empty DataFrame if no data.
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )

        empty = self.validate_api_response(response_json, entity_id, "gdd-offset")
        if empty is not None:
            return empty

        row = {}
        if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
            row["entity_id"] = self.get_entity_value(entity_data, "id")

        for item in response_json:
            off = item.get("offset")
            if off is None:
                continue
            raw_date = item.get("date")
            try:
                date_value = pd.to_datetime(raw_date).strftime("%Y-%m-%d") if raw_date else None
            except (ValueError, TypeError):
                self.logger.warning(f"Entity {entity_id}: could not parse date '{raw_date}' for offset {off}")
                date_value = raw_date
            row[f"offset_{int(off)}"] = date_value

        if len(row) == (1 if "entity_id" in row else 0):
            # Only the metadata column present → no offsets parsed
            self.logger.warning(f"Entity {entity_id}: no offsets parsed from response")
            return pd.DataFrame()

        return pd.DataFrame([row])

    # ----------------------- Single-entity process -------------------

    def process_single_entity_gdd_offset(self, row, params=None):
        """
        Process GDD offset extraction for a single entity with retry logic.

        Args:
            row (dict | pd.Series | pd.DataFrame): Entity data.
            params (dict, optional): Override for ``self.gdd_offset_params``.

        Returns:
            dict: ``{"data": DataFrame or None, "error": dict or None}``.
        """
        if params is None:
            params = self.gdd_offset_params

        # Normalise row to dict
        try:
            if isinstance(row, pd.DataFrame):
                if len(row) > 1:
                    self.logger.warning(f"DataFrame with {len(row)} rows — processing first row only.")
                row = row.iloc[0].to_dict()
            elif isinstance(row, pd.Series):
                row = row.to_dict()
            elif not isinstance(row, dict):
                raise TypeError(f"Expected dict/Series/DataFrame, got {type(row).__name__}")
        except Exception as e:
            return {
                "data": None,
                "error": {"message": f"Invalid input: {e}", "entity_id": "unknown"},
            }

        entity_id = self.get_entity_value(row, "id", "unknown")

        try:

            def _call_api():
                return self.get_gdd_offset(row)

            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            gdd_df = self.format_gdd_offset_json(raw_json, entity_data=row)
            if gdd_df is None or gdd_df.empty:
                return {
                    "data": None,
                    "error": {"message": "No GDD offset data returned", "entity_id": entity_id},
                }

            result_df = normalize_with_metadata(
                row,
                gdd_df,
            )

            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: GDD offset processing failed — {e}")
            return {
                "data": None,
                "error": {"message": str(e), "entity_id": entity_id},
            }

    # ----------------------- Bulk parallel process -------------------

    def process_entity_gdd_offset_bulk_parallel(
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
        prefix="gdd_offset",
        generate_report=False,
        report_options=None,
        use_cache=None,
        spatial_grouping=False,
        spatial_precision=DEFAULT_SPATIAL_PRECISION,
    ):
        """
        Bulk GDD offset extraction with threading, partial saves, and merge logic.

        Args:
            entity_list (pd.DataFrame): Entities with at least ``id`` and
                ``geometry``; may provide per-row ``start_date``/``date`` and
                ``offsets`` overrides.
            spatial_grouping (bool): If True, group fields by ``geohash(centroid) ×
                start_date × offsets × provider``, call the API once per group and
                broadcast. The forecast reached-date is a function of the centroid,
                base date, thresholds and offsets — safe to dedup across neighbours.
            spatial_precision (int): Geohash precision for the bucket (default
                ``DEFAULT_SPATIAL_PRECISION``).
        """

        def _dispatch(
            el, *, skip_export_=skip_export, generate_report_=generate_report, report_options_=report_options
        ):
            if use_cache is not None and use_cache:
                return self._bulk_with_cache(
                    bulk_method=self._process_entity_gdd_offset_bulk_parallel_inner,
                    entity_list=el,
                    params=self.gdd_offset_params,
                    max_workers=max_workers,
                    output_path=output_path,
                    partial_frequency=partial_frequency,
                    fail_safe=fail_safe,
                    filter_column=filter_column,
                    filter_value=filter_value,
                    filter_type=filter_type,
                    merge_existing=merge_existing,
                    skip_export=skip_export_,
                    prefix=prefix,
                    generate_report=generate_report_,
                    report_options=report_options_,
                )
            return self._process_entity_gdd_offset_bulk_parallel_inner(
                entity_list=el,
                params_kw=params,
                max_workers=max_workers,
                output_path=output_path,
                partial_frequency=partial_frequency,
                fail_safe=fail_safe,
                filter_column=filter_column,
                filter_value=filter_value,
                filter_type=filter_type,
                merge_existing=merge_existing,
                skip_export=skip_export_,
                prefix=prefix,
                generate_report=generate_report_,
                report_options=report_options_,
            )

        if spatial_grouping:
            return self._run_bulk_spatially_grouped(
                entity_list=entity_list,
                run_representatives=lambda reps: _dispatch(
                    reps, skip_export_=True, generate_report_=False, report_options_=None
                ),
                signature_columns=["start_date", "offsets", "provider"],
                precision=spatial_precision,
                prefix=prefix,
                output_path=output_path,
                skip_export=skip_export,
                generate_report=generate_report,
                report_options=report_options,
                report_params=getattr(self, "gdd_offset_params", {}),
            )
        return _dispatch(entity_list)

    @requires_token
    @require_gdd_offset_params
    def _process_entity_gdd_offset_bulk_parallel_inner(
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
        prefix="gdd_offset",
        generate_report=False,
        report_options=None,
    ):
        params = params_kw
        self.logger.info(f"🚀 Starting bulk GDD offset extraction for {len(entity_list)} entities")

        if merge_existing is None:
            merge_existing = self.merge_existing
        if merge_existing not in ("auto", "preserve", "mark"):
            raise ValueError(f"Invalid merge_existing='{merge_existing}'. Choose from ['auto','preserve','mark']")

        filtered, skipped, skip_count = filter_entities(entity_list, filter_column, filter_value, filter_type)

        # Resume mode is explicit (self.retry_failed_only) and never implied
        # by fail_safe -- see BaseExtractor._resolve_retry_entity_list.
        filtered = self._resolve_retry_entity_list(filtered, prefix, fail_safe=fail_safe, log=self.logger)

        all_rows = []
        global_errors = []
        successful = 0
        total = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        print(f"🔄 Processing {prefix.replace('_', ' ').title()} for {len(filtered)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_gdd_offset, row, params): self.get_entity_value(row, "id")
                for _, row in filtered.iterrows()
            }

            with tqdm(
                total=len(future_to_id),
                desc=f"🌡️ Processing {prefix.replace('_', ' ').title()}",
                unit="entity",
            ) as pbar:
                for future in as_completed(future_to_id):
                    total += 1
                    entity_id = future_to_id[future]
                    try:
                        result = future.result()
                        df = result.get("data")
                        error = result.get("error")

                        if df is not None and not df.empty:
                            successful += 1
                            all_rows.append(df)
                            buffer_rows.append(df)
                        else:
                            failed_ids.append(entity_id)

                        if error:
                            record = {"entity_id": entity_id, **error}
                            global_errors.append(record)
                            buffer_errors.append(record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)

                    except Exception as e:
                        record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(record)
                        buffer_errors.append(record)
                        failed_ids.append(entity_id)

                    if self.partial_path and partial_frequency > 0 and total % partial_frequency == 0:
                        partial_df = pd.concat(buffer_rows, ignore_index=True) if buffer_rows else pd.DataFrame()
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

        elapsed = time.time() - start_time
        print(f"\n⏱️ Total processing time: {elapsed:.2f} seconds")
        print(f"✅ Successful calculations: {successful}/{total}")

        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df,
            skipped_entities_df=skipped,
            merge_mode=merge_existing,
            verbose=True,
        )

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
                "total_calculations": total,
                "successful": successful,
                "failed": total - successful,
                "elapsed_seconds": elapsed,
                "parameters": getattr(self, "gdd_offset_params", {}),
                "entity_df": entity_list,
            },
        )

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total,
            "successful_calculations": successful,
            "failed_calculations": total - successful,
            "failed_ids": failed_ids,
        }
