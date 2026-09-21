# gdd_functions.py
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
from earthdaily.agriculture.core.base_extractor import (
    DEFAULT_MAX_WINDOW_DAYS,
    DEFAULT_SPATIAL_PRECISION,
    BaseExtractor,
    requires_token,
)
from earthdaily.agriculture.core.geometry import get_centroid_wkt, validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

# --- Valid values ---
VALID_PROVIDERS = {"GLOBAL1"}


def require_gdd_params(func):
    """Decorator to ensure gdd parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "gdd_params") or self.gdd_params is None:
            self.logger.error("No GDD parameters found")
            raise RuntimeError("❌ No GDD parameters found. Call setup_gdd_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class GDDExtractor(BaseExtractor):
    """
    Extracts GDD (Growing Degree Days) analytics for agricultural entities.

    Computes daily and cumulated Growing Degree Days from a start date to a
    last date, using a lower (and optional upper) temperature threshold. The
    output is a long DataFrame: one row per entity per day, with columns for
    minimum/maximum temperature, daily GDD and cumulated GDD.

    Endpoint: ``{weather_url}/analytics/gdd``

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgro_growing_degree_days.ipynb

    Args (setup_gdd_parameters):
        provider (str): Weather data provider. Default: 'GLOBAL1'
        lower_threshold (float): Lower GDD temperature threshold (required by API). Default: 10.0
        upper_threshold (float): Optional upper temperature threshold (capped GDD). Default: None
        start_date (str): Period start in YYYY-MM-DD (maps to API StartDate). Default: None
        end_date (str): Period end in YYYY-MM-DD (maps to API LastDate). Default: None
        reset_cumulative_every_year (bool): Reset the cumulated GDD at each year boundary. Default: False
        extrapolate_forecast_data (bool): Extend the series with forecast data. Default: False
        use_cache (bool): Reuse cached API responses and cache new results to avoid
            re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required)
        start_date → optional per-entity override for the API ``StartDate``
        end_date   → optional per-entity override for the API ``LastDate``

    Output columns:
        entity_id, date, minimum, maximum, daily_gdd, cumulated_gdd
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.gdd_params = None
        self.weather_url = agro_urls["weather_url"][self.env]

        self.logger.info(f"🌡️ GDDExtractor initialized for env: {self.env}")
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
    def setup_gdd_parameters(
        self,
        provider="GLOBAL1",
        lower_threshold=10.0,
        upper_threshold=None,
        start_date=None,
        end_date=None,
        reset_cumulative_every_year=False,
        extrapolate_forecast_data=False,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for GDD extraction.

        Args:
            provider (str): Weather data provider (default ``'GLOBAL1'``).
            lower_threshold (float): Lower GDD temperature threshold (required by API).
            upper_threshold (float, optional): Upper GDD temperature threshold.
            start_date (str, optional): Start date (``YYYY-MM-DD``). Can be
                overwritten per entity by ``start_date``.
            end_date (str, optional): Last date (``YYYY-MM-DD``). Can be
                overwritten per entity by ``end_date``.
            reset_cumulative_every_year (bool): Reset the cumulative GDD at
                the start of each calendar year.
            extrapolate_forecast_data (bool): Extrapolate forecast data when
                the range extends beyond observations.
            partial_frequency (int): How often to save partial bulk results.
            column_mapping (dict): Custom entity column mapping.
            output_mapping / exclude_columns / output_columns: Output formatting.
            use_cache (bool, optional): Per-call cache override.
        """
        # Provider validation
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider '{provider}'. Must be one of {VALID_PROVIDERS}.")

        # Threshold validation
        if not isinstance(lower_threshold, (int, float)) or isinstance(lower_threshold, bool):
            raise ValueError(f"lower_threshold must be a number, got {type(lower_threshold).__name__}.")
        if upper_threshold is not None:
            if not isinstance(upper_threshold, (int, float)) or isinstance(upper_threshold, bool):
                raise ValueError(f"upper_threshold must be a number or None, got {type(upper_threshold).__name__}.")
            if lower_threshold >= upper_threshold:
                raise ValueError(f"lower_threshold ({lower_threshold}) must be < upper_threshold ({upper_threshold}).")

        # Date validation (optional — can be overridden per entity)
        for label, val in (("start_date", start_date), ("end_date", end_date)):
            if val is not None:
                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Invalid {label} '{val}'. Expected YYYY-MM-DD.")
        if start_date is not None and end_date is not None:
            if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
                raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date}).")

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.configure_output(
            output_mapping=output_mapping,
            exclude_columns=exclude_columns,
            output_columns=output_columns,
        )

        self.apply_cache_setting(use_cache)

        self.gdd_params = {
            "provider": provider,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "start_date": start_date,
            "end_date": end_date,
            "reset_cumulative_every_year": bool(reset_cumulative_every_year),
            "extrapolate_forecast_data": bool(extrapolate_forecast_data),
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        self.logger.info("GDD parameters configured successfully")
        print("🌡️ GDD parameters configured:")
        for k, v in self.gdd_params.items():
            print(f"   {k}: {v}")

    # ----------------------------- API call --------------------------

    @require_gdd_params
    @requires_token
    def get_gdd(self, entity_data):
        """
        Call the GDD endpoint for one entity.

        Resolves ``StartDate`` / ``LastDate`` from the entity row first, then
        falls back to the extractor params.

        Args:
            entity_data (dict | pd.Series): Must contain ``id`` and
                ``geometry``. May carry ``start_date`` / ``end_date`` overrides.

        Returns:
            dict: Raw API response (``GrowingDegreeDayResult``).
        """
        params = self.gdd_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        # Resolve start_date / end_date: entity row > params
        start_date = self.get_entity_value(entity_data, "start_date") or params.get("start_date")
        end_date = self.get_entity_value(entity_data, "end_date") or params.get("end_date")

        if not start_date:
            raise ValueError(f"Entity {entity_id}: no start_date provided (row.start_date or params.start_date).")
        if not end_date:
            raise ValueError(f"Entity {entity_id}: no end_date provided (row.end_date or params.end_date).")

        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Entity {entity_id}: invalid start_date '{start_date}'. Expected YYYY-MM-DD.")
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Entity {entity_id}: invalid end_date '{end_date}'. Expected YYYY-MM-DD.")
        if start_dt > end_dt:
            raise ValueError(f"Entity {entity_id}: start_date ({start_date}) must be <= end_date ({end_date}).")

        # Geometry → centroid POINT
        geometry = self.get_entity_value(entity_data, "geometry")
        geometry = validate_wkt(geometry)
        location_wkt = get_centroid_wkt(geometry)

        # Build URL + query params
        url = f"{self.weather_url}/analytics/gdd"
        query = [
            ("StartDate", start_date),
            ("LastDate", end_date),
            ("Provider", params["provider"]),
            ("Location", location_wkt),
            ("LowerThreshold", params["lower_threshold"]),
            ("ResetCumulativeEveryYear", str(params["reset_cumulative_every_year"]).lower()),
            ("ExtrapolateForecastData", str(params["extrapolate_forecast_data"]).lower()),
        ]
        if params.get("upper_threshold") is not None:
            query.append(("UpperThreshold", params["upper_threshold"]))

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }

        self.logger.debug(f"Entity {entity_id}: GDD URL: {url}")
        self.logger.debug(f"Entity {entity_id}: GDD params: {query}")

        try:
            response = requests.get(url, headers=headers, params=query, timeout=60)
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                self.logger.debug(f"Entity {entity_id}: API returned empty response (204)")
                return None
            json_response = response.json()
            self.logger.debug(f"Entity {entity_id}: API response received")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | url={url} | params={query}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: GDD API call failed - {e} | url={url} | params={query}")
            raise

    @require_gdd_params
    @requires_token
    def get_gdd_safe(self, entity_data) -> dict:
        """
        Safe wrapper around :meth:`get_gdd`.

        Returns:
            dict: ``{success, data, error, entity_id}``.
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        try:
            data = self.get_gdd(entity_data)
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

    def format_gdd_json(self, response_json, entity_data=None):
        """
        Format the API response into a **long DataFrame** (one row per day).

        Columns:
            ``entity_id`` (if provided), ``date``, ``minimum``, ``maximum``,
            ``daily_gdd``, ``cumulated_gdd``.

        Args:
            response_json (dict): ``GrowingDegreeDayResult`` API response.
            entity_data (dict, optional): Original entity row (for id column).

        Returns:
            pd.DataFrame: long DataFrame, or empty DataFrame if no data.
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )

        empty = self.validate_api_response(response_json, entity_id, "gdd")
        if empty is not None:
            return empty

        elements = response_json.get("elements") or []
        if not elements:
            self.logger.warning(f"Entity {entity_id}: no GDD elements in response")
            return pd.DataFrame()

        entity_id_provided = self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id")
        entity_id_value = self.get_entity_value(entity_data, "id") if entity_id_provided else None

        rows = []
        for item in elements:
            raw_date = item.get("date")
            try:
                date_value = pd.to_datetime(raw_date).strftime("%Y-%m-%d") if raw_date else None
            except (ValueError, TypeError):
                self.logger.warning(f"Entity {entity_id}: could not parse date '{raw_date}'")
                date_value = raw_date

            row = {}
            if entity_id_provided:
                row["entity_id"] = entity_id_value
            row["date"] = date_value
            row["minimum"] = item.get("minimum")
            row["maximum"] = item.get("maximum")
            row["daily_gdd"] = item.get("dailyGrowingDegreeDay")
            row["cumulated_gdd"] = item.get("cumulatedGrowingDegreeDay")
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ----------------------- Single-entity process -------------------

    def process_single_entity_gdd(self, row, params=None):
        """
        Process GDD extraction for a single entity with retry logic.

        Args:
            row (dict | pd.Series | pd.DataFrame): Entity data.
            params (dict, optional): Override for ``self.gdd_params``.

        Returns:
            dict: ``{"data": DataFrame or None, "error": dict or None}``.
        """
        if params is None:
            params = self.gdd_params

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
                return self.get_gdd(row)

            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            gdd_df = self.format_gdd_json(raw_json, entity_data=row)
            if gdd_df is None or gdd_df.empty:
                return {
                    "data": None,
                    "error": {"message": "No GDD data returned", "entity_id": entity_id},
                }

            result_df = normalize_with_metadata(
                row,
                gdd_df,
            )

            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: GDD processing failed — {e}")
            return {
                "data": None,
                "error": {"message": str(e), "entity_id": entity_id},
            }

    # ----------------------- Bulk parallel process -------------------

    def process_entity_gdd_bulk_parallel(
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
        prefix="gdd",
        generate_report=False,
        report_options=None,
        use_cache=None,
        spatial_grouping=False,
        spatial_precision=DEFAULT_SPATIAL_PRECISION,
        spatial_max_window_days=DEFAULT_MAX_WINDOW_DAYS,
    ):
        """
        Bulk GDD extraction with threading, partial saves, and merge logic.

        Args:
            entity_list (pd.DataFrame): Entities with at least ``id`` and
                ``geometry``; may provide per-row ``start_date`` / ``end_date``
                overrides.
            spatial_grouping (bool): If True, group fields by ``geohash(centroid) ×
                provider``, call the API once per group over the *union* of the
                group's date windows, and slice each member out of that single
                response. GDD depends only on the centroid + date window +
                thresholds, so neighbouring fields dedup safely. ``cumulated_gdd``
                accumulates from the *request* start, so each member's series is
                re-zeroed to its own start date (year by year when
                ``reset_cumulative_every_year`` is set) — the output matches an
                ungrouped run row for row. With ``use_cache``, the geohash-keyed
                spatial cache is consulted per cell before any request.
            spatial_precision (int): Geohash precision for the bucket (default
                ``DEFAULT_SPATIAL_PRECISION``). Lower = coarser cells = more dedup.
            spatial_max_window_days (int): Cap on a group's widened window (default
                ``DEFAULT_MAX_WINDOW_DAYS``). Cells whose union exceeds it are split
                into sub-groups, so one long-history field cannot force a decade-long
                pull on every field sharing its cell.
        """

        def _dispatch(
            el, *, skip_export_=skip_export, generate_report_=generate_report, report_options_=report_options
        ):
            if use_cache is not None and use_cache:
                return self._bulk_with_cache(
                    bulk_method=self._process_entity_gdd_bulk_parallel_inner,
                    entity_list=el,
                    params=self.gdd_params,
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
            return self._process_entity_gdd_bulk_parallel_inner(
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
            # Grouped runs use the geohash-keyed spatial cache instead of the
            # entity-keyed one: the representative fetch goes straight to the API and
            # _run_bulk_spatially_grouped handles cache lookup/store per cell.
            def _fetch_representatives(reps):
                return self._process_entity_gdd_bulk_parallel_inner(
                    entity_list=reps,
                    params_kw=params,
                    max_workers=max_workers,
                    output_path=output_path,
                    partial_frequency=partial_frequency,
                    fail_safe=fail_safe,
                    filter_column=filter_column,
                    filter_value=filter_value,
                    filter_type=filter_type,
                    merge_existing=merge_existing,
                    skip_export=True,
                    prefix=prefix,
                    generate_report=False,
                    report_options=None,
                )

            return self._run_bulk_spatially_grouped(
                entity_list=entity_list,
                run_representatives=_fetch_representatives,
                signature_columns=["provider"],
                window_columns=("start_date", "end_date"),
                max_window_days=spatial_max_window_days,
                cumulative_columns=["cumulated_gdd"],
                reset_yearly=bool((self.gdd_params or {}).get("reset_cumulative_every_year", False)),
                precision=spatial_precision,
                prefix=prefix,
                output_path=output_path,
                skip_export=skip_export,
                generate_report=generate_report,
                report_options=report_options,
                report_params=getattr(self, "gdd_params", {}),
                use_cache=use_cache,
                cache_params=getattr(self, "gdd_params", {}),
            )
        return _dispatch(entity_list)

    @requires_token
    @require_gdd_params
    def _process_entity_gdd_bulk_parallel_inner(
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
        prefix="gdd",
        generate_report=False,
        report_options=None,
    ):
        params = params_kw
        self.logger.info(f"🚀 Starting bulk GDD extraction for {len(entity_list)} entities")

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
                executor.submit(self.process_single_entity_gdd, row, params): self.get_entity_value(row, "id")
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
                "parameters": getattr(self, "gdd_params", {}),
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
