"""Location-Based Field Border extractor.

Wraps the Geosys ``/field-borders/v1/AutomaticBoundary`` endpoint, which takes a
``longitude,latitude`` pair and returns the field polygon containing that point.
The extractor consumes a DataFrame whose ``geometry`` column carries a point WKT
and emits one row per input entity with a ``polygon_geometry`` column (WKT).

Mirrors the rest of the extractor family (BaseExtractor inheritance, setup +
get_*_api + format_*_json + process_single + bulk-parallel pattern, retry with
backoff, partial exports, optional cache, optional HTML report).
"""

#  Standard Library
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

import pandas as pd

#  Third-Party Libraries
import requests
from shapely import wkt as shapely_wkt
from shapely.errors import ShapelyError
from shapely.geometry import shape as shapely_shape
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


def require_location_based_border_params(func):
    """Decorator to ensure location-based-border parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "location_based_border_params") or self.location_based_border_params is None:
            error_msg = (
                "No location-based-border parameters found. Call setup_location_based_border_parameters() first."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class LocationBasedBorderExtractor(BaseExtractor):
    """
    Extracts a field-border polygon from a single longitude/latitude location.

    Calls the Geosys ``/field-borders/v1/AutomaticBoundary`` endpoint with a
    ``location=lon,lat`` query parameter and returns the polygon (in WKT) of
    the field that contains that point. Useful for converting a list of
    geocoded locations (CSV / API output) into proper field geometries.

    Documentation: https://api.geosys-na.net/field-borders/v1/swagger/index.html
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_LocationBasedBorder.ipynb

    Args (setup_location_based_border_parameters):
        simplified_geom (bool): If True, returns a simplified field geometry
            (lower shape-point count). Default: True. Maps to the API's
            ``simplified_geom`` query parameter.
        partial_frequency (int): How often to flush partial results. Default: 50.
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id (required) - used to stamp results, log lines, and cache keys.
        geometry (required) - **must be a Point WKT** (e.g. ``POINT(-93.6 41.5)``).
            If the row carries a non-point geometry the entity is rejected with
            a clean error rather than being silently degraded to a centroid.

    Output columns:
        entity_id, point_geometry (echo of input point as WKT),
        polygon_geometry (returned field border as WKT),
        plus any flat scalar fields the API returns alongside ``geometry``
        (e.g. area, sourceId).
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.location_based_border_params = None
        self.location_based_border_url = agro_urls["location_based_border_url"][self.env]

        self.logger.debug(f"Location-based border URL configured: {self.location_based_border_url}")
        self.logger.info(f"LocationBasedBorderExtractor ready for {self.env} environment")

    def get_new_token(self):
        """Implements token refresh logic for LocationBasedBorderExtractor."""
        self.logger.debug("Getting new token for LocationBasedBorderExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------

    def setup_location_based_border_parameters(
        self,
        simplified_geom: bool = True,
        partial_frequency: int = 50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """Configure parameters for location-based-border extraction.

        Args:
            simplified_geom (bool): If True, returns simplified field geometry. Default: True.
            partial_frequency (int): How often to flush partial results. Default: 50.
            column_mapping (dict): Column name overrides (e.g. ``{"geometry": "point_wkt"}``).
            output_mapping (dict): Output column renames.
            exclude_columns (list): Columns to exclude from output.
            output_columns (list): Whitelist of output columns.
            use_cache (bool): Whether to use caching for this run.
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring location-based-border parameters...")

        if not isinstance(simplified_geom, bool):
            error_msg = f"Invalid simplified_geom={simplified_geom}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(partial_frequency, int) or partial_frequency < 0 or isinstance(partial_frequency, bool):
            error_msg = f"Invalid partial_frequency={partial_frequency}. Must be a non-negative integer."
            log.error(error_msg)
            raise ValueError(error_msg)

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.apply_cache_setting(use_cache)

        self.location_based_border_params = {
            "simplified_geom": simplified_geom,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        geom_col = self.get_mapped_column("geometry")
        self.cache_key_columns = [id_col, geom_col]

        self.configure_output(
            output_mapping=output_mapping,
            exclude_columns=exclude_columns,
            output_columns=output_columns,
        )

        log.success("Location-based-border parameters configured successfully")
        log.debug(f"Parameters: {self.location_based_border_params}")

    # -------------------------------------------------------------------------
    # API CALL
    # -------------------------------------------------------------------------

    @requires_token
    @require_location_based_border_params
    def get_location_based_border_api(self, entity_data):
        """Call the AutomaticBoundary API for a single entity.

        Args:
            entity_data (dict|Series): Entity data with ``id`` and ``geometry``
                (a POINT WKT, e.g. ``POINT(-93.6 41.5)``).

        Returns:
            dict: Parsed JSON response from the API. Always carries the input
            ``id`` and ``point_wkt`` echoed back via ``setdefault`` so the
            downstream formatter can stamp them.
        """
        params = self.location_based_border_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

        # Required fields
        if not self.has_entity_field(entity_data, "id"):
            error_msg = "entity_data must include an 'id' field."
            log.error(error_msg)
            raise ValueError(error_msg)
        if not self.has_entity_field(entity_data, "geometry"):
            error_msg = f"entity_data for {entity_id} must include a 'geometry' field (point WKT)."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate + parse the point WKT.
        raw_geom = self.get_entity_value(entity_data, "geometry")
        try:
            validate_wkt(raw_geom)
            geom_obj = shapely_wkt.loads(raw_geom) if isinstance(raw_geom, str) else raw_geom
        except (ValueError, ShapelyError) as e:
            log.error(f"Invalid geometry for entity {entity_id}: {e}")
            raise ValueError(f"Invalid geometry for entity {entity_id}: {e}") from e

        if geom_obj.geom_type != "Point":
            error_msg = (
                f"Entity {entity_id}: geometry must be a Point (got {geom_obj.geom_type}). "
                "Pre-process polygons to a centroid via core.geometry.get_centroid_wkt()."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        lon, lat = geom_obj.x, geom_obj.y
        point_wkt = geom_obj.wkt

        # Build URL (location=lon,lat&simplified_geom=...).
        full_url = (
            f"{self.location_based_border_url}"
            f"?location={lon},{lat}"
            f"&simplified_geom={str(bool(params['simplified_geom'])).lower()}"
        )

        log.info(f"Requesting field border for entity {entity_id} (location={lon},{lat})")
        log.debug(f"API URL: {full_url}")

        try:
            response = requests.get(
                full_url,
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
            json_response = response.json()
            log.success(f"Field border retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")

            # The endpoint occasionally double-encodes — `response.json()` gives
            # back a string that is itself JSON. Parse it through one more time
            # so downstream code always sees a dict.
            if isinstance(json_response, str):
                try:
                    json_response = json.loads(json_response)
                except (json.JSONDecodeError, ValueError) as e:
                    log.warning(
                        f"Entity {entity_id}: response was a string and not parseable as JSON ({e}); "
                        "passing through unchanged."
                    )

            # Stamp inputs for the formatter. Preserve the API's own ``id``
            # (the field-border identifier) under a different key so it isn't
            # lost when we overwrite ``id`` with the row's entity id.
            if isinstance(json_response, dict):
                if "id" in json_response and json_response["id"] != entity_id:
                    json_response.setdefault("field_border_id", json_response["id"])
                json_response["id"] = entity_id
                json_response["point_wkt"] = point_wkt
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"Timeout requesting field border for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            log.error(f"HTTP error for entity {entity_id}: {status} - {text}")
            raise
        except Exception as e:
            log.error(f"Unexpected error for entity {entity_id}: {e}")
            raise

    def get_location_based_border_api_safe(self, entity_data):
        """Safe wrapper around get_location_based_border_api().

        Returns:
            dict: ``{"success": bool, "data": dict|None, "error": str|None, "entity_id": str}``
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")
        try:
            response = self.get_location_based_border_api(entity_data)
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

    @require_location_based_border_params
    def format_location_based_border_json(self, response_json, params=None):
        """Normalize an AutomaticBoundary response into a single-row DataFrame.

        Accepts three shapes (the API's response schema isn't documented in the
        Swagger so we tolerate the common variants):

        1. Flat dict with a ``geometry`` field — string WKT or GeoJSON geometry.
        2. GeoJSON Feature: ``{"type": "Feature", "geometry": {...}, "properties": {...}}``.
        3. GeoJSON FeatureCollection — uses ``features[0]``.

        Args:
            response_json (dict): API response (augmented with ``id`` and
                ``point_wkt`` by ``get_location_based_border_api``).
            params (dict, optional): Override params (unused here, kept for symmetry).

        Returns:
            pd.DataFrame: Single-row DataFrame with ``entity_id``,
            ``point_geometry``, ``polygon_geometry``, and any flat scalar
            properties returned by the API.
        """
        log = self.get_contextualized_logger("FORMAT")

        # Tolerate the double-encoded shape: if the API method (or a manual
        # caller passing safe['data']) hands us a JSON string instead of a
        # dict, parse it once. Plain non-JSON strings still fall through to
        # the dict-required raise below.
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"response_json must be a dictionary; received a string that is not valid JSON ({e})."
                log.error(error_msg)
                raise ValueError(error_msg) from e

        if not isinstance(response_json, dict):
            error_msg = "response_json must be a dictionary."
            log.error(error_msg)
            raise ValueError(error_msg)

        entity_id = response_json.get("id")
        validation = self.validate_api_response(response_json, entity_id, "location_based_border")
        if isinstance(validation, pd.DataFrame):
            return validation

        # Resolve the geometry source from any of the supported shapes.
        feature: dict | None = None
        geom_payload = None
        properties: dict = {}

        if response_json.get("type") == "Feature":
            feature = response_json
        elif response_json.get("type") == "FeatureCollection":
            features = response_json.get("features") or []
            if features:
                feature = features[0]

        if feature is not None:
            geom_payload = feature.get("geometry")
            properties = feature.get("properties") or {}
        else:
            # Flat dict — geometry can be a WKT string or a GeoJSON dict.
            geom_payload = response_json.get("geometry")

        polygon_wkt = self._geometry_payload_to_wkt(geom_payload, entity_id, log)

        row: dict = {
            "entity_id": entity_id,
            "point_geometry": response_json.get("point_wkt"),
            "polygon_geometry": polygon_wkt,
        }

        # Promote flat scalar properties (numbers, strings, bools) onto the row
        # — useful when the API returns area, sourceId, etc.
        reserved = {"id", "point_wkt", "type", "geometry", "features", "properties"}
        for k, v in response_json.items():
            if k in reserved or k in row:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        # When promoting GeoJSON ``properties``, skip ``id`` — it's the API's
        # field-border id which we've already preserved as ``field_border_id``
        # at the top level.
        for k, v in properties.items():
            if k in row or k == "id":
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v

        log.debug(f"Formatted location-based-border row for entity {entity_id} with {len(row)} columns")
        return pd.DataFrame([row])

    @staticmethod
    def _geometry_payload_to_wkt(geom_payload, entity_id, log) -> str | None:
        """Convert a geometry payload (WKT string, GeoJSON dict, or None) to WKT."""
        if geom_payload is None:
            log.warning(f"Entity {entity_id}: no geometry in API response")
            return None
        if isinstance(geom_payload, str):
            try:
                return validate_wkt(geom_payload)
            except (ValueError, ShapelyError) as e:
                log.warning(f"Entity {entity_id}: returned geometry is not valid WKT ({e})")
                return None
        if isinstance(geom_payload, dict):
            try:
                return shapely_shape(geom_payload).wkt
            except (ValueError, ShapelyError, KeyError, TypeError) as e:
                log.warning(f"Entity {entity_id}: cannot convert GeoJSON geometry to WKT ({e})")
                return None
        log.warning(f"Entity {entity_id}: unexpected geometry payload type {type(geom_payload).__name__}")
        return None

    # -------------------------------------------------------------------------
    # PROCESS SINGLE ENTITY
    # -------------------------------------------------------------------------

    @requires_token
    @require_location_based_border_params
    @cache_single_entity("location_based_border_params")
    def process_single_entity_location_based_border(self, row, params=None):
        """Process location-based-border extraction for a single entity with retry logic.

        Returns:
            dict: ``{"data": DataFrame or None, "error": dict or None}``
        """
        if params is None:
            params = self.location_based_border_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Validate geometry early so failures don't burn retries.
        try:
            geom_raw = self.get_entity_value(row, "geometry")
            validate_wkt(geom_raw)
            geom_obj = shapely_wkt.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
            if geom_obj.geom_type != "Point":
                raise ValueError(
                    f"geometry must be a Point (got {geom_obj.geom_type}). Pre-process polygons to a centroid."
                )
        except Exception as e:
            log.error(f"Validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        def _call_api():
            return self.get_location_based_border_api(row)

        try:
            log.info(f"Calling AutomaticBoundary API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(
                func=_call_api,
                max_retries=5,
                base_delay=1.0,
                max_delay=60.0,
            )

            border_df = self.format_location_based_border_json(raw_json)
            if border_df is None or border_df.empty:
                log.warning(f"No border returned for entity {entity_id}")
                return {
                    "data": None,
                    "error": {"message": "No border returned for entity", "entity_id": entity_id},
                }

            log.success(f"Successfully processed entity {entity_id}")
            return {
                "data": normalize_with_metadata(row, border_df),
                "error": None,
            }

        except Exception as e:
            log.error(f"Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    # -------------------------------------------------------------------------
    # BULK PROCESSING
    # -------------------------------------------------------------------------

    def process_location_based_border_bulk_extraction_parallel(
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
        prefix="location_based_border",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """Bulk processing of location-based-border requests using threads + progress bar.

        Args mirror the family pattern (``coverage_function`` / ``difference_functions`` /
        ``processor_change_index_functions``). Returns the standard
        ``{results_df, global_errors, total_entities, total_calculations,
        successful_calculations, failed_calculations, failed_ids}`` dict.
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_location_based_border_bulk_extraction_parallel_inner,
                entity_list=entity_list,
                params=self.location_based_border_params,
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
        return self._process_location_based_border_bulk_extraction_parallel_inner(
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
    @require_location_based_border_params
    def _process_location_based_border_bulk_extraction_parallel_inner(
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
        prefix="location_based_border",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing of location-based-border requests."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk location-based-border extraction: {prefix}")
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
        failed_ids: list = []

        log.info(f"Processing {len(filtered_entity_list)} entities in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_location_based_border, row, params): self.get_entity_value(
                    row, "id"
                )
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"Processing {prefix.upper()}", unit="entity") as pbar:
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
        log.success(f"Successful: {successful_calculations}/{total_calculations}")
        log.info(f"Failed: {total_calculations - successful_calculations}/{total_calculations}")
        log.info("=" * 60)

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
                "parameters": getattr(self, "location_based_border_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("Bulk location-based-border extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
