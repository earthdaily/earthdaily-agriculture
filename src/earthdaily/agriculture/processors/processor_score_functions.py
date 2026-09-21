# processor_score_functions.py - Historical and in-season potential / risk score processors
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
    normalize_date,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
    validate_historical_years,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

available_crops = {"CORN", "SECOND CORN", "SOYBEANS", "SUGARCANE", "COTTON", "OTHERS"}


def validate_crop(crop: str, available_crops: set):
    """
    Validate and normalize the crop type for historical_score requests.

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


def validate_historical_seasons(historical_seasons):
    """
    Validate historical_seasons parameter.
    Delegates to the central validate_historical_years() in api_utils.

    Args:
        historical_seasons (list, str, or None): List of historical years, comma-separated string, or None.

    Returns:
        list or None: Validated historical_seasons
    """
    result = validate_historical_years(historical_seasons)
    # validate_historical_years may return int (N-year lookback) or "ALL",
    # but historical_seasons only accepts list or None.
    if isinstance(result, int) or result == "ALL":
        raise ValueError(f"historical_seasons must be a list of specific years or None, got: {historical_seasons}")
    return result


def require_historical_score_params(func):
    """Decorator to ensure historical_score parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "historical_score_params"):
            self.logger.error("No historical_score parameters found")
            raise RuntimeError(
                "❌ No historical_score parameters found. Call setup_historical_score_parameters() first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def require_inseason_score_params(func):
    """Decorator to ensure inseason_score parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "inseason_score_params"):
            self.logger.error("No inseason_score parameters found")
            raise RuntimeError("❌ No inseason_score parameters found. Call setup_inseason_score_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class HistoricalScoreExtractor(BaseExtractor):
    """
    Extracts historical potential/risk score analytics for agricultural entities.

    Computes historical performance-based risk assessment scores by comparing vegetation
    index patterns across multiple years. Supports configurable season windows, historical
    season comparison, and multiple detail levels.

    Documentation: https://docs.earthdaily.com/agro/library/Historical_Potential_Score/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_historical_score.ipynb

    Args (setup_historical_score_parameters):
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        threshold_start (float): Score threshold start. Default: 0.7
        year (int): Target year. Default: 2025
        historical_seasons (list): Historical season list for comparison. Default: None
        data_source (str): Data source ('LR', 'MR'). Default: 'LR'
        detail_level (str): Output detail ('full' or 'summary'). Default: 'full'
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, sowing_date (optional); historical_seasons (optional)

    Output columns:
        entity_id, score, percentile, rank, + historical comparison metrics
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.historical_score_params = None

        # Get ressource URL
        self.historical_score_url = agro_urls["historical_score_url"][self.env]

        self.logger.info(f"🛰️ HistoricalScoreExtractor initialized for env: {self.env}")
        self.logger.debug(f"API endpoint: {self.historical_score_url}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for historical_scoreExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.info("🔑 Refreshing API token for HistoricalScoreExtractor...")
        try:
            new_token, exp_time = EDAuthenticator.get_new_token(env=self.env)
            self.logger.info("✅ Token refreshed successfully")
            self.logger.debug(f"New token expires at: {datetime.fromtimestamp(exp_time)}")
            return new_token, exp_time
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {str(e)}")
            raise

    def setup_historical_score_parameters(
        self,
        season_duration=120,
        season_start_day=1,
        season_start_month=4,
        threshold_start=0.7,
        year=2025,
        historical_seasons=None,
        data_source="LR",
        publish_af=False,
        partial_frequency=50,
        detail_level="full",
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for historical_score extraction

        Args:
            season_duration (int): Crop cycle duration in days
            season_start_day (int): Crop cycle start day (1-31)
            season_start_month (int): Crop cycle start month (1-12)
            threshold_start (float): Threshold for season start detection (0.0-1.0). Default: 0.7
            year (int): Current year to analyze
            historical_seasons (list or None, optional): List of historical years for comparison (e.g., [2024, 2023, 2022]).
                                                        If None, no historical comparison. Default: None
            data_source (str): Imagery type - 'LR' (low resolution) or 'MR' (medium resolution)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
            detail_level (str): Output detail level - 'summary' (only averages) or 'full' (all per-season data). Default: 'full'
        """
        self.logger.info("⚙️ Setting up historical score parameters...")

        # Month validation
        if not (1 <= season_start_month <= 12):
            self.logger.error(f"Invalid season_start_month={season_start_month}")
            raise ValueError(f"Invalid season_start_month={season_start_month}. Must be between 1 and 12.")

        # Day validation
        if not (1 <= season_start_day <= 31):
            self.logger.error(f"Invalid season_start_day={season_start_day}")
            raise ValueError(f"Invalid season_start_day={season_start_day}. Must be between 1 and 31.")

        # Threshold start validation
        if not isinstance(threshold_start, (int, float)):
            self.logger.error(f"Invalid threshold_start type: {type(threshold_start)}")
            raise ValueError(f"Invalid threshold_start={threshold_start}. Must be a numeric value (int or float).")

        if not (0 <= threshold_start <= 1):
            self.logger.error(f"Invalid threshold_start value: {threshold_start}")
            raise ValueError(f"Invalid threshold_start={threshold_start}. Must be between 0 and 1.")

        # Historical seasons validation - using module-level helper
        try:
            historical_seasons = validate_historical_seasons(historical_seasons)
            if historical_seasons:
                self.logger.debug(f"Historical seasons: {historical_seasons}")
        except Exception as e:
            self.logger.error(f"Historical seasons validation failed: {str(e)}")
            raise

        # Data source validation
        available_sources = {"LR", "MR"}
        if data_source not in available_sources:
            self.logger.error(f"Invalid data_source '{data_source}'")
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from: {available_sources}")

        # Publish AF validation
        if not isinstance(publish_af, bool):
            self.logger.error(f"Invalid publish_af type: {type(publish_af)}")
            raise ValueError(f"Invalid publish_af={publish_af}. Must be True or False.")

        # Detail level validation
        valid_detail_levels = ["summary", "full"]
        if detail_level not in valid_detail_levels:
            self.logger.error(f"Invalid detail_level '{detail_level}'")
            raise ValueError(f"Invalid detail_level='{detail_level}'. Choose from: {valid_detail_levels}")

        self.apply_cache_setting(use_cache)

        self.historical_score_params = {
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "threshold_start": threshold_start,
            "year": year,
            "historical_seasons": historical_seasons,
            "data_source": data_source,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
            "detail_level": detail_level,
        }

        # Configure cache key columns for historical score results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.logger.info("✅ Historical score parameters configured:")
        for k, v in self.historical_score_params.items():
            self.logger.debug(f"   {k}: {v}")

        print("🌍 Historical score parameters configured:")
        for k, v in self.historical_score_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_historical_score_params
    def get_historical_score_api(self, entity_data: dict):
        """
        Request historical_score for an entity

        Args:
            entity_data (dict): Entity data containing:
                - 'id' (str): Entity identifier
                - 'geometry' (str): WKT geometry
                - 'historical_seasons' (list or None, optional): List of historical years (e.g., [2024, 2023, 2022]).
                If not provided, defaults to None (no historical comparison).

        Returns:
            dict: API response JSON
        """
        entity_id = self.get_entity_value(entity_data, "id")
        self.logger.debug(f"Requesting historical score for entity: {entity_id}")

        params = self.historical_score_params

        # Step 1: Input validation
        if not self.has_entity_field(entity_data, "id"):
            self.logger.error("Entity data missing 'id' field")
            raise ValueError("❌ entity_data must include a 'id' field.")

        # Geometry validation
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

        # Historical seasons validation - using module-level helper
        try:
            historical_seasons = validate_historical_seasons(self.get_entity_value(entity_data, "historical_seasons"))
            if historical_seasons:
                self.logger.debug(f"Entity {entity_id}: Historical seasons - {historical_seasons}")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Historical seasons validation failed - {str(e)}")
            raise

        # Step 2: Build API URL
        url = f"{self.historical_score_url}/launch"

        params_list = [
            f"seasonDuration={params['season_duration']}",
            f"seasonStartDay={params['season_start_day']}",
            f"seasonStartMonth={params['season_start_month']}",
            f"thresholdStart={params['threshold_start']}",
            f"year={params['year']}",
            f"dataSource={params['data_source']}",
        ]

        full_url = f"{url}?{'&'.join(params_list)}"
        self.logger.debug(f"Entity {entity_id}: API URL constructed")

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Step 3: Build payload
        payload = {"geometry": geometry}

        # Add id field
        if params.get("publish_af", False):
            payload["id"] = f"SeasonField:{self.get_entity_value(entity_data, 'id')}@LEGACY_ID_NA"
        else:
            payload["id"] = ""

        # Add historicalSeasons only if it's not None
        if historical_seasons is not None:
            payload["historicalSeasons"] = historical_seasons

        # Step 4: Request data
        try:
            self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
            self.logger.debug(f"Entity {entity_id}: Payload: {payload}")
            response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=60)
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

    def get_historical_score_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_historical_score_api().
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
            response_json = self.get_historical_score_api(entity_data)
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

    @require_historical_score_params
    def format_historical_score_json(self, response_historical_score_json, params=None, detail_level="full"):
        """
        Normalize historical_score API response into a clean pandas DataFrame.

        Args:
            response_historical_score_json (dict): API response JSON with keys 'id' and 'data'.
            params (dict, optional): historical_score parameters (defaults to self.historical_score_params).
            detail_level (str, optional): Output detail level:
                - 'summary': Returns only summary metrics (one row)
                - 'full': Returns summary + all per-season data in one row (wide format). Default: 'full'

        Returns:
            pd.DataFrame: Normalized historical_score data (single row).
                - 'summary': Columns: entity_id, average_potential_score, olympic_mean_potential_score,
                            standard_deviation, risk_score
                - 'full': Summary + potential_score_YYYY, season_break_YYYY for each season
        """
        if params is None:
            params = self.historical_score_params

        entity_id = response_historical_score_json.get("id", None)
        self.logger.debug(f"Entity {entity_id}: Formatting JSON response with detail_level='{detail_level}'")

        # Validate detail_level
        valid_levels = ["summary", "full"]
        if detail_level not in valid_levels:
            self.logger.error(f"Invalid detail_level '{detail_level}'")
            raise ValueError(f"Invalid detail_level='{detail_level}'. Choose from: {valid_levels}")

        if not isinstance(response_historical_score_json, dict):
            self.logger.error(f"Entity {entity_id}: Response is not a dictionary")
            raise ValueError("❌ response_historical_score_json must be a dictionary.")

        data = response_historical_score_json.get("data", None)

        if data is None:
            self.logger.warning(f"Entity {entity_id}: No 'data' key found in response")
            print(f"⚠️ No 'data' key found in response for entity {entity_id}")
            return pd.DataFrame()

        if not isinstance(data, dict):
            self.logger.warning(f"Entity {entity_id}: Unexpected 'data' format: {type(data)}")
            print(f"⚠️ Unexpected 'data' format: {type(data)}")
            return pd.DataFrame()

        # Extract summary metrics
        row = {
            "entity_id": entity_id,
            "average_potential_score": data.get("AveragePotentialScore"),
            "olympic_mean_potential_score": data.get("OlympicMeanPotentialScore"),
            "standard_deviation": data.get("StandardDeviation"),
            "risk_score": data.get("RiskScore"),
        }

        # If summary only, return single row with just summary metrics
        if detail_level == "summary":
            df = pd.DataFrame([row])
            self.logger.debug(f"Entity {entity_id}: Formatted summary-only data")
            return df

        # Full detail: Add per-season columns
        # Parse PotentialScores (it's a JSON string, not a list)
        potential_scores_raw = data.get("PotentialScores", "[]")
        try:
            potential_scores = (
                json.loads(potential_scores_raw) if isinstance(potential_scores_raw, str) else potential_scores_raw
            )
        except json.JSONDecodeError:
            self.logger.warning(f"Entity {entity_id}: Failed to parse PotentialScores JSON")
            potential_scores = []

        for score_entry in potential_scores:
            season = score_entry.get("season")
            potential_score = score_entry.get("potentialScore")
            row[f"potential_score_{season}"] = potential_score

        # Parse SeasonBreaks (it's also a JSON string)
        season_breaks_raw = data.get("SeasonBreaks", "[]")
        try:
            season_breaks = json.loads(season_breaks_raw) if isinstance(season_breaks_raw, str) else season_breaks_raw
        except json.JSONDecodeError:
            self.logger.warning(f"Entity {entity_id}: Failed to parse SeasonBreaks JSON")
            season_breaks = []

        for break_entry in season_breaks:
            season = break_entry.get("season")
            season_break = break_entry.get("seasonBreak")
            row[f"season_break_{season}"] = season_break

        # Convert to DataFrame (single row)
        df = pd.DataFrame([row])
        self.logger.debug(f"Entity {entity_id}: Formatted full data with {len(potential_scores)} seasons")

        return df

    @requires_token
    @require_historical_score_params
    @cache_single_entity("historical_score_params")
    def process_single_entity_historical_score(self, row, params=None):
        if params is None:
            params = self.historical_score_params

        entity_id = self.get_entity_value(row, "id")
        self.logger.debug(f"Entity {entity_id}: Processing single entity...")

        # Step 1: validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            crop = validate_crop(self.get_entity_value(row, "crop"), available_crops)
            self.logger.debug(f"Entity {entity_id}: Input validation passed")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Validation failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 2: define API call wrapper
        def _call_api():
            return self.get_historical_score_api(
                {
                    self.get_mapped_column("id"): entity_id,
                    self.get_mapped_column("crop"): crop,
                    self.get_mapped_column("geometry"): geometry,
                }
            )

        try:
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Step 4: package response and errors
            historical_score_df = self.format_historical_score_json(raw_json)

            if historical_score_df is None or historical_score_df.empty:
                self.logger.warning(f"Entity {entity_id}: No historical_score results found")
                return {"data": None, "error": {"message": "No historical_score results found", "entity_id": entity_id}}
            else:
                self.logger.info(f"Entity {entity_id}: Successfully processed")
                return {"data": normalize_with_metadata(row, historical_score_df), "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_historical_score_params
    def process_historical_score_bulk_extraction_parallel(
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
        prefix="historical_score",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of historical_score requests using threads + progress bar,
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "historical_score".
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
                bulk_method=self._process_historical_score_bulk_extraction_parallel_inner,
                params=self.historical_score_params,
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

        return self._process_historical_score_bulk_extraction_parallel_inner(
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
    @require_historical_score_params
    def _process_historical_score_bulk_extraction_parallel_inner(
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
        prefix="historical_score",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        self.logger.info(f"🚀 Starting bulk historical score extraction for {len(entity_list)} entities")
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

        print(f"🔄 Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(self.process_single_entity_historical_score, row, params): self.get_entity_value(
                    row, "id"
                )
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
                "parameters": getattr(self, "historical_score_params", {}),
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

        self.logger.info("✅ Bulk historical score extraction complete")
        return summary


class InseasonScoreExtractor(BaseExtractor):
    """
    Extracts in-season potential/risk score analytics for agricultural entities.

    Computes current-season risk assessment using predictive analytics based on vegetation
    index trends. Provides real-time scoring and comparison with historical baselines.

    Documentation: https://docs.earthdaily.com/agro/library/In-season_Potential_Score/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_inseason_score.ipynb

    Args (setup_inseason_score_parameters):
        season_duration (int): Season length in days. Default: 120
        season_start_day (int): Season start day of month. Default: 1
        season_start_month (int): Season start month. Default: 4
        threshold_start (float): Score threshold start. Default: 0.7
        year (int): Target year. Default: 2025
        data_source (str): Data source ('LR', 'MR'). Default: 'LR'
        detail_level (str): Output detail ('full' or 'summary'). Default: 'full'
        historical_seasons (list): Explicit prior season years for the baseline. Default: None
        nb_historical_year (int): Number of historical years used for the baseline. Default: 1
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, sowing_date (optional)

    Output columns:
        entity_id, score, status, trend, + current season metrics
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        #  Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.inseason_score_params = None

        # Get ressource URL
        self.inseason_score_url = agro_urls["inseason_score_url"][self.env]

        self.logger.info(f"🛰️ InseasonScoreExtractor initialized for env: {self.env}")
        self.logger.debug(f"API endpoint: {self.inseason_score_url}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for InseasonScoreExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.info("🔑 Refreshing API token for InseasonScoreExtractor...")
        try:
            new_token, exp_time = EDAuthenticator.get_new_token(env=self.env)
            self.logger.info("✅ Token refreshed successfully")
            self.logger.debug(f"New token expires at: {datetime.fromtimestamp(exp_time)}")
            return new_token, exp_time
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {str(e)}")
            raise

    def setup_inseason_score_parameters(
        self,
        season_duration=120,
        season_start_day=1,
        season_start_month=4,
        nb_historical_year=1,
        threshold_start=0.7,
        historical_seasons=None,
        data_source="LR",
        publish_af=False,
        partial_frequency=50,
        detail_level="full",
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for inseason_score extraction

        Args:
            season_duration (int): Crop cycle duration in days
            season_start_day (int): Crop cycle start day (1-31)
            season_start_month (int): Crop cycle start month (1-12)
            nb_historical_year (int): Number of historical years to include (must be >= 1). Default: 1
            threshold_start (float): Threshold for season start detection (0.0-1.0). Default: 0.7
            historical_seasons (list or None, optional): List of historical years for comparison (e.g., [2024, 2023, 2022]).
                                                        If None, no historical comparison. Default: None
            data_source (str): Imagery type - 'LR' (low resolution) or 'MR' (medium resolution)
            publish_af (bool): Whether to publish to AF (includes id in payload if True). Default: False
            partial_frequency (int): How often to save partial results
            detail_level (str): Output detail level - 'summary' (only averages) or 'full' (all per-season data). Default: 'full'
        """
        self.logger.info("⚙️ Setting up in-season score parameters...")

        # Month validation
        if not (1 <= season_start_month <= 12):
            self.logger.error(f"Invalid season_start_month={season_start_month}")
            raise ValueError(f"Invalid season_start_month={season_start_month}. Must be between 1 and 12.")

        # Day validation
        if not (1 <= season_start_day <= 31):
            self.logger.error(f"Invalid season_start_day={season_start_day}")
            raise ValueError(f"Invalid season_start_day={season_start_day}. Must be between 1 and 31.")

        # Number of historical years validation
        if not isinstance(nb_historical_year, int):
            self.logger.error(f"Invalid nb_historical_year type: {type(nb_historical_year)}")
            raise ValueError(f"Invalid nb_historical_year={nb_historical_year}. Must be an integer.")

        if nb_historical_year < 1:
            self.logger.error(f"Invalid nb_historical_year value: {nb_historical_year}")
            raise ValueError(f"Invalid nb_historical_year={nb_historical_year}. Must be at least 1.")

        # Threshold start validation
        if not isinstance(threshold_start, (int, float)):
            self.logger.error(f"Invalid threshold_start type: {type(threshold_start)}")
            raise ValueError(f"Invalid threshold_start={threshold_start}. Must be a numeric value (int or float).")

        if not (0 <= threshold_start <= 1):
            self.logger.error(f"Invalid threshold_start value: {threshold_start}")
            raise ValueError(f"Invalid threshold_start={threshold_start}. Must be between 0 and 1.")

        # Historical seasons validation - using module-level helper
        try:
            historical_seasons = validate_historical_seasons(historical_seasons)
            if historical_seasons:
                self.logger.debug(f"Historical seasons: {historical_seasons}")
        except Exception as e:
            self.logger.error(f"Historical seasons validation failed: {str(e)}")
            raise

        # Data source validation
        available_sources = {"LR", "MR"}
        if data_source not in available_sources:
            self.logger.error(f"Invalid data_source '{data_source}'")
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from: {available_sources}")

        # Publish AF validation
        if not isinstance(publish_af, bool):
            self.logger.error(f"Invalid publish_af type: {type(publish_af)}")
            raise ValueError(f"Invalid publish_af={publish_af}. Must be True or False.")

        # Detail level validation
        valid_detail_levels = ["summary", "full"]
        if detail_level not in valid_detail_levels:
            self.logger.error(f"Invalid detail_level '{detail_level}'")
            raise ValueError(f"Invalid detail_level='{detail_level}'. Choose from: {valid_detail_levels}")

        self.apply_cache_setting(use_cache)

        self.inseason_score_params = {
            "season_duration": season_duration,
            "season_start_month": season_start_month,
            "season_start_day": season_start_day,
            "nb_historical_year": nb_historical_year,
            "threshold_start": threshold_start,
            "historical_seasons": historical_seasons,
            "data_source": data_source,
            "publish_af": publish_af,
            "partial_frequency": partial_frequency,
            "detail_level": detail_level,
        }

        # Configure cache key columns for in-season score results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.logger.info("✅ In-season score parameters configured:")
        for k, v in self.inseason_score_params.items():
            self.logger.debug(f"   {k}: {v}")

        print("🌍 In-Season score parameters configured:")
        for k, v in self.inseason_score_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_inseason_score_params
    def get_inseason_score_api(self, entity_data: dict):
        """
        Request inseason_score for an entity

        Args:
            entity_data (dict): Entity data containing:
                - 'id' (str): Entity identifier
                - 'geometry' (str): WKT geometry
                - 'crop' (str): Crop code (e.g., 'CORN', 'WHEAT', 'OTHERS')
                - 'sowing_date' (str): Sowing date in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format
                - 'end_date' (str, optional): End date in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format.
                If not provided, calculated as sowing_date + season_duration.
                - 'historical_seasons' (list or None, optional): List of historical years (e.g., [2024, 2023, 2022]).
                If not provided, defaults to None (no historical comparison).

        Returns:
            dict: API response JSON
        """
        params = self.inseason_score_params
        entity_id = self.get_entity_value(entity_data, "id")
        self.logger.debug(f"Requesting in-season score for entity: {entity_id}")

        # Step 1: Input validation
        if not self.has_entity_field(entity_data, "id"):
            self.logger.error("Entity data missing 'id' field")
            raise ValueError("❌ entity_data must include a 'id' field.")

        # Geometry validation
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

        # Crop validation - using module-level helper
        if not self.has_entity_field(entity_data, "crop"):
            self.logger.error(f"Entity {entity_id}: Missing crop field")
            raise ValueError("❌ entity_data must include a 'crop' field.")

        crop = validate_crop(self.get_entity_value(entity_data, "crop"), available_crops)
        self.logger.debug(f"Entity {entity_id}: Crop validated - {crop}")

        # Sowing date validation and normalization - using module-level helper
        if not self.has_entity_field(entity_data, "sowing_date"):
            self.logger.error(f"Entity {entity_id}: Missing sowing_date field")
            raise ValueError("❌ entity_data must include a 'sowing_date' field.")

        sowing_date = normalize_date(self.get_entity_value(entity_data, "sowing_date"), field_name="sowing_date")
        sowing_dt = datetime.strptime(sowing_date, "%Y-%m-%d")  # For end_date calculation

        # End date validation (optional - calculate if not provided)
        end_date = self.get_entity_value(entity_data, "end_date")
        if end_date:
            # Normalize format if provided - using module-level helper
            end_date = normalize_date(end_date, field_name="end_date")
            self.logger.debug(f"Entity {entity_id}: End date provided - {end_date}")
        else:
            # Calculate end_date as sowing_date + season_duration
            season_duration = params["season_duration"]
            end_dt = sowing_dt + timedelta(days=season_duration)
            end_date = end_dt.strftime("%Y-%m-%d")
            self.logger.debug(
                f"Entity {entity_id}: End date calculated as {end_date} (sowing_date + {season_duration} days)"
            )
            print(f"ℹ️  end_date not provided, calculated as: {end_date} (sowing_date + {season_duration} days)")

        # Historical seasons validation - using module-level helper
        try:
            historical_seasons = validate_historical_seasons(self.get_entity_value(entity_data, "historical_seasons"))
            if historical_seasons:
                self.logger.debug(f"Entity {entity_id}: Historical seasons - {historical_seasons}")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Historical seasons validation failed - {str(e)}")
            raise

        # Step 2: Build API URL
        url = f"{self.inseason_score_url}/launch"

        # Compute effective nb_historical_year: if historical_seasons is a list,
        # use its length so the API fetches enough years to cover the specified seasons.
        nb_historical_year = params["nb_historical_year"]
        if historical_seasons is not None:
            nb_historical_year = max(nb_historical_year, len(historical_seasons))

        params_list = [
            f"seasonDuration={params['season_duration']}",
            f"seasonStartDay={params['season_start_day']}",
            f"seasonStartMonth={params['season_start_month']}",
            f"sowingDate={sowing_date}",
            f"numberHistoricalYears={nb_historical_year}",
            f"threshold={params['threshold_start']}",
            f"dataSource={params['data_source']}",
            f"endDate={end_date}",
            f"crop={crop}",
        ]

        full_url = f"{url}?{'&'.join(params_list)}"
        self.logger.debug(f"Entity {entity_id}: API URL constructed")

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Step 3: Build payload
        payload = {"geometry": geometry}

        # Add id field
        if params.get("publish_af", False):
            payload["id"] = f"SeasonField:{self.get_entity_value(entity_data, 'id')}@LEGACY_ID_NA"
        else:
            payload["id"] = ""

        # Add historicalSeasons only if it's not None
        if historical_seasons is not None:
            payload["historicalSeasons"] = historical_seasons

        # Step 4: Request data
        try:
            self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
            self.logger.debug(f"Entity {entity_id}: Payload: {payload}")
            response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=60)
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

    def get_inseason_score_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_inseason_score_api().
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
            response_json = self.get_inseason_score_api(entity_data)
            self.logger.debug(f"Entity {entity_id}: Safe API call succeeded")
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.warning(f"Entity {entity_id}: HTTP {status} error")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "seasonfield_id": entity_id}
        except Exception as e:
            self.logger.warning(f"Entity {entity_id}: Error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "seasonfield_id": entity_id}

    @require_inseason_score_params
    def format_inseason_score_json(self, response_inseason_score_json, params=None, detail_level="full"):
        """
        Normalize inseason_score API response into a clean pandas DataFrame.

        Args:
            response_inseason_score_json (dict): API response JSON with keys 'id' and 'data'.
            params (dict, optional): inseason_score parameters (defaults to self.inseason_score_params).
            detail_level (str, optional): Output detail level:
                - 'summary': Returns only the three main scores (one row)
                - 'full': Returns all scores (same as summary for in-season score). Default: 'full'

        Returns:
            pd.DataFrame: Normalized inseason_score data (single row).
                Columns: entity_id, historical_potential_score, inseason_potential_score,
                        relative_potential_score
        """
        if params is None:
            params = self.inseason_score_params

        entity_id = response_inseason_score_json.get("id", None)
        self.logger.debug(f"Entity {entity_id}: Formatting JSON response with detail_level='{detail_level}'")

        # Validate detail_level
        valid_levels = ["summary", "full"]
        if detail_level not in valid_levels:
            self.logger.error(f"Invalid detail_level '{detail_level}'")
            raise ValueError(f"Invalid detail_level='{detail_level}'. Choose from: {valid_levels}")

        if not isinstance(response_inseason_score_json, dict):
            self.logger.error(f"Entity {entity_id}: Response is not a dictionary")
            raise ValueError("❌ response_inseason_score_json must be a dictionary.")

        data = response_inseason_score_json.get("data", None)

        if data is None:
            self.logger.warning(f"Entity {entity_id}: No 'data' key found in response")
            print(f"⚠️ No 'data' key found in response for entity {entity_id}")
            return pd.DataFrame()

        if not isinstance(data, dict):
            self.logger.warning(f"Entity {entity_id}: Unexpected 'data' format: {type(data)}")
            print(f"⚠️ Unexpected 'data' format: {type(data)}")
            return pd.DataFrame()

        # Extract the three score metrics
        row = {
            "entity_id": entity_id,
            "historical_potential_score": data.get("historical_potential_score"),
            "inseason_potential_score": data.get("inseason_potential_score"),
            "relative_potential_score": data.get("relative_potential_score"),
        }

        # Convert to DataFrame (single row)
        df = pd.DataFrame([row])
        self.logger.debug(f"Entity {entity_id}: Formatted in-season score data")

        return df

    @requires_token
    @require_inseason_score_params
    @cache_single_entity("inseason_score_params")
    def process_single_entity_inseason_score(self, row, params=None):
        """
        Process a single entity for in-season score extraction.

        Args:
            row (pd.Series or dict): Entity data containing:
                - 'id': Entity identifier
                - 'geometry': WKT geometry
                - 'crop': Crop code
                - 'sowing_date': Sowing date in YYYY-MM-DD format
                - 'end_date' (optional): End date in YYYY-MM-DD format
                - 'historical_seasons' (optional): List of historical years
            params (dict, optional): Processing parameters

        Returns:
            dict: {'data': DataFrame or None, 'error': dict or None}
        """
        if params is None:
            params = self.inseason_score_params

        entity_id = self.get_entity_value(row, "id")
        self.logger.debug(f"Entity {entity_id}: Processing single entity...")

        # Step 1: Validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            crop = validate_crop(self.get_entity_value(row, "crop"), available_crops)
            self.logger.debug(f"Entity {entity_id}: Input validation passed")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Validation failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Get sowing_date
        sowing_date = self.get_entity_value(row, "sowing_date")
        if not sowing_date:
            self.logger.error(f"Entity {entity_id}: Missing sowing_date field")
            return {"data": None, "error": {"message": "Missing 'sowing_date' field in row", "entity_id": entity_id}}

        # Step 2: Build entity_data dict for API call
        entity_data = {
            self.get_mapped_column("id"): self.get_entity_value(row, "id"),
            self.get_mapped_column("crop"): crop,
            self.get_mapped_column("geometry"): geometry,
            self.get_mapped_column("sowing_date"): sowing_date,
        }

        # Add optional fields if present
        if self.has_entity_field(row, "end_date") and self.get_entity_value(row, "end_date"):
            entity_data[self.get_mapped_column("end_date")] = self.get_entity_value(row, "end_date")

        if (
            self.has_entity_field(row, "historical_seasons")
            and self.get_entity_value(row, "historical_seasons") is not None
        ):
            entity_data[self.get_mapped_column("historical_seasons")] = self.get_entity_value(row, "historical_seasons")

        # Step 3: Define API call wrapper
        def _call_api():
            return self.get_inseason_score_api(entity_data)

        try:
            # Step 4: Call API with retry logic
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(func=_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Ensure the response contains the entity ID
            if not raw_json.get("id"):
                raw_json["id"] = entity_id

            # Step 5: Format response
            inseason_score_df = self.format_inseason_score_json(raw_json)

            if inseason_score_df is None or inseason_score_df.empty:
                self.logger.warning(f"Entity {entity_id}: No inseason_score results found")
                return {"data": None, "error": {"message": "No inseason_score results found", "entity_id": entity_id}}
            else:
                self.logger.info(f"Entity {entity_id}: Successfully processed")
                return {"data": normalize_with_metadata(row, inseason_score_df), "error": None}

        except Exception as e:
            # Step 6: Handle error cleanly
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_inseason_score_params
    def process_inseason_score_bulk_extraction_parallel(
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
        prefix="inseason_score",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of inseason_score requests using threads + progress bar,
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "inseason_score".
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
                bulk_method=self._process_inseason_score_bulk_extraction_parallel_inner,
                params=self.inseason_score_params,
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

        return self._process_inseason_score_bulk_extraction_parallel_inner(
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
    @require_inseason_score_params
    def _process_inseason_score_bulk_extraction_parallel_inner(
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
        prefix="inseason_score",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        self.logger.info(f"🚀 Starting bulk in-season score extraction for {len(entity_list)} entities")
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

        print(f"🔄 Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(self.process_single_entity_inseason_score, row, params): self.get_entity_value(
                    row, "id"
                )
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
                "parameters": getattr(self, "inseason_score_params", {}),
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

        self.logger.info("✅ Bulk in-season score extraction complete")
        return summary
