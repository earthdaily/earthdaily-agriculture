#  Standard Library
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
    filter_timeseries_kpi,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
    validate_historical_years,
    validate_kpi_filter,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator


def require_vegetation_ts_params(func):
    """Decorator to ensure vegetation ts parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "vegetation_ts_params"):
            self.logger.error("No vegetation time series parameters found")
            raise RuntimeError(
                "❌ No vegetation time series parameters found. Call setup_vegetation_ts_parameters() first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def require_mrts_params(func):
    """Decorator to ensure MRTS parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "mrts_params") or self.mrts_params is None:
            self.logger.error("No MRTS parameters found")
            raise RuntimeError("❌ No MRTS parameters found. Call setup_mrts_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class VegationTsExtractor(BaseExtractor):
    """
    Extracts high-resolution vegetation time series for agricultural entities.

    Retrieves historical and near-real-time vegetation index time series (NDVI, EVI, LAI, etc.)
    at field level. Supports period-based and target-date extraction modes, KPI aggregation,
    and multi-year historical comparison.

    Documentation: https://docs.earthdaily.com/agro/library/Vegetation_time_series/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_VTS.ipynb

    Args (setup_vegetation_ts_parameters):
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        vegetation_index (str): Index type ('NDVI', 'EVI', 'CVI', 'GNDVI', 'NDWI', 'LAI'). Default: 'NDVI'
        is_extrapolated (bool): Include extrapolated values. Default: True
        limit (int): Max number of records. Default: 3000
        historical_years (int): Number of historical years. Default: 10
        extraction_mode (str): 'period' or 'target_dates'. Default: 'period'
        target_dates (list): Specific dates to extract (for target_dates mode)
        kpi_filter (dict): KPI aggregation config (kpi_name, aggregation, threshold)
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop (required for LAI); start_date, end_date (optional overrides)

    Output columns:
        entity_id, date, value (vegetation index), + historical year columns when applicable
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.vegetation_ts_params = None

        # Specific API endpoint for coverage
        self.vegetation_ts_url = agro_urls["vts_urls"][self.env]

        self.logger.info(f"🛰️ VegationTsExtractor initialized for env: {self.env}")
        self.logger.debug(f"API endpoint: {self.vegetation_ts_url}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for VegationTsExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.info("🔑 Refreshing API token for VegationTsExtractor...")
        try:
            new_token, exp_time = EDAuthenticator.get_new_token(env=self.env)
            self.logger.info("✅ Token refreshed successfully")
            self.logger.debug(f"New token expires at: {datetime.fromtimestamp(exp_time)}")
            return new_token, exp_time
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {str(e)}")
            raise

    def setup_vegetation_ts_parameters(
        self,
        start_date: str = None,
        end_date: str = None,
        vegetation_index="NDVI",
        is_extrapolated=True,
        limit=3000,
        historical_years=10,
        partial_frequency=50,
        extraction_mode="period",
        target_dates=None,
        kpi_filter=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for vegetation index extraction.

        Args:
            start_date (str, optional): Baseline start date in YYYY-MM-DD format.
                Used as default when entity_data does not provide its own start_date.
            end_date (str, optional): Baseline end date in YYYY-MM-DD format.
                Used as default when entity_data does not provide its own end_date.
            vegetation_index (str): Vegetation index to extract (NDVI, EVI)
            is_extrapolated (bool): Whether to include extrapolated values
            limit (int): Maximum number of records to retrieve
            historical_years (int): Number of historical years to include (max 15)
            partial_frequency (int): How often to save partial results
            extraction_mode (str): Extraction mode.
                - 'period': seasonal MM-DD window extraction. Uses start_date/end_date as baseline
                  (entity_data can override). Expands by historical_years for KPI comparison.
                - 'windows': absolute date range extraction. Uses start_date/end_date as baseline
                  (entity_data can override). No historical expansion.
                - 'specific_dates': punctual extraction at exact target dates.
            target_dates (list, optional):
                - List of specific dates (YYYY-MM-DD) when mode='specific_dates'
                - Not used for 'period' or 'windows' modes
            kpi_filter (dict, optional): KPI configuration
        """
        self.logger.info("⚙️ Setting up vegetation time series parameters...")

        # Validate baseline dates if provided
        if start_date is not None:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD")

        if end_date is not None:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD")

        # Validate vegetation index
        valid_indexes = {"NDVI", "EVI"}
        if vegetation_index not in valid_indexes:
            self.logger.error(f"Invalid vegetation_index '{vegetation_index}'")
            raise ValueError(f"Invalid vegetation_index '{vegetation_index}'. Must be one of {valid_indexes}")

        if not isinstance(limit, int) or limit < 1:
            self.logger.error(f"Invalid limit={limit}")
            raise ValueError(f"Invalid limit={limit}. Must be a positive integer.")

        if not isinstance(is_extrapolated, bool):
            self.logger.error(f"Invalid is_extrapolated type: {type(is_extrapolated)}")
            raise ValueError(f"Invalid is_extrapolated={is_extrapolated}. Must be True or False.")

        # Validate extraction_mode
        valid_modes = {"period", "specific_dates", "windows"}
        if extraction_mode not in valid_modes:
            self.logger.error(f"Invalid extraction_mode '{extraction_mode}'")
            raise ValueError(f"Invalid extraction_mode '{extraction_mode}'. Must be one of {valid_modes}")

        # Validate target_dates if specific_dates mode
        if extraction_mode == "specific_dates":
            if target_dates is None or len(target_dates) == 0:
                self.logger.error("target_dates required for specific_dates mode")
                raise ValueError("❌ target_dates must be provided when extraction_mode='specific_dates'")

            # Validate date formats
            validated_dates = []
            for date_str in target_dates:
                try:
                    parsed = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
                    validated_dates.append(parsed.strftime("%Y-%m-%d"))
                except ValueError:
                    self.logger.warning(f"Invalid date format skipped: {date_str}")

            if not validated_dates:
                raise ValueError("❌ No valid dates in target_dates list")

            target_dates = sorted(validated_dates)
            self.logger.info(f"📅 Specific dates mode: {len(target_dates)} dates configured")

        # Windows mode — uses start_date/end_date baseline, entity_data can override
        elif extraction_mode == "windows":
            if start_date is not None and end_date is not None:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                duration = (end_dt - start_dt).days
                self.logger.info(f"📅 Windows mode: {start_date} to {end_date} ({duration} days)")
            else:
                self.logger.info("📅 Windows mode: dates will be resolved from entity_data")

        # Validate historical_years (int or list of ints, also accept comma-separated string)
        if isinstance(historical_years, str):
            historical_years = [int(y) for y in historical_years.split(",") if y.strip()]
        if isinstance(historical_years, list):
            if len(historical_years) == 0:
                raise ValueError("historical_years list must not be empty")
            if not all(isinstance(y, int) for y in historical_years):
                raise ValueError(f"historical_years list must contain integers, got: {historical_years}")
            self.logger.info(f"Historical years set to specific years: {historical_years}")
        elif isinstance(historical_years, int):
            if historical_years < 0 or historical_years > 15:
                raise ValueError(f"Invalid historical_years={historical_years}. Must be integer 0-15.")
        else:
            raise ValueError(f"historical_years must be int or list of ints, got {type(historical_years)}")

        # Validate KPI filter (centralized rules — see api_utils.validate_kpi_filter)
        validate_kpi_filter(kpi_filter, logger=self.logger)

        self.apply_cache_setting(use_cache)

        # Store parameters
        self.vegetation_ts_params = {
            "start_date": start_date,
            "end_date": end_date,
            "vegetation_index": vegetation_index,
            "is_extrapolated": is_extrapolated,
            "limit": limit,
            "historical_years": historical_years,
            "partial_frequency": partial_frequency,
            "extraction_mode": extraction_mode,
            "target_dates": target_dates,
            "kpi_filter": kpi_filter,
        }

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure cache key columns for vegetation time series results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.logger.success("✅ Vegetation time series parameters configured successfully")
        self.logger.debug(f"Parameters: {self.vegetation_ts_params}")

        print("🌿 Vegetation time series parameters configured:")
        for k, v in self.vegetation_ts_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_vegetation_ts_params
    @cache_single_entity("vegetation_ts_params")
    def process_single_entity_specific_dates(self, row, params=None):
        """
        Process a single entity for vegetation time series extraction at specific dates.

        Args:
            row (pd.Series or dict): Entity data with 'id' and 'name'
            params (dict, optional): Override vegetation_ts_params

        Returns:
            dict: Contains 'data' (DataFrame with columns: date, field_name, ndvi_value) and 'error'
        """
        if params is None:
            params = self.vegetation_ts_params

        # Convert pandas Series to dict if needed
        if isinstance(row, pd.Series):
            row = row.to_dict()

        entity_id = self.get_entity_value(row, "id")
        entity_name = row.get("name", entity_id)
        target_dates = params.get("target_dates", [])
        vegetation_index = params.get("vegetation_index", "NDVI")

        self.logger.debug(f"Entity {entity_id}: Processing {len(target_dates)} specific dates...")

        if not target_dates:
            return {"data": None, "error": {"message": "No target dates configured", "entity_id": entity_id}}

        # Calculate date range to query API (min to max of target dates)
        min_date = min(target_dates)
        max_date = max(target_dates)

        # Prepare entity data for API call
        entity_data = {"id": entity_id, "start_date": min_date, "end_date": max_date}

        try:
            # Call API to get all data in date range
            def _call_api():
                return self.get_vegetation_api(entity_data)

            self.logger.debug(f"Entity {entity_id}: Calling API for range {min_date} to {max_date}...")
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Format response - extract value list
            if raw_json and "value" in raw_json:
                api_data = raw_json["value"]
            else:
                api_data = raw_json if isinstance(raw_json, list) else []

            # Create lookup dict for API results (date -> value)
            date_value_map = {}
            for record in api_data:
                record_date = record.get("date", "")[:10]  # Get YYYY-MM-DD part
                record_value = record.get("value")
                if record_date:
                    date_value_map[record_date] = record_value

            self.logger.debug(f"Entity {entity_id}: API returned {len(date_value_map)} dates")

            # Build result for each target date
            results = []
            value_col = f"{vegetation_index.lower()}_value"

            for target_date in target_dates:
                value = date_value_map.get(target_date)

                # Format value: number or "No data"
                if value is not None:
                    formatted_value = round(value, 4) if isinstance(value, (int, float)) else value
                else:
                    formatted_value = "No data"

                results.append({"date": target_date, "field_name": entity_name, value_col: formatted_value})

            result_df = pd.DataFrame(results)

            self.logger.info(f"Entity {entity_id}: Extracted {len(results)} date records")
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")

            # Return "No data" for all dates on error
            value_col = f"{vegetation_index.lower()}_value"
            results = []
            for target_date in target_dates:
                results.append({"date": target_date, "field_name": entity_name, value_col: "No data"})

            return {"data": pd.DataFrame(results), "error": {"message": str(e), "entity_id": entity_id}}

    @requires_token
    @require_vegetation_ts_params
    def get_vegetation_api(self, entity_data: dict):
        """
        Get vegetation time series data from VTS API.

        Args:
            entity_data (dict): Entity data containing:
                - 'id' (str): Season field ID
                - 'start_date' (str): Start date in YYYY-MM-DD format (required for 'period' mode)
                - 'end_date' (str): End date in YYYY-MM-DD format (required for 'period' mode)
                Note: For 'windows' mode, start_date and end_date are optional and will use
                    target_dates from vegetation_ts_params if not provided

        Returns:
            dict: JSON response with vegetation time series data
        """
        entity_id = self.get_entity_value(entity_data, "id")
        self.logger.debug(f"Requesting vegetation time series for entity: {entity_id}")

        params = self.vegetation_ts_params
        extraction_mode = params.get("extraction_mode", "period")

        # Validate entity has required ID field
        if not self.has_entity_field(entity_data, "id"):
            self.logger.error("Entity data missing 'id' field")
            raise ValueError("❌ entity_data must include an 'id' field (season field ID).")

        season_field_id = self.get_entity_value(entity_data, "id")

        # ==== Date resolution: params baseline, entity_data overrides ====
        start_date = self.get_entity_value(entity_data, "start_date", params.get("start_date"))
        end_date = self.get_entity_value(entity_data, "end_date", params.get("end_date"))

        if start_date is None or end_date is None:
            self.logger.error(f"Entity {entity_id}: No dates resolved — check setup params or entity_data")
            raise ValueError("❌ Could not resolve start_date/end_date from entity_data or params")

        if self.has_entity_field(entity_data, "start_date") or self.has_entity_field(entity_data, "end_date"):
            self.logger.debug(f"Entity {entity_id}: Using entity-level dates: {start_date} to {end_date}")
        else:
            self.logger.debug(f"Entity {entity_id}: Using setup-level dates: {start_date} to {end_date}")

        # Validate date formats
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            self.logger.debug(f"Entity {entity_id}: Start date validated - {start_date}")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid start_date format '{start_date}'")
            raise ValueError(f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD")

        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            self.logger.debug(f"Entity {entity_id}: End date validated - {end_date}")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid end_date format '{end_date}'")
            raise ValueError(f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD")

        # Build query parameters based on extraction mode
        query_params = [
            "$offset=0",
            f"$limit={params['limit']}",
            "$count=false",
            f"SeasonField.Id={season_field_id}",
            f"index={params['vegetation_index']}",
            "$sort=-date",
            f"IsExtrapolted={str(params['is_extrapolated']).lower()}",
            "$fields=date,value",
        ]

        # Add filter based on mode
        if extraction_mode == "windows":
            # For windows mode: use $between filter
            filter_param = f"Date=$between:{start_date}|{end_date}"
            query_params.append(filter_param)
            self.logger.debug(
                f"Entity {entity_id}: Using windows mode filter: Date between {start_date} and {end_date}"
            )

        else:  # period mode
            # For period mode: use historical years and isInTimeFrame
            historical_years_param = params.get("historical_years", 0)
            if isinstance(historical_years_param, list):
                effective_lookback = start_dt.year - min(historical_years_param)
            else:
                effective_lookback = historical_years_param
            earliest_year = start_dt.year - effective_lookback
            earliest_date = f"{earliest_year}-{start_dt.month:02d}-{start_dt.day:02d}"
            self.logger.debug(
                f"Entity {entity_id}: Historical data range from {earliest_date} ({effective_lookback} years lookback)"
            )

            # Extract MM-DD from start and end dates for isInTimeFrame
            start_mmdd = f"{start_dt.month:02d}-{start_dt.day:02d}"
            end_mmdd = f"{end_dt.month:02d}-{end_dt.day:02d}"

            filter_param = f"$filter=Date>='{earliest_date}' and isInTimeFrame(Date,'{start_mmdd}','{end_mmdd}')"
            query_params.append(filter_param)
            self.logger.debug(f"Entity {entity_id}: Using period mode filter with isInTimeFrame")

        full_url = f"{self.vegetation_ts_url}/season-fields/values?{'&'.join(query_params)}"
        self.logger.debug(f"Entity {entity_id}: API URL constructed with {params['vegetation_index']}")

        # Headers
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.bearer_token}"}

        # Request data
        try:
            self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
            response = requests.get(full_url, headers=headers, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.info(f"Entity {entity_id}: API request successful (mode: {extraction_mode})")
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            body = e.response.text[:500] if e.response is not None else "No response body"
            self.logger.error(f"Entity {entity_id}: HTTP error {status} — {body}")
            raise
        except requests.exceptions.Timeout:
            self.logger.error(f"Entity {entity_id}: Request timeout after 60s")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Unexpected error - {str(e)}")
            raise

    def get_vegetation_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_vegetation_api().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain 'id', 'start_date', and 'end_date' keys

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "entity_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id")
        try:
            response_json = self.get_vegetation_api(entity_data)
            self.logger.debug(f"Entity {entity_id}: Safe API call succeeded")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.warning(f"Entity {entity_id}: HTTP {status} error")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "entity_id": entity_id}
        except ValueError as e:
            self.logger.warning(f"Entity {entity_id}: Validation error - {str(e)}")
            return {"success": False, "data": None, "error": f"Validation error: {str(e)}", "entity_id": entity_id}
        except Exception as e:
            self.logger.warning(f"Entity {entity_id}: Error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "entity_id": entity_id}

    @require_vegetation_ts_params
    def format_vegetation_ts_json(self, response_vegetation_ts_json, entity_data: dict = None):
        """
        Process vegetation API response into a DataFrame with date and value.

        Args:
            response_vegetation_ts_json (list): JSON response from get_vegetation_api (list of date/value records)
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame: DataFrame with columns: entity_id (optional), date, value
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )
        self.logger.debug(f"Entity {entity_id}: Formatting vegetation time series JSON response...")

        rows = []

        # Extract date and value from each record
        for entry in response_vegetation_ts_json:
            row = {"date": entry.get("date"), "value": entry.get("value")}

            # Add entity_id if entity_data provided
            if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
                row["entity_id"] = self.get_entity_value(entity_data, "id")

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns if entity_id exists
        if "entity_id" in df.columns:
            df = df[["entity_id", "date", "value"]]
        else:
            df = df[["date", "value"]]

        # Convert date to datetime
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        self.logger.debug(f"Entity {entity_id}: Formatted {len(df)} time series records")
        return df

    @requires_token
    @require_vegetation_ts_params
    @cache_single_entity("vegetation_ts_params")
    def process_single_entity_vegetation_ts(self, row, params=None):
        """
        Process a single entity for vegetation time series extraction with optional KPI filtering.

        Args:
            row (pd.Series or dict): Entity data with required fields:
                - 'id': Entity identifier
                - 'geometry': WKT geometry string
                - 'start_date': Start date (YYYY-MM-DD)
                - 'end_date': End date (YYYY-MM-DD)
                - 'years' (optional): List of historical years for KPI comparison
            params (dict, optional): Override vegetation_ts_params (includes kpi_filter config)

        Returns:
            dict: Contains 'data' (DataFrame) and 'error'
                - If KPI requested: DataFrame with metadata + KPI values (one row per entity)
                - If NO KPI: DataFrame with metadata + daily date/value data (multiple rows)
        """
        if params is None:
            params = self.vegetation_ts_params

        # Convert pandas Series to dict if needed
        if isinstance(row, pd.Series):
            row = row.to_dict()

        entity_id = self.get_entity_value(row, "id")
        self.logger.debug(f"Entity {entity_id}: Processing vegetation time series...")

        start_date = self.get_entity_value(row, "start_date", params.get("start_date"))
        end_date = self.get_entity_value(row, "end_date", params.get("end_date"))
        years = validate_historical_years(self.get_entity_value(row, "years"))

        # Get KPI filter configuration from params
        kpi_filter = params.get("kpi_filter")
        if kpi_filter:
            self.logger.debug(f"Entity {entity_id}: KPI filter enabled - {kpi_filter.get('aggregation', 'unknown')}")

        # Step 2: Define API call wrapper
        def _call_api():
            return self.get_vegetation_api(row)

        # Step 3: Call API
        try:
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Step 4: Format response to DataFrame
            vegetation_df = self.format_vegetation_ts_json(raw_json, entity_data=row)

            # Check if results exist
            if vegetation_df is None or vegetation_df.empty:
                self.logger.warning(f"Entity {entity_id}: No vegetation data found")
                return {"data": None, "error": {"message": "No vegetation data found", "entity_id": entity_id}}

            # Step 5: Determine output format based on KPI filter
            if kpi_filter:
                # KPI requested - compute KPI and return single row with metadata + KPI values
                self.logger.debug(f"Entity {entity_id}: Computing KPI values...")
                try:
                    years = validate_historical_years(self.get_entity_value(row, "years"))
                    if years is None:
                        historical_years_param = params.get("historical_years", 0)
                        if isinstance(historical_years_param, list):
                            years = historical_years_param
                        elif isinstance(historical_years_param, int) and historical_years_param > 0:
                            years = "ALL"
                    if years:
                        self.logger.debug(f"Entity {entity_id}: KPI historical years: {years}")

                    # Use explicit value_column from kpi_filter, or default to 'value'
                    kpi_value_column = kpi_filter.get("value_column", "value")

                    # Call as standalone function, not as method
                    kpi_result = filter_timeseries_kpi(
                        timeseries_df=vegetation_df,
                        start_date=start_date,
                        end_date=end_date,
                        kpi_name=kpi_filter.get("kpi_name", "Vegetation KPI"),
                        aggregation=kpi_filter.get("aggregation", "accumulation"),
                        threshold=kpi_filter.get("threshold"),
                        window=kpi_filter.get("window"),
                        years=years,
                        date_column="date",
                        value_column=kpi_value_column,
                    )

                    self.logger.debug(
                        f"Entity {entity_id}: KPI computed - current: {kpi_result['current_period']['value']}, historical avg: {kpi_result['historical_avg']['value']}"
                    )

                    # Create DataFrame from KPI result
                    kpi_df = pd.DataFrame(
                        [
                            {
                                "kpi_name": kpi_result["kpi_name"],
                                "aggregation": kpi_result["aggregation"],
                                "start_date": start_date,
                                "end_date": end_date,
                                "current_value": kpi_result["current_period"]["value"],
                                "current_num_records": kpi_result["current_period"]["num_records"],
                                "historical_avg": kpi_result["historical_avg"]["value"],
                                "historical_num_years": kpi_result["historical_avg"]["num_years"],
                                "difference": kpi_result["comparison"]["difference"],
                                "percent_change": kpi_result["comparison"]["percent_change"],
                            }
                        ]
                    )

                    # Use normalize_with_metadata to add entity metadata (including years)
                    result_df = normalize_with_metadata(row, kpi_df)

                    self.logger.info(f"Entity {entity_id}: KPI processing successful")

                except Exception as kpi_error:
                    self.logger.error(f"Entity {entity_id}: KPI computation failed - {str(kpi_error)}")
                    return {
                        "data": None,
                        "error": {"message": f"KPI computation failed: {str(kpi_error)}", "entity_id": entity_id},
                    }

            else:
                # NO KPI - return all daily data with metadata
                self.logger.debug(f"Entity {entity_id}: Returning raw time series data ({len(vegetation_df)} records)")
                result_df = normalize_with_metadata(row, vegetation_df)

            # Step 6: Return result
            self.logger.info(f"Entity {entity_id}: Processing successful")
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_vegetation_ts_params
    def process_entity_vegetation_ts_bulk_parallel(
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
        prefix="vegetation_ts",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity vegetation time series using threads + progress bar,
        with optional fail-safe retry, filter capabilities, and caching.

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain 'id' and 'geometry'.
            params (dict, optional): Vegetation time series parameters override.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final CSV results.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "vegetation_ts".
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
                bulk_method=self._process_entity_vegetation_ts_bulk_parallel_inner,
                params=self.vegetation_ts_params,
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

        return self._process_entity_vegetation_ts_bulk_parallel_inner(
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
    @require_vegetation_ts_params
    def _process_entity_vegetation_ts_bulk_parallel_inner(
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
        prefix="vegetation_ts",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        self.logger.info(f"🚀 Starting bulk vegetation time series extraction for {len(entity_list)} entities")
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

        print(
            f"🔄 Processing {prefix.replace('_', ' ').title()} for {len(filtered_entity_list)} entities in parallel..."
        )
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(self.process_single_entity_vegetation_ts, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(
                total=len(future_to_id), desc=f"🌱 Processing {prefix.replace('_', ' ').title()}", unit="entity"
            ) as pbar:
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
                "parameters": getattr(self, "vegetation_ts_params", {}),
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

        self.logger.info("✅ Bulk vegetation time series extraction complete")
        return summary

    @require_vegetation_ts_params
    def process_specific_dates_bulk_parallel(
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
        skip_export=False,
        prefix="vegetation_specific_dates",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of vegetation time series at specific dates using parallel threads,
        with optional caching.

        Args:
            entity_list (pd.DataFrame): Entities with 'id' and 'name' columns
            params (dict, optional): Override vegetation_ts_params
            max_workers (int): Number of parallel threads
            output_path (str, optional): Directory to save results
            partial_frequency (int): Save partial results every N entities
            fail_safe (bool): Retry previously failed entities
            filter_column (str, optional): Column to filter by
            filter_value (any, optional): Value to filter
            filter_type (str): 'exclude' or 'include'
            skip_export (bool): Skip final export
            prefix (str): Output filename prefix
            use_cache (bool, optional): Override instance-level cache setting. Default: None (uses self.use_cache).

        Returns:
            dict: Results DataFrame, errors, and statistics.
                  When cache is active, also includes cache_hit and cache_miss counts.
        """
        # Route through cache wrapper if caching is enabled
        cache_enabled = use_cache if use_cache is not None else self.use_cache
        if cache_enabled and self.cache_key_columns is not None:
            return self._bulk_with_cache(
                entity_list=entity_list,
                bulk_method=self._process_specific_dates_bulk_parallel_inner,
                params=self.vegetation_ts_params,
                use_cache=True,
                params_kw=params,
                max_workers=max_workers,
                output_path=output_path,
                partial_frequency=partial_frequency,
                fail_safe=fail_safe,
                filter_column=filter_column,
                filter_value=filter_value,
                filter_type=filter_type,
                skip_export=skip_export,
                prefix=prefix,
                generate_report=generate_report,
                report_options=report_options,
            )

        return self._process_specific_dates_bulk_parallel_inner(
            entity_list=entity_list,
            params_kw=params,
            max_workers=max_workers,
            output_path=output_path,
            partial_frequency=partial_frequency,
            fail_safe=fail_safe,
            filter_column=filter_column,
            filter_value=filter_value,
            filter_type=filter_type,
            skip_export=skip_export,
            prefix=prefix,
            generate_report=generate_report,
            report_options=report_options,
        )

    @requires_token
    @require_vegetation_ts_params
    def _process_specific_dates_bulk_parallel_inner(
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
        skip_export=False,
        prefix="vegetation_specific_dates",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        if params is None:
            params = self.vegetation_ts_params

        target_dates = params.get("target_dates", [])
        vegetation_index = params.get("vegetation_index", "NDVI")

        if not target_dates:
            raise ValueError("❌ No target_dates configured. Call setup_vegetation_ts_parameters first.")

        self.logger.info(f"🚀 Starting bulk specific dates extraction for {len(entity_list)} entities")
        self.logger.info(f"📅 Target dates: {len(target_dates)} ({target_dates[0]} to {target_dates[-1]})")

        # Apply filter
        filtered_entity_list, skipped_entities, skip_count = filter_entities(
            entity_list, filter_column, filter_value, filter_type
        )

        if skip_count > 0:
            self.logger.info(f"📊 Filter applied: {skip_count} skipped, {len(filtered_entity_list)} to process")

        all_rows = []
        global_errors = []
        successful_calculations = 0
        total_calculations = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        print(
            f"📄 Processing {vegetation_index} for {len(filtered_entity_list)} entities at {len(target_dates)} specific dates..."
        )
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_specific_dates, row, params): self.get_entity_value(
                    row, "id"
                )
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🌱 Processing {vegetation_index}", unit="entity") as pbar:
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
                        else:
                            failed_ids.append(entity_id)

                        if error:
                            error_record = {"entity_id": entity_id, **error}
                            global_errors.append(error_record)
                            buffer_errors.append(error_record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)

                    except Exception as e:
                        error_record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(error_record)
                        failed_ids.append(entity_id)

                    # Partial export
                    if self.partial_path and partial_frequency > 0 and total_calculations % partial_frequency == 0:
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

        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Successful: {successful_calculations}/{total_calculations}")

        # Concatenate results
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

        # Ensure column order: date, field_name, ndvi_value
        value_col = f"{vegetation_index.lower()}_value"
        if not results_df.empty and all(col in results_df.columns for col in ["date", "field_name", value_col]):
            results_df = results_df[["date", "field_name", value_col]]

        # Sort by date, then field_name
        if not results_df.empty:
            results_df = results_df.sort_values(["date", "field_name"]).reset_index(drop=True)

        # Finalize
        if not skip_export:
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
                    "parameters": getattr(self, "vegetation_ts_params", {}),
                    "entity_df": entity_list,
                },
            )

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
            "target_dates": target_dates,
        }


### MRSTEXtractor
# require_mrts_params is defined at module top (line 48); the duplicate
# definition that used to live here has been removed.


class MRTSExtractor(BaseExtractor):
    """
    Extracts Medium Resolution Time Series (MRTS) vegetation data for agricultural entities.

    Retrieves multi-sensor vegetation index time series using geometry-based queries.
    Supports raw and smoothed data extraction, temporal consistency checks, denoising,
    and end-of-curve extrapolation. Works with Sentinel-2, Landsat-8/9, and other sensors.

    Documentation: https://docs.earthdaily.com/agro/library/Vegetation_time_series/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_MRTS_extraction_functions.ipynb

    Args (setup_mrts_parameters):
        start_date (str): Start date in YYYY-MM-DD format. Default: '2025-05-01'
        end_date (str): End date in YYYY-MM-DD format. Default: '2025-10-15'
        sensors (list): Sensor list (Sentinel_2, Landsat_8, Landsat_9, etc.). Default: None (all)
        vegetation_index (str): Index type ('NDVI', 'EVI', 'CVI', 'GNDVI', 'NDWI', 'LAI'). Default: 'NDVI'
        aggregation (str): Aggregation method ('average', 'median', 'max', 'min'). Default: 'average'
        smoothing_method (str): Smoothing ('Whittaker', 'SavitzkyGolay', 'None'). Default: 'Whittaker'
        apply_denoiser (bool): Apply denoising filter. Default: True
        apply_end_of_curve (bool): Extrapolate end of curve. Default: True
        clear_cover_min (int): Minimum clear sky percentage (0-100). Default: 100
        mask (str): Cloud mask used to discard NotClear pixels ('Native', 'ACM', 'Auto', 'ML',
            'MLCirrus'). 'All' is NOT accepted here (coverage-only). Default: 'Auto' — best
            mask picked per image. Pass None to omit the key entirely (same behaviour).
        mode (str): Output mode ('full' = raw+smoothed, 'raw' = raw only). Default: 'full'
        compute_temporal_consistency (bool): Compute consistency checks. Default: True
        temporal_consistency_threshold (dict): Per-index thresholds. Example: {"Ndvi": 0.06, "Lai": 0.3}
        output_saturation (bool): Include index saturation flags in the output. Default: True
        extract_raw_datasets (bool): Include the raw per-image datasets alongside smoothed values. Default: True
        historical_years (int): Number of prior years to fetch for historical comparison. Default: 10
        kpi_filter (dict): KPI aggregation rules applied to the time series. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop (required for LAI); start_date, end_date (optional overrides)

    Output columns:
        entity_id, date, raw_value, noised, temporalConsistencyCheck, image_id, smoothed_value (full mode)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        """
        Initialize Vegetation Time Series Extractor

        Args:
            bearer_token (str): Bearer token for API authentication
            token_expiration (datetime): Token expiration datetime
            config (dict): Configuration dictionary with env, paths, etc.
            workflow_ref (optional): Reference to WorkflowManager for token refresh
        """
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        # Available vegetation indices
        self.available_indices = {"NDVI", "EVI", "CVI", "GNDVI", "NDWI", "LAI", "NDRE", "NDMI", "S2REP"}

        # Available sensors
        self.available_sensors = {"Sentinel_2", "Landsat_8", "Landsat_9", "HJ2B_CCD4", "GAOFEN_6_WFV3"}

        # Available smoothing methods
        self.available_smoothing = {"Whittaker", "SavitzkyGolay", "None"}

        # Available aggregation methods
        self.available_aggregation = {"average", "median", "max", "min", "accumulation", "std"}

        # Available cloud masks (TimeSeriesMaskType) — canonical API casing keyed by lowercase input
        self.available_masks = {
            "native": "Native",
            "acm": "ACM",
            "auto": "Auto",
            "ml": "ML",
            "mlcirrus": "MLCirrus",
        }

        # Vegetation time series parameters (initially None)
        self.vegetation_ts_params = None

        # MRTS parameters (initially None)
        self.mrts_params = None

        # Specific API endpoint for vegetation time series
        self.mrts_url = agro_urls["map_products_url"][self.env]

        self.logger.info(f"🛰️ VegationTsExtractor initialized for env: {self.env}")
        self.logger.debug(f"MRTS API endpoint: {self.mrts_url}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for cropidExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for cropidExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_mrts_parameters(
        self,
        start_date="2025-05-01",
        end_date="2025-10-15",
        sensors=None,
        vegetation_index="NDVI",
        aggregation="average",
        smoothing_method="Whittaker",
        apply_denoiser=True,
        apply_end_of_curve=True,
        clear_cover_min=100,
        mask="Auto",
        output_saturation=True,
        extract_raw_datasets=True,
        compute_temporal_consistency=True,
        temporal_consistency_threshold=None,
        mode="full",
        historical_years=10,
        partial_frequency=50,
        kpi_filter=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for MRTS (Multi-Resolution Time Series) extraction.

        Args:
            start_date (str): Start date of the period (YYYY-MM-DD). Default: '2025-05-01'
            end_date (str): End date of the period (YYYY-MM-DD). Default: '2025-10-15'
            sensors (list, optional): List of sensors to use.
                                     Available: Sentinel_2, Landsat_8, Landsat_9, HJ2B_CCD4, GAOFEN_6_WFV3
                                     Default: None (omitted from payload → API uses all available sensors)
            vegetation_index (str): Vegetation index ('NDVI', 'EVI', 'CVI', 'GNDVI', 'NDWI', 'LAI'). Default: 'NDVI'
            aggregation (str): Aggregation method ('average', 'median', 'max', 'min'). Default: 'average'
            smoothing_method (str): Smoothing method ('Whittaker', 'SavitzkyGolay', 'None'). Default: 'Whittaker'
            apply_denoiser (bool): Apply denoising. Default: True
            apply_end_of_curve (bool): Apply end of curve extrapolation. Default: True
            clear_cover_min (int): Minimum clear sky coverage percentage (0-100). Default: 100
            mask (str, optional): Cloud mask applied before aggregation (TimeSeriesMaskType).
                One of 'Native', 'ACM', 'Auto', 'ML', 'MLCirrus' (case-insensitive; normalized to
                the API casing). Unlike the coverage/catalog-imagery endpoint, 'All' is rejected
                by this endpoint (HTTP 400) — it is a coverage-only value.
                Default: 'Auto' — the API selects the best available mask per image, which gives
                the most complete series. This is also what the API does when the key is absent,
                so the default is explicit, not a behaviour change; pass mask=None to omit it.
                Notes from live probing (prod, 2026-08-25, 3 sites x 2 seasons):
                  - omitting mask is identical to mask='Auto' (same observations, not just
                    the same count) — it is NOT Native and NOT ML
                  - 'Auto' picks the best mask per image, so a single response can mix
                    Native/ACM/ML labels; the 'mask' field on each raw point reports the
                    mask actually applied, not the one requested
                  - on fields where Auto happens to resolve to one mask throughout, omitted /
                    'Auto' / that mask are indistinguishable — compare observations, not counts
                  - 'ACM' returns noticeably fewer raw observations than Native/ML
                  - 'MLCirrus' is accepted but returned no data on any field tested
            output_saturation (bool): Include saturation information. Default: True
            extract_raw_datasets (bool): Extract raw datasets in addition to smoothed data. Default: True
            compute_temporal_consistency (bool): Compute temporal consistency metrics. Default: True
            temporal_consistency_threshold (dict, optional): Threshold values for temporal consistency by index.
                Example: {"Ndvi": 0.06, "Lai": 0.3, "S2Rep": 2.5}
                If None and compute_temporal_consistency=True, uses default thresholds.
            mode (str): Output mode. Default: 'full'
                - 'full': Returns both raw and smoothed data merged on date
                - 'raw': Returns only raw observation data (no smoothed interpolation)
            partial_frequency (int): Partial save frequency during bulk processing. Default: 50
            kpi_filter (dict, optional): KPI configuration for aggregating time series data:
                - 'kpi_name' (str): Name of the KPI
                - 'aggregation' (str): Type of aggregation (accumulation, average, max, min, count_gt, count_lt, count_between, std)
                - 'threshold' (float or tuple, optional): Threshold value(s) for count operations
                    - Single float for count_gt/count_lt
                    - Tuple of two floats (min, max) for count_between
        """
        self.logger.info("⚙️ Setting up MRTS parameters...")

        # Vegetation index validation - use class-level available indices
        if vegetation_index not in self.available_indices:
            self.logger.error(f"Invalid vegetation_index '{vegetation_index}'")
            raise ValueError(f"Invalid vegetation_index '{vegetation_index}'. Must be one of {self.available_indices}")

        # LAI requires crop in entity data
        if vegetation_index == "LAI":
            self.logger.warning(
                "⚠️ Vegetation index 'LAI' requires 'crop' field in entity_data. "
                "Ensure each entity includes a 'crop' value."
            )
            print("⚠️ LAI index selected: 'crop' field is required in entity_data for each entity.")

        # Sensors: None means all sensors (omit from payload), otherwise validate list
        if sensors is not None:
            invalid_sensors = [s for s in sensors if s not in self.available_sensors]
            if invalid_sensors:
                self.logger.error(f"Invalid sensors: {invalid_sensors}")
                raise ValueError(f"Invalid sensors: {invalid_sensors}. Choose from: {self.available_sensors}")
            self.logger.debug(f"Using selected sensors: {sensors}")
        else:
            self.logger.debug("Sensors set to None → all available sensors will be used (omitted from payload)")

        # Validate smoothing method
        if smoothing_method not in self.available_smoothing:
            self.logger.error(f"Invalid smoothing method '{smoothing_method}'")
            raise ValueError(f"Invalid smoothing method '{smoothing_method}'. Choose from: {self.available_smoothing}")

        # Validate aggregation method
        if aggregation not in self.available_aggregation:
            self.logger.error(f"Invalid aggregation '{aggregation}'")
            raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from: {self.available_aggregation}")

        # Date validation
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            self.logger.debug(f"Date range validated: {start_date} to {end_date}")
            if start_dt >= end_dt:
                self.logger.error(f"Invalid date range: start_date ({start_date}) >= end_date ({end_date})")
                raise ValueError("start_date must be before end_date")
        except ValueError as e:
            self.logger.error(f"Date validation failed: {str(e)}")
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        # Clear sky coverage validation
        if not (0 <= clear_cover_min <= 100):
            self.logger.error(f"Invalid clear_cover_min={clear_cover_min}")
            raise ValueError("clear_cover_min must be between 0 and 100")

        # Cloud mask validation — normalize to the casing the API expects
        if mask is not None:
            if not isinstance(mask, str) or mask.lower() not in self.available_masks:
                self.logger.error(f"Invalid mask '{mask}'")
                raise ValueError(f"Invalid mask '{mask}'. Choose from: {sorted(self.available_masks.values())}")
            mask = self.available_masks[mask.lower()]
            self.logger.debug(f"Cloud mask set to: {mask}")
        else:
            self.logger.debug("mask=None → key omitted from payload (API resolves it as 'Auto')")

        # Validate historical_years (int or list of ints, also accept comma-separated string)
        if isinstance(historical_years, str):
            historical_years = [int(y) for y in historical_years.split(",") if y.strip()]
        if isinstance(historical_years, list):
            if len(historical_years) == 0:
                self.logger.error("historical_years list is empty")
                raise ValueError("historical_years list must not be empty")
            if not all(isinstance(y, int) for y in historical_years):
                self.logger.error(f"historical_years list contains non-integers: {historical_years}")
                raise ValueError(f"historical_years list must contain integers, got: {historical_years}")
            self.logger.info(f"Historical years set to specific years: {historical_years}")
        elif isinstance(historical_years, int):
            if historical_years < 0:
                self.logger.error(f"Invalid historical_years value: {historical_years}")
                raise ValueError(f"Invalid historical_years={historical_years}. Must be non-negative.")
            if historical_years > 15:
                self.logger.error(f"Historical_years exceeds maximum: {historical_years}")
                raise ValueError(f"Invalid historical_years={historical_years}. Must be 15 or less.")
        else:
            self.logger.error(f"Invalid historical_years type: {type(historical_years)}")
            raise ValueError(f"historical_years must be int or list of ints, got {type(historical_years)}")
        # Partial frequency validation
        if not isinstance(partial_frequency, int) or partial_frequency < 0:
            self.logger.error(f"Invalid partial_frequency={partial_frequency}")
            raise ValueError("partial_frequency must be a non-negative integer")

        # Validate boolean parameters
        for param_name, param_value in [
            ("apply_denoiser", apply_denoiser),
            ("apply_end_of_curve", apply_end_of_curve),
            ("output_saturation", output_saturation),
            ("extract_raw_datasets", extract_raw_datasets),
            ("compute_temporal_consistency", compute_temporal_consistency),
        ]:
            if not isinstance(param_value, bool):
                self.logger.error(f"Invalid {param_name} type: {type(param_value)}")
                raise ValueError(f"{param_name} must be True or False")

        # Set default temporal consistency thresholds if enabled but not provided
        if compute_temporal_consistency and temporal_consistency_threshold is None:
            temporal_consistency_threshold = {"Ndvi": 0.06, "Lai": 0.3, "S2Rep": 2.5}
            self.logger.debug(f"Using default temporal consistency thresholds: {temporal_consistency_threshold}")

        # Validate temporal_consistency_threshold if provided
        if temporal_consistency_threshold is not None:
            if not isinstance(temporal_consistency_threshold, dict):
                self.logger.error(
                    f"Invalid temporal_consistency_threshold type: {type(temporal_consistency_threshold)}"
                )
                raise ValueError(
                    "temporal_consistency_threshold must be a dictionary (e.g., {'Ndvi': 0.06, 'Lai': 0.3, 'S2Rep': 2.5})"
                )

        # Validate mode
        valid_modes = {"full", "raw"}
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode '{mode}'")
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        self.logger.debug(f"Output mode: {mode}")

        # Validate KPI filter (centralized rules — see api_utils.validate_kpi_filter)
        validate_kpi_filter(kpi_filter, logger=self.logger)

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.apply_cache_setting(use_cache)

        # Store parameters
        self.mrts_params = {
            "start_date": start_date,
            "end_date": end_date,
            "sensors": sensors,
            "vegetation_index": vegetation_index,
            "aggregation": aggregation,
            "smoothing_method": smoothing_method,
            "apply_denoiser": apply_denoiser,
            "apply_end_of_curve": apply_end_of_curve,
            "clear_cover_min": clear_cover_min,
            "mask": mask,
            "output_saturation": output_saturation,
            "extract_raw_datasets": extract_raw_datasets,
            "compute_temporal_consistency": compute_temporal_consistency,
            "temporal_consistency_threshold": temporal_consistency_threshold,
            "mode": mode,
            "historical_years": historical_years,
            "partial_frequency": partial_frequency,
            "kpi_filter": kpi_filter,
        }

        # Configure cache key columns for MRTS results
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        self.logger.info("✅ MRTS parameters configured:")
        for k, v in self.mrts_params.items():
            if k == "kpi_filter" and v:
                self.logger.debug(f"   {k}: {v}")
            else:
                self.logger.debug(f"   {k}: {v}")

        print("🌍 MRTS parameters configured:")
        for k, v in self.mrts_params.items():
            if k == "kpi_filter" and v:
                print(f"   {k}:")
                for kpi_key, kpi_val in v.items():
                    print(f"      {kpi_key}: {kpi_val}")
            else:
                print(f"   {k}: {v}")

    @requires_token
    @require_mrts_params
    def get_mrts_api(self, entity_data: dict):
        """
        Get MRTS (Medium Resolution Time Series) data from API using geometry.

        Args:
            entity_data (dict): Entity data containing:
                - 'geometry' (str): WKT geometry string
                - 'id' (str, optional): Season field ID
                - 'crop' (str, optional): Crop type
                - 'sowing_date' (str, optional): Sowing date in YYYY-MM-DD format
                - 'start_date' (str, optional): Start date in YYYY-MM-DD format
                - 'end_date' (str, optional): End date in YYYY-MM-DD format
                If start_date/end_date not provided, uses dates from mrts_params

        Returns:
            dict: JSON response with MRTS data
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        self.logger.debug(f"Requesting MRTS data for entity: {entity_id}")

        params = self.mrts_params

        # Validate entity has required fields
        if not self.has_entity_field(entity_data, "geometry"):
            self.logger.error("Entity data missing 'geometry' field")
            raise ValueError("❌ entity_data must include a 'geometry' field in WKT format.")

        geometry = self.get_entity_value(entity_data, "geometry")

        # Validate geometry
        try:
            geometry = validate_wkt(geometry)
            self.logger.debug(f"Entity {entity_id}: Geometry validated")
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Invalid geometry - {str(e)}")
            raise

        # Get dates from entity_data if present, otherwise from params
        if self.has_entity_field(entity_data, "start_date") and self.has_entity_field(entity_data, "end_date"):
            start_date = self.get_entity_value(entity_data, "start_date")
            end_date = self.get_entity_value(entity_data, "end_date")
            self.logger.info(f"Entity {entity_id}: MRTS API using entity dates: {start_date} to {end_date}")
        elif "start_date" in params and "end_date" in params:
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            self.logger.info(f"Entity {entity_id}: MRTS API using param dates: {start_date} to {end_date}")
        else:
            self.logger.error(f"Entity {entity_id}: No dates found in entity_data or params")
            raise ValueError("❌ start_date and end_date must be provided either in entity_data or in mrts_params")

        # Validate and convert date formats to ISO format
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_date_iso = start_dt.strftime("%Y-%m-%dT00:00:00.000Z")
            self.logger.debug(f"Entity {entity_id}: Start date validated and converted - {start_date_iso}")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid start_date format '{start_date}'")
            raise ValueError(f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD")

        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_iso = end_dt.strftime("%Y-%m-%dT23:59:59.999Z")
            self.logger.debug(f"Entity {entity_id}: End date validated and converted - {end_date_iso}")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid end_date format '{end_date}'")
            raise ValueError(f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD")

        # Build seasonfield object based on vegetation index
        is_lai = params.get("vegetation_index") == "LAI"

        if is_lai:
            # LAI requires geometry + crop, no id
            validation = self.validate_entity(entity_data, ["geometry", "crop"], context="LAI extraction")
            if not validation["valid"]:
                self.logger.error(validation["details"])
                raise ValueError(validation["details"])
            seasonfield = {
                "geometry": geometry,
                "crop": self.get_entity_value(entity_data, "crop"),
            }
            self.logger.debug(f"Entity {entity_id}: LAI mode - crop={seasonfield['crop']}, id omitted from seasonfield")
        else:
            # Other indices: geometry + optional id
            seasonfield = {
                "geometry": geometry,
            }
            # no id as easier and not adherence with platform and access control linked to identity
            # if "id" in entity_data and entity_data.get("id"):
            #     seasonfield["id"] = entity_data.get("id")
        # no sowing date as not impacting index
        # # Add sowing date if provided
        # if "sowing_date" in entity_data:
        #     try:
        #         sowing_dt = datetime.strptime(entity_data.get("sowing_date"), "%Y-%m-%d")
        #         seasonfield["sowingDate"] = sowing_dt.strftime("%Y-%m-%dT00:00:00.000Z")
        #         self.logger.debug(f"Entity {entity_id}: Sowing date included - {seasonfield['sowingDate']}")
        #     except ValueError:
        #         self.logger.warning(f"Entity {entity_id}: Invalid sowing_date format, skipping")

        # Build API URL
        full_url = f"{self.mrts_url}/time-serie"
        self.logger.debug(f"Entity {entity_id}: API URL constructed (MRTS endpoint)")

        # Headers
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.bearer_token}"}

        # Build request body with all parameters
        body = {
            "seasonfield": seasonfield,
            "vegetationIndex": params.get("vegetation_index", "NDVI"),
            "aggregation": params.get("aggregation", "average"),
            "collections": params.get("collections", []),
            "clearCoverMin": params.get("clear_cover_min", 100),
            "startDate": start_date_iso,
            "endDate": end_date_iso,
            "applyDenoiser": params.get("apply_denoiser", True),
            "smoothingMethod": params.get("smoothing_method", "Whittaker"),
            "applyEndOfCurve": params.get("apply_end_of_curve", True),
            "outputSaturation": params.get("output_saturation", True),
            "extractRawDatasets": params.get("extract_raw_datasets", True),
            "computeTemporalConsistency": params.get("compute_temporal_consistency", True),
        }

        # Only include sensors if explicitly specified; None → omit to use all available sensors
        if params.get("sensors") is not None:
            body["sensors"] = params["sensors"]

        # Only include mask if explicitly specified; None → omit to use the API default mask
        if params.get("mask") is not None:
            body["mask"] = params["mask"]

        # Add temporal consistency threshold only if compute_temporal_consistency is True
        if params.get("compute_temporal_consistency", True) and params.get("temporal_consistency_threshold"):
            body["temporalConsistencyThreshold"] = params.get("temporal_consistency_threshold")

        sensors_info = params.get("sensors")
        self.logger.debug(
            f"Entity {entity_id}: Request body prepared with {len(sensors_info) if sensors_info else 'all'} sensors"
        )

        # Request data (POST with geometry and parameters in body)
        try:
            self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
            self.logger.debug(f"Entity {entity_id}: Payload: {body}")
            response = requests.post(full_url, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.info(f"Entity {entity_id}: MRTS API request successful")
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | url={full_url} | payload={body}")
            raise
        except requests.exceptions.Timeout:
            self.logger.error(f"Entity {entity_id}: Request timeout after 60s | url={full_url} | payload={body}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Unexpected error - {str(e)} | url={full_url} | payload={body}")
            raise

    def get_mrts_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_mrts_api().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain 'geometry' and optionally 'id', 'start_date', 'end_date'

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "entity_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        try:
            response_json = self.get_mrts_api(entity_data)
            self.logger.debug(f"Entity {entity_id}: Safe API call succeeded")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.warning(f"Entity {entity_id}: HTTP {status} error")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "entity_id": entity_id}
        except ValueError as e:
            self.logger.warning(f"Entity {entity_id}: Validation error - {str(e)}")
            return {"success": False, "data": None, "error": f"Validation error: {str(e)}", "entity_id": entity_id}
        except Exception as e:
            self.logger.warning(f"Entity {entity_id}: Error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "entity_id": entity_id}

    @require_mrts_params
    def format_mrts_json(self, response_mrts_json, entity_data: dict = None):
        """
        Process MRTS API response into a DataFrame with date, raw and smoothed values.

        Args:
            response_mrts_json (dict): JSON response with 'rawData' and 'smoothedData' keys
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame:
                - mode='full': entity_id (optional), date, raw_value, noised, temporalConsistencyCheck, mask, coveragePercent, image_id, smoothed_value
                - mode='raw': entity_id (optional), date, raw_value, noised, temporalConsistencyCheck, mask, coveragePercent, image_id
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )
        mode = self.mrts_params.get("mode", "full")
        self.logger.debug(f"Entity {entity_id}: Formatting MRTS JSON response (mode={mode})...")

        # Extract raw and smoothed data lists (handle None values explicitly)
        raw_data = response_mrts_json.get("rawData") or []
        smoothed_data = response_mrts_json.get("smoothedData") or []

        # Create DataFrames for each
        df_raw = pd.DataFrame(raw_data)
        df_smoothed = pd.DataFrame(smoothed_data)

        # Rename value columns to distinguish them and keep noised, temporalConsistencyCheck, mask, coveragePercent, image_id
        if not df_raw.empty:
            df_raw = df_raw.rename(columns={"value": "raw_value"})
            # Flatten nested image dict to extract image_id
            if "image" in df_raw.columns:
                df_raw["image_id"] = df_raw["image"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
            raw_cols = ["date", "raw_value"]
            if "noised" in df_raw.columns:
                raw_cols.append("noised")
            if "temporalConsistencyCheck" in df_raw.columns:
                raw_cols.append("temporalConsistencyCheck")
            if "mask" in df_raw.columns:
                raw_cols.append("mask")
            if "coveragePercent" in df_raw.columns:
                raw_cols.append("coveragePercent")
            if "image_id" in df_raw.columns:
                raw_cols.append("image_id")
            df_raw = df_raw[raw_cols]

        # Build output based on mode
        if mode == "raw":
            # Raw mode: only raw observation data, no smoothed
            if not df_raw.empty:
                df = df_raw
            else:
                df = pd.DataFrame(
                    columns=[
                        "date",
                        "raw_value",
                        "noised",
                        "temporalConsistencyCheck",
                        "mask",
                        "coveragePercent",
                        "image_id",
                    ]
                )
            self.logger.debug(f"Entity {entity_id}: Raw mode - {len(df)} raw records")

        else:
            # Full mode: merge raw + smoothed on date
            if not df_smoothed.empty:
                df_smoothed = df_smoothed.rename(columns={"value": "smoothed_value"})
                df_smoothed = df_smoothed[["date", "smoothed_value"]]

            if not df_raw.empty and not df_smoothed.empty:
                df = pd.merge(df_raw, df_smoothed, on="date", how="outer")
            elif not df_raw.empty:
                df = df_raw
                df["smoothed_value"] = None
            elif not df_smoothed.empty:
                df = df_smoothed
                df["raw_value"] = None
            else:
                df = pd.DataFrame(
                    columns=[
                        "date",
                        "raw_value",
                        "noised",
                        "temporalConsistencyCheck",
                        "mask",
                        "coveragePercent",
                        "image_id",
                        "smoothed_value",
                    ]
                )

        # Add entity_id if provided
        if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
            df.insert(0, "entity_id", self.get_entity_value(entity_data, "id"))

        # Convert date to datetime and sort
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        self.logger.debug(
            f"Entity {entity_id}: Formatted {len(df)} MRTS records (raw: {len(raw_data)}, smoothed: {len(smoothed_data)})"
        )
        return df

    @requires_token
    @require_mrts_params
    @cache_single_entity("mrts_params")
    def process_single_entity_mrts(self, row, params=None):
        """
        Process a single entity for medium resolution time series extraction with optional KPI filtering.

        Args:
            row (pd.DataFrame, pd.Series, or dict): Entity data with required fields:
                - 'id': Entity identifier
                - 'geometry': WKT geometry string
                - 'start_date' (optional): Start date (YYYY-MM-DD) - falls back to params
                - 'end_date' (optional): End date (YYYY-MM-DD) - falls back to params
                - 'years' (optional): List of historical years for KPI comparison
                If DataFrame with multiple rows: only first row is processed
            params (dict, optional): Override mrts_params (includes kpi_filter config and fallback dates)

        Returns:
            dict: Contains 'data' (DataFrame) and 'error'
                - If KPI requested: DataFrame with metadata + KPI values (one row per entity)
                - If NO KPI: DataFrame with metadata + daily date/value data (multiple rows)
        """
        if params is None:
            params = self.mrts_params

        # Handle different input types - convert all to dict
        try:
            if isinstance(row, pd.DataFrame):
                if len(row) > 1:
                    self.logger.warning(f"DataFrame contains {len(row)} rows. Processing only the first row.")
                row = row.iloc[0].to_dict()
            elif isinstance(row, pd.Series):
                row = row.to_dict()
            elif not isinstance(row, dict):
                raise TypeError(f"row must be DataFrame, Series, or dict. Got {type(row)}")
        except Exception as e:
            self.logger.error(f"Failed to convert input to dict: {str(e)}")
            return {"data": None, "error": {"message": f"Invalid input type: {str(e)}", "entity_id": "unknown"}}

        # Now process as dict
        entity_id = self.get_entity_value(row, "id", "unknown")
        self.logger.debug(f"Entity {entity_id}: Processing medium resolution time series...")

        try:
            # Get start_date from row, fall back to params
            start_date = self.get_entity_value(row, "start_date") or params.get("start_date")
            if not start_date:
                self.logger.error(f"Entity {entity_id}: Missing start_date in both row and params")
                return {
                    "data": None,
                    "error": {"message": "Missing start_date in both row and params", "entity_id": entity_id},
                }

            # Get end_date from row, fall back to params
            end_date = self.get_entity_value(row, "end_date") or params.get("end_date")
            if not end_date:
                self.logger.error(f"Entity {entity_id}: Missing end_date in both row and params")
                return {
                    "data": None,
                    "error": {"message": "Missing end_date in both row and params", "entity_id": entity_id},
                }

            # Log where dates came from
            date_source = []
            if self.get_entity_value(row, "start_date"):
                date_source.append("start_date from row")
            else:
                date_source.append("start_date from params")

            if self.get_entity_value(row, "end_date"):
                date_source.append("end_date from row")
            else:
                date_source.append("end_date from params")

            self.logger.debug(f"Entity {entity_id}: Using {', '.join(date_source)} ({start_date} to {end_date})")

            years = validate_historical_years(self.get_entity_value(row, "years"))

            # Get KPI filter configuration from params
            kpi_filter = params.get("kpi_filter")
            if kpi_filter:
                self.logger.debug(
                    f"Entity {entity_id}: KPI filter enabled - {kpi_filter.get('aggregation', 'unknown')}"
                )

            # Expand API date range for historical KPI comparison
            # MRTS API takes an absolute date range, so we push start_date back
            # to fetch both current and historical data.
            # Also resolve per-entity historical_years override via column_mapping.
            historical_years_param = self.get_entity_value(row, "historical_years", params.get("historical_years", 0))
            if isinstance(historical_years_param, str):
                historical_years_param = [int(y) for y in historical_years_param.split(",") if y.strip()]

            api_start_date = start_date
            if kpi_filter and historical_years_param:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if isinstance(historical_years_param, list):
                    effective_lookback = start_dt.year - min(historical_years_param)
                else:
                    effective_lookback = historical_years_param
                if effective_lookback > 0:
                    expanded_start = start_dt.replace(year=start_dt.year - effective_lookback)
                    api_start_date = expanded_start.strftime("%Y-%m-%d")
                    self.logger.info(
                        f"Entity {entity_id}: Expanding MRTS date range for KPI "
                        f"({api_start_date} to {end_date}, {effective_lookback} years lookback)"
                    )

            # Inject expanded dates into row for the API call
            # Both start_date AND end_date must be present on the row, otherwise
            # get_mrts_api falls back to params (ignoring the expanded date).
            start_date_col = self.get_mapped_column("start_date")
            end_date_col = self.get_mapped_column("end_date")
            original_start_date = row.get(start_date_col)
            original_end_date = row.get(end_date_col)
            row[start_date_col] = api_start_date
            row[end_date_col] = end_date
            self.logger.info(
                f"Entity {entity_id}: API dates injected: "
                f"row[{start_date_col}]={api_start_date}, row[{end_date_col}]={end_date}"
            )

            # Step 2: Define API call wrapper
            def _call_api():
                return self.get_mrts_api(row)

            # Step 3: Call API
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Restore original dates on row after API call
            if original_start_date is not None:
                row[start_date_col] = original_start_date
            elif start_date_col in row:
                del row[start_date_col]
            if original_end_date is not None:
                row[end_date_col] = original_end_date
            elif end_date_col in row and end_date_col != start_date_col:
                del row[end_date_col]

            # Step 4: Format response to DataFrame
            vegetation_df = self.format_mrts_json(raw_json, entity_data=row)

            # Check if results exist
            if vegetation_df is None or vegetation_df.empty:
                self.logger.warning(f"Entity {entity_id}: No vegetation data found")
                return {"data": None, "error": {"message": "No MRTS data found", "entity_id": entity_id}}

            # Step 5: Determine output format based on KPI filter
            if kpi_filter:
                # KPI requested - compute KPI and return single row with metadata + KPI values
                self.logger.debug(f"Entity {entity_id}: Computing KPI values...")
                try:
                    years = validate_historical_years(self.get_entity_value(row, "years"))
                    if years is None:
                        if isinstance(historical_years_param, list):
                            years = historical_years_param
                        elif isinstance(historical_years_param, int) and historical_years_param > 0:
                            years = "ALL"
                    if years:
                        self.logger.debug(f"Entity {entity_id}: KPI historical years: {years}")

                    # Use explicit value_column from kpi_filter, or auto-detect from mode
                    kpi_value_column = kpi_filter.get("value_column")
                    if not kpi_value_column:
                        mrts_mode = params.get("mode", "full")
                        if mrts_mode == "raw" or "smoothed_value" not in vegetation_df.columns:
                            kpi_value_column = "raw_value"
                        else:
                            kpi_value_column = "smoothed_value"

                    # Call as standalone function, not as method
                    kpi_result = filter_timeseries_kpi(
                        timeseries_df=vegetation_df,
                        start_date=start_date,
                        end_date=end_date,
                        kpi_name=kpi_filter.get("kpi_name", "Vegetation KPI"),
                        aggregation=kpi_filter.get("aggregation", "accumulation"),
                        threshold=kpi_filter.get("threshold"),
                        window=kpi_filter.get("window"),
                        years=years,
                        date_column="date",
                        value_column=kpi_value_column,
                    )

                    self.logger.debug(
                        f"Entity {entity_id}: KPI computed - current: {kpi_result['current_period']['value']}, historical avg: {kpi_result['historical_avg']['value']}"
                    )

                    # Create DataFrame from KPI result
                    kpi_df = pd.DataFrame(
                        [
                            {
                                "kpi_name": kpi_result["kpi_name"],
                                "aggregation": kpi_result["aggregation"],
                                "start_date": start_date,
                                "end_date": end_date,
                                "current_value": kpi_result["current_period"]["value"],
                                "current_num_records": kpi_result["current_period"]["num_records"],
                                "historical_avg": kpi_result["historical_avg"]["value"],
                                "historical_num_years": kpi_result["historical_avg"]["num_years"],
                                "difference": kpi_result["comparison"]["difference"],
                                "percent_change": kpi_result["comparison"]["percent_change"],
                            }
                        ]
                    )

                    # Use normalize_with_metadata to add entity metadata (including years)
                    result_df = normalize_with_metadata(row, kpi_df)

                    self.logger.info(f"Entity {entity_id}: KPI processing successful")

                except Exception as kpi_error:
                    self.logger.error(f"Entity {entity_id}: KPI computation failed - {str(kpi_error)}")
                    return {
                        "data": None,
                        "error": {"message": f"KPI computation failed: {str(kpi_error)}", "entity_id": entity_id},
                    }

            else:
                # NO KPI - return all daily data with metadata
                self.logger.debug(f"Entity {entity_id}: Returning raw time series data ({len(vegetation_df)} records)")
                result_df = normalize_with_metadata(row, vegetation_df)

            # Step 6: Return result
            self.logger.info(f"Entity {entity_id}: Processing successful")
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_mrts_params
    def process_mrts_bulk_parallel(
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
        prefix="mrts",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity medium resolution vegetation time series using threads + progress bar,
        with optional fail-safe retry, filter capabilities, and caching.

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain 'id' and 'geometry'.
            params (dict, optional): Vegetation time series parameters override.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final CSV results.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "mrts".
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
                bulk_method=self._process_mrts_bulk_parallel_inner,
                params=self.mrts_params,
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

        return self._process_mrts_bulk_parallel_inner(
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
    @require_mrts_params
    def _process_mrts_bulk_parallel_inner(
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
        prefix="mrts",
        generate_report=False,
        report_options=None,
    ):
        """Inner bulk processing logic (called directly or via cache wrapper)."""
        params = params_kw
        self.logger.info(
            f"🚀 Starting bulk medium resolution vegetation time series extraction for {len(entity_list)} entities"
        )
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

        print(
            f"🔄 Processing {prefix.replace('_', ' ').title()} for {len(filtered_entity_list)} entities in parallel..."
        )
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(self.process_single_entity_mrts, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(
                total=len(future_to_id), desc=f"🌱 Processing {prefix.replace('_', ' ').title()}", unit="entity"
            ) as pbar:
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
                "parameters": getattr(self, "mrts_params", {}),
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

        self.logger.info("✅ Bulk mrts extraction complete")
        return summary
