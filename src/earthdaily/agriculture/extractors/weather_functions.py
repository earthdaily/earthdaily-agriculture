# weather_functions.py
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
    autosize_row_limit,
    export_results,
    filter_entities,
    filter_timeseries_kpi,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
    validate_historical_years,
    validate_kpi_filter,
)
from earthdaily.agriculture.core.base_extractor import (
    DEFAULT_MAX_WINDOW_DAYS,
    DEFAULT_SPATIAL_PRECISION,
    BaseExtractor,
    cache_single_entity,
    requires_token,
)
from earthdaily.agriculture.core.geometry import get_centroid_wkt, validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

# Weather parameters available
available_weather_parameters = {
    "cloudcover.min",
    "cloudcover",
    "cloudcover.max",
    "dewpoint",
    "evapotranspiration",
    "humidity.relativemin",
    "humidity",
    "humidity.relativemax",
    "precipitation.cumulative",
    "solarradiation",
    "sunshineduration",
    "Temperature.standardmin",
    "Temperature.standard",
    "Temperature.standardmax",
    "wind",
}
available_weather_type = {"HISTORICAL_DAILY", "FORECAST_DAILY", "FORECAST_HOURLY"}


def require_weather_params(func):
    """Decorator to ensure weather parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "weather_params"):
            self.logger.error("No weather parameters found")
            raise RuntimeError("❌ No weather parameters found. Call setup_weather_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class WeatherExtractor(BaseExtractor):
    """
    Extracts field-level weather data for agricultural entities.

    Retrieves historical daily weather data and agro-meteorological indices for a given
    geometry and date range. Supports configurable weather parameters and KPI aggregation.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_weather.ipynb

    Args (setup_weather_parameters):
        weather_type (str): Weather data type ('HISTORICAL_DAILY'). Default: 'HISTORICAL_DAILY'
        weather_parameters (str|list): Parameters to retrieve; 'none' expands to all available parameters. Default: 'precipitation.cumulative'
        kpi_filter (dict): KPI aggregation config (kpi_name, aggregation, threshold)
        historical_years (int): Number of prior years of weather history to fetch. Default: 0
        page_limit (int): Override the API row limit ($limit); by default it is auto-sized to the query span so multi-year ranges are not truncated. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); start_date, end_date (optional overrides)

    Output columns:
        entity_id, date, + dynamic weather parameter columns (temperature, precipitation, etc.)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.weather_params = None

        # Specific API endpoint for weather
        self.weather_url = agro_urls["weather_url"][self.env]

        self.logger.info(f"🌤️ WeatherExtractor initialized for env: {self.env}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for WeatherExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Refreshing authentication token...")
        return EDAuthenticator.get_new_token(env=self.env)

    @requires_token
    def setup_weather_parameters(
        self,
        weather_type="HISTORICAL_DAILY",
        weather_parameters="precipitation.cumulative",
        historical_years=0,
        partial_frequency=50,
        kpi_filter=None,
        page_limit=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure params for weather extraction for an entity

        Args:
            weather_type (str): Type of weather data (HISTORICAL_DAILY, FORECAST_DAILY, FORECAST_HOURLY)
            weather_parameters (str or list): Weather parameters to extract. Defaults to
                'precipitation.cumulative'. Pass 'none' to retrieve every available parameter.
            historical_years (int or list): Number of historical years for KPI comparison, or list of specific years
            partial_frequency (int): How often to save partial results
            kpi_filter (dict, optional): KPI configuration
            page_limit (int, optional): Override the API row limit ($limit). By default the
                limit is auto-sized to the query span (see get_weather_data) so long
                multi-year ranges are not truncated; set this to force an explicit value.
                Must be a positive integer. Default: None (auto-size).
        """
        self.logger.info("Setting up weather parameters...")

        if weather_type not in available_weather_type:
            self.logger.error(f"Invalid weather type: {weather_type}")
            raise ValueError(f"Invalid weather '{weather_type}'. Must be one of {available_weather_type}")

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

        # Validate weather parameters — accepts string, list, or "none" (all params)
        if weather_parameters != "none":
            params_list = weather_parameters if isinstance(weather_parameters, list) else [weather_parameters]
            invalid = [p for p in params_list if p not in available_weather_parameters]
            if invalid:
                self.logger.error(f"Invalid weather parameter(s): {invalid}")
                raise ValueError(f"Invalid weather parameter(s) {invalid}. Must be from {available_weather_parameters}")

        # Validate KPI filter (centralized rules — see api_utils.validate_kpi_filter)
        validate_kpi_filter(kpi_filter, logger=self.logger)

        # Validate page_limit override (None = auto-size the limit to the query span)
        if page_limit is not None and (
            not isinstance(page_limit, int) or isinstance(page_limit, bool) or page_limit < 1
        ):
            self.logger.error(f"Invalid page_limit: {page_limit}")
            raise ValueError(f"Invalid page_limit={page_limit}. Must be a positive integer or None.")

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.apply_cache_setting(use_cache)

        # ✅ store in self.weather_params
        self.weather_params = {
            "weather_type": weather_type,
            "weather_parameters": weather_parameters,
            "historical_years": historical_years,
            "partial_frequency": partial_frequency,
            "kpi_filter": kpi_filter,
            "page_limit": page_limit,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        self.logger.info("Weather parameters configured successfully")
        self.logger.debug(f"Weather type: {weather_type}")
        self.logger.debug(f"Parameters: {weather_parameters}")
        if kpi_filter:
            self.logger.debug(f"KPI filter enabled: {kpi_filter.get('aggregation')}")

        print("🌍 Weather parameters configured:")
        for k, v in self.weather_params.items():
            if k == "kpi_filter" and v:
                print(f"   {k}:")
                for kpi_key, kpi_val in v.items():
                    print(f"      {kpi_key}: {kpi_val}")
            else:
                print(f"   {k}: {v}")

    @require_weather_params
    @requires_token
    def get_weather_data(self, entity_data):
        """
        Get weather data for selected parameters using self.weather_params

        The API returns one row per day in ascending date order and caps the response at
        ``$limit`` rows, so a fixed limit silently truncates long ranges to the *earliest* N
        days (dropping the most recent data — the current-period window for KPIs). The row
        limit is therefore auto-sized to the query span via
        ``api_utils.autosize_row_limit`` (floor 1000 keeps short single-season/forecast
        requests byte-identical). A ``page_limit`` set in ``setup_weather_parameters`` wins
        over the auto-sized value.

        Args:
            entity_data (dict): Entity data with geometry, start_date, end_date

        Returns:
            dict: Weather API response
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        if not hasattr(self, "weather_params") or self.weather_params is None:
            self.logger.error("Weather parameters not configured")
            raise RuntimeError("❌ Weather parameters not configured. Call setup_weather_parameters() first.")

        params = self.weather_params

        # Step 1: Date validation
        start_date = self.get_entity_value(entity_data, "start_date")
        end_date = self.get_entity_value(entity_data, "end_date")

        self.logger.debug(f"Entity {entity_id}: Validating dates ({start_date} to {end_date})")

        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")

        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid start_date format - {start_date}")
            raise ValueError(f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD")
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            self.logger.error(f"Entity {entity_id}: Invalid end_date format - {end_date}")
            raise ValueError(f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD")

        # Step 2: Geometry validation
        geometry = self.get_entity_value(entity_data, "geometry")
        self.logger.debug(f"Entity {entity_id}: Validating geometry...")
        geometry = validate_wkt(geometry)
        centroid = get_centroid_wkt(geometry)
        self.logger.debug(f"Entity {entity_id}: Centroid calculated")

        # Step 3: Handle weather_params - ensure it's a list
        weather_params = params["weather_parameters"]
        if isinstance(weather_params, str):
            weather_params = [weather_params]

        # The "none" sentinel is documented as "all parameters" — translate it
        # to the full set here. Sending it literally produces `$fields=none,Date`,
        # which the prod Weather API rejects with 400 unselectable_field_error.
        if weather_params == ["none"]:
            weather_params = sorted(available_weather_parameters)

        # Step 4: API URL
        url = f"{self.weather_url}/weather"

        # Auto-size the row limit to the [start_date, end_date] span so long (multi-year)
        # ranges are not truncated to the earliest 1000 days. An explicit page_limit wins.
        page_limit = params.get("page_limit")
        row_limit = page_limit if page_limit is not None else autosize_row_limit(start_date, end_date)

        # Build query parameters
        query_params = [
            "$offset=0",
            "$count=false",
            f"$limit={row_limit}",
            f"$fields={','.join(weather_params)},Date",
            "Provider=GLOBAL1",
            f"WeatherType={params['weather_type']}",
            f"Location={centroid}",
            f"Date=$between:{start_date}T00:00:00.0000000Z|{end_date}T23:59:59.0000000Z",
        ]

        # --- Full URL ---
        full_url = f"{url}?{'&'.join(query_params)}"
        self.logger.debug(f"Entity {entity_id}: Calling weather API...")
        self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")

        # --- Headers ---
        headers = {"Authorization": f"Bearer {self.bearer_token}", "Accept": "application/json"}

        # Step 5: Request data
        try:
            response = requests.get(full_url, headers=headers, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.debug(f"Entity {entity_id}: Weather API call successful")
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | url={full_url}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Weather API call failed - {str(e)} | url={full_url}")
            raise

    @require_weather_params
    @requires_token
    def get_weather_data_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_weather_data().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain 'id', 'geometry', 'start_date', 'end_date' keys

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
            response_json = self.get_weather_data(entity_data)
            self.logger.debug(f"Entity {entity_id}: Weather data retrieved successfully")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            # HTTPError from requests includes response object
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text}")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "entity_id": entity_id}
        except ValueError as e:
            # Catches date format errors and geometry validation errors
            self.logger.error(f"Entity {entity_id}: Validation error - {str(e)}")
            return {"success": False, "data": None, "error": f"Validation error: {str(e)}", "entity_id": entity_id}
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Unexpected error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "entity_id": entity_id}

    def format_weather_json(self, response_weather_json, entity_data: dict = None):
        """
        Format weather API response into a DataFrame.
        Dynamically handles all weather parameters returned by the API.

        Args:
            response_weather_json (list): JSON response from get_weather_data
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame: DataFrame with date and all weather parameter columns
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )
        self.logger.debug(f"Entity {entity_id}: Formatting weather JSON response...")

        empty = self.validate_api_response(response_weather_json, entity_id, "weather")
        if empty is not None:
            return empty

        rows = []

        # Process each weather record
        for entry in response_weather_json:
            row = {}

            # Add entity_id first if provided
            if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
                row["entity_id"] = self.get_entity_value(entity_data, "id")

            # Add date
            row["date"] = entry.get("date")

            # Flatten all nested structures and add all weather parameters
            for key, value in entry.items():
                if key == "date":
                    continue  # Already added

                # If value is a nested dict (like temperature), flatten it
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        # Create column name: e.g., "Temperature.standardMin"
                        column_name = f"{key}.{sub_key}"
                        row[column_name] = sub_val
                else:
                    # Direct value (precipitation, wind, etc.)
                    row[key] = value

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        if df.empty:
            self.logger.warning(f"Entity {entity_id}: No weather data to format")
            return df

        # Sort columns: entity_id (if exists) → date → alphabetically sorted weather params
        if "entity_id" in df.columns:
            weather_cols = sorted([col for col in df.columns if col not in ["entity_id", "date"]])
            df = df[["entity_id", "date"] + weather_cols]
        else:
            weather_cols = sorted([col for col in df.columns if col != "date"])
            df = df[["date"] + weather_cols]

        # Convert date to datetime and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self.logger.debug(
            f"Entity {entity_id}: Formatted {len(df)} weather records with {len(weather_cols)} parameters"
        )
        return df

    @require_weather_params
    @requires_token
    @cache_single_entity("weather_params")
    def process_single_entity_weather(self, row, params=None):
        """
        Process a single entity for weather data extraction with optional KPI filtering.

        Args:
            row (pd.DataFrame, pd.Series, or dict): Entity data with required fields
            params (dict, optional): Override weather_params

        Returns:
            dict: Contains 'data' (DataFrame) and 'error'
        """
        if params is None:
            params = self.weather_params

        # Handle different input types - convert all to dict
        try:
            if isinstance(row, pd.DataFrame):
                if len(row) > 1:
                    self.logger.warning(f"DataFrame contains {len(row)} rows. Processing only the first row.")
                row = row.iloc[0].to_dict()
            elif isinstance(row, pd.Series):
                row = row.to_dict()
            elif not isinstance(row, dict):
                self.logger.error(f"Invalid input type: {type(row)}")
                raise TypeError(f"row must be DataFrame, Series, or dict. Got {type(row)}")
        except Exception as e:
            self.logger.error(f"Failed to convert input to dict: {str(e)}")
            return {"data": None, "error": {"message": f"Invalid input type: {str(e)}", "entity_id": "unknown"}}

        entity_id = self.get_entity_value(row, "id", "unknown")
        self.logger.debug(f"Entity {entity_id}: Processing weather data...")

        try:
            # Get dates from row, fall back to params
            start_date = self.get_entity_value(row, "start_date") or params.get("start_date")
            end_date = self.get_entity_value(row, "end_date") or params.get("end_date")

            if not start_date:
                self.logger.error(f"Entity {entity_id}: Missing start_date in both row and params")
                return {
                    "data": None,
                    "error": {"message": "Missing start_date in both row and params", "entity_id": entity_id},
                }

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
            # Weather API uses absolute $between date range, so we push start_date back.
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
                        f"Entity {entity_id}: Expanding Weather date range for KPI "
                        f"({api_start_date} to {end_date}, {effective_lookback} years lookback)"
                    )

            # Inject expanded dates into row for the API call
            start_date_col = self.get_mapped_column("start_date")
            end_date_col = self.get_mapped_column("end_date")
            original_start_date = row.get(start_date_col)
            original_end_date = row.get(end_date_col)
            row[start_date_col] = api_start_date
            row[end_date_col] = end_date

            # Step 2: Define API call wrapper
            def _call_api():
                return self.get_weather_data(row)

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
            weather_df = self.format_weather_json(raw_json, entity_data=row)

            # Check if results exist
            if weather_df is None or weather_df.empty:
                self.logger.warning(f"Entity {entity_id}: No weather data found")
                return {"data": None, "error": {"message": "No weather data found", "entity_id": entity_id}}

            # Step 5: Determine output format based on KPI filter
            if kpi_filter:
                # KPI requested - compute KPI and return single row with metadata + KPI values
                self.logger.debug(f"Entity {entity_id}: Computing KPI values...")
                try:
                    # Handle years - priority: entity years > historical_years list > "ALL"
                    if years is None:
                        if isinstance(historical_years_param, list):
                            years = historical_years_param
                        elif isinstance(historical_years_param, int) and historical_years_param > 0:
                            years = "ALL"
                    if years:
                        self.logger.debug(f"Entity {entity_id}: KPI historical years: {years}")

                    # Determine which column to use for KPI
                    kpi_column = kpi_filter.get("value_column")
                    if not kpi_column:
                        kpi_column = row.get("weather_kpi_column")
                    if not kpi_column:
                        # Use first weather parameter column (skip entity_id and date)
                        numeric_cols = [
                            col
                            for col in weather_df.columns
                            if col not in ["entity_id", "date"] and pd.api.types.is_numeric_dtype(weather_df[col])
                        ]
                        if not numeric_cols:
                            self.logger.error(f"Entity {entity_id}: No numeric weather columns found for KPI")
                            raise ValueError("No numeric weather columns found for KPI computation")
                        kpi_column = numeric_cols[0]

                    self.logger.debug(f"Entity {entity_id}: Using column '{kpi_column}' for KPI computation")

                    # Call standalone KPI function
                    kpi_result = filter_timeseries_kpi(
                        timeseries_df=weather_df,
                        start_date=start_date,
                        end_date=end_date,
                        kpi_name=kpi_filter.get("kpi_name", f"Weather KPI ({kpi_column})"),
                        aggregation=kpi_filter.get("aggregation", "average"),
                        threshold=kpi_filter.get("threshold"),
                        window=kpi_filter.get("window"),
                        years=years,
                        date_column="date",
                        value_column=kpi_column,
                    )

                    self.logger.debug(
                        f"Entity {entity_id}: KPI computed - current: {kpi_result['current_period']['value']}, historical avg: {kpi_result['historical_avg']['value']}"
                    )

                    # Create DataFrame from KPI result
                    kpi_df = pd.DataFrame(
                        [
                            {
                                "kpi_name": kpi_result["kpi_name"],
                                "kpi_column": kpi_column,
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

                    # Use normalize_with_metadata to add entity metadata
                    result_df = normalize_with_metadata(row, kpi_df)

                    self.logger.info(f"Entity {entity_id}: KPI processing successful")

                except Exception as kpi_error:
                    self.logger.error(f"Entity {entity_id}: KPI computation failed - {str(kpi_error)}")
                    return {
                        "data": None,
                        "error": {"message": f"KPI computation failed: {str(kpi_error)}", "entity_id": entity_id},
                    }

            else:
                # NO KPI - return all daily weather data with metadata
                self.logger.debug(f"Entity {entity_id}: Returning raw weather data ({len(weather_df)} records)")
                result_df = normalize_with_metadata(row, weather_df)

            # Step 6: Return result
            self.logger.info(f"Entity {entity_id}: Processing successful")
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_weather_params
    def process_entity_weather_bulk_parallel(
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
        prefix="weather",
        generate_report=False,
        report_options=None,
        use_cache=None,
        spatial_grouping=False,
        spatial_precision=DEFAULT_SPATIAL_PRECISION,
        spatial_max_window_days=DEFAULT_MAX_WINDOW_DAYS,
    ):
        """
        Bulk processing of entity weather data using threads + progress bar,
        with optional fail-safe retry and filter capabilities.

        Args:
            entity_list (pd.DataFrame): List of entities to process
            params (dict, optional): Weather parameters override
            max_workers (int): Number of threads to use
            output_path (str, optional): Directory to save results
            partial_frequency (int): How often to save partial results
            fail_safe (bool): If True, retries only previously failed entities
            filter_column (str, optional): Column name to filter entities by
            filter_value (any, optional): Value to filter
            filter_type (str): 'exclude' or 'include'
            merge_existing (str, optional): Merge strategy
            skip_export (bool): If True, skip final export
            prefix (str): Prefix for output filenames
            use_cache (bool, optional): Enable/disable caching for this call
            spatial_grouping (bool): If True, group fields by ``geohash(centroid) ×
                historical_years``, call the API once per group over the *union* of
                the group's date windows, and slice each member out of that single
                response. Weather is queried by field centroid
                (``Location=<centroid>``), so neighbouring fields return identical
                values and dedup safely; members are cut back to their own window, so
                the output matches an ungrouped run row for row. Weather parameters are
                all per-day — including ``precipitation.cumulative``, which despite its
                name is measured to be window-independent — so no value correction is
                applied.
                With ``use_cache``, the geohash-keyed spatial cache is consulted per
                cell before any request — a cell whose window is already on disk costs
                no API call, whatever field set asks for it.
            spatial_precision (int): Geohash precision for the bucket (default
                ``DEFAULT_SPATIAL_PRECISION``).
            spatial_max_window_days (int): Cap on a group's widened window (default
                ``DEFAULT_MAX_WINDOW_DAYS``). Cells whose union exceeds it are split
                into sub-groups, so one long-history field cannot force a decade-long
                pull on every field sharing its cell.

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics
        """

        def _dispatch(
            el, *, skip_export_=skip_export, generate_report_=generate_report, report_options_=report_options
        ):
            if use_cache is not None and use_cache:
                return self._bulk_with_cache(
                    bulk_method=self._process_entity_weather_bulk_parallel_inner,
                    entity_list=el,
                    params=self.weather_params,
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
            return self._process_entity_weather_bulk_parallel_inner(
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
                return self._process_entity_weather_bulk_parallel_inner(
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
                signature_columns=["historical_years", "years"],
                window_columns=("start_date", "end_date"),
                max_window_days=spatial_max_window_days,
                # No cumulative columns: `precipitation.cumulative` is a per-day total
                # despite its name — measured identical across overlapping windows, so
                # rebasing it would corrupt the values rather than correct them.
                cumulative_columns=None,
                precision=spatial_precision,
                prefix=prefix,
                output_path=output_path,
                skip_export=skip_export,
                generate_report=generate_report,
                report_options=report_options,
                report_params=getattr(self, "weather_params", {}),
                use_cache=use_cache,
                cache_params=getattr(self, "weather_params", {}),
            )
        return _dispatch(entity_list)

    @requires_token
    @require_weather_params
    def _process_entity_weather_bulk_parallel_inner(
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
        prefix="weather",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of entity weather data using threads + progress bar,
        with optional fail-safe retry and filter capabilities.
        """
        params = params_kw
        self.logger.info(f"🚀 Starting bulk weather extraction for {len(entity_list)} entities")
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
                executor.submit(self.process_single_entity_weather, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(
                total=len(future_to_id), desc=f"🌤️ Processing {prefix.replace('_', ' ').title()}", unit="entity"
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
                "parameters": getattr(self, "weather_params", {}),
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

        self.logger.info("✅ Bulk weather extraction complete")
        return summary
