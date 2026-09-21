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


def require_disease_params(func):
    """Decorator to ensure disease parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "disease_params") or self.disease_params is None:
            self.logger.error("No disease parameters found")
            raise RuntimeError("❌ No disease parameters found. Call setup_disease_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class DiseaseExtractor(BaseExtractor):
    """
    Extracts crop disease risk analytics for agricultural entities.

    Evaluates disease risk based on weather conditions and crop type. Returns daily disease
    parameter values (e.g., infection risk, sporulation, severity) for supported crops
    (corn, soybeans).

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_disease.ipynb

    Args (setup_disease_parameters):
        start_date (str): Start date in YYYY-MM-DD format. Default: None
        end_date (str): End date in YYYY-MM-DD format. Default: None
        kpi_filter (dict): KPI aggregation rules applied to the daily disease series. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); crop, start_date, end_date (optional overrides)

    Output columns:
        entity_id, date, + dynamic disease parameter columns (flattened from API response)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.disease_params = None

        # Specific API endpoint for disease
        self.disease_url = agro_urls["disease_url"][self.env]

        self.logger.info(f"🦠 DiseaseExtractor initialized for env: {self.env}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def get_new_token(self):
        """
        Implements token refresh logic for DiseaseExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Refreshing authentication token...")
        return EDAuthenticator.get_new_token(env=self.env)

    @requires_token
    def setup_disease_parameters(
        self,
        start_date=None,
        end_date=None,
        partial_frequency=50,
        kpi_filter=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure params for disease extraction for an entity

        Args:
            start_date (str, optional): Default start date (YYYY-MM-DD) used when entity row has no start_date
            end_date (str, optional): Default end date (YYYY-MM-DD) used when entity row has no end_date
            partial_frequency (int): How often to save partial results
            kpi_filter (dict, optional): KPI aggregation config. See api_utils.validate_kpi_filter
                for the supported aggregations and their threshold/window contracts.
            column_mapping (dict, optional): Override input column mapping for this extractor.
                Example: {"id": "entity_id", "geometry": "wkt", "crop": "crop_type"}
        """
        self.logger.info("Setting up disease parameters...")

        # Validate KPI filter (centralized rules — see api_utils.validate_kpi_filter)
        validate_kpi_filter(kpi_filter, logger=self.logger)

        self.apply_cache_setting(use_cache)

        self.disease_params = {
            "start_date": start_date,
            "end_date": end_date,
            "partial_frequency": partial_frequency,
            "kpi_filter": kpi_filter,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.logger.info("Disease parameters configured successfully")

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        print("🦠 Disease parameters configured:")
        for k, v in self.disease_params.items():
            print(f"   {k}: {v}")

    @require_disease_params
    @requires_token
    def get_disease_data(self, entity_data):
        """
        Get disease data for selected parameters using POST request

        Args:
            entity_data (dict): Entity data with id, geometry, start_date, end_date

        Returns:
            dict: Disease API response
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        # Step 1: Date validation (get_entity_value normalizes dates to YYYY-MM-DD strings)
        start_date = self.get_entity_value(entity_data, "start_date")
        end_date = self.get_entity_value(entity_data, "end_date")

        self.logger.debug(f"Entity {entity_id}: Validating dates ({start_date} to {end_date})")

        if not start_date or not isinstance(start_date, str):
            self.logger.error(f"Entity {entity_id}: Missing or invalid start_date - {start_date}")
            raise ValueError(f"Missing or invalid start_date '{start_date}'. Expected YYYY-MM-DD string.")
        if not end_date or not isinstance(end_date, str):
            self.logger.error(f"Entity {entity_id}: Missing or invalid end_date - {end_date}")
            raise ValueError(f"Missing or invalid end_date '{end_date}'. Expected YYYY-MM-DD string.")

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

        # Step 3: Build API URL with query parameters
        url = f"{self.disease_url}/launch"
        query_params = [
            f"startDate={start_date}",
            f"endDate={end_date}",
        ]

        full_url = f"{url}?{'&'.join(query_params)}"
        self.logger.debug(f"Entity {entity_id}: Calling disease API...")
        self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")

        # Step 4: Build request body
        request_body = {"id": entity_id, "geometry": geometry}

        # Step 5: Headers
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Step 6: POST request
        self.logger.debug(f"Entity {entity_id}: API URL: {full_url}")
        self.logger.debug(f"Entity {entity_id}: Payload: {request_body}")
        try:
            response = requests.post(full_url, json=request_body, headers=headers, timeout=60)
            response.raise_for_status()
            json_response = response.json()
            self.logger.debug(f"Entity {entity_id}: Disease API call successful")
            self.logger.debug(f"Entity {entity_id}: API response: {json_response}")
            return json_response
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            self.logger.error(f"Entity {entity_id}: HTTP error - {e.response.status_code}")
            self.logger.error(f"Entity {entity_id}: Response: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Disease API call failed - {str(e)}")
            raise

    @require_disease_params
    @requires_token
    def get_disease_data_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_disease_data().
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
            response_json = self.get_disease_data(entity_data)
            self.logger.debug(f"Entity {entity_id}: Disease data retrieved successfully")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            self.logger.error(f"Entity {entity_id}: HTTP {status} - {text}")
            return {"success": False, "data": None, "error": f"HTTP {status} - {text}", "entity_id": entity_id}
        except ValueError as e:
            self.logger.error(f"Entity {entity_id}: Validation error - {str(e)}")
            return {"success": False, "data": None, "error": f"Validation error: {str(e)}", "entity_id": entity_id}
        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Unexpected error - {str(e)}")
            return {"success": False, "data": None, "error": str(e), "entity_id": entity_id}

    def format_disease_json(self, response_disease_json, entity_data: dict = None):
        """
        Format disease API response into a DataFrame.
        Dynamically handles all disease parameters returned by the API.

        Args:
            response_disease_json (list or dict): JSON response from get_disease_data
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame: DataFrame with date and all disease parameter columns
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )
        self.logger.debug(f"Entity {entity_id}: Formatting disease JSON response...")

        empty = self.validate_api_response(response_disease_json, entity_id, "disease")
        if empty is not None:
            return empty

        # Handle if response is dict with data inside
        if isinstance(response_disease_json, dict):
            if "data" in response_disease_json:
                response_disease_json = response_disease_json["data"]
            elif "results" in response_disease_json:
                response_disease_json = response_disease_json["results"]

        # Ensure we have a list
        if not isinstance(response_disease_json, list):
            self.logger.warning(f"Entity {entity_id}: Unexpected response format")
            return pd.DataFrame()

        rows = []

        # Process each disease record
        for entry in response_disease_json:
            row = {}

            # Add entity_id first if provided
            if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
                row["entity_id"] = self.get_entity_value(entity_data, "id")

            # Add date
            row["date"] = entry.get("date")

            # Flatten all nested structures and add all disease parameters
            for key, value in entry.items():
                if key == "date":
                    continue

                # If value is a nested dict, flatten it
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        column_name = f"{key}.{sub_key}"
                        row[column_name] = sub_val
                else:
                    row[key] = value

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        if df.empty:
            self.logger.warning(f"Entity {entity_id}: No disease data to format")
            return df

        # Sort columns: entity_id (if exists) → date → alphabetically sorted disease params
        if "entity_id" in df.columns:
            disease_cols = sorted([col for col in df.columns if col not in ["entity_id", "date"]])
            df = df[["entity_id", "date"] + disease_cols]
        else:
            disease_cols = sorted([col for col in df.columns if col != "date"])
            df = df[["date"] + disease_cols]

        # Convert date to datetime and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self.logger.debug(
            f"Entity {entity_id}: Formatted {len(df)} disease records with {len(disease_cols)} parameters"
        )
        return df

    @require_disease_params
    @requires_token
    @cache_single_entity("disease_params")
    def process_single_entity_disease(self, row, params=None):
        """
        Process a single entity for disease data extraction with optional KPI filtering.

        Args:
            row (pd.DataFrame, pd.Series, or dict): Entity data with required fields
            params (dict, optional): Override disease_params

        Returns:
            dict: Contains 'data' (DataFrame) and 'error'
        """
        if params is None:
            params = self.disease_params

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
        self.logger.debug(f"Entity {entity_id}: Processing disease data...")

        try:
            # Get dates from row (using mapped column), fall back to params
            row_start = self.get_entity_value(row, "start_date")
            row_end = self.get_entity_value(row, "end_date")
            start_date = row_start or params.get("start_date")
            end_date = row_end or params.get("end_date")

            # Log date resolution with mapped column names for debugging
            start_col = self.get_mapped_column("start_date")
            end_col = self.get_mapped_column("end_date")

            if not start_date:
                self.logger.error(
                    f"Entity {entity_id}: Missing start_date — "
                    f"not found in row['{start_col}'] and not set in setup_disease_parameters(start_date=...)"
                )
                return {
                    "data": None,
                    "error": {
                        "message": f"Missing start_date: column '{start_col}' not in row and no default in params",
                        "entity_id": entity_id,
                    },
                }

            if not end_date:
                self.logger.error(
                    f"Entity {entity_id}: Missing end_date — "
                    f"not found in row['{end_col}'] and not set in setup_disease_parameters(end_date=...)"
                )
                return {
                    "data": None,
                    "error": {
                        "message": f"Missing end_date: column '{end_col}' not in row and no default in params",
                        "entity_id": entity_id,
                    },
                }

            self.logger.debug(
                f"Entity {entity_id}: start_date={start_date} ({'from row' if row_start else 'from params'}), "
                f"end_date={end_date} ({'from row' if row_end else 'from params'})"
            )

            years = validate_historical_years(self.get_entity_value(row, "years"))

            # Get KPI filter configuration from params
            kpi_filter = params.get("kpi_filter")
            if kpi_filter:
                self.logger.debug(
                    f"Entity {entity_id}: KPI filter enabled - {kpi_filter.get('aggregation', 'unknown')}"
                )

            # Inject resolved dates into row so get_disease_data can find them
            row[self.get_mapped_column("start_date")] = start_date
            row[self.get_mapped_column("end_date")] = end_date

            # Step 2: Define API call wrapper
            def _call_api():
                return self.get_disease_data(row)

            # Step 3: Call API
            self.logger.debug(f"Entity {entity_id}: Calling API with retry logic...")
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Step 4: Format response to DataFrame
            disease_df = self.format_disease_json(raw_json, entity_data=row)

            # Check if results exist
            if disease_df is None or disease_df.empty:
                self.logger.warning(f"Entity {entity_id}: No disease data found")
                return {"data": None, "error": {"message": "No disease data found", "entity_id": entity_id}}

            # Step 5: Determine output format based on KPI filter
            if kpi_filter:
                # KPI requested - compute KPI and return single row with metadata + KPI values
                self.logger.debug(f"Entity {entity_id}: Computing KPI values...")
                try:
                    # Handle years - if not provided, use "ALL" for all historical data
                    if years is None and params.get("historical_years", 0) > 0:
                        years = "ALL"
                        self.logger.debug(f"Entity {entity_id}: Using all available historical data")

                    # Determine which column to use for KPI
                    kpi_column = kpi_filter.get("value_column")
                    if not kpi_column:
                        kpi_column = row.get("disease_kpi_column")
                    if not kpi_column:
                        # Use first disease parameter column (skip entity_id and date)
                        numeric_cols = [
                            col
                            for col in disease_df.columns
                            if col not in ["entity_id", "date"] and pd.api.types.is_numeric_dtype(disease_df[col])
                        ]
                        if not numeric_cols:
                            self.logger.error(f"Entity {entity_id}: No numeric disease columns found for KPI")
                            raise ValueError("No numeric disease columns found for KPI computation")
                        kpi_column = numeric_cols[0]

                    self.logger.debug(f"Entity {entity_id}: Using column '{kpi_column}' for KPI computation")

                    # Call standalone KPI function
                    kpi_result = filter_timeseries_kpi(
                        timeseries_df=disease_df,
                        start_date=start_date,
                        end_date=end_date,
                        kpi_name=kpi_filter.get("kpi_name", f"Disease KPI ({kpi_column})"),
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
                # NO KPI - return all daily disease data with metadata
                self.logger.debug(f"Entity {entity_id}: Returning raw disease data ({len(disease_df)} records)")
                result_df = normalize_with_metadata(row, disease_df)

            # Step 6: Return result
            self.logger.info(f"Entity {entity_id}: Processing successful")
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: Processing failed - {str(e)}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_disease_params
    def process_entity_disease_bulk_parallel(
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
        prefix="disease",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity disease data using threads + progress bar,
        with optional fail-safe retry and filter capabilities.

        Args:
            entity_list (pd.DataFrame): List of entities to process
            params (dict, optional): Disease parameters override
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

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_entity_disease_bulk_parallel_inner,
                entity_list=entity_list,
                params=self.disease_params,
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
        return self._process_entity_disease_bulk_parallel_inner(
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
    @require_disease_params
    def _process_entity_disease_bulk_parallel_inner(
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
        prefix="disease",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of entity disease data using threads + progress bar,
        with optional fail-safe retry and filter capabilities.
        """
        params = params_kw
        self.logger.info(f"🚀 Starting bulk disease extraction for {len(entity_list)} entities")
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
                executor.submit(self.process_single_entity_disease, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(
                total=len(future_to_id), desc=f"🦠 Processing {prefix.replace('_', ' ').title()}", unit="entity"
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
                "parameters": getattr(self, "disease_params", {}),
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

        self.logger.info("✅ Bulk disease extraction complete")
        return summary
