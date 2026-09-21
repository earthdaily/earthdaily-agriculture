#  Standard Library
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd

#  Third-Party Libraries
import requests
from tqdm import tqdm

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.api_utils import (
    decode_day_of_year,
    export_results,
    filter_entities,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.identity import EDAuthenticator


def require_regional_params(func):
    """Decorator to ensure regional parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "regional_params") or self.regional_params is None:
            error_msg = "❌ No regional parameters found. Call setup_regional_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class RegionalExtractor(BaseExtractor):
    """
    Extracts regional-scale time series analytics for administrative boundaries.

    Retrieves aggregated vegetation and crop monitoring indices at regional level
    (countries, states, municipalities). Supports multiple indicator types and
    historical gap-filling.

    Documentation: https://docs.earthdaily.com/agro/library/Regional_Monitoring/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_VTS.ipynb

    Args (setup_regional_parameters):
        index (str): Regional index type. Default: 'vegetation-vigor-index'
        start_date (str): Start date in YYYY-MM-DD format. Default: '2025-01-01'
        end_date (str): End date in YYYY-MM-DD format. Default: None (last day of current year)
        fillyeargap (bool): Fill year gaps in data. Default: False
        idblock (str): Block identifier. Default: None
        idpixeltype (str): Pixel type identifier. Default: None
        indicatorTypeIds (list): Indicator type IDs — accepted values are 1-5 only
            (1 VVI, 2/3 weather observed, 4/5 weather forecast). Default: None
            (auto-set to [1] when index='vegetation-vigor-index')
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        amu_id (required — regional AMU entity ID)

    Output columns:
        entity_id, date, year, regional index values
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.regional_params = None

        # Specific API endpoint for coverage
        self.regional_url = agro_urls["regional_url"][self.env]

        self.logger.debug(f"Regional URL configured: {self.regional_url}")
        self.logger.info(f"🛰️ Regional Extractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for RegionalExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for RegionalExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_regional_parameters(
        self,
        index="vegetation-vigor-index",
        start_date="2025-01-01",
        end_date=None,
        fillyeargap=False,
        idblock=None,
        idpixeltype=None,
        indicatorTypeIds=None,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure params for regional analytic extraction for an entity id.

        Args:
            index: Type of vegetation index (default: "vegetation-vigor-index")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: last day of current year)
            fillyeargap: Whether to fill year gaps in data
            idblock: Block ID (267 or 141)
            idpixeltype: Pixel type ID
                - [1] ALL_VEGETATIONS
                - [2] SUMMER
                - [3] WINTER
                - [7] GRASSLANDS
                - [8] ALL CROPS
                - [10] CORN
                - [11] SOYBEAN
                - [301] BR SOYBEAN
                - [400] EUR CORN
                - [401] EUR WINTER WHEAT
                - [406] EUR BARLEY
                - [407] EUR RAPESEED
                - [402] EUR SURGARBEET

            indicatorTypeIds: List of indicator type IDs. Only 1-5 are accepted; any
                other value raises ValueError:
                - [1] for VVI
                - [2] for WEATHER OBSERVED ECMWF all blocks
                - [3] for WEATHER OBSERVED AROME on over France
                - [4] for WEATHER FORECAST ECMWF all blocks
                - [5] for WEATHER FORECAST GFS all blocks
                WEATHER REANALYSIS (10) and WEATHER RAINFALL ESTIMATES HI-RES / CHIRPS (11)
                are documented by the API but are **not** in this extractor's accepted set —
                widen valid_indicatorTypeIds below before passing them.
            partial_frequency: Frequency threshold for partial data
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring regional parameters...")

        # Default end_date to last day of current year if not provided
        if end_date is None:
            end_date = f"{datetime.now().year}-12-31"
            log.info(f"end_date defaulted to {end_date}")

        # Validate index
        valid_indexes = {
            "vegetation-vigor-index",
            "daily-precipitation",
            "soil-moisture",
            "min-temperature",
            "max-temperature",
            "average-temperature",
            "surface-temperature",
            "etp",
            "max-wind-speed",
            "p-etp",
            "relative-humidity",
            "snow-depth",
            "solar-radiation",
        }
        if index not in valid_indexes:
            error_msg = f"Invalid index '{index}'. Must be one of {valid_indexes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate start_date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            error_msg = f"Invalid start_date format '{start_date}'. Expected YYYY-MM-DD"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate end_date format if provided
        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                error_msg = f"Invalid end_date format '{end_date}'. Expected YYYY-MM-DD"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate idblock if provided
        if idblock is not None:
            valid_idblock = {301, 281, 267, 226, 207, 204, 202, 197, 181, 141, 140, 135, 131, 130, 129, 127, 125, 115}
            if idblock not in valid_idblock:
                error_msg = f"Invalid block selected '{idblock}'. Must be one of {valid_idblock}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate idpixeltype if provided
        if idpixeltype is not None:
            valid_idpixeltype = {1, 2, 3, 7, 8, 10, 11, 301, 400, 401, 402, 406, 407}
            if idpixeltype not in valid_idpixeltype:
                error_msg = f"Invalid pixel type selected '{idpixeltype}'. Must be one of {valid_idpixeltype}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate indicatorTypeIds if provided
        if indicatorTypeIds is not None:
            if not isinstance(indicatorTypeIds, list):
                error_msg = f"indicatorTypeIds must be a list, got {type(indicatorTypeIds)}"
                log.error(error_msg)
                raise ValueError(error_msg)

            valid_indicatorTypeIds = {1, 2, 3, 4, 5}
            for indicator_id in indicatorTypeIds:
                if indicator_id not in valid_indicatorTypeIds:
                    error_msg = f"Invalid indicator ID '{indicator_id}'. Must be one of {valid_indicatorTypeIds}"
                    log.error(error_msg)
                    raise ValueError(error_msg)

        # Auto-set indicatorTypeIds for vegetation-vigor-index, and validate consistency
        if index == "vegetation-vigor-index":
            if indicatorTypeIds is None:
                indicatorTypeIds = [1]
                log.info("indicatorTypeIds auto-set to [1] for vegetation-vigor-index")
            elif 1 not in indicatorTypeIds:
                log.warning(
                    f"indicatorTypeIds={indicatorTypeIds} does not contain 1 (VVI) but index is 'vegetation-vigor-index'. "
                    "Auto-adding 1 to indicatorTypeIds. Please check your parameters."
                )
                indicatorTypeIds = [1] + indicatorTypeIds
        else:
            if indicatorTypeIds is not None and 1 in indicatorTypeIds:
                log.warning(
                    f"indicatorTypeIds contains 1 (VVI) but index is '{index}', not 'vegetation-vigor-index'. "
                    "Removing 1 from indicatorTypeIds. Please check your parameters."
                )
                indicatorTypeIds = [i for i in indicatorTypeIds if i != 1]

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.apply_cache_setting(use_cache)

        # Store in self.regional_params
        self.regional_params = {
            "index": index,
            "start_date": start_date,
            "end_date": end_date,
            "fillyeargap": fillyeargap,
            "idblock": idblock,
            "idpixeltype": idpixeltype,
            "indicatorTypeIds": indicatorTypeIds,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "date"]

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("✅ Regional parameters configured successfully")
        log.debug(f"Parameters: {self.regional_params}")

        print("🌍 Regional parameters configured:")
        for k, v in self.regional_params.items():
            print(f"   {k}: {v}")

    @staticmethod
    def safe_convert_amu_id(amu_id: Union[str, int, float]) -> int:
        """
        Safely convert amu_id to integer, handling various input formats.

        Handles:
        - Already an int: returns as-is
        - String integer: "141" -> 141
        - String tuple representation: '(141;"County";...)' -> 141
        - Float: 141.0 -> 141

        Args:
            amu_id: The AMU ID in any format

        Returns:
            int: Cleaned integer AMU ID

        Raises:
            ValueError: If amu_id cannot be converted to int

        Examples:
            >>> safe_convert_amu_id(141)
            141
            >>> safe_convert_amu_id("141")
            141
            >>> safe_convert_amu_id('(141;"County";"Missouri")')
            141
            >>> safe_convert_amu_id(141.0)
            141
        """
        # Already an integer
        if isinstance(amu_id, int):
            return amu_id

        # Handle float (like 141.0)
        if isinstance(amu_id, float):
            if amu_id.is_integer():
                return int(amu_id)
            else:
                raise ValueError(f"Float amu_id must be a whole number: {amu_id}")

        # Handle None or empty
        if amu_id is None or amu_id == "":
            raise ValueError("amu_id cannot be None or empty")

        # Convert to string for processing
        amu_str = str(amu_id).strip()

        # Check if it's a tuple-like string: starts with '(' or contains ';'
        if amu_str.startswith("(") or ";" in amu_str:
            # Extract first number from tuple representation
            # Match first number after '(' or at start
            match = re.search(r"^\(?(\d+)", amu_str)
            if match:
                return int(match.group(1))
            else:
                raise ValueError(f"Cannot extract integer from tuple string: {amu_str}")

        # Simple string integer
        try:
            return int(amu_str)
        except ValueError:
            raise ValueError(f"Cannot convert amu_id to integer: {amu_str}")

    @requires_token
    @require_regional_params
    def get_regional_ts_by_id(self, entity_data: dict):
        """
        Query the regional analytic API to get cumulative vegetation vigor data for a given entity id.

        Args:
            entity_data (dict): Must contain the mapped 'amu_id' column (configurable via column_mapping).

        Returns:
            dict: Raw JSON response from API.
        """
        params = self.regional_params
        raw_entity_id = self.get_entity_value(entity_data, "amu_id", "unknown")
        log = self.get_contextualized_logger("API")

        # Step 1: Convert amu_id to integer
        try:
            entity_id = self.safe_convert_amu_id(raw_entity_id)
            log.debug(f"Converted amu_id: {raw_entity_id} -> {entity_id}")
        except ValueError as e:
            log.error(f"❌ Invalid amu_id format: {e}")
            raise ValueError(f"Invalid amu_id in entity_data: {e}")

        # Step 2: API URL construction
        url = f"{self.regional_url}/api/{params['index']}"
        # print(url)
        log.debug(f"API URL: {url}")

        # Step 3: Construct payload using configured parameters
        payload = {
            "amuIds": [entity_id],  # List with integer ID
        }

        # Add required parameters
        if params["idblock"] is not None:
            payload["idBlock"] = params["idblock"]

        if params["idpixeltype"] is not None:
            payload["idPixelType"] = params["idpixeltype"]

        if params["indicatorTypeIds"] is not None:
            payload["indicatorTypeIds"] = params["indicatorTypeIds"]

        payload["startDate"] = params["start_date"]

        if params["end_date"]:
            payload["endDate"] = params["end_date"]

        payload["fillYearGaps"] = params["fillyeargap"]

        log.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Step 4: Request data
        log.info(f"Requesting regional data for entity {entity_id}")
        log.debug(f"API URL: {url}")
        log.debug(f"Payload: {payload}")
        try:
            response_regional = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,  # type: ignore[arg-type]
                timeout=60,
            )
            response_regional.raise_for_status()
            json_response = response_regional.json()
            log.success(f"✅ Regional data retrieved for entity {entity_id}")
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting regional data for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_regional_ts_by_id_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_regional_ts_by_id().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain the mapped 'amu_id' column (configurable via column_mapping).

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "seasonfield_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "amu_id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response_json = self.get_regional_ts_by_id(entity_data)
            log.debug(f"Successful safe API call for entity {entity_id}")
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}

    def format_regional_json(self, response_json, entity_id="unknown", base_year=1900):
        """
        Process API time series response to extract daily and average data.

        Args:
            response_json (str | dict): API response containing time series data
            entity_id (str): Entity ID for logging purposes
            base_year (int): Base year for converting dayOfYear codes (default 1900)

        Returns:
            tuple: (observed_df, daily_avg_df) - Two DataFrames with formatted data
        """
        log = self.get_contextualized_logger("FORMAT")

        # Step 1: Parse JSON if needed
        if isinstance(response_json, str):
            log.debug(f"Parsing string JSON for entity {entity_id}")
            data = json.loads(response_json)
        elif isinstance(response_json, dict):
            data = response_json
            log.debug(f"Processing dict response for entity {entity_id}")
        else:
            error_msg = f"Unsupported response type: {type(response_json)}"
            log.error(error_msg)
            raise TypeError(error_msg)

        # Step 2: Extract dailyAverage data
        daily_avg_records = []
        if "dailyAverage" in data and data["dailyAverage"]:
            log.debug(f"Processing {len(data['dailyAverage'])} daily average records")
            for record in data["dailyAverage"]:
                day_code = record["dayOfYear"]
                date_obj = decode_day_of_year(day_code, base_year)

                daily_avg_records.append(
                    {
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "date_obj": date_obj,
                        "day_of_year": day_code,
                        "month": date_obj.month,
                        "day": date_obj.day,
                        "value": record["value"],
                    }
                )

        daily_avg_df = pd.DataFrame(daily_avg_records)

        # Step 3: Extract observedMeasures data
        observed_records = []
        if "observedMeasures" in data and data["observedMeasures"]:
            log.debug(f"Processing {len(data['observedMeasures'])} observed measure records")
            for record in data["observedMeasures"]:
                # Parse ISO datetime
                iso_date = record["time"]
                parsed_date = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))

                observed_records.append(
                    {
                        "date": parsed_date.strftime("%Y-%m-%d"),
                        "date_obj": parsed_date,
                        "day_id": record["dayId"],
                        "indicator_type_id": record["indicatorTypeId"],
                        "value": record["value"],
                    }
                )

        observed_df = pd.DataFrame(observed_records)

        # Step 4: Sort by date and drop helper columns
        if not daily_avg_df.empty:
            daily_avg_df = daily_avg_df.sort_values("date_obj").reset_index(drop=True)
            daily_avg_df = daily_avg_df.drop("date_obj", axis=1)

        if not observed_df.empty:
            observed_df = observed_df.sort_values("date_obj").reset_index(drop=True)
            observed_df = observed_df.drop("date_obj", axis=1)

        log.success(
            f"✅ Formatted {len(observed_df)} observed values and {len(daily_avg_df)} daily averages for entity {entity_id}"
        )

        return observed_df, daily_avg_df

    @requires_token
    @require_regional_params
    @cache_single_entity("regional_params")
    def process_single_entity_regional(self, row, params=None):
        """
        Process regional extraction for a single region with retry logic.

        Args:
            row (dict | pd.Series): Entity data containing the mapped 'amu_id' column
            params (dict, optional): regional parameters

        Returns:
            dict: {
                "observed_data": DataFrame or None,    # Time series observations
                "daily_avg_data": DataFrame or None,   # Daily averages (climatology)
                "error": dict or None
            }
        """
        if params is None:
            params = self.regional_params

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

        # Add start_date and end_date from params if not in row
        mapped_start = self.get_mapped_column("start_date")
        mapped_end = self.get_mapped_column("end_date")
        if mapped_start not in row and "start_date" in params:
            row[mapped_start] = params["start_date"]
        if mapped_end not in row and "end_date" in params:
            row[mapped_end] = params["end_date"]

        entity_id = self.get_entity_value(row, "amu_id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Step 2: define API call wrapper
        def _call_api():
            return self.get_regional_ts_by_id(row)

        # Step 3: call API with retry
        try:
            log.info(f"Calling regional API for entity {entity_id} (with retry logic)")
            raw_json = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Format returns TWO dataframes: observed time series + daily averages
            observed_df, daily_avg_df = self.format_regional_json(raw_json, entity_id=entity_id)

            # Step 4: Check if we got any data
            has_observed = observed_df is not None and not observed_df.empty
            has_daily_avg = daily_avg_df is not None and not daily_avg_df.empty

            if not has_observed and not has_daily_avg:
                log.warning(f"No data found for entity {entity_id}")
                return {
                    "observed_data": None,
                    "daily_avg_data": None,
                    "error": {"message": "No data found", "entity_id": entity_id},
                }

            # Step 5: Normalize each dataset with metadata
            observed_normalized = None
            if has_observed:
                observed_normalized = normalize_with_metadata(row, observed_df)
                log.debug(f"Normalized {len(observed_normalized)} observed records")

            daily_avg_normalized = None
            if has_daily_avg:
                daily_avg_normalized = normalize_with_metadata(row, daily_avg_df)
                log.debug(f"Normalized {len(daily_avg_normalized)} daily average records")

            # Step 6: Success logging
            log.success(
                f"✅ Successfully processed entity {entity_id}: "
                f"{len(observed_df) if has_observed else 0} observations, "
                f"{len(daily_avg_df) if has_daily_avg else 0} daily averages"
            )

            return {"observed_data": observed_normalized, "daily_avg_data": daily_avg_normalized, "error": None}

        except Exception as e:
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"observed_data": None, "daily_avg_data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_regional_params
    def process_entity_regional_bulk_parallel(
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
        prefix="regional",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity regional data using threads + progress bar,
        with optional fail-safe retry and filter capabilities.
        Returns TWO normalized dataframes: observed time series and daily averages (climatology).

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain the mapped 'amu_id' column.
            params (dict, optional): Regional parameters override.
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
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "regional".
            use_cache (bool, optional): If True, use caching for bulk extraction. Default: None (uses instance setting).

        Returns:
            dict: Contains TWO results DataFrames (observed_data and daily_avg_data), errors list, and summary statistics.
        """
        if use_cache or (use_cache is None and getattr(self, "_cache_enabled", False)):
            return self._bulk_with_cache(
                bulk_method=self._process_entity_regional_bulk_parallel_inner,
                entity_list=entity_list,
                params=self.regional_params,
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
        return self._process_entity_regional_bulk_parallel_inner(
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
    @require_regional_params
    def _process_entity_regional_bulk_parallel_inner(
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
        prefix="regional",
        generate_report=False,
        report_options=None,
    ):
        """Inner implementation of process_entity_regional_bulk_parallel."""
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk regional extraction: {prefix}")
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

        # Separate tracking for both dataframes
        all_observed_rows = []
        all_daily_avg_rows = []
        global_errors = []
        successful_calculations = 0
        total_calculations = 0
        buffer_observed_rows = []
        buffer_daily_avg_rows = []
        buffer_errors = []
        failed_ids = []

        log.info(f"Processing {len(filtered_entity_list)} entities in parallel...")
        print(f"🔄 Processing {prefix.title()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            amu_col = self.get_mapped_column("amu_id")
            future_to_id = {
                executor.submit(self.process_single_entity_regional, row.to_dict(), params): row.get(amu_col)
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🔍 Processing {prefix.title()}", unit="entity") as pbar:
                for future in as_completed(future_to_id):
                    total_calculations += 1
                    entity_id = future_to_id[future]

                    try:
                        result = future.result()
                        log.debug(f"Entity {entity_id} - Got result from future")

                        # Validate result structure
                        if result is None:
                            log.error(f"Entity {entity_id} - Result is None!")
                            failed_ids.append(entity_id)
                            continue

                        if not isinstance(result, dict):
                            log.error(f"Entity {entity_id} - Result is not a dict: {type(result)}")
                            failed_ids.append(entity_id)
                            continue

                        log.debug(f"Entity {entity_id} - Result keys: {list(result.keys())}")

                        observed_df = result.get("observed_data")
                        daily_avg_df = result.get("daily_avg_data")
                        error = result.get("error")

                        # Detailed logging of what we got
                        if observed_df is not None:
                            log.debug(
                                f"Entity {entity_id} - observed_df type: {type(observed_df)}, shape: {getattr(observed_df, 'shape', 'N/A')}, empty: {getattr(observed_df, 'empty', 'N/A')}"
                            )
                        else:
                            log.debug(f"Entity {entity_id} - observed_df is None")

                        if daily_avg_df is not None:
                            log.debug(
                                f"Entity {entity_id} - daily_avg_df type: {type(daily_avg_df)}, shape: {getattr(daily_avg_df, 'shape', 'N/A')}, empty: {getattr(daily_avg_df, 'empty', 'N/A')}"
                            )
                        else:
                            log.debug(f"Entity {entity_id} - daily_avg_df is None")

                        # Track success if either dataframe has data
                        has_observed = observed_df is not None and not observed_df.empty
                        has_daily_avg = daily_avg_df is not None and not daily_avg_df.empty

                        log.debug(f"Entity {entity_id} - has_observed: {has_observed}, has_daily_avg: {has_daily_avg}")

                        if has_observed or has_daily_avg:
                            successful_calculations += 1
                            log.debug(
                                f"Entity {entity_id} - Incrementing successful_calculations to {successful_calculations}"
                            )

                            if has_observed:
                                all_observed_rows.append(observed_df)
                                buffer_observed_rows.append(observed_df)
                                log.debug(
                                    f"Entity {entity_id} - Added {len(observed_df)} observed rows. Total observed lists: {len(all_observed_rows)}"
                                )

                            if has_daily_avg:
                                all_daily_avg_rows.append(daily_avg_df)
                                buffer_daily_avg_rows.append(daily_avg_df)
                                log.debug(
                                    f"Entity {entity_id} - Added {len(daily_avg_df)} daily avg rows. Total daily avg lists: {len(all_daily_avg_rows)}"
                                )

                            log.info(
                                f"Entity {entity_id} - ✅ Success (Observed: {has_observed}, Daily Avg: {has_daily_avg})"
                            )
                        else:
                            failed_ids.append(entity_id)
                            log.warning(
                                f"Entity {entity_id} - ⚠️ No data returned (observed exists: {observed_df is not None}, daily_avg exists: {daily_avg_df is not None})"
                            )

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

                        # Export both dataframes separately
                        partial_observed_df = (
                            pd.concat(buffer_observed_rows, ignore_index=True)
                            if buffer_observed_rows
                            else pd.DataFrame()
                        )
                        partial_daily_avg_df = (
                            pd.concat(buffer_daily_avg_rows, ignore_index=True)
                            if buffer_daily_avg_rows
                            else pd.DataFrame()
                        )

                        # Save observed data
                        if not partial_observed_df.empty:
                            export_results(
                                results_df=partial_observed_df,
                                errors=buffer_errors,
                                output_path=self.partial_path,
                                prefix=f"{prefix}_observed",
                                partial=True,
                                verbose=False,
                            )

                        # Save daily average data
                        if not partial_daily_avg_df.empty:
                            export_results(
                                results_df=partial_daily_avg_df,
                                errors=[],  # Only save errors once with observed data
                                output_path=self.partial_path,
                                prefix=f"{prefix}_daily_avg",
                                partial=True,
                                verbose=False,
                            )

                        log.info(
                            f"💾 Partial results saved: {len(buffer_observed_rows)} observed, {len(buffer_daily_avg_rows)} daily avg rows"
                        )
                        buffer_observed_rows.clear()
                        buffer_daily_avg_rows.clear()
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
        observed_results_df = pd.concat(all_observed_rows, ignore_index=True) if all_observed_rows else pd.DataFrame()
        daily_avg_results_df = (
            pd.concat(all_daily_avg_rows, ignore_index=True) if all_daily_avg_rows else pd.DataFrame()
        )

        log.info(f"Observed data results shape: {observed_results_df.shape}")
        log.info(f"Daily average data results shape: {daily_avg_results_df.shape}")

        # Merge back skipped entities to maintain all input rows in output (for observed data)
        log.debug("Merging with skipped entities...")
        observed_results_df = self._merge_with_skipped_entities(
            new_results_df=observed_results_df,
            skipped_entities_df=skipped_entities,
            merge_mode=merge_existing,
            verbose=True,
        )

        # Finalize extraction: export both dataframes, store failed IDs, cleanup partials
        if not skip_export:
            log.info("Finalizing extraction...")

            # Export observed data
            if not observed_results_df.empty:
                observed_results_df, finalization_status_observed = self._finalize_extraction(
                    results_df=observed_results_df,
                    global_errors=global_errors,
                    failed_ids=failed_ids,
                    output_path=output_path,
                    prefix=f"{prefix}_observed",
                    skip_export=False,
                    verbose=True,
                    generate_report=generate_report,
                    report_options=report_options,
                    extraction_stats={
                        "total_entities": len(entity_list),
                        "total_calculations": total_calculations,
                        "successful": successful_calculations,
                        "failed": total_calculations - successful_calculations,
                        "elapsed_seconds": elapsed_time,
                        "parameters": getattr(self, "regional_params", {}),
                        "entity_df": entity_list,
                    },
                )
                log.info(f"Observed data finalization status: {finalization_status_observed}")

            # Export daily average data
            if not daily_avg_results_df.empty:
                daily_avg_results_df, finalization_status_daily_avg = self._finalize_extraction(
                    results_df=daily_avg_results_df,
                    global_errors=[],  # Only save errors once with observed data
                    failed_ids=[],
                    output_path=output_path,
                    prefix=f"{prefix}_daily_avg",
                    skip_export=False,
                    verbose=True,
                    generate_report=generate_report,
                    report_options=report_options,
                    extraction_stats={
                        "total_entities": len(entity_list),
                        "total_calculations": total_calculations,
                        "successful": successful_calculations,
                        "failed": total_calculations - successful_calculations,
                        "elapsed_seconds": elapsed_time,
                        "parameters": getattr(self, "regional_params", {}),
                        "entity_df": entity_list,
                    },
                )
                log.info(f"Daily average data finalization status: {finalization_status_daily_avg}")
        else:
            log.info("Skipping export as requested")

        log.success("🎉 Bulk regional extraction completed!")

        return {
            "results_df": observed_results_df,  # primary frame, per the standard workflow convention
            "observed_results_df": observed_results_df,  # back-compat alias for existing readers
            "daily_avg_results_df": daily_avg_results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
