import math
import os
import random
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

# Supported result-export formats. Errors are always written as CSV (tiny,
# human-read) regardless of this setting.
SUPPORTED_EXPORT_FORMATS = ("csv", "parquet")


def resolve_export_format(export_format=None):
    """
    Resolve the results export format.

    Precedence: explicit ``export_format`` arg > ``EDAGRO_EXPORT_FORMAT`` env var
    > default ``"csv"``. Unknown values fall back to ``"csv"`` with a warning so a
    typo never aborts an extraction.

    Returns one of :data:`SUPPORTED_EXPORT_FORMATS`.
    """
    fmt = (export_format or os.getenv("EDAGRO_EXPORT_FORMAT", "csv")).lower()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        # Plain ASCII — this is a defensive fallback path and must never itself
        # raise (e.g. UnicodeEncodeError on a cp1252 console).
        tqdm.write(
            f"Warning: unknown export format '{fmt}' - falling back to 'csv'. Supported: {SUPPORTED_EXPORT_FORMATS}"
        )
        fmt = "csv"
    return fmt


def normalize_date(date_input, field_name="date"):
    """
    Normalize date to YYYY-MM-DD format.

    Handles multiple input formats:
    - 'YYYY-MM-DD' (ISO date string)
    - 'YYYY-MM-DDTHH:MM:SS' (ISO datetime string)
    - pandas Timestamp objects
    - datetime objects

    Args:
        date_input: Date in various formats
        field_name (str): Name of the field (for error messages)

    Returns:
        str: Date in YYYY-MM-DD format

    Raises:
        ValueError: If date format is invalid
    """
    if date_input is None:
        return None

    # Convert to string if it's not already
    if not isinstance(date_input, str):
        date_input = str(date_input)

    # Remove any extra whitespace
    date_input = date_input.strip()

    # Parse and normalize date (handles both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats)
    try:
        # Try parsing with time component first (more specific)
        if "T" in date_input:
            date_obj = datetime.strptime(date_input, "%Y-%m-%dT%H:%M:%S")
        else:
            date_obj = datetime.strptime(date_input, "%Y-%m-%d")

        # Return normalized YYYY-MM-DD format
        return date_obj.strftime("%Y-%m-%d")

    except ValueError:
        raise ValueError(f"❌ Invalid {field_name} format '{date_input}'. Expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")


def decode_day_of_year(day_code, base_year=1900):
    """
    Convert dayOfYear code (1MMDD format) to datetime.

    Args:
        day_code (int): Day code in 1MMDD format (e.g., 10106 = Jan 6, 10201 = Feb 1)
        base_year (int): Base year for date conversion (default 1900)

    Returns:
        datetime: Parsed date
    """
    day_str = str(day_code)

    if len(day_str) != 5:
        raise ValueError(f"Invalid day_code format: {day_code}. Expected 5 digits (1MMDD)")

    # Skip first digit (prefix), parse month and day
    month = int(day_str[1:3])  # Digits 2-3 = month
    day = int(day_str[3:5])  # Digits 4-5 = day

    # Handle leap year edge case - Feb 29
    # If base_year is not a leap year but we have Feb 29, use Feb 28
    if month == 2 and day == 29:
        is_leap = (base_year % 4 == 0 and base_year % 100 != 0) or (base_year % 400 == 0)
        if not is_leap:
            day = 28  # Use Feb 28 for non-leap years

    try:
        return datetime(base_year, month, day)
    except ValueError as e:
        raise ValueError(
            f"Invalid date from day_code {day_code}: month={month}, day={day}, year={base_year}. Error: {e}"
        )


def filter_entities(entity_list, filter_column=None, filter_value=None, filter_type="exclude"):
    """
    Filter entities by including or excluding rows based on a column value.
    Handles empty values (None, NaN, empty strings, whitespace).

    Args:
        entity_list (pd.DataFrame): DataFrame of entities to filter.
        filter_column (str, optional): Column name to apply filter on.
        filter_value (any, optional): Value to filter by. Use "" or "empty" to filter empty/null values.
        filter_type (str, optional): 'exclude' to skip rows with filter_value,
                                     'include' to process only rows with filter_value.
                                     Defaults to 'exclude'.

    Returns:
        tuple: (filtered_df, skipped_df, skip_count)
            - filtered_df: DataFrame with rows to process
            - skipped_df: DataFrame with skipped rows
            - skip_count: Number of rows skipped
    """
    if filter_column is None or filter_value is None:
        # No filtering applied
        return entity_list, pd.DataFrame(), 0

    if filter_column not in entity_list.columns:
        print(f"⚠️ Warning: Filter column '{filter_column}' not found in entity_list. No filtering applied.")
        return entity_list, pd.DataFrame(), 0

    if filter_type not in ["exclude", "include"]:
        print(f"⚠️ Warning: Invalid filter_type '{filter_type}'. Must be 'exclude' or 'include'. No filtering applied.")
        return entity_list, pd.DataFrame(), 0

    # Check if filtering for empty/null values
    is_empty_filter = filter_value == "" or filter_value == "empty"

    if is_empty_filter:
        # Create mask for empty/null values (handles NaN, None, empty strings, whitespace)
        mask = (
            entity_list[filter_column].isna()  # NaN or None
            | (entity_list[filter_column] == "")  # Empty string
            | (entity_list[filter_column].astype(str).str.strip() == "")  # Whitespace only
        )
        filter_description = "empty/null"
    else:
        # Create mask for specific value
        mask = entity_list[filter_column] == filter_value
        filter_description = f"'{filter_value}'"

    # Apply filter based on type
    if filter_type == "exclude":
        # Skip rows where condition is True
        skipped_df = entity_list[mask].copy()
        filtered_df = entity_list[~mask].copy()
        action = "Excluding"
    else:  # filter_type == 'include'
        # Process only rows where condition is True
        filtered_df = entity_list[mask].copy()
        skipped_df = entity_list[~mask].copy()
        action = "Including only"

    skip_count = len(skipped_df)
    process_count = len(filtered_df)

    if skip_count > 0:
        print(
            f"🔍 Filter applied ({filter_type}): {action} {process_count} entities where {filter_column}={filter_description} ({skip_count} skipped)"
        )

    return filtered_df, skipped_df, skip_count


def normalize_with_metadata(row, df, error=None, include_geometry=True):
    """
    Normalize API results by attaching entity metadata.

    Args:
        row (pd.Series | dict): Input entity row.
        df (pd.DataFrame | None): Results DataFrame (may be None or empty).
        error (dict | None): Optional error details to tag.
        include_geometry (bool): If True, includes geometry in metadata. Default is True.
            Callers normally leave this as-is and drop geometry downstream via
            BaseExtractor.exclude_columns (defaults to ["geometry"]).

    Returns:
        pd.DataFrame: Always returns a DataFrame with metadata attached.
    """
    # Attach metadata (optionally exclude geometry)
    if include_geometry:
        metadata = {k: row[k] for k in row.keys()}
    else:
        metadata = {k: row[k] for k in row.keys() if k != "geometry"}

    # Convert list values to comma-separated strings
    for key, val in metadata.items():
        if isinstance(val, list):
            metadata[key] = ",".join(map(str, val))

    if df is None or df.empty:
        # Return just metadata with optional error message
        error_row = metadata.copy()
        if error:
            error_row.update(error)
        return pd.DataFrame([error_row])

    # Get extraction columns (columns that came from API processing)
    extraction_columns = set(df.columns)

    # Filter metadata to exclude extraction columns
    # This prevents overwriting newly extracted data with old NaN values
    metadata_only = {col: val for col, val in metadata.items() if col not in extraction_columns}

    # Attach filtered metadata to extraction results
    for col, val in metadata_only.items():
        df[col] = val

    # Add error info if provided
    if error:
        for k, v in error.items():
            df[k] = v

    return df


def safe_parse_date(date_str):
    """
    Convert a string date to ISO format (YYYY-MM-DD), or return None if invalid or placeholder.
    """
    if not date_str or date_str in ["0001-01-01", "0000-00-00"]:
        return None
    try:
        # Try parsing ISO-like formats
        return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    except ValueError:
        # Try fallback MM-DD format (as in historical averages). Pin a leap year
        # (2000) so 02-29 validates and behaviour is stable across Python versions:
        # a bare "%m-%d" defaults to year 1900 (non-leap, so 02-29 would raise) and
        # is deprecated from Python 3.15.
        try:
            parsed = datetime.strptime(f"2000-{date_str}", "%Y-%m-%d")
            return parsed.strftime("%m-%d")
        except Exception:
            return date_str  # leave as-is if unrecognized


def retry_with_backoff_no_retry_on_400(func, max_retries=5, base_delay=1.0, max_delay=60.0, *args, **kwargs):
    """
    Retries a function call with exponential backoff and jitter.
    Skips retry if a 400-series client error is returned.

    Args:
        func (callable): The function to call.
        max_retries (int): Max retry attempts before failing.
        base_delay (float): Base delay (in seconds) for exponential backoff.
        max_delay (float): Maximum backoff delay in seconds.
        *args, **kwargs: Arguments passed to func.

    Returns:
        Any: The result of func(*args, **kwargs).

    Raises:
        ValueError: For 400-series client errors (non-retriable).
        RuntimeError: If all retries fail.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except requests.exceptions.HTTPError as http_err:
            # `is not None`, never a truthiness test: requests.Response.__bool__ is
            # self.ok, so every 4xx/5xx response object is FALSY. `if http_err.response`
            # therefore yielded status_code=None for precisely the client errors this
            # function exists to short-circuit, and a 403 was retried 5x with backoff
            # (~83s) while logging "HTTP None". Same trap as truthiness on a pd.Series.
            status_code = http_err.response.status_code if http_err.response is not None else None
            if status_code is not None and 400 <= status_code < 500:
                # Non-retriable client-side error. The body carries the only
                # actionable detail (e.g. which entitlement a 403 is missing);
                # `str(http_err)` gives status + URL and drops it.
                body = http_err.response.text[:500] if http_err.response is not None else ""
                print(f"❌ HTTP {status_code} error (non-retriable): {http_err}")
                if body:
                    print(f"   Response body: {body}")
                raise ValueError(f"HTTP {status_code}: {http_err}{f' | {body}' if body else ''}")
            else:
                wait = min(base_delay * (2**attempt) + random.uniform(0, base_delay), max_delay)
                print(f"⚠️ HTTP {status_code} → Retry {attempt + 1}/{max_retries} in {wait:.1f}s...")
                time.sleep(wait)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
            wait = min(base_delay * (2**attempt) + random.uniform(0, base_delay), max_delay)
            print(f"🌐 Network issue: {conn_err} → Retry {attempt + 1}/{max_retries} in {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            wait = min(base_delay * (2**attempt) + random.uniform(0, base_delay), max_delay)
            print(f"⚠️ Retry {attempt + 1}/{max_retries} failed: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Function {func.__name__} failed after {max_retries} retries.")


def export_results(results_df, errors, output_path, prefix="coverage", partial=False, verbose=True, export_format=None):
    """
    Export results to CSV or Parquet, and errors to CSV.

    ``output_path`` may be a local directory (existing behaviour) or a remote
    URI like ``s3://bucket/prefix``. Remote URIs route through fsspec via
    pandas' ``storage_options`` argument; credentials flow through the
    standard chain (env vars / IAM role / ``AWS_ENDPOINT_URL`` for MinIO).

    The results file format is selected by ``export_format`` (or the
    ``EDAGRO_EXPORT_FORMAT`` env var, default ``"csv"``). Errors are always
    written as CSV — they are tiny and meant to be eyeballed. Parquet requires
    ``pyarrow`` (already a dependency, used by the extraction cache).

    Args:
        results_df (pd.DataFrame): Successful results.
        errors (list[dict]): Error logs.
        output_path (str): Base output folder. Local path or remote URI.
        prefix (str): Prefix for filenames (e.g., "coverage", "inseason").
        partial (bool): If True, mark file as partial.
        verbose (bool): If True, print messages. Suppress for partials.
        export_format (str, optional): "csv" or "parquet". When None, falls back
            to ``EDAGRO_EXPORT_FORMAT`` then "csv". Inline partial flushes that
            omit this arg therefore still honour the env var automatically.
    """
    from earthdaily.agriculture.core._fs import ensure_dir, join_path, storage_options_for

    if results_df is None or results_df.empty:
        if verbose:
            tqdm.write("⚠️ No results to export.")
        return None, None

    # Local path → makedirs; remote URI → no-op (object stores have no dirs).
    ensure_dir(output_path)
    storage_options = storage_options_for(output_path)

    fmt = resolve_export_format(export_format)
    ext = "parquet" if fmt == "parquet" else "csv"

    # Microseconds suffix prevents collisions when partials are flushed faster
    # than one per second (test loops, fast extractors, or future-parallel writes).
    # Partial-cleanup globs match `{prefix}_*_partial.{csv,parquet}` so the longer
    # timestamp and either extension are both handled.
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "partial" if partial else "final"

    results_path = join_path(output_path, f"{prefix}_results_{timestamp}_{suffix}.{ext}")
    if ext == "parquet":
        try:
            results_df.to_parquet(results_path, index=False, storage_options=storage_options)
        except ImportError as exc:
            raise ImportError(
                "Parquet export requires 'pyarrow'. Install it (`pip install pyarrow`) "
                "or set EDAGRO_EXPORT_FORMAT=csv / export_format='csv'."
            ) from exc
    else:
        results_df.to_csv(results_path, index=False, storage_options=storage_options)
    if verbose:
        tqdm.write(f"📄 Results saved to: {results_path}")

    errors_path = None
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = join_path(output_path, f"{prefix}_errors_{timestamp}_{suffix}.csv")
        errors_df.to_csv(errors_path, index=False, storage_options=storage_options)
        if verbose:
            tqdm.write(f"📄 Errors saved to: {errors_path}")

    return results_path, errors_path


def validate_historical_years(years):
    """
    Validate and normalize a historical years parameter.

    Handles all the formats that flow through the pipeline:
    - None → None
    - "ALL" → "ALL"
    - [2024, 2023, 2022] → validated list
    - "2024,2023,2022" → parsed to [2024, 2023, 2022] (normalize_with_metadata flattening)
    - int (e.g. 5) → returned as-is (N-year lookback, not a year list)

    Used by: VTS, MRTS, LRTS, Weather, Disease (as 'years' / 'historical_years'),
             Score, Coverage (as 'historical_seasons').

    Args:
        years: None, "ALL", int, list[int], or comma-separated string.

    Returns:
        None, "ALL", int, or list[int]

    Raises:
        ValueError: If the value cannot be interpreted as a valid years specification.
    """
    if years is None:
        return None

    if isinstance(years, int):
        return years

    if isinstance(years, str):
        if years.upper() == "ALL":
            return "ALL"
        # Parse comma-separated string
        try:
            years = [int(y) for y in years.split(",") if y.strip()]
        except ValueError:
            raise ValueError(
                f"Cannot parse historical years string '{years}'. "
                f"Expected 'ALL' or comma-separated integers (e.g. '2024,2023,2022')."
            )

    if not isinstance(years, list):
        raise ValueError(
            f"Invalid historical years type '{type(years).__name__}'. Must be None, 'ALL', int, or list of ints."
        )

    if len(years) == 0:
        raise ValueError("Historical years list must not be empty.")

    current_year = datetime.now().year
    for yr in years:
        if not isinstance(yr, int):
            raise ValueError(f"Invalid year '{yr}'. All years must be integers.")
        if not (1900 <= yr <= current_year):
            raise ValueError(f"Invalid year {yr}. Must be between 1900 and {current_year}.")

    if len(years) != len(set(years)):
        raise ValueError(f"Duplicate years found: {years}. Each year must be unique.")

    return years


def parse_matching_seasons(value):
    """
    Normalize a per-field "matching seasons" input into a ``set[int]`` of calendar years.

    Used by the HISTORICAL branches of the Emergence and Harvest processors, where a field
    supplies the calendar years it actually grew the target crop so off-years can be dropped
    from ``<event>_year_N``. Delegates string/list parsing and year validation to
    :func:`validate_historical_years` (for parity with the Score extractors), with
    empty-vs-absent semantics layered on top:

      - ``None`` / NaN (field missing the mapped column) -> ``None``  (no filtering)
      - empty string / empty list                        -> ``set()`` (keep none)
      - ``"2020,2022,2024"`` or ``[2020, 2022, 2024]``    -> ``{2020, 2022, 2024}``

    Args:
        value: None, NaN, comma-separated string, or list of ints.

    Returns:
        set[int] | None: matching-season years, ``set()`` for an explicit empty input, or
        ``None`` when no season set was supplied (no filtering).

    Raises:
        ValueError: on non-integer tokens, or an int / "ALL" spec (a matching-season set must
            be an explicit list of years, not an N-year lookback).
    """
    # Absent mapped column: get_entity_value returns None, or scalar NaN from a pandas row.
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    # Explicitly empty -> keep none (validate_historical_years rejects empty inputs).
    if isinstance(value, str):
        if not [tok for tok in value.split(",") if tok.strip()]:
            return set()
    elif isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return set()
        value = list(value)

    validated = validate_historical_years(value)
    if isinstance(validated, int) or validated == "ALL":
        raise ValueError(
            f"❌ Invalid matching seasons {value!r}: expected a list of calendar years "
            "or a comma-separated string (e.g. '2020,2022,2024'), not an N-year lookback."
        )
    return set(validated)


def _month_day(date_str):
    """Extract ``(month, day)`` from an ISO ``"YYYY-MM-DD"`` or bare ``"MM-DD"`` string."""
    parts = str(date_str).split("-")
    try:
        if len(parts) == 3:  # YYYY-MM-DD
            return int(parts[1]), int(parts[2])
        if len(parts) == 2:  # MM-DD
            return int(parts[0]), int(parts[1])
    except ValueError:
        pass
    return None, None


def average_dates_mmdd(dates):
    """
    Average a set of dates year-agnostically and return an ``"MM-DD"`` string.

    Each date's month/day is re-based onto a fixed non-leap sentinel year (1900) so leap
    years do not shift the day-of-year; the day-of-year values are then averaged, rounded to
    a whole day, and reformatted as ``%m-%d``. Averaging the day-of-year (rather than raw
    epoch nanoseconds) keeps the sentinel-year arithmetic correct.

    Args:
        dates: iterable of date strings ("YYYY-MM-DD" or "MM-DD") and/or None.

    Returns:
        str | None: mean date as ``"MM-DD"``, or ``None`` if there are no non-null dates.
    """
    sentinel_year = 1900  # non-leap
    day_of_years = []
    for d in dates:
        if not d:
            continue
        month, day = _month_day(d)
        if month is None:
            continue
        try:
            sentinel = date(sentinel_year, month, day)
        except ValueError:
            # Feb 29 has no equivalent in a non-leap year — clamp to Feb 28.
            sentinel = date(sentinel_year, month, min(day, 28))
        day_of_years.append(sentinel.timetuple().tm_yday)

    if not day_of_years:
        return None

    avg_doy = round(sum(day_of_years) / len(day_of_years))
    mean_date = date(sentinel_year, 1, 1) + timedelta(days=avg_doy - 1)
    return mean_date.strftime("%m-%d")


def autosize_row_limit(start_date, end_date, *, floor=1000, buffer=366):
    """Size a daily-timeseries row limit to cover an inclusive ``[start_date, end_date]`` span.

    Daily endpoints (e.g. the Weather API) return one row per day in ascending date order and
    cap the response at ``$limit`` rows — a range longer than the limit is silently truncated
    to the *earliest* N days. A fixed limit therefore drops the most recent data on long
    (multi-year) requests. This computes a limit that covers the whole span::

        days  = (end_date - start_date).days + 1        # inclusive
        limit = max(floor, days + buffer)

    The ``floor`` keeps short-range calls (a 10-day forecast, a single season) at the historic
    default so their requests stay byte-identical. The ``buffer`` absorbs inclusive-boundary
    and timezone edges.

    Args:
        start_date: ``"YYYY-MM-DD"`` string (or ``date`` / ``datetime``).
        end_date: ``"YYYY-MM-DD"`` string (or ``date`` / ``datetime``).
        floor: Minimum limit (default ``1000`` — the historic hardcoded value).
        buffer: Extra days added on top of the span (default ``366``).

    Returns:
        int: The row limit to request.
    """

    def _to_date(value):
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()

    span_days = (_to_date(end_date) - _to_date(start_date)).days + 1
    if span_days < 1:
        span_days = 1
    return max(floor, span_days + buffer)


# Single source of truth for KPI aggregation contracts.
# - threshold: None | "number" | "positive_int" | "tuple_min_max" | "percentile_rank"
# - window:    None | "positive_int"
# Adding a new aggregation = one row here + a branch in `compute_kpi`.
KPI_AGGREGATION_RULES = {
    "accumulation": {"threshold": None, "window": None},
    "top_accumulation": {"threshold": "positive_int", "window": None},
    "average": {"threshold": None, "window": None},
    "max": {"threshold": None, "window": None},
    "min": {"threshold": None, "window": None},
    "std": {"threshold": None, "window": None},
    "count_gt": {"threshold": "number", "window": None},
    "count_lt": {"threshold": "number", "window": None},
    "count_between": {"threshold": "tuple_min_max", "window": None},
    "rolling_avg": {"threshold": None, "window": "positive_int"},
    "rolling_avg_gt": {"threshold": "number", "window": "positive_int"},
    "rolling_avg_lt": {"threshold": "number", "window": "positive_int"},
    "percentile": {"threshold": "percentile_rank", "window": None},
    "percentile_gt": {"threshold": "percentile_rank", "window": None},
    "percentile_lt": {"threshold": "percentile_rank", "window": None},
}


def _check_threshold(threshold, spec, agg):
    if spec == "number":
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"For {agg} aggregation, threshold must be a single numeric value (int or float)")
    elif spec == "positive_int":
        if not isinstance(threshold, int) or threshold < 1:
            raise ValueError(f"For {agg} aggregation, threshold must be a positive integer (number of top days)")
    elif spec == "tuple_min_max":
        if not isinstance(threshold, (tuple, list)) or len(threshold) != 2:
            raise ValueError(f"For {agg} aggregation, threshold must be a tuple/list of two values (min, max)")
        if not all(isinstance(t, (int, float)) for t in threshold):
            raise ValueError("Threshold values must be numeric (int or float)")
        if threshold[0] >= threshold[1]:
            raise ValueError(f"Threshold min ({threshold[0]}) must be less than max ({threshold[1]})")
    elif spec == "percentile_rank":
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 100):
            raise ValueError(f"For {agg} aggregation, threshold must be a percentile rank in 0–100")


def validate_kpi_filter(kpi_filter, logger=None):
    """
    Validate a kpi_filter dict ahead of an extractor run (fail-fast at setup time).

    Centralized so all KPI-supporting extractors share one rule set instead of each
    re-implementing the same checks. The contracts live in `KPI_AGGREGATION_RULES`.

    Args:
        kpi_filter (dict | None): The KPI configuration from a `setup_*_parameters` call.
            None is accepted as a no-op.
        logger: Optional loguru-style logger; errors are logged through it if provided.

    Raises:
        ValueError: when kpi_filter is malformed, the aggregation is unsupported,
            or threshold/window don't match the aggregation's contract.
    """
    if kpi_filter is None:
        return

    def _err(msg):
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg)

    if not isinstance(kpi_filter, dict):
        _err("kpi_filter must be a dictionary")
    if "aggregation" not in kpi_filter:
        _err("kpi_filter must include 'aggregation' field")

    agg = kpi_filter["aggregation"]
    if agg not in KPI_AGGREGATION_RULES:
        valid = sorted(KPI_AGGREGATION_RULES.keys())
        _err(f"Invalid aggregation '{agg}'. Must be one of {valid}")

    rule = KPI_AGGREGATION_RULES[agg]

    has_threshold = kpi_filter.get("threshold") is not None
    if rule["threshold"] is None:
        if has_threshold:
            _err(f"Aggregation '{agg}' does not require a 'threshold' parameter")
    else:
        if not has_threshold:
            _err(f"Aggregation '{agg}' requires a 'threshold' parameter")
        try:
            _check_threshold(kpi_filter["threshold"], rule["threshold"], agg)
        except ValueError as e:
            _err(str(e))

    has_window = kpi_filter.get("window") is not None
    if rule["window"] is None:
        if has_window:
            _err(f"Aggregation '{agg}' does not require a 'window' parameter")
    else:
        if not has_window:
            _err(
                f"Aggregation '{agg}' requires a 'window' parameter (positive integer — integration period in records)"
            )
        window = kpi_filter["window"]
        if not isinstance(window, int) or window < 1:
            _err(f"For {agg}, 'window' must be a positive integer (got {window!r})")

    if logger is not None:
        logger.info(f"KPI filter validated: {kpi_filter.get('kpi_name', 'unnamed')} using {agg}")


def filter_timeseries_kpi(
    timeseries_df,
    start_date,
    end_date,
    kpi_name,
    aggregation="accumulation",
    threshold=None,
    window=None,
    years=None,
    date_column="date",
    value_column="value",
):
    """
    Filter time series DataFrame and compute KPI for current period and historical average.
    Works with any time series data (vegetation, weather, radar indices, etc.)

    Args:
        timeseries_df (pd.DataFrame): DataFrame with date and value columns
        start_date (str): Start date for KPI period (YYYY-MM-DD)
        end_date (str): End date for KPI period (YYYY-MM-DD)
        kpi_name (str): Name of the KPI for labeling
        aggregation (str): Type of aggregation. Supported:
            - 'accumulation', 'average', 'max', 'min', 'std'
            - 'top_accumulation' (threshold = number of top days)
            - 'count_gt', 'count_lt' (threshold = value cutoff)
            - 'count_between' (threshold = (min, max))
            - 'rolling_avg' (window = integration period in records)
            - 'rolling_avg_gt', 'rolling_avg_lt' (window = integration period, threshold = value cutoff)
            - 'percentile' (threshold = percentile rank 0-100)
            - 'percentile_gt', 'percentile_lt' (threshold = percentile rank 0-100)
        threshold (float or tuple, optional): Cutoff value for count/rolling operations,
            (min, max) tuple for count_between, or percentile rank 0-100 for percentile_*.
        window (int, optional): Integration period (number of records) for rolling_avg* aggregations.
        years (list, str, or None, optional): Historical years for comparison:
            - None: No historical comparison (historical fields will be None)
            - "ALL": Use all available historical data in the timeframe
            - List of years: Use only specified years (e.g., [2024, 2023, 2022])
        date_column (str): Name of the date column (default: 'date')
        value_column (str): Name of the value column (default: 'value')

    Returns:
        dict: KPI results with current period and historical comparison
    """
    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    # Validate columns exist
    if date_column not in timeseries_df.columns:
        raise ValueError(f"DataFrame must contain '{date_column}' column")
    if value_column not in timeseries_df.columns:
        raise ValueError(f"DataFrame must contain '{value_column}' column")

    # Ensure date column is datetime
    timeseries_df = timeseries_df.copy()
    timeseries_df[date_column] = pd.to_datetime(timeseries_df[date_column])

    # Helper function to compute KPI
    def compute_kpi(values, agg_method):
        if not values or len(values) == 0:
            return None

        if agg_method == "accumulation":
            return round(sum(values), 4)
        elif agg_method == "average":
            return round(sum(values) / len(values), 4)
        elif agg_method == "max":
            return round(max(values), 4)
        elif agg_method == "min":
            return round(min(values), 4)
        elif agg_method == "std":
            if len(values) < 2:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return round(variance**0.5, 4)
        elif agg_method == "top_accumulation":
            if threshold is None or not isinstance(threshold, int) or threshold < 1:
                raise ValueError("threshold must be a positive integer (number of top days) for top_accumulation")
            top_n = min(threshold, len(values))
            top_values = sorted(values, reverse=True)[:top_n]
            return round(sum(top_values), 4)
        elif agg_method == "count_gt":
            if threshold is None:
                raise ValueError("threshold required for count_gt aggregation")
            return sum(1 for v in values if v > threshold)
        elif agg_method == "count_lt":
            if threshold is None:
                raise ValueError("threshold required for count_lt aggregation")
            return sum(1 for v in values if v < threshold)
        elif agg_method == "count_between":
            if not isinstance(threshold, (tuple, list)) or len(threshold) != 2:
                raise ValueError("threshold must be a tuple (min, max) for count_between")
            min_val, max_val = threshold
            return sum(1 for v in values if min_val <= v <= max_val)
        elif agg_method == "rolling_avg":
            if not isinstance(window, int) or window < 1:
                raise ValueError("window must be a positive integer for rolling_avg")
            rolling = pd.Series(values).rolling(window=window, min_periods=window).mean().dropna()
            return round(float(rolling.mean()), 4) if not rolling.empty else None
        elif agg_method in ("rolling_avg_gt", "rolling_avg_lt"):
            if not isinstance(window, int) or window < 1:
                raise ValueError(f"window must be a positive integer for {agg_method}")
            if threshold is None:
                raise ValueError(f"threshold required for {agg_method}")
            rolling = pd.Series(values).rolling(window=window, min_periods=window).mean().dropna()
            if agg_method == "rolling_avg_gt":
                return int((rolling > threshold).sum())
            return int((rolling < threshold).sum())
        elif agg_method == "percentile":
            if threshold is None or not (0 <= threshold <= 100):
                raise ValueError("threshold (percentile rank 0-100) required for percentile")
            return round(float(pd.Series(values).quantile(threshold / 100)), 4)
        elif agg_method in ("percentile_gt", "percentile_lt"):
            if threshold is None or not (0 <= threshold <= 100):
                raise ValueError(f"threshold (percentile rank 0-100) required for {agg_method}")
            cutoff = float(pd.Series(values).quantile(threshold / 100))
            if agg_method == "percentile_gt":
                return sum(1 for v in values if v > cutoff)
            return sum(1 for v in values if v < cutoff)
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

    # Extract current year
    current_year = end_dt.year
    month = end_dt.month
    day = end_dt.day

    # Get values for current period
    current_mask = (timeseries_df[date_column].dt.date >= start_dt) & (timeseries_df[date_column].dt.date <= end_dt)
    current_values = timeseries_df[current_mask][value_column].dropna().tolist()

    # Compute current KPI
    kpi_current = compute_kpi(current_values, aggregation)

    # Compute historical KPIs based on years parameter
    past_kpis = []
    years_included = []

    if years is None:
        # No historical comparison
        kpi_past_avg = None

    elif isinstance(years, str) and years.upper() == "ALL":
        # Use ALL available historical data within the same timeframe
        # Get all unique years in the data (excluding current year)
        all_years = timeseries_df[date_column].dt.year.unique()
        historical_years = [y for y in all_years if y < current_year]

        for y in sorted(historical_years):
            try:
                # Create corresponding date range for historical year
                hist_end = date(y, month, day)
                days_diff = (end_dt - start_dt).days
                hist_start = hist_end - timedelta(days=days_diff)

                # Filter data for this historical period
                hist_mask = (timeseries_df[date_column].dt.date >= hist_start) & (
                    timeseries_df[date_column].dt.date <= hist_end
                )
                hist_values = timeseries_df[hist_mask][value_column].dropna().tolist()

                if hist_values:
                    hist_kpi = compute_kpi(hist_values, aggregation)
                    if hist_kpi is not None:
                        past_kpis.append(hist_kpi)
                        years_included.append(y)

            except ValueError:
                # Handle invalid dates (e.g., Feb 29 on non-leap years)
                continue

        kpi_past_avg = round(sum(past_kpis) / len(past_kpis), 4) if past_kpis else None

    elif isinstance(years, list):
        # Use specific list of years
        for y in years:
            if y >= current_year:
                continue  # Skip current and future years

            try:
                # Create corresponding date range for historical year
                hist_end = date(y, month, day)
                days_diff = (end_dt - start_dt).days
                hist_start = hist_end - timedelta(days=days_diff)

                # Filter data for this historical period
                hist_mask = (timeseries_df[date_column].dt.date >= hist_start) & (
                    timeseries_df[date_column].dt.date <= hist_end
                )
                hist_values = timeseries_df[hist_mask][value_column].dropna().tolist()

                if hist_values:
                    hist_kpi = compute_kpi(hist_values, aggregation)
                    if hist_kpi is not None:
                        past_kpis.append(hist_kpi)
                        years_included.append(y)

            except ValueError:
                # Handle invalid dates (e.g., Feb 29 on non-leap years)
                print(f"⚠️ Skipping invalid date for year {y}: {month:02d}-{day:02d}")

        kpi_past_avg = round(sum(past_kpis) / len(past_kpis), 4) if past_kpis else None

    else:
        raise ValueError(f"Invalid years parameter: {years}. Must be None, 'ALL', or a list of years.")

    # Calculate comparison metrics
    if kpi_current is not None and kpi_past_avg is not None and kpi_past_avg != 0:
        difference = round(kpi_current - kpi_past_avg, 4)
        percent_change = round(((kpi_current - kpi_past_avg) / kpi_past_avg) * 100, 2)
    else:
        difference = None
        percent_change = None

    return {
        "kpi_name": kpi_name,
        "aggregation": aggregation,
        "current_period": {
            "start_date": start_date,
            "end_date": end_date,
            "value": kpi_current,
            "num_records": len(current_values),
        },
        "historical_avg": {"value": kpi_past_avg, "num_years": len(past_kpis), "years_included": years_included},
        "comparison": {"difference": difference, "percent_change": percent_change},
    }


def format_kpi_results(kpi_result):
    """
    Format KPI result dictionary into a readable DataFrame.

    Args:
        kpi_result (dict): Output from filter_timeseries_kpi

    Returns:
        pd.DataFrame: Formatted KPI comparison
    """
    data = {
        "KPI": [kpi_result["kpi_name"]],
        "Aggregation": [kpi_result["aggregation"]],
        "Period Start": [kpi_result["current_period"]["start_date"]],
        "Period End": [kpi_result["current_period"]["end_date"]],
        "Current Value": [kpi_result["current_period"]["value"]],
        "Historical Avg": [kpi_result["historical_avg"]["value"]],
        "Difference": [kpi_result["comparison"]["difference"]],
        "Change %": [kpi_result["comparison"]["percent_change"]],
        "Num Records": [kpi_result["current_period"]["num_records"]],
    }

    return pd.DataFrame(data)
