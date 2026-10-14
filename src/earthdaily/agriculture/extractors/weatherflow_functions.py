# weatherflow_functions.py
"""WeatherFlow API: field-level weather from the new weather service.

Built as a near-copy of ``weather_functions.py`` (Geosys Weather v1) so a caller switches
with few edits: same setup → get → format → process → bulk flow, same ``historical_years`` /
``kpi_filter`` / ``use_cache`` / ``spatial_grouping`` behavior. What differs is the API:

- **Auth** is an ``X-API-Key`` header, not the Geosys token. The key helpers
  (:func:`resolve_api_key`, :func:`api_key_headers`) are plain functions with no extractor
  state, so they can move to ``core/`` unchanged if another API needs them.
- **Models**: ``era5`` and ``seamless`` (history), ``ifs`` and ``icon_eu`` (history and
  forecast). One call covers at most 366 days (daily) or 10 local days (hourly), so
  longer ranges are split.
- **Columns** are normalized by default, so the same column means the same thing whatever
  the model: names follow Weather v1's ``parameter.subParameter`` vocabulary
  (``temperature.standardMax``, ``precipitation.cumulative``; ``naming="api"`` keeps the
  API's ``temperature_2m_max``), and values follow a unit profile (``units="metric"``:
  °C, mm, km/h, W/m²; ``"imperial"``: °F, in, mph, W/m²; ``"native"``: as the API sends
  them). Each column's source unit is read from :meth:`WeatherFlowExtractor.describe_model`;
  conversions use ``core.units`` (pint). ``output_units`` lists the units of the output.
- **Status codes**: 200 rows, 204 no rows, 400 / 422 refused request (not retried), 503
  busy (retried after ``Retry-After``).
- **Time**: ``date`` is the location's local civil time, without a zone: the local day for
  daily rows (Weather v1's days are local days too, so the two join on the date), the step
  start for hourly rows. Hourly rows also carry ``date_utc`` (tz-aware UTC), unique even
  when daylight saving ends and the local hour repeats: use it to merge countries on one
  timeline, join other timestamped data, or deduplicate.
"""

#  Standard Library
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_type
from datetime import datetime, timedelta
from functools import wraps
from typing import Literal

import pandas as pd

#  Third-Party Libraries
import requests
from shapely import wkt as shapely_wkt
from tqdm import tqdm

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.config.weather_normalization import SNOWFALL_COLUMNS
from earthdaily.agriculture.config.weather_normalization import WEATHERFLOW_VOCABULARY as VOCABULARY
from earthdaily.agriculture.core.api_utils import (
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
from earthdaily.agriculture.core.choices import MergeMode
from earthdaily.agriculture.core.geometry import get_centroid_wkt, validate_wkt
from earthdaily.agriculture.core.results import SpatialBulkResult
from earthdaily.agriculture.core.units import UnitProfile
from earthdaily.agriculture.core.units import convert as convert_units

# Dataset and aggregation in one value, as Weather v1's `weather_type` (plus HISTORICAL_HOURLY).
WeatherFlowType = Literal["HISTORICAL_DAILY", "HISTORICAL_HOURLY", "FORECAST_DAILY", "FORECAST_HOURLY"]
# Every model the API serves. What each one serves (dataset × aggregation) is checked
# against GET /models at setup; FORECAST_MODELS is the fallback when that call fails.
WeatherFlowModel = Literal["era5", "ifs", "icon_eu", "seamless"]
FORECAST_MODELS = ("ifs", "icon_eu")
#: Model used when `model` is left to None, per dataset.
DEFAULT_MODELS: dict[str, WeatherFlowModel] = {"historical": "era5", "forecast": "ifs"}
#: Longest date range one call accepts, per aggregation; longer ranges are split into
#: consecutive calls. Hourly is 10 local days: beyond, the API answers 400 "covers at most
#: 10 local days (WEATHERFLOW_HOURLY_POINT_MAX_DAYS)" (measured on era5, 2026-10-09).
MAX_DAYS_PER_CALL = {"daily": 366, "hourly": 10}
#: Columns describing where and how a row was computed, kept next to the weather values.
METADATA_COLUMNS = ["geohash", "model", "aggregation_timezone"]

#: Output column names: Weather v1's ``parameter.subParameter`` vocabulary, or the API's own.
WeatherFlowNaming = Literal["weather_v1", "api"]


def describe_column(column: str, unit_label: str | None) -> tuple[str, str]:
    """``(Weather v1 name, kind)`` of an API column; an unknown column keeps its name, kind ``unknown``."""
    if column in SNOWFALL_COLUMNS:
        label = unit_label or ""
        if "fresh" in label or label.startswith("cm"):
            return "snow.freshSnow", "fresh_snow"
        return "snow.waterEquivalent", "snow_water_equivalent"
    name, kind = VOCABULARY.get(column, (None, "unknown"))
    return name or column, kind


def api_columns_for(requested: list[str]) -> set[str]:
    """API columns behind requested names, which may be Weather v1 names or API names."""
    wanted = set(requested)
    columns = {api for api, (v1, _) in VOCABULARY.items() if api in wanted or (v1 and v1 in wanted)}
    if wanted & ({"snow.freshSnow", "snow.waterEquivalent"} | SNOWFALL_COLUMNS):
        columns |= SNOWFALL_COLUMNS
    return columns | wanted


# -------------------------------------------------------------------------------------------
# API key (temporary auth). Standalone on purpose: no extractor state, so these two functions
# can move to core/ as they are if another API-key service needs them.
# -------------------------------------------------------------------------------------------

API_KEY_HEADER = "X-API-Key"
_ENV_PREFIX = {"prod": "PROD", "preprod": "PREPROD"}


def resolve_api_key(service: str, env: str, api_key: str | None = None, config: dict | None = None) -> str | None:
    """Find the API key of ``service`` for ``env``; None when there is none.

    First match wins:

    1. ``api_key`` (explicit argument);
    2. ``config["<service>_api_key"]``;
    3. ``PROD_<SERVICE>_API_KEY`` / ``PREPROD_<SERVICE>_API_KEY`` (keys are environment-specific);
    4. ``<SERVICE>_API_KEY``.

    Environment variables are read when called, so a ``.env`` loaded before the call is seen.
    """
    upper = service.upper()
    candidates = (
        api_key,
        (config or {}).get(f"{service.lower()}_api_key"),
        os.getenv(f"{_ENV_PREFIX.get(env, env.upper())}_{upper}_API_KEY"),
        os.getenv(f"{upper}_API_KEY"),
    )
    return next((c for c in candidates if c), None)


def api_key_headers(api_key: str) -> dict:
    """Request headers for an API-key service: the key header plus ``Accept: application/json``."""
    return {API_KEY_HEADER: api_key, "Accept": "application/json"}


# -------------------------------------------------------------------------------------------


def split_weather_type(weather_type: str) -> tuple[str, str]:
    """``"FORECAST_HOURLY"`` → ``("forecast", "hourly")``: the API's dataset and aggregation."""
    dataset, aggregation = weather_type.lower().split("_")
    return dataset, aggregation


def date_chunks(start_date: str, end_date: str, max_days: int = MAX_DAYS_PER_CALL["daily"]) -> list[tuple[str, str]]:
    """Split an inclusive ``[start_date, end_date]`` range into consecutive ranges of ``max_days`` at most."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    chunks = []
    while start <= end:
        chunk_end: date_type = min(start + timedelta(days=max_days - 1), end)
        chunks.append((start.isoformat(), chunk_end.isoformat()))
        start = chunk_end + timedelta(days=1)
    return chunks


def missing_day_ranges(rows: list[dict], start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Local days of ``[start_date, end_date]`` with no row, as inclusive ``(first, last)`` runs.

    Only days up to the last day returned are checked: a series that simply stops early
    (forecast horizon, publication delay) is reported separately. Hourly rows count for
    their local day. Returns ``[]`` when there are no rows at all.
    """
    days = {str(r.get("date", ""))[:10] for r in rows if r.get("date")}
    if not days:
        return []
    last = min(end_date, max(days))
    missing = pd.date_range(start_date, last).strftime("%Y-%m-%d").difference(sorted(days))
    runs: list[tuple[str, str]] = []
    for day in missing:
        if runs and (pd.Timestamp(day) - pd.Timestamp(runs[-1][1])).days == 1:
            runs[-1] = (runs[-1][0], day)
        else:
            runs.append((day, day))
    return runs


def require_weatherflow_params(func):
    """Decorator to ensure WeatherFlow parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "weatherflow_params"):
            self.logger.error("No WeatherFlow parameters found")
            raise RuntimeError("❌ No WeatherFlow parameters found. Call setup_weatherflow_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class WeatherFlowExtractor(BaseExtractor):
    """
    Extracts field-level weather data from the WeatherFlow API.

    Retrieves historical (ERA5, IFS, ICON-EU, seamless) and forecast (IFS, ICON-EU) weather,
    daily or hourly, at each field's centroid for a date range. Mirrors WeatherExtractor
    (same flow, KPI aggregation and spatial grouping); columns use Weather v1's names and
    metric units by default (``naming="api"`` keeps the API's names).

    Documentation: https://weatherflow.aws.geosys.com/prod/docs
    OpenAPI: https://weatherflow.aws.geosys.com/prod/openapi.json
    Operation: GET /models/{model}/historical
    Operation: GET /models/{model}/forecast

    Dev notebook: notebooks/EDAgriculture_WeatherFlow_Service_Function_Dev.ipynb

    Args (setup_weatherflow_parameters):
        weather_type (WeatherFlowType): Historical or forecast data, daily or hourly. Default: 'HISTORICAL_DAILY'
        model (WeatherFlowModel | None): Weather model; None uses 'era5' for history and 'ifs' for forecasts. Forecasts need 'ifs' or 'icon_eu'. Default: None
        weather_parameters (str | list[str]): Columns to keep, as Weather v1 names or API names; 'none' keeps every column. Default: 'none'
        units (UnitProfile): Unit profile of the output: 'metric' (°C, mm, km/h, W/m²), 'imperial' (°F, in, mph, W/m²) or 'native' (the API's units). Default: 'metric'
        naming (WeatherFlowNaming): Output column names: 'weather_v1' (temperature.standardMax, ...) or 'api' (temperature_2m_max, ...). Default: 'weather_v1'
        kpi_filter (dict): KPI aggregation config (kpi_name, aggregation, threshold)
        historical_years (int): Number of prior years of weather history to fetch. Default: 0
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required); start_date, end_date (optional overrides)

    Output columns:
        entity_id, date, date_utc (hourly only), geohash, model, aggregation_timezone, + the model's weather columns named by `naming` in the `units` profile (output_units lists their units)

    Constraints:
        The model must serve the weather_type's dataset and aggregation, checked against GET /models at setup (icon_eu: daily history and hourly forecasts only).
        One call covers at most 366 days (daily) or 10 local days (hourly); longer ranges are split.
        A forecast starts no earlier than today at the location, which ahead of UTC can be tomorrow.
        ifs history covers the last 15 complete local days only.
        Needs a WeatherFlow API key: api_key=, config["weatherflow_api_key"], PROD_/PREPROD_WEATHERFLOW_API_KEY or WEATHERFLOW_API_KEY.
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None, api_key=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.weatherflow_params = None
        self._served = None  # GET /models as {model: {(dataset, aggregation)}}, fetched once
        self._column_units = {}  # {(model, dataset, aggregation): {column: unit label}}
        self.output_units = {}  # {output column: unit} of the last formatted responses

        # Specific API endpoint for WeatherFlow
        self.weatherflow_url = agro_urls["weatherflow_urls"][self.env]

        # X-API-Key auth: resolved once here so a missing key fails at construction, not per entity.
        self.api_key = resolve_api_key("weatherflow", self.env, api_key=api_key, config=config)
        if not self.api_key:
            raise ValueError(
                f"❌ No WeatherFlow API key for env={self.env!r}: pass api_key=, or set "
                f"{_ENV_PREFIX.get(self.env, self.env.upper())}_WEATHERFLOW_API_KEY or WEATHERFLOW_API_KEY "
                "in src/.env. The LRTS and EDS keys are not valid here."
            )

        self.logger.info(f"🌤️ WeatherFlowExtractor initialized for env: {self.env}")
        if self.output_path:
            self.logger.info(f"📁 Output path set to: {self.output_path}")
        if self.partial_path:
            self.logger.info(f"📦 Partial results path: {self.partial_path}")

    def ensure_token_valid(self):
        """No-op: WeatherFlow authenticates with ``X-API-Key``, not the Geosys token.

        Kept so the ``@requires_token`` decorators copied from WeatherExtractor stay harmless
        and the two modules stay comparable line by line.
        """
        return self.bearer_token, self.token_expiration

    def get_new_token(self):
        """No Geosys token to refresh (see :meth:`ensure_token_valid`)."""
        return self.bearer_token, self.token_expiration

    def _get(self, path, params=None, timeout=60):
        """``GET <weatherflow_url><path>`` with the API key; returns the decoded JSON, or None on 204."""
        response = requests.get(
            f"{self.weatherflow_url}{path}", params=params, headers=api_key_headers(self.api_key), timeout=timeout
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()

    def list_models(self) -> list[dict]:
        """``GET /models``: every model with the dataset / aggregation combinations it serves."""
        return (self._get("/models") or {}).get("models") or []

    def describe_model(self, model: str) -> dict:
        """``GET /models/{model}``: per dataset and aggregation, the time step and each column's unit."""
        return self._get(f"/models/{model}") or {}

    def served_combinations(self) -> dict | None:
        """``GET /models`` as ``{model: {(dataset, aggregation), ...}}``, fetched once per extractor.

        Returns None when the call fails, so setup falls back to the static check instead of
        refusing to configure offline.
        """
        if self._served is None:
            try:
                self._served = {
                    m["name"]: {(c["dataset"], c["aggregation"]) for c in m.get("combinations") or []}
                    for m in self.list_models()
                }
            except requests.RequestException as e:
                self.logger.warning(f"GET /models failed ({e}); model checks fall back to the static list")
                return None
        return self._served

    def column_units(self, model: str, dataset: str, aggregation: str) -> dict[str, str | None]:
        """``{API column: unit label}`` for one model / dataset / aggregation, from ``describe_model``.

        Fetched once per model and cached. Returns ``{}`` when the call fails: values are then
        kept in the API's units, with a warning, rather than converted from a guessed unit.
        """
        key = (model, dataset, aggregation)
        if key not in self._column_units:
            try:
                info = self.describe_model(model)
            except requests.RequestException as e:
                self.logger.warning(f"GET /models/{model} failed ({e}); units cannot be read")
                return {}
            if not isinstance(info, dict):
                self.logger.warning(f"GET /models/{model} returned no model description; units cannot be read")
                return {}
            for combo in info.get("combinations") or []:
                self._column_units[(model, combo["dataset"], combo["aggregation"])] = {
                    col["name"]: col.get("unit") for col in combo.get("columns") or []
                }
            self._column_units.setdefault(key, {})
        return self._column_units[key]

    @requires_token
    def setup_weatherflow_parameters(
        self,
        weather_type: WeatherFlowType = "HISTORICAL_DAILY",
        model: WeatherFlowModel | None = None,
        weather_parameters: str | list[str] = "none",
        units: UnitProfile = "metric",
        naming: WeatherFlowNaming = "weather_v1",
        historical_years=0,
        partial_frequency=50,
        kpi_filter=None,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure params for WeatherFlow extraction for an entity

        Args:
            weather_type (WeatherFlowType): Historical or forecast data, daily or hourly
            model (WeatherFlowModel | None): Weather model. None uses the dataset's default
                ('era5' for history, 'ifs' for forecasts). Forecasts need 'ifs' or 'icon_eu'.
            weather_parameters (str | list[str]): Columns to keep, as Weather v1 names
                ('temperature.standardMax') or API names ('temperature_2m_max'). Defaults to 'none',
                which keeps every column the model returns.
            units (UnitProfile): Unit profile of the output: 'metric' (°C, mm, km/h, W/m²),
                'imperial' (°F, in, mph, W/m²) or 'native' (the API's own units). Default 'metric'.
            naming (WeatherFlowNaming): Output column names: 'weather_v1' (Weather v1's
                parameter.subParameter names) or 'api' (the API's names). Default 'weather_v1'.
            historical_years (int or list): Number of historical years for KPI comparison, or list of specific years
            partial_frequency (int): How often to save partial results
            kpi_filter (dict, optional): KPI configuration
        """
        self.logger.info("Setting up WeatherFlow parameters...")

        weather_type = self._check_choice("weather_type", weather_type, WeatherFlowType)
        units = self._check_choice("units", units, UnitProfile)
        naming = self._check_choice("naming", naming, WeatherFlowNaming)
        dataset, aggregation = split_weather_type(weather_type)
        if model is None:
            model = DEFAULT_MODELS[dataset]
        model = self._check_choice("model", model, WeatherFlowModel)
        served = self.served_combinations()
        if served is not None:
            if (dataset, aggregation) not in served.get(model, set()):
                offers = ", ".join(f"{d}/{a}" for d, a in sorted(served.get(model, set()))) or "nothing"
                self.logger.error(f"Model {model} does not serve {dataset}/{aggregation}")
                raise ValueError(
                    f"Invalid model '{model}' for {weather_type}: it does not serve {dataset}/{aggregation} "
                    f"(it serves: {offers})."
                )
        elif dataset == "forecast" and model not in FORECAST_MODELS:
            self.logger.error(f"Model {model} serves no forecast")
            raise ValueError(
                f"Invalid model '{model}' for {weather_type}. Forecasts need one of {list(FORECAST_MODELS)}."
            )

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

        # Weather parameters are Weather v1 or API column names; which exist depends on the
        # model and aggregation (describe_model() lists them), so only their type is checked
        # here; a name the response lacks is reported when the response is formatted.
        params_list = weather_parameters if isinstance(weather_parameters, list) else [weather_parameters]
        if not params_list or not all(isinstance(p, str) and p for p in params_list):
            raise ValueError(
                f"weather_parameters must be 'none', a column name or a list of them, got {weather_parameters!r}"
            )

        # Validate KPI filter (centralized rules — see api_utils.validate_kpi_filter)
        validate_kpi_filter(kpi_filter, logger=self.logger)

        if column_mapping:
            self.set_column_mapping(column_mapping)

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        self.apply_cache_setting(use_cache)

        # Output units describe the current configuration only: start afresh at each setup.
        self.output_units = {}

        # ✅ store in self.weatherflow_params
        self.weatherflow_params = {
            "weather_type": weather_type,
            "model": model,
            "weather_parameters": weather_parameters,
            "units": units,
            "naming": naming,
            "historical_years": historical_years,
            "partial_frequency": partial_frequency,
            "kpi_filter": kpi_filter,
        }

        id_col = self.get_mapped_column("id")
        # Hourly rows are keyed on the instant: when daylight saving ends the local hour
        # repeats, and keying on `date` would deduplicate a real hour away.
        self.cache_key_columns = [id_col, "date_utc" if aggregation == "hourly" else "date"]

        self.logger.info("WeatherFlow parameters configured successfully")
        self.logger.debug(f"Weather type: {weather_type} ({dataset}, {aggregation}), model: {model}")
        self.logger.debug(f"Parameters: {weather_parameters}")
        if kpi_filter:
            self.logger.debug(f"KPI filter enabled: {kpi_filter.get('aggregation')}")

        print("🌍 WeatherFlow parameters configured:")
        for k, v in self.weatherflow_params.items():
            if k == "kpi_filter" and v:
                print(f"   {k}:")
                for kpi_key, kpi_val in v.items():
                    print(f"      {kpi_key}: {kpi_val}")
            else:
                print(f"   {k}: {v}")

    @require_weatherflow_params
    @requires_token
    def get_weatherflow_data(self, entity_data):
        """
        Get WeatherFlow rows for the entity's centroid and date range using self.weatherflow_params

        The API accepts at most 366 days per daily call and 10 local days per hourly call
        (``MAX_DAYS_PER_CALL``), so the range is split into consecutive calls
        (``date_chunks``) and their rows concatenated. A 204 (no rows for a chunk)
        contributes nothing; any other non-200 status raises ``requests.HTTPError``, which
        the caller's retry wrapper retries (503, 429) or not (400, 422 and other 4xx).

        Args:
            entity_data (dict): Entity data with geometry, start_date, end_date

        Returns:
            list[dict]: WeatherFlow rows, one per day (daily) or time step (hourly)
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        if not hasattr(self, "weatherflow_params") or self.weatherflow_params is None:
            self.logger.error("WeatherFlow parameters not configured")
            raise RuntimeError("❌ WeatherFlow parameters not configured. Call setup_weatherflow_parameters() first.")

        params = self.weatherflow_params

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
        centroid = shapely_wkt.loads(get_centroid_wkt(geometry))
        self.logger.debug(f"Entity {entity_id}: Centroid calculated")

        # Step 3: API URL (the model and dataset are path segments)
        dataset, aggregation = split_weather_type(params["weather_type"])
        path = f"/models/{params['model']}/{dataset}"

        # Step 4: Request data, one call per date chunk
        rows = []
        for chunk_start, chunk_end in date_chunks(start_date, end_date, MAX_DAYS_PER_CALL[aggregation]):
            query = {
                "latitude": round(centroid.y, 6),
                "longitude": round(centroid.x, 6),
                "start": chunk_start,
                "end": chunk_end,
                "aggregation": aggregation,
            }
            self.logger.debug(f"Entity {entity_id}: Calling WeatherFlow {path} {query}")
            try:
                chunk_rows = self._get(path, params=query)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                text = e.response.text if e.response is not None else str(e)
                self.logger.error(f"Entity {entity_id}: HTTP {status} - {text} | {path} {query}")
                if status == 400 and "today at the location" in text:
                    # Kept an HTTPError carrying the response, so it is still not retried.
                    raise requests.exceptions.HTTPError(
                        f"Forecast start {chunk_start} is before today at the field's location. Forecasts "
                        "start at the LOCAL date, which ahead of UTC (Europe, Asia) can already be "
                        "tomorrow: use that date or later.",
                        response=e.response,
                    ) from e
                raise
            except Exception as e:
                self.logger.error(f"Entity {entity_id}: WeatherFlow call failed - {str(e)} | {path} {query}")
                raise
            rows.extend(chunk_rows or [])

        # A forecast past the model's horizon (or history not yet published) comes back as a
        # 200 with fewer days than asked; say so instead of returning a silently short series.
        last_day = max((str(r.get("date", ""))[:10] for r in rows), default=None)
        if last_day and last_day < end_date:
            self.logger.warning(
                f"Entity {entity_id}: {params['model']} {dataset} data ends on {last_day}, before the "
                f"requested end {end_date} (forecast horizon or publication delay)"
            )

        # Days missing INSIDE the range also come back as a 200 (e.g. era5 and seamless have no
        # data for 2023-09-11 to 2023-12-07, found 2026-10-12): warn rather than return a series
        # with a silent hole.
        gaps = missing_day_ranges(rows, start_date, end_date)
        if gaps:
            n_missing = sum((pd.Timestamp(b) - pd.Timestamp(a)).days + 1 for a, b in gaps)
            shown = ", ".join(a if a == b else f"{a} to {b}" for a, b in gaps[:3])
            more = f" and {len(gaps) - 3} more gap(s)" if len(gaps) > 3 else ""
            self.logger.warning(
                f"Entity {entity_id}: {params['model']} {dataset} {aggregation} has no data for "
                f"{n_missing} day(s) of the requested range: {shown}{more}"
            )

        self.logger.debug(f"Entity {entity_id}: WeatherFlow returned {len(rows)} rows")
        return rows

    @require_weatherflow_params
    @requires_token
    def get_weatherflow_data_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_weatherflow_data().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain 'id', 'geometry', 'start_date', 'end_date' keys

        Returns:
            dict: {
                "success": bool,
                "data": list | None,
                "error": str | None,
                "entity_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")

        try:
            response_json = self.get_weatherflow_data(entity_data)
            self.logger.debug(f"Entity {entity_id}: WeatherFlow data retrieved successfully")
            return {"success": True, "data": response_json, "error": None, "entity_id": entity_id}
        except requests.exceptions.HTTPError as e:
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

    def format_weatherflow_json(self, response_weatherflow_json, entity_data: dict = None):
        """
        Format WeatherFlow rows into a DataFrame.

        Keeps the API's column names. ``date`` becomes a naive timestamp in the location's
        local civil time (a day for daily rows, the step start for hourly rows, whose ISO
        offsets are dropped after parsing); ``aggregation_timezone`` names that zone.
        Hourly rows also get ``date_utc``, the tz-aware UTC instant, and are sorted on it:
        when daylight saving ends, two rows share a local ``date`` but not a ``date_utc``.
        With ``weather_parameters`` other than 'none', only those weather columns are kept
        (metadata columns always are), and requested names the response lacks are logged.

        Args:
            response_weatherflow_json (list): Rows from get_weatherflow_data
            entity_data (dict, optional): Original entity data to include ID in output

        Returns:
            pd.DataFrame: entity_id, date, date_utc (hourly only), geohash, model, aggregation_timezone,
            then the weather columns
        """
        entity_id = (
            self.get_entity_value(entity_data, "id", "unknown") if self._is_entity_provided(entity_data) else "unknown"
        )
        self.logger.debug(f"Entity {entity_id}: Formatting WeatherFlow response...")

        empty = self.validate_api_response(response_weatherflow_json, entity_id, "weatherflow")
        if empty is not None:
            return empty

        df = pd.DataFrame(response_weatherflow_json)

        if df.empty or "date" not in df.columns:
            self.logger.warning(f"Entity {entity_id}: No WeatherFlow data to format")
            return pd.DataFrame()

        # Every weather column; the requested ones are kept after normalization, when each
        # column's Weather v1 name is known from its real unit (snowfall depends on it).
        weather_cols = sorted(c for c in df.columns if c != "date" and c not in METADATA_COLUMNS)

        meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
        df = df[["date"] + meta_cols + weather_cols]

        # Add entity_id first if provided
        if self._is_entity_provided(entity_data) and self.has_entity_field(entity_data, "id"):
            df.insert(0, "entity_id", self.get_entity_value(entity_data, "id"))

        # Local civil time: parse the ISO value (offsets vary with DST), then drop the offset.
        # Hourly values carry an offset, so they also get the exact instant in UTC.
        raw_dates = df["date"].astype(str)
        hourly = raw_dates.str.contains("T").any()
        if hourly:
            df.insert(df.columns.get_loc("date") + 1, "date_utc", pd.to_datetime(raw_dates, utc=True, format="ISO8601"))
        df["date"] = pd.to_datetime(raw_dates.str[:19], format="ISO8601")
        df = df.sort_values("date_utc" if hourly else "date").reset_index(drop=True)

        df = self._normalize_columns(df, weather_cols, hourly, entity_id)

        self.logger.debug(
            f"Entity {entity_id}: Formatted {len(df)} WeatherFlow records with {len(weather_cols)} parameters"
        )
        return df

    def _normalize_columns(self, df, weather_cols, hourly, entity_id):
        """Convert the weather columns to the unit profile and rename them per ``naming``.

        The source unit of each column is the one ``describe_model`` reports for the row's
        model, dataset and aggregation. Records the output units in ``self.output_units``,
        ``df.attrs["units"]`` and the export manifest (``manifest_metadata``).
        """
        params = getattr(self, "weatherflow_params", None) or {}
        profile = params.get("units", "native")
        naming = params.get("naming", "api")
        dataset, aggregation = split_weather_type(params.get("weather_type", "HISTORICAL_DAILY"))
        model = df["model"].iloc[0] if "model" in df.columns and len(df) else params.get("model")
        labels = self.column_units(model, dataset, aggregation) if model else {}
        period = "hour" if hourly else "day"

        # Keep only the requested columns ('none' keeps them all), named either way.
        requested = params.get("weather_parameters", "none")
        if requested != "none":
            requested = requested if isinstance(requested, list) else [requested]
            names = {col: describe_column(col, labels.get(col))[0] for col in weather_cols}
            missing = [r for r in requested if r not in weather_cols and r not in names.values()]
            if missing:
                self.logger.warning(
                    f"Entity {entity_id}: columns not in the response, skipped: {missing} "
                    f"(available: {sorted(names.values()) if naming == 'weather_v1' else weather_cols})"
                )
            dropped = [col for col in weather_cols if col not in requested and names[col] not in requested]
            df = df.drop(columns=dropped)
            weather_cols = [col for col in weather_cols if col not in dropped]

        renames, units_out, unknown = {}, {}, []
        for col in weather_cols:
            label = labels.get(col)
            v1_name, kind = describe_column(col, label)
            if profile != "native" and label is None and kind not in ("code", "unknown"):
                unknown.append(col)
            df[col], shown = convert_units(df[col], label, kind, profile, period=period)
            out = v1_name if naming == "weather_v1" else col
            renames[col] = out
            units_out[out] = shown
        if unknown:
            self.logger.warning(
                f"Entity {entity_id}: no unit known for {unknown} ({model} {dataset}/{aggregation}); "
                "kept as the API sent them"
            )

        df = df.rename(columns=renames)
        leading = [c for c in df.columns if c not in renames.values()]
        df = df[leading + sorted(renames.values())]
        df.attrs["units"] = dict(units_out)
        self.output_units.update(units_out)
        self.manifest_metadata = {
            **(getattr(self, "manifest_metadata", None) or {}),
            "units": dict(self.output_units),
            "unit_profile": profile,
            "naming": naming,
        }
        return df

    @require_weatherflow_params
    @requires_token
    @cache_single_entity("weatherflow_params")
    def process_single_entity_weatherflow(self, row, params=None):
        """
        Process a single entity for weather data extraction with optional KPI filtering.

        Args:
            row (pd.DataFrame, pd.Series, or dict): Entity data with required fields
            params (dict, optional): Override weatherflow_params

        Returns:
            dict: Contains 'data' (DataFrame) and 'error'
        """
        if params is None:
            params = self.weatherflow_params

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
            # WeatherFlow takes an absolute start/end range, so we push start_date back.
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
                return self.get_weatherflow_data(row)

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
            weather_df = self.format_weatherflow_json(raw_json, entity_data=row)

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

    @require_weatherflow_params
    def process_entity_weatherflow_bulk_parallel(
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
        merge_existing: MergeMode | None = None,
        skip_export=False,
        prefix="weatherflow",
        generate_report=False,
        report_options=None,
        use_cache=None,
        spatial_grouping=False,
        spatial_precision=DEFAULT_SPATIAL_PRECISION,
        spatial_max_window_days=DEFAULT_MAX_WINDOW_DAYS,
        rate_limit=None,
    ) -> SpatialBulkResult:
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
                response. WeatherFlow is queried by field centroid and serves one
                value per geohash-5 cell (the ``geohash`` column), the same precision as
                the default bucket, so fields sharing a cell return identical values and
                dedup safely; members are cut back to their own window, so the output
                matches an ungrouped run row for row. Every WeatherFlow column is
                per-step (``precipitation_sum`` is a daily total), so no value correction
                is applied.
                With ``use_cache``, the geohash-keyed spatial cache is consulted per
                cell before any request — a cell whose window is already on disk costs
                no API call, whatever field set asks for it.
            spatial_precision (int): Geohash precision for the bucket (default
                ``DEFAULT_SPATIAL_PRECISION``).
            spatial_max_window_days (int): Cap on a group's widened window (default
                ``DEFAULT_MAX_WINDOW_DAYS``). Cells whose union exceeds it are split
                into sub-groups, so one long-history field cannot force a decade-long
                pull on every field sharing its cell.
            rate_limit (float, optional): Requests per second for this run, shared by all
                ``max_workers`` threads. None (default) falls back to ``self.rate_limit``, then
                ``DEFAULT_RATE_LIMIT``; unset means no limit.

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics
        """

        self._bulk_rate_limit = rate_limit  # read by _paced()

        def _dispatch(
            el, *, skip_export_=skip_export, generate_report_=generate_report, report_options_=report_options
        ):
            if use_cache is not None and use_cache:
                return self._bulk_with_cache(
                    bulk_method=self._process_entity_weatherflow_bulk_parallel_inner,
                    entity_list=el,
                    params=self.weatherflow_params,
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
            return self._process_entity_weatherflow_bulk_parallel_inner(
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
                return self._process_entity_weatherflow_bulk_parallel_inner(
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
                # No cumulative columns: every WeatherFlow column is per-step
                # (`precipitation_sum` is a daily total), so nothing needs rebasing.
                cumulative_columns=None,
                precision=spatial_precision,
                prefix=prefix,
                output_path=output_path,
                skip_export=skip_export,
                generate_report=generate_report,
                report_options=report_options,
                report_params=getattr(self, "weatherflow_params", {}),
                use_cache=use_cache,
                cache_params=getattr(self, "weatherflow_params", {}),
            )
        return _dispatch(entity_list)

    @requires_token
    @require_weatherflow_params
    def _process_entity_weatherflow_bulk_parallel_inner(
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
        merge_existing: MergeMode | None = None,
        skip_export=False,
        prefix="weatherflow",
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
        merge_existing = self._check_choice("merge_existing", merge_existing, MergeMode)

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

        # One limiter per run, shared by every worker (a no-op without a rate limit).
        paced = self._paced(self.process_single_entity_weatherflow)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.logger.debug(f"ThreadPoolExecutor started with {max_workers} workers")
            future_to_id = {
                executor.submit(paced, row, params): self.get_entity_value(row, "id")
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
                "parameters": getattr(self, "weatherflow_params", {}),
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
