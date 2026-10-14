"""WeatherFlowExtractor — API-key helpers, request building, formatting and the bulk path.

No network: ``requests.get`` is patched and the request it receives is what these tests
assert on. Responses are the shapes recorded from the prod API on 2026-10-09 (ERA5 daily,
Ames IA); the hourly sample follows the documented ISO-with-offset convention.
"""

import time
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from earthdaily.agriculture.extractors import weatherflow_functions as wf
from earthdaily.agriculture.extractors.weatherflow_functions import (
    WeatherFlowExtractor,
    api_key_headers,
    date_chunks,
    missing_day_ranges,
    resolve_api_key,
)

pytestmark = pytest.mark.public

GET = "earthdaily.agriculture.extractors.weatherflow_functions.requests.get"
# Small square around Ames, IA: centroid (-93.63, 42.03).
FIELD = "POLYGON ((-93.635 42.025, -93.625 42.025, -93.625 42.035, -93.635 42.035, -93.635 42.025))"

DAILY_ROWS = [
    {
        "date": "2025-06-01",
        "geohash": "9zmrj",
        "aggregation_timezone": "America/Chicago",
        "temperature_2m_mean": 21.3354,
        "temperature_2m_max": 27.9,
        "temperature_2m_min": 14.0,
        "precipitation_sum": 0.0,
        "swvl1": 0.2452,
        "model": "era5",
    },
    {
        "date": "2025-06-02",
        "geohash": "9zmrj",
        "aggregation_timezone": "America/Chicago",
        "temperature_2m_mean": 24.1479,
        "temperature_2m_max": 31.9,
        "temperature_2m_min": 15.9,
        "precipitation_sum": 0.6,
        "swvl1": 0.2239,
        "model": "era5",
    },
]

HOURLY_ROWS = [
    {
        "date": "2025-03-09T01:00:00-06:00",
        "geohash": "9zmrj",
        "aggregation_timezone": "America/Chicago",
        "temperature_2m": 1.5,
        "model": "era5",
    },
    {
        "date": "2025-03-09T03:00:00-05:00",  # DST starts: the offset changes mid-series
        "geohash": "9zmrj",
        "aggregation_timezone": "America/Chicago",
        "temperature_2m": 1.1,
        "model": "era5",
    },
]


# GET /models as recorded on prod, 2026-10-09: what each model serves.
SERVED = {
    "era5": {("historical", "daily"), ("historical", "hourly")},
    "ifs": {("historical", "daily"), ("historical", "hourly"), ("forecast", "daily"), ("forecast", "hourly")},
    "icon_eu": {("historical", "daily"), ("forecast", "hourly")},
    "seamless": {("historical", "daily"), ("historical", "hourly")},
}
MODELS_PAYLOAD = {
    "models": [
        {"name": name, "combinations": [{"dataset": d, "aggregation": a} for d, a in sorted(combos)]}
        for name, combos in SERVED.items()
    ]
}

# Paris, 26 Oct 2025: daylight saving ends at 03:00 CEST, so the local hour 02:00 happens twice.
DST_END_ROWS = [
    {
        "date": f"2025-10-26T{t}",
        "geohash": "u09tv",
        "aggregation_timezone": "Europe/Paris",
        "temperature_2m": v,
        "model": "era5",
    }
    for t, v in [("01:00:00+02:00", 9.0), ("02:00:00+02:00", 8.5), ("02:00:00+01:00", 8.1), ("03:00:00+01:00", 7.9)]
]


# Two full local days in Paris around the DST end: 24 hours on the 25th, 25 on the 26th.
_PARIS_HOURS = pd.date_range("2025-10-24 22:00", "2025-10-26 22:00", freq="h", tz="UTC").tz_convert("Europe/Paris")
TWO_DAYS_HOURLY = [
    {
        "date": t.isoformat(),
        "geohash": "9zmrj",
        "aggregation_timezone": "Europe/Paris",
        "temperature_2m": float(i),
        "model": "era5",
    }
    for i, t in enumerate(_PARIS_HOURS)
]
assert len(TWO_DAYS_HOURLY) == 49


# Column units per model / dataset / aggregation, as GET /models/{model} reported them on prod
# (2026-10-09). The ifs forecast is the odd one: wind in km/h, snowfall as fresh snow in cm.
MODEL_UNITS = {
    ("era5", "historical", "daily"): {
        "temperature_2m_mean": "degC",
        "temperature_2m_max": "degC",
        "temperature_2m_min": "degC",
        "precipitation_sum": "mm",
        "precipitation_hours": "h",
        "snowfall_sum": "mm (water equivalent)",
        "wind_speed_10m_max": "m/s",
        "wind_gusts_10m_max": "m/s",
        "wind_direction_10m_dominant": "degrees (meteorological)",
        "shortwave_radiation_sum": "MJ/m2",
        "swvl1": "m3/m3",
        "swvl2": "m3/m3",
        "swvl3": "m3/m3",
        "swvl4": "m3/m3",
    },
    ("era5", "historical", "hourly"): {
        "temperature_2m": "degC",
        "dew_point_2m": "degC",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "snowfall_water_equivalent": "mm",
        "wind_speed_10m": "m/s",
        "wind_direction_10m": "degrees (meteorological)",
        "wind_gusts_10m": "m/s",
        "shortwave_radiation": "W/m2",
    },
    ("icon_eu", "historical", "daily"): {
        "temperature_2m_mean": "degC",
        "temperature_2m_max": "degC",
        "temperature_2m_min": "degC",
        "relative_humidity_2m_mean": "%",
        "relative_humidity_2m_max": "%",
        "relative_humidity_2m_min": "%",
        "precipitation_sum": "mm",
        "precipitation_hours": "h",
        "rain_sum": "mm",
        "showers_sum": "mm",
        "snowfall_water_equivalent_sum": "mm",
        "snow_depth_max": "m",
        "wind_speed_10m_max": "m/s",
        "wind_direction_10m_dominant": "degrees (meteorological)",
        "wind_gusts_10m_max": "m/s",
        "direct_radiation_sum": "MJ/m2",
        "diffuse_radiation_sum": "MJ/m2",
        "shortwave_radiation_sum": "MJ/m2",
        "soil_temperature_0cm_mean": "degC",
        "soil_moisture_0_to_1cm_mean": "m3/m3",
        "cloud_cover_mean": "%",
        "cloud_cover_max": "%",
        "cloud_cover_min": "%",
        "visibility_min": "m",
        "weather_code_max": "WMO code",
    },
    ("ifs", "forecast", "daily"): {
        "temperature_2m_mean": "degC",
        "temperature_2m_max": "degC",
        "temperature_2m_min": "degC",
        "relative_humidity_2m_mean": "%",
        "relative_humidity_2m_max": "%",
        "relative_humidity_2m_min": "%",
        "cloud_cover_mean": "%",
        "cloud_cover_max": "%",
        "cloud_cover_min": "%",
        "precipitation_sum": "mm",
        "precipitation_hours": "h",
        "snowfall_sum": "cm (fresh snow)",
        "wind_speed_10m_mean": "km/h",
        "wind_speed_10m_max": "km/h",
        "wind_speed_10m_min": "km/h",
        "wind_gusts_10m_mean": "km/h",
        "wind_gusts_10m_max": "km/h",
        "wind_gusts_10m_min": "km/h",
        "wind_direction_10m_dominant": "degrees (meteorological)",
        "shortwave_radiation_sum": "MJ/m2",
    },
    ("ifs", "forecast", "hourly"): {
        "temperature_2m": "degC",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "snowfall": "cm (fresh snow)",
        "snow_depth": "m",
        "cloud_cover": "%",
        "cloud_cover_low": "%",
        "cloud_cover_mid": "%",
        "cloud_cover_high": "%",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "degrees (meteorological)",
        "wind_gusts_10m": "km/h",
        "shortwave_radiation": "W/m2",
    },
}


def _resp(status=200, body=None, retry_after=None):
    resp = Mock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = body
    resp.text = "" if body is None else str(body)
    resp.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} error", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def extractor(tmp_path):
    ext = WeatherFlowExtractor(
        "fake-token",
        time.time() + 3600,
        {"env": "prod", "output_result_dir": str(tmp_path), "project_root": str(tmp_path)},
        api_key="test-key",
    )
    ext._served = {m: set(c) for m, c in SERVED.items()}  # recorded GET /models: no network in setup
    ext._column_units = {k: dict(v) for k, v in MODEL_UNITS.items()}  # recorded units: no network either
    return ext


@pytest.fixture
def configured(extractor, tmp_path):
    extractor.setup_weatherflow_parameters()
    extractor.output_path = extractor.partial_path = str(tmp_path)
    extractor.use_cache = False
    return extractor


def _entity(**extra):
    return {"id": "f1", "geometry": FIELD, "start_date": "2025-06-01", "end_date": "2025-06-02", **extra}


# ---------------------------------------------------------------------------
# API-key helpers: standalone, no extractor state
# ---------------------------------------------------------------------------


class TestApiKeyHelpers:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in ("PROD_WEATHERFLOW_API_KEY", "PREPROD_WEATHERFLOW_API_KEY", "WEATHERFLOW_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("PROD_WEATHERFLOW_API_KEY", "env-key")
        assert resolve_api_key("weatherflow", "prod", api_key="arg", config={"weatherflow_api_key": "cfg"}) == "arg"

    def test_config_before_environment(self, monkeypatch):
        monkeypatch.setenv("PROD_WEATHERFLOW_API_KEY", "env-key")
        assert resolve_api_key("weatherflow", "prod", config={"weatherflow_api_key": "cfg"}) == "cfg"

    def test_prefixed_variable_before_bare_one(self, monkeypatch):
        monkeypatch.setenv("WEATHERFLOW_API_KEY", "bare")
        monkeypatch.setenv("PREPROD_WEATHERFLOW_API_KEY", "pre")
        assert resolve_api_key("weatherflow", "preprod") == "pre"
        assert resolve_api_key("weatherflow", "prod") == "bare"

    def test_none_when_absent(self):
        assert resolve_api_key("weatherflow", "prod") is None

    def test_service_name_is_a_parameter(self, monkeypatch):
        """Reusable for another API-key service without edits."""
        monkeypatch.setenv("PROD_OTHERSVC_API_KEY", "other")
        assert resolve_api_key("othersvc", "prod") == "other"

    def test_headers(self):
        assert api_key_headers("k") == {"X-API-Key": "k", "Accept": "application/json"}


class TestConstruction:
    def test_missing_key_fails_at_construction(self, tmp_path, monkeypatch):
        for name in ("PROD_WEATHERFLOW_API_KEY", "WEATHERFLOW_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ValueError, match="No WeatherFlow API key"):
            WeatherFlowExtractor("t", time.time() + 3600, {"env": "prod", "project_root": str(tmp_path)})

    def test_prod_url(self, extractor):
        assert extractor.weatherflow_url == "https://weatherflow.aws.geosys.com/prod"

    def test_no_geosys_token_refresh(self, extractor):
        """The copied @requires_token decorators must not reach the Geosys identity service."""
        extractor.token_expiration = time.time() - 10  # an expired Geosys token
        with patch("earthdaily.agriculture.core.identity.EDAuthenticator.get_new_token") as refresh:
            extractor.ensure_token_valid()
        refresh.assert_not_called()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TestSetup:
    def test_defaults(self, extractor):
        extractor.setup_weatherflow_parameters()
        p = extractor.weatherflow_params
        assert (p["weather_type"], p["model"], p["weather_parameters"]) == ("HISTORICAL_DAILY", "era5", "none")

    def test_forecast_defaults_to_ifs(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_HOURLY")
        assert extractor.weatherflow_params["model"] == "ifs"

    def test_forecast_rejects_history_only_model(self, extractor):
        with pytest.raises(ValueError, match="does not serve forecast/daily"):
            extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY", model="era5")

    @pytest.mark.parametrize("weather_type", ["HISTORICAL_HOURLY", "FORECAST_DAILY"])
    def test_icon_eu_combinations_it_does_not_serve(self, extractor, weather_type):
        """The API answers 400 to these; setup now refuses them first, naming what icon_eu serves."""
        with pytest.raises(ValueError, match=r"it serves: forecast/hourly, historical/daily"):
            extractor.setup_weatherflow_parameters(weather_type=weather_type, model="icon_eu")

    def test_icon_eu_hourly_forecast_is_accepted(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_HOURLY", model="icon_eu")
        assert extractor.weatherflow_params["model"] == "icon_eu"

    def test_models_are_fetched_once(self, extractor):
        extractor._served = None
        with patch(GET, return_value=_resp(200, MODELS_PAYLOAD)) as get:
            extractor.setup_weatherflow_parameters()
            extractor.setup_weatherflow_parameters(weather_type="FORECAST_HOURLY")
        assert get.call_count == 1
        assert extractor.served_combinations()["icon_eu"] == SERVED["icon_eu"]

    def test_falls_back_to_static_check_when_models_unreachable(self, extractor):
        extractor._served = None
        with patch(GET, side_effect=requests.exceptions.ConnectionError("down")):
            extractor.setup_weatherflow_parameters(weather_type="FORECAST_HOURLY", model="icon_eu")
            with pytest.raises(ValueError, match="Forecasts need one of"):
                extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY", model="era5")

    def test_hourly_cache_key_is_the_instant(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="HISTORICAL_HOURLY")
        assert extractor.cache_key_columns == ["id", "date_utc"]
        extractor.setup_weatherflow_parameters()
        assert extractor.cache_key_columns == ["id", "date"]

    def test_invalid_weather_type(self, extractor):
        with pytest.raises(ValueError, match="Invalid weather_type"):
            extractor.setup_weatherflow_parameters(weather_type="DAILY")

    def test_invalid_model(self, extractor):
        with pytest.raises(ValueError, match="Invalid model"):
            extractor.setup_weatherflow_parameters(model="gfs")

    def test_weather_parameters_must_be_names(self, extractor):
        with pytest.raises(ValueError, match="weather_parameters"):
            extractor.setup_weatherflow_parameters(weather_parameters=[1, 2])


# ---------------------------------------------------------------------------
# Date chunks
# ---------------------------------------------------------------------------


class TestDateChunks:
    def test_short_range_is_one_call(self):
        assert date_chunks("2025-01-01", "2025-12-31") == [("2025-01-01", "2025-12-31")]

    def test_long_range_is_split_without_gaps_or_overlap(self):
        chunks = date_chunks("2023-01-01", "2025-06-30")
        assert chunks[0] == ("2023-01-01", "2024-01-01")  # 366 days inclusive
        assert chunks[-1][1] == "2025-06-30"
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            assert pd.Timestamp(next_start) - pd.Timestamp(prev_end) == pd.Timedelta(days=1)

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="before start_date"):
            date_chunks("2025-02-01", "2025-01-01")


class TestMissingDays:
    def _rows(self, *days):
        return [{"date": d} for d in days]

    def test_complete_series_has_no_gap(self):
        assert (
            missing_day_ranges(self._rows("2025-06-01", "2025-06-02", "2025-06-03"), "2025-06-01", "2025-06-03") == []
        )

    def test_gaps_are_grouped_into_runs(self):
        rows = self._rows("2023-09-09", "2023-09-10", "2023-09-14", "2023-09-16")
        assert missing_day_ranges(rows, "2023-09-09", "2023-09-16") == [
            ("2023-09-11", "2023-09-13"),
            ("2023-09-15", "2023-09-15"),
        ]

    def test_missing_first_days_count(self):
        assert missing_day_ranges(self._rows("2025-06-03"), "2025-06-01", "2025-06-03") == [
            ("2025-06-01", "2025-06-02")
        ]

    def test_early_end_is_not_a_gap(self):
        # Reported by the "ends on" warning instead.
        assert missing_day_ranges(self._rows("2025-06-01", "2025-06-02"), "2025-06-01", "2025-06-10") == []

    def test_hourly_rows_count_for_their_local_day(self):
        rows = self._rows("2025-03-09T01:00:00-06:00", "2025-03-09T23:00:00-05:00", "2025-03-11T00:00:00-05:00")
        assert missing_day_ranges(rows, "2025-03-09", "2025-03-11") == [("2025-03-10", "2025-03-10")]

    def test_no_rows_is_not_reported_as_a_gap(self):
        assert missing_day_ranges([], "2025-06-01", "2025-06-03") == []


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestRequest:
    def test_historical_request(self, configured):
        with patch(GET, return_value=_resp(200, DAILY_ROWS)) as get:
            rows = configured.get_weatherflow_data(_entity())
        assert rows == DAILY_ROWS
        url = get.call_args.args[0]
        kwargs = get.call_args.kwargs
        assert url == "https://weatherflow.aws.geosys.com/prod/models/era5/historical"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        assert "Authorization" not in kwargs["headers"]
        q = kwargs["params"]
        assert (q["start"], q["end"], q["aggregation"]) == ("2025-06-01", "2025-06-02", "daily")
        assert q["latitude"] == pytest.approx(42.03) and q["longitude"] == pytest.approx(-93.63)

    def test_forecast_path_and_aggregation(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_HOURLY", model="icon_eu")
        with patch(GET, return_value=_resp(200, [])) as get:
            extractor.get_weatherflow_data(_entity())
        assert get.call_args.args[0].endswith("/models/icon_eu/forecast")
        assert get.call_args.kwargs["params"]["aggregation"] == "hourly"

    def test_hourly_ranges_are_split_into_10_day_calls(self, extractor):
        """Hourly calls cover at most 10 local days; a longer one gets a 400 from the API."""
        extractor.setup_weatherflow_parameters(weather_type="HISTORICAL_HOURLY")
        with patch(GET, return_value=_resp(200, [])) as get:
            extractor.get_weatherflow_data(_entity(start_date="2025-06-01", end_date="2025-06-25"))
        spans = [(c.kwargs["params"]["start"], c.kwargs["params"]["end"]) for c in get.call_args_list]
        assert spans == [("2025-06-01", "2025-06-10"), ("2025-06-11", "2025-06-20"), ("2025-06-21", "2025-06-25")]

    def test_400_is_not_retried(self, configured):
        with patch(GET, return_value=_resp(400, {"detail": "split the range"})) as get:
            out = configured.process_single_entity_weatherflow(_entity())
        assert out["error"] is not None and get.call_count == 1

    def test_long_range_calls_once_per_chunk(self, configured):
        with patch(GET, return_value=_resp(200, DAILY_ROWS[:1])) as get:
            rows = configured.get_weatherflow_data(_entity(start_date="2023-01-01", end_date="2025-06-30"))
        assert get.call_count == 3
        assert len(rows) == 3

    def test_204_is_no_rows(self, configured):
        with patch(GET, return_value=_resp(204)):
            assert configured.get_weatherflow_data(_entity()) == []

    def test_422_raises(self, configured):
        with patch(GET, return_value=_resp(422, {"detail": "bad"})):
            with pytest.raises(requests.exceptions.HTTPError):
                configured.get_weatherflow_data(_entity())

    def test_forecast_start_before_local_today_explains(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY")
        body = {"detail": "start must be >= today at the location (2026-10-10)"}
        with patch(GET, return_value=_resp(400, body)) as get:
            out = extractor.process_single_entity_weatherflow(_entity())
        assert "LOCAL date" in out["error"]["message"] and "2026-10-10" in out["error"]["message"]
        assert get.call_count == 1  # still a 400: not retried

    def test_short_response_is_warned(self, configured):
        configured.logger = Mock()
        with patch(GET, return_value=_resp(200, DAILY_ROWS[:1])):  # asked 06-01..06-02, got 06-01
            configured.get_weatherflow_data(_entity())
        warnings = " ".join(str(c.args[0]) for c in configured.logger.warning.call_args_list)
        assert "ends on 2025-06-01, before the requested end 2025-06-02" in warnings

    def test_missing_days_inside_the_range_are_warned(self, configured):
        configured.logger = Mock()
        rows = [{**DAILY_ROWS[0], "date": d} for d in ("2023-09-10", "2023-12-08")]
        with patch(GET, return_value=_resp(200, rows)):
            out = configured.get_weatherflow_data(_entity(start_date="2023-09-10", end_date="2023-12-08"))
        assert len(out) == 2  # rows are returned as sent; only a warning is added
        warnings = " ".join(str(c.args[0]) for c in configured.logger.warning.call_args_list)
        assert "no data for 88 day(s)" in warnings and "2023-09-11 to 2023-12-07" in warnings

    def test_full_response_is_not_warned(self, configured):
        configured.logger = Mock()
        with patch(GET, return_value=_resp(200, DAILY_ROWS)):
            configured.get_weatherflow_data(_entity())
        configured.logger.warning.assert_not_called()

    def test_list_and_describe_models(self, extractor):
        with patch(GET, return_value=_resp(200, {"models": [{"name": "era5"}]})) as get:
            assert extractor.list_models() == [{"name": "era5"}]
        assert get.call_args.args[0].endswith("/models")
        with patch(GET, return_value=_resp(200, {"name": "era5", "combinations": []})) as get:
            assert extractor.describe_model("era5")["name"] == "era5"
        assert get.call_args.args[0].endswith("/models/era5")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormat:
    def test_weather_v1_names_by_default_metadata_first(self, configured):
        df = configured.format_weatherflow_json(DAILY_ROWS, entity_data=_entity())
        assert list(df.columns[:5]) == ["entity_id", "date", "geohash", "model", "aggregation_timezone"]
        assert {"temperature.standardMax", "precipitation.cumulative", "soil.moisture0to7cm"} <= set(df.columns)
        assert "temperature_2m_max" not in df.columns

    def test_api_names_on_request(self, extractor):
        extractor.setup_weatherflow_parameters(naming="api")
        df = extractor.format_weatherflow_json(DAILY_ROWS, entity_data=_entity())
        assert {"temperature_2m_max", "precipitation_sum", "swvl1"} <= set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].iloc[0] == pd.Timestamp("2025-06-01")

    def test_selected_parameters_only(self, configured):
        meta = ("entity_id", "date", "geohash", "model", "aggregation_timezone")
        for requested in (["precipitation_sum", "not_a_column"], ["precipitation.cumulative"]):  # either naming
            configured.weatherflow_params["weather_parameters"] = requested
            df = configured.format_weatherflow_json(DAILY_ROWS, entity_data=_entity())
            assert [c for c in df.columns if c not in meta] == ["precipitation.cumulative"]

    def test_hourly_dates_are_local_wall_time_across_dst(self, configured):
        df = configured.format_weatherflow_json(HOURLY_ROWS, entity_data=_entity())
        assert list(df["date"]) == [pd.Timestamp("2025-03-09 01:00"), pd.Timestamp("2025-03-09 03:00")]

    def test_daily_rows_have_no_date_utc(self, configured):
        assert "date_utc" not in configured.format_weatherflow_json(DAILY_ROWS, entity_data=_entity()).columns

    def test_hourly_rows_get_date_utc(self, configured):
        df = configured.format_weatherflow_json(HOURLY_ROWS, entity_data=_entity())
        assert list(df.columns[:3]) == ["entity_id", "date", "date_utc"]
        assert str(df["date_utc"].dt.tz) == "UTC"
        assert list(df["date_utc"]) == [
            pd.Timestamp("2025-03-09 07:00", tz="UTC"),
            pd.Timestamp("2025-03-09 08:00", tz="UTC"),
        ]

    def test_dst_end_repeated_hour_stays_distinct_and_ordered(self, configured):
        df = configured.format_weatherflow_json(list(reversed(DST_END_ROWS)), entity_data=_entity())
        assert list(df["date"].dt.strftime("%H:%M")) == ["01:00", "02:00", "02:00", "03:00"]  # labels repeat
        assert df["date_utc"].is_unique and df["date_utc"].is_monotonic_increasing
        assert list(df["temperature.standard"]) == [9.0, 8.5, 8.1, 7.9]  # chronological, both 02:00 hours kept

    def test_empty_response(self, configured):
        assert configured.format_weatherflow_json([], entity_data=_entity()).empty


# ---------------------------------------------------------------------------
# Single entity and bulk
# ---------------------------------------------------------------------------


class TestProcessing:
    def test_single_entity_keeps_input_columns(self, configured):
        with patch(GET, return_value=_resp(200, DAILY_ROWS)):
            out = configured.process_single_entity_weatherflow(_entity(farm="North"))
        df = out["data"]
        assert out["error"] is None
        assert (df["farm"] == "North").all() and len(df) == 2

    def test_503_is_retried_after_retry_after(self, configured):
        responses = [_resp(503, retry_after="0"), _resp(200, DAILY_ROWS)]
        with patch(GET, side_effect=responses) as get:
            out = configured.process_single_entity_weatherflow(_entity())
        assert out["error"] is None and get.call_count == 2

    def test_kpi(self, configured):
        configured.setup_weatherflow_parameters(
            kpi_filter={"kpi_name": "rain", "aggregation": "accumulation", "value_column": "precipitation.cumulative"}
        )
        with patch(GET, return_value=_resp(200, DAILY_ROWS)):
            out = configured.process_single_entity_weatherflow(_entity())
        df = out["data"]
        assert df["kpi_column"].iloc[0] == "precipitation.cumulative"
        assert df["current_value"].iloc[0] == pytest.approx(0.6)

    def test_bulk_keeps_input_columns(self, configured):
        entities = pd.DataFrame([_entity(farm="North"), {**_entity(farm="South"), "id": "f2"}])
        with patch(GET, return_value=_resp(200, DAILY_ROWS)):
            out = configured.process_entity_weatherflow_bulk_parallel(entities, skip_export=True, max_workers=1)
        df = out["results_df"]
        assert out["successful_calculations"] == 2
        assert set(df["farm"]) == {"North", "South"} and "geometry" not in df.columns

    @pytest.mark.parametrize("use_cache", [False, True])
    @pytest.mark.parametrize("spatial_grouping", [False, True])
    def test_every_hour_of_every_day_survives(self, extractor, tmp_path, spatial_grouping, use_cache):
        """Two full local days around the DST end (24 + 25 hours), two fields in one cell.

        Guards three bugs found 2026-10-09: spatial grouping clipped windows at midnight of the
        last day (25 rows instead of 49), and the entity and spatial caches deduplicated on the
        local `date`, which repeats when daylight saving ends.
        """
        extractor.setup_weatherflow_parameters(weather_type="HISTORICAL_HOURLY", use_cache=use_cache)
        extractor.output_path = extractor.partial_path = str(tmp_path)
        entities = pd.DataFrame(
            [{"id": i, "geometry": FIELD, "start_date": "2025-10-25", "end_date": "2025-10-26"} for i in ("f1", "f2")]
        )
        kwargs = dict(skip_export=True, max_workers=1, spatial_grouping=spatial_grouping, use_cache=use_cache)
        runs = []
        for _ in range(2):
            with patch(GET, return_value=_resp(200, TWO_DAYS_HOURLY)) as get:
                runs.append(
                    (
                        extractor.process_entity_weatherflow_bulk_parallel(entities, **kwargs)["results_df"],
                        get.call_count,
                    )
                )
        for df, _calls in runs:
            assert df.groupby("entity_id").size().to_dict() == {"f1": 49, "f2": 49}
            assert df.groupby("entity_id")["date_utc"].nunique().to_dict() == {"f1": 49, "f2": 49}
            assert df["date"].max() == pd.Timestamp("2025-10-26 23:00")
        expected_first = 1 if spatial_grouping else 2
        assert runs[0][1] == expected_first
        assert runs[1][1] == (0 if use_cache else expected_first)

    def test_spatial_grouping_calls_once_per_cell(self, configured):
        """Two fields in the same geohash-5 cell: one API call, both fields in the output."""
        entities = pd.DataFrame([_entity(), {**_entity(), "id": "f2"}])
        with patch(GET, return_value=_resp(200, DAILY_ROWS)) as get:
            out = configured.process_entity_weatherflow_bulk_parallel(
                entities, skip_export=True, max_workers=1, spatial_grouping=True
            )
        assert get.call_count == 1
        assert set(out["results_df"]["entity_id"]) == {"f1", "f2"}


# One ERA5 day (m/s, MJ/m², mm water equivalent) and one IFS forecast day (km/h, cm of fresh snow).
ERA5_DAY = {
    "date": "2025-06-01",
    "geohash": "9zmrj",
    "aggregation_timezone": "America/Chicago",
    "model": "era5",
    "temperature_2m_max": 27.9,
    "precipitation_sum": 25.4,
    "wind_speed_10m_max": 5.0,
    "shortwave_radiation_sum": 8.64,
    "snowfall_sum": 2.0,
    "swvl1": 0.25,
}
IFS_FORECAST_DAY = {
    "date": "2026-10-12",
    "geohash": "9zmrj",
    "aggregation_timezone": "America/Chicago",
    "model": "ifs",
    "temperature_2m_max": 20.0,
    "wind_speed_10m_max": 18.0,
    "snowfall_sum": 3.0,
}


class TestNormalization:
    def _format(self, extractor, rows, **setup):
        extractor.setup_weatherflow_parameters(**setup)
        return extractor.format_weatherflow_json(rows, entity_data=_entity())

    def test_metric_converts_wind_and_radiation_only(self, extractor):
        df = self._format(extractor, [ERA5_DAY])
        row = df.iloc[0]
        assert row["temperature.standardMax"] == pytest.approx(27.9)  # °C already
        assert row["precipitation.cumulative"] == pytest.approx(25.4)  # mm already
        assert row["wind.speed.maxAt10M"] == pytest.approx(18.0)  # 5 m/s → km/h
        assert row["solarRadiation.surfaceDownwardShortWave"] == pytest.approx(100.0)  # 8.64 MJ/m²/day → W/m²
        assert row["snow.waterEquivalent"] == pytest.approx(2.0)  # ERA5 snowfall is mm of water
        assert extractor.output_units["wind.speed.maxAt10M"] == "km/h"
        assert extractor.output_units["solarRadiation.surfaceDownwardShortWave"] == "W/m²"
        assert df.attrs["units"]["temperature.standardMax"] == "°C"

    def test_imperial(self, extractor):
        row = self._format(extractor, [ERA5_DAY], units="imperial").iloc[0]
        assert row["temperature.standardMax"] == pytest.approx(82.22, abs=0.01)  # a value: × 9/5 + 32
        assert row["precipitation.cumulative"] == pytest.approx(1.0)  # 25.4 mm
        assert row["wind.speed.maxAt10M"] == pytest.approx(11.18, abs=0.01)  # mph
        assert row["solarRadiation.surfaceDownwardShortWave"] == pytest.approx(100.0)  # W/m² in both profiles
        assert extractor.output_units["temperature.standardMax"] == "°F"
        assert extractor.output_units["precipitation.cumulative"] == "in"

    def test_native_api_names_is_the_raw_response(self, extractor):
        df = self._format(extractor, [ERA5_DAY], units="native", naming="api")
        assert df.iloc[0]["wind_speed_10m_max"] == pytest.approx(5.0)
        assert extractor.output_units["wind_speed_10m_max"] == "m/s"

    def test_history_and_forecast_share_one_unit(self, extractor):
        """ERA5 sends wind in m/s, the IFS forecast in km/h: after metric both are km/h."""
        hist = self._format(extractor, [ERA5_DAY])
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY")
        fore = extractor.format_weatherflow_json([IFS_FORECAST_DAY], entity_data=_entity())
        assert hist.iloc[0]["wind.speed.maxAt10M"] == pytest.approx(18.0)
        assert fore.iloc[0]["wind.speed.maxAt10M"] == pytest.approx(18.0)

    def test_fresh_snow_is_never_turned_into_water(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY")
        df = extractor.format_weatherflow_json([IFS_FORECAST_DAY], entity_data=_entity())
        assert "snow.freshSnow" in df.columns and "snow.waterEquivalent" not in df.columns
        assert df.iloc[0]["snow.freshSnow"] == pytest.approx(3.0)
        assert extractor.output_units["snow.freshSnow"] == "cm"

    def test_unknown_units_are_kept_as_sent(self, extractor):
        extractor._column_units = {}
        extractor.logger = Mock()
        with patch.object(WeatherFlowExtractor, "describe_model", return_value=[]):
            row = self._format(extractor, [ERA5_DAY]).iloc[0]
        assert row["wind.speed.maxAt10M"] == pytest.approx(5.0)  # not converted from a guessed unit
        assert "no unit known" in " ".join(str(c.args[0]) for c in extractor.logger.warning.call_args_list)

    def test_output_units_describe_the_current_setup_only(self, extractor):
        extractor.setup_weatherflow_parameters(weather_type="FORECAST_DAILY")
        extractor.format_weatherflow_json([IFS_FORECAST_DAY], entity_data=_entity())
        assert "snow.freshSnow" in extractor.output_units
        self._format(extractor, [ERA5_DAY])
        assert "snow.freshSnow" not in extractor.output_units

    def test_units_reach_the_manifest(self, extractor):
        self._format(extractor, [ERA5_DAY], units="imperial")
        meta = extractor.manifest_metadata
        assert meta["unit_profile"] == "imperial" and meta["naming"] == "weather_v1"
        assert meta["units"]["temperature.standardMax"] == "°F"

    def test_invalid_profile_and_naming(self, extractor):
        with pytest.raises(ValueError, match="Invalid units"):
            extractor.setup_weatherflow_parameters(units="si")
        with pytest.raises(ValueError, match="Invalid naming"):
            extractor.setup_weatherflow_parameters(naming="v2")

    def test_every_vocabulary_kind_has_a_unit_in_each_profile(self):
        from earthdaily.agriculture.core.units import PROFILES, UNCHANGED_KINDS

        kinds = {kind for _, kind in wf.VOCABULARY.values()} | {"fresh_snow", "snow_water_equivalent"}
        for profile in PROFILES.values():
            assert kinds - UNCHANGED_KINDS - set(profile) == set()


def test_module_mirrors_weather_v1_pipeline():
    """Same method shape as WeatherExtractor, so a caller switches by renaming."""
    for name in (
        "setup_weatherflow_parameters",
        "get_weatherflow_data",
        "get_weatherflow_data_safe",
        "format_weatherflow_json",
        "process_single_entity_weatherflow",
        "process_entity_weatherflow_bulk_parallel",
    ):
        assert callable(getattr(WeatherFlowExtractor, name)), name
    assert wf.MAX_DAYS_PER_CALL == {"daily": 366, "hourly": 10}
