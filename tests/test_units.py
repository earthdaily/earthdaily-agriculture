"""core.units: pint-backed conversions behind the metric / imperial output profiles."""

import numpy as np
import pandas as pd
import pytest

from earthdaily.agriculture.core.units import convert, display, pint_unit, target_unit

pytestmark = pytest.mark.public


def test_temperature_values_use_the_offset():
    values, unit = convert(pd.Series([0.0, 20.0, -40.0]), "degC", "temperature", "imperial")
    assert list(values) == pytest.approx([32.0, 68.0, -40.0]) and unit == "°F"


def test_radiation_sum_becomes_mean_irradiance_over_the_period():
    values, unit = convert(pd.Series([8.64]), "MJ/m2", "radiation_sum", "metric", period="day")
    assert values.iloc[0] == pytest.approx(100.0) and unit == "W/m²"


def test_radiation_sum_needs_a_period():
    with pytest.raises(ValueError, match="period"):
        convert(pd.Series([1.0]), "MJ/m2", "radiation_sum", "metric")


def test_nan_and_index_survive():
    series = pd.Series([np.nan, 10.0], index=[7, 9])
    values, _ = convert(series, "km/h", "wind_speed", "imperial")
    assert list(values.index) == [7, 9] and np.isnan(values.iloc[0])
    assert values.iloc[1] == pytest.approx(6.2137, abs=1e-4)


@pytest.mark.parametrize("kind", ["relative", "volumetric", "direction", "duration", "code"])
def test_kinds_without_profile_units_are_unchanged(kind):
    series = pd.Series([1.5])
    values, _ = convert(series, "%", kind, "imperial")
    assert values is series


def test_native_profile_is_a_no_op():
    series = pd.Series([5.0])
    values, unit = convert(series, "m/s", "wind_speed", "native")
    assert values is series and unit == "m/s"


def test_api_labels():
    assert pint_unit("mm (water equivalent)") == "mm"
    assert pint_unit("cm (fresh snow)") == "cm"
    assert pint_unit("WMO code") is None
    assert pint_unit("not a unit at all") is None
    assert display("W/m**2") == "W/m²" and display(None, "WMO code") == "WMO code"


def test_fresh_snow_and_water_equivalent_are_different_kinds():
    assert target_unit("fresh_snow", "metric") == "cm"
    assert target_unit("snow_water_equivalent", "metric") == "mm"


# MDM unit profiles EU and US, weather categories (GET /master-data-management/v6/unit-profiles/{id},
# prod, 2026-10-10) → the kinds they govern. `metric` / `imperial` must match them.
MDM_WEATHER = {
    "EU": {
        "WEATHER_TEMPERATURE": "degC",
        "PRECIPITATION": "mm",
        "SNOW_DEPTH": "cm",
        "WIND_SPEED": "km/h",
        "SOLAR_RADIATION": "W/m**2",
    },
    "US": {
        "WEATHER_TEMPERATURE": "degF",
        "PRECIPITATION": "inch",
        "SNOW_DEPTH": "inch",
        "WIND_SPEED": "mph",
        "SOLAR_RADIATION": "W/m**2",
    },
}
CATEGORY_KINDS = {
    "WEATHER_TEMPERATURE": ["temperature"],
    "PRECIPITATION": ["precipitation", "snow_water_equivalent"],
    "SNOW_DEPTH": ["snow_depth", "fresh_snow"],
    "WIND_SPEED": ["wind_speed"],
    "SOLAR_RADIATION": ["irradiance", "radiation_sum"],
}


@pytest.mark.parametrize("profile, mdm", [("metric", "EU"), ("imperial", "US")])
def test_profiles_match_the_mdm_unit_profiles(profile, mdm):
    for category, unit in MDM_WEATHER[mdm].items():
        for kind in CATEGORY_KINDS[category]:
            assert target_unit(kind, profile) == unit, (profile, kind)


def test_metric_snow_depth_is_cm():
    values, unit = convert(pd.Series([0.25]), "m", "snow_depth", "metric")
    assert values.iloc[0] == pytest.approx(25.0) and unit == "cm"
