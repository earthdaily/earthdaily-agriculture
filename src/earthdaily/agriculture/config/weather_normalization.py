"""
Weather output normalization: column names and unit profiles, as data.

Read by ``extractors.weatherflow_functions`` (names) and ``core.units`` (units). Like the rest of
``config/``, this holds data only; the logic that applies it lives in those modules.

Two tables drive WeatherFlow's normalized output:

- ``WEATHERFLOW_VOCABULARY``: each WeatherFlow API column → its Weather v1
  ``parameter.subParameter`` name (``naming="weather_v1"``) and the kind of quantity it holds.
  The column's *source* unit is not stored here: it is read from the API at run time
  (``GET /models/{model}``), because the same column differs between models (wind is m/s in
  ERA5 history, km/h in the IFS forecast).
- ``UNIT_PROFILES``: each kind of quantity → its unit in the ``metric`` and ``imperial``
  profiles (pint unit strings).

To support a new API column: add it to ``WEATHERFLOW_VOCABULARY`` with its Weather v1 name (or
None to keep the API name) and its kind. A new kind needs a unit in every profile of
``UNIT_PROFILES`` (or an entry in ``UNCHANGED_KINDS``); ``tests/test_weatherflow_functions.py``
checks that. A new API unit label needs an entry in ``API_UNIT_LABELS``.

Last updated: 2026-10-10 (vocabulary from GET /models on prod, 2026-10-09).
"""

# ---------------------------------------------------------------------------
# Column names: WeatherFlow API column → (Weather v1 name, kind of quantity)
# ---------------------------------------------------------------------------

#: API column → (Weather v1 name, kind of quantity). A None name keeps the API's name: Weather
#: v1 has no equivalent, and only soil moisture and precipitation hours get extended names.
#: The kind picks the unit a profile gives the column (``UNIT_PROFILES`` below).
WEATHERFLOW_VOCABULARY: dict[str, tuple[str | None, str]] = {
    # --- daily
    "temperature_2m_mean": ("temperature.standard", "temperature"),
    "temperature_2m_max": ("temperature.standardMax", "temperature"),
    "temperature_2m_min": ("temperature.standardMin", "temperature"),
    "precipitation_sum": ("precipitation.cumulative", "precipitation"),
    "precipitation_hours": ("precipitation.hours", "duration"),
    "rain_sum": (None, "precipitation"),
    "showers_sum": (None, "precipitation"),
    "snowfall_water_equivalent_sum": ("snow.waterEquivalent", "snow_water_equivalent"),
    "snow_depth_max": (None, "snow_depth"),
    "relative_humidity_2m_mean": ("humidity.relative", "relative"),
    "relative_humidity_2m_max": ("humidity.relativeMax", "relative"),
    "relative_humidity_2m_min": ("humidity.relativeMin", "relative"),
    "cloud_cover_mean": ("cloudCover.value", "relative"),
    "cloud_cover_max": ("cloudCover.max", "relative"),
    "cloud_cover_min": ("cloudCover.min", "relative"),
    "wind_speed_10m_mean": ("wind.speed.at10M", "wind_speed"),
    "wind_speed_10m_max": ("wind.speed.maxAt10M", "wind_speed"),
    "wind_speed_10m_min": ("wind.speed.minAt10M", "wind_speed"),
    "wind_gusts_10m_mean": ("wind.gust.at10M", "wind_speed"),
    "wind_gusts_10m_max": ("wind.gust.maxAt10M", "wind_speed"),
    "wind_gusts_10m_min": (None, "wind_speed"),
    "wind_direction_10m_dominant": ("wind.direction.at10M", "direction"),
    "shortwave_radiation_sum": ("solarRadiation.surfaceDownwardShortWave", "radiation_sum"),
    "direct_radiation_sum": (None, "radiation_sum"),
    "diffuse_radiation_sum": (None, "radiation_sum"),
    # ECMWF soil layers (ERA5, IFS, seamless): 0-7, 7-28, 28-100 and 100-289 cm.
    "swvl1": ("soil.moisture0to7cm", "volumetric"),
    "swvl2": ("soil.moisture7to28cm", "volumetric"),
    "swvl3": ("soil.moisture28to100cm", "volumetric"),
    "swvl4": ("soil.moisture100to289cm", "volumetric"),
    "soil_moisture_0_to_1cm_mean": ("soil.moisture0to1cm", "volumetric"),
    "soil_temperature_0cm_mean": (None, "temperature"),
    "visibility_min": (None, "distance"),
    "weather_code_max": (None, "code"),
    # --- hourly
    "temperature_2m": ("temperature.standard", "temperature"),
    "dew_point_2m": ("dewPoint.value", "temperature"),
    "relative_humidity_2m": ("humidity.relative", "relative"),
    "precipitation": ("precipitation.cumulative", "precipitation"),
    "rain": (None, "precipitation"),
    "showers": (None, "precipitation"),
    "snowfall_water_equivalent": ("snow.waterEquivalent", "snow_water_equivalent"),
    "snow_depth": ("snow.depth", "snow_depth"),
    "cloud_cover": ("cloudCover.value", "relative"),
    "cloud_cover_low": (None, "relative"),
    "cloud_cover_mid": (None, "relative"),
    "cloud_cover_high": (None, "relative"),
    "wind_speed_10m": ("wind.speed.at10M", "wind_speed"),
    "wind_gusts_10m": ("wind.gust.at10M", "wind_speed"),
    "wind_direction_10m": ("wind.direction.at10M", "direction"),
    "wind_u_component_10m": (None, "wind_speed"),
    "wind_v_component_10m": (None, "wind_speed"),
    "shortwave_radiation": ("solarRadiation.surfaceDownwardShortWave", "irradiance"),
    "direct_radiation": (None, "irradiance"),
    "diffuse_radiation": (None, "irradiance"),
    "soil_moisture_0_to_1cm": ("soil.moisture0to1cm", "volumetric"),
    "soil_temperature_0cm": (None, "temperature"),
    "visibility": (None, "distance"),
    "weather_code": (None, "code"),
}
#: Snowfall columns carry fresh snow (cm) in some models and water equivalent (mm) in others:
#: different quantities, so the name and kind follow the unit the model reports.
SNOWFALL_COLUMNS = {"snowfall_sum", "snowfall"}


# ---------------------------------------------------------------------------
# Units: kind of quantity → unit per profile, and how API unit labels read
# ---------------------------------------------------------------------------

#: Kind of quantity → unit, per profile (pint unit strings).
#:
#: The two profiles follow the MDM unit profiles (GET /master-data-management/v6/unit-profiles,
#: checked 2026-10-10): ``metric`` = ``EU`` and ``imperial`` = ``US`` for every weather category
#: they define (WEATHER_TEMPERATURE, PRECIPITATION, SNOW_DEPTH, WIND_SPEED, SOLAR_RADIATION).
#: Fresh snow uses SNOW_DEPTH's unit. Visibility has no MDM category. Other MDM profiles differ
#: (CIS/Kazakhstan wind m/s; Australia/Canada radiation J/m²; Canada snow depth mm) and are
#: not offered yet.
UNIT_PROFILES: dict[str, dict[str, str]] = {
    "metric": {
        "temperature": "degC",
        "precipitation": "mm",
        "snow_water_equivalent": "mm",
        "fresh_snow": "cm",
        "snow_depth": "cm",
        "wind_speed": "km/h",
        "irradiance": "W/m**2",
        "radiation_sum": "W/m**2",
        "distance": "m",
    },
    "imperial": {
        "temperature": "degF",
        "precipitation": "inch",
        "snow_water_equivalent": "inch",
        "fresh_snow": "inch",
        "snow_depth": "inch",
        "wind_speed": "mph",
        "irradiance": "W/m**2",
        "radiation_sum": "W/m**2",
        "distance": "mile",
    },
}

#: Kinds whose unit is the same in every profile (no conversion, label kept).
UNCHANGED_KINDS = {"relative", "volumetric", "direction", "duration", "code"}

#: Unit labels as APIs write them → pint unit strings; None = not a physical unit (a code).
API_UNIT_LABELS: dict[str, str | None] = {
    "degC": "degC",
    "degF": "degF",
    "mm": "mm",
    "mm (water equivalent)": "mm",
    "cm": "cm",
    "cm (fresh snow)": "cm",
    "m": "m",
    "meters": "m",
    "km/h": "km/h",
    "m/s": "m/s",
    "mph": "mph",
    "MJ/m2": "MJ/m**2",
    "W/m2": "W/m**2",
    "%": "percent",
    "m3/m3": "m**3/m**3",
    "degrees (meteorological)": "degree",
    "h": "hour",
    "hours": "hour",
    "WMO code": None,
}

#: How each pint unit is written in output metadata.
UNIT_DISPLAY = {
    "degC": "°C",
    "degF": "°F",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "inch": "in",
    "mile": "mi",
    "km/h": "km/h",
    "m/s": "m/s",
    "mph": "mph",
    "MJ/m**2": "MJ/m²",
    "W/m**2": "W/m²",
    "percent": "%",
    "m**3/m**3": "m³/m³",
    "degree": "°",
    "hour": "h",
}
