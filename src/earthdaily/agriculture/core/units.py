"""Unit conversion for extractor outputs: one value set, delivered in a chosen unit profile.

An API states each column's unit (WeatherFlow: ``GET /models/{model}``), and the units differ
between models and datasets: wind is m/s in ERA5 history but km/h in the IFS forecast. A
**profile** fixes the unit of each kind of quantity, so the same column always carries the
same unit whatever model produced it:

==========================  ==================  ==================
Kind                        ``metric`` (EU)     ``imperial`` (US)
==========================  ==================  ==================
temperature                 °C                  °F
precipitation, snow (w.e.)  mm                  in
fresh snow                  cm                  in
snow depth                  cm                  in
wind speed                  km/h                mph
irradiance, radiation sum   W/m² (mean)         W/m² (mean)
distance (visibility)       m                   mi
==========================  ==================  ==================

Percentages, volumetric soil moisture (m³/m³), directions (°), durations (h) and codes are the
same in both. ``native`` keeps the API's own units. ``metric`` and ``imperial`` match the
MDM unit profiles ``EU`` and ``US``. The tables themselves are data, in
``config/weather_normalization.py``; this module only converts.

Conversions use `pint <https://pint.readthedocs.io>`_, which handles what a hand-written table
gets wrong: a temperature *value* is ``°C × 9/5 + 32`` in °F, not ``× 9/5``. Two things are
decisions rather than conversions, and are made explicit here:

- A **radiation sum** (MJ/m² over a day) becomes the **mean irradiance** over that period
  (W/m²): divide by the period, as Weather v1 reports daily solar radiation.
- **Fresh snow** (cm of snow) and **snow water equivalent** (mm of water) are different
  quantities; neither is ever converted into the other.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd

from earthdaily.agriculture.config.weather_normalization import API_UNIT_LABELS, UNCHANGED_KINDS
from earthdaily.agriculture.config.weather_normalization import UNIT_DISPLAY as DISPLAY
from earthdaily.agriculture.config.weather_normalization import UNIT_PROFILES as PROFILES

#: Output unit profile: metric (EU), imperial (US), or the API's own units.
UnitProfile = Literal["metric", "imperial", "native"]


@lru_cache(maxsize=1)
def registry():
    """The shared pint registry (built once; importing pint is not free)."""
    import pint

    return pint.UnitRegistry()


def pint_unit(label: str | None) -> str | None:
    """The pint unit for an API unit label; None for codes, empty and unknown labels."""
    if not label:
        return None
    if label in API_UNIT_LABELS:
        return API_UNIT_LABELS[label]
    try:
        registry().Unit(label)
    except Exception:  # noqa: BLE001 - pint raises several error types for a bad label
        return None
    return label


def display(unit: str | None, fallback: str | None = None) -> str | None:
    """A unit as written in metadata (``"W/m**2"`` → ``"W/m²"``)."""
    if unit is None:
        return fallback
    return DISPLAY.get(unit, unit)


def target_unit(kind: str, profile: str) -> str | None:
    """The unit a kind of quantity takes in a profile; None when it is never converted."""
    if profile == "native" or kind in UNCHANGED_KINDS:
        return None
    return PROFILES[profile].get(kind)


def convert(
    values,
    from_label: str | None,
    kind: str,
    profile: str,
    period: str | None = None,
) -> tuple[pd.Series | np.ndarray, str | None]:
    """Convert ``values`` from an API unit label to ``profile``; returns ``(values, unit shown)``.

    ``period`` (e.g. ``"day"``) is needed for a ``radiation_sum``: the sum over that period
    becomes the mean irradiance over it. Values whose unit cannot be read, and kinds a profile
    does not convert, come back unchanged with their original label.

    Raises:
        ValueError: ``kind`` is a radiation sum and no ``period`` was given.
    """
    source = pint_unit(from_label)
    target = target_unit(kind, profile)
    if source is None or target is None:
        return values, display(source, from_label)

    index = values.index if isinstance(values, pd.Series) else None
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    quantity = registry().Quantity(numbers, source)
    if kind == "radiation_sum":
        if period is None:
            raise ValueError("a radiation sum needs the period it was summed over (e.g. 'day')")
        quantity = quantity / registry().Quantity(1, period)
    converted = quantity.to(target).magnitude
    if index is not None:
        converted = pd.Series(converted, index=index)
    return converted, display(target)
