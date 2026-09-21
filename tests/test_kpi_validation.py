"""
Tests for the centralized validate_kpi_filter helper in earthdaily.agriculture.core.api_utils.

This is the single source of truth for KPI parameter validation; every
KPI-supporting extractor delegates to it. Tests cover:
  - Top-level kpi_filter shape (None → no-op, non-dict, missing/invalid aggregation)
  - Threshold contract per aggregation (required when needed, rejected when not,
    type / range checks)
  - Window contract for rolling_avg* aggregations
  - A drift guard: every entry in KPI_AGGREGATION_RULES has a happy path covered
"""

import pytest

from earthdaily.agriculture.core.api_utils import (
    KPI_AGGREGATION_RULES,
    validate_kpi_filter,
)

pytestmark = pytest.mark.public

# ===================================================================
# Top-level shape
# ===================================================================


class TestKpiFilterShape:
    def test_none_is_noop(self):
        validate_kpi_filter(None)  # must not raise

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="dictionary"):
            validate_kpi_filter("bogus")

    def test_missing_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation"):
            validate_kpi_filter({"kpi_name": "x"})

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Invalid aggregation"):
            validate_kpi_filter({"aggregation": "bogus_agg"})


# ===================================================================
# Coverage: every aggregation in the rule table has a happy path
# ===================================================================


HAPPY_CASES = {
    "accumulation": {},
    "average": {},
    "max": {},
    "min": {},
    "std": {},
    "top_accumulation": {"threshold": 30},
    "count_gt": {"threshold": 0.5},
    "count_lt": {"threshold": 0.5},
    "count_between": {"threshold": (0.2, 0.8)},
    "rolling_avg": {"window": 5},
    "rolling_avg_gt": {"window": 5, "threshold": 0.6},
    "rolling_avg_lt": {"window": 5, "threshold": 0.4},
    "percentile": {"threshold": 90},
    "percentile_gt": {"threshold": 90},
    "percentile_lt": {"threshold": 10},
}


def test_happy_cases_cover_every_supported_aggregation():
    """Drift guard — adding to KPI_AGGREGATION_RULES means adding a HAPPY_CASES row."""
    assert set(HAPPY_CASES.keys()) == set(KPI_AGGREGATION_RULES.keys())


@pytest.mark.parametrize("agg, extra", list(HAPPY_CASES.items()))
def test_happy_path(agg, extra):
    validate_kpi_filter({"aggregation": agg, "kpi_name": "test", **extra})


# ===================================================================
# Threshold contract
# ===================================================================


THRESHOLD_REQUIRED = [
    "count_gt",
    "count_lt",
    "count_between",
    "top_accumulation",
    "rolling_avg_gt",
    "rolling_avg_lt",
    "percentile",
    "percentile_gt",
    "percentile_lt",
]
THRESHOLD_NOT_SUPPORTED = ["accumulation", "average", "max", "min", "std", "rolling_avg"]


def _with_required_window(cfg):
    """Add window=3 if the aggregation needs it, so we don't trip the window check."""
    if cfg["aggregation"] in {"rolling_avg", "rolling_avg_gt", "rolling_avg_lt"}:
        cfg["window"] = 3
    return cfg


class TestThresholdRequired:
    @pytest.mark.parametrize("agg", THRESHOLD_REQUIRED)
    def test_missing_threshold_raises(self, agg):
        with pytest.raises(ValueError, match="threshold"):
            validate_kpi_filter(_with_required_window({"aggregation": agg}))

    @pytest.mark.parametrize("agg", THRESHOLD_REQUIRED)
    def test_threshold_none_is_treated_as_missing(self, agg):
        with pytest.raises(ValueError, match="threshold"):
            validate_kpi_filter(_with_required_window({"aggregation": agg, "threshold": None}))


class TestThresholdNotSupported:
    @pytest.mark.parametrize("agg", THRESHOLD_NOT_SUPPORTED)
    def test_unwanted_threshold_raises(self, agg):
        with pytest.raises(ValueError, match="threshold"):
            validate_kpi_filter(_with_required_window({"aggregation": agg, "threshold": 0.5}))

    @pytest.mark.parametrize("agg", THRESHOLD_NOT_SUPPORTED)
    def test_threshold_none_is_accepted(self, agg):
        validate_kpi_filter(_with_required_window({"aggregation": agg, "threshold": None}))


class TestNumericThreshold:
    @pytest.mark.parametrize("agg", ["count_gt", "count_lt", "rolling_avg_gt", "rolling_avg_lt"])
    def test_string_threshold_raises(self, agg):
        with pytest.raises(ValueError, match="numeric"):
            validate_kpi_filter(_with_required_window({"aggregation": agg, "threshold": "high"}))


class TestTopAccumulationThreshold:
    @pytest.mark.parametrize("bad", [0, -1, 0.5, "ten"])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            validate_kpi_filter({"aggregation": "top_accumulation", "threshold": bad})


class TestCountBetweenThreshold:
    def test_non_tuple_raises(self):
        with pytest.raises(ValueError, match="tuple"):
            validate_kpi_filter({"aggregation": "count_between", "threshold": 0.5})

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="tuple"):
            validate_kpi_filter({"aggregation": "count_between", "threshold": (0.1, 0.2, 0.3)})

    def test_min_ge_max_raises(self):
        with pytest.raises(ValueError, match="max"):
            validate_kpi_filter({"aggregation": "count_between", "threshold": (0.8, 0.2)})

    def test_non_numeric_values_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            validate_kpi_filter({"aggregation": "count_between", "threshold": ("a", "b")})


class TestPercentileThreshold:
    @pytest.mark.parametrize("agg", ["percentile", "percentile_gt", "percentile_lt"])
    @pytest.mark.parametrize("bad", [-1, 101, 200, "ninety"])
    def test_out_of_range_or_nonnumeric_raises(self, agg, bad):
        with pytest.raises(ValueError, match="percentile rank"):
            validate_kpi_filter({"aggregation": agg, "threshold": bad})

    @pytest.mark.parametrize("agg", ["percentile", "percentile_gt", "percentile_lt"])
    @pytest.mark.parametrize("good", [0, 50, 90, 100, 25.5])
    def test_in_range_accepted(self, agg, good):
        validate_kpi_filter({"aggregation": agg, "threshold": good})


# ===================================================================
# Window contract
# ===================================================================


ROLLING_AGGS = ["rolling_avg", "rolling_avg_gt", "rolling_avg_lt"]
NON_ROLLING_AGGS = [
    "accumulation",
    "average",
    "max",
    "min",
    "std",
    "count_gt",
    "count_lt",
    "count_between",
    "top_accumulation",
    "percentile",
    "percentile_gt",
    "percentile_lt",
]


def _with_required_threshold(cfg):
    """Add a valid threshold if the aggregation needs one, so we don't trip the threshold check."""
    agg = cfg["aggregation"]
    if agg in {"count_gt", "count_lt", "rolling_avg_gt", "rolling_avg_lt"}:
        cfg["threshold"] = 0.5
    elif agg == "count_between":
        cfg["threshold"] = (0.2, 0.8)
    elif agg == "top_accumulation":
        cfg["threshold"] = 10
    elif agg.startswith("percentile"):
        cfg["threshold"] = 90
    return cfg


class TestWindowRequired:
    @pytest.mark.parametrize("agg", ROLLING_AGGS)
    def test_missing_window_raises(self, agg):
        with pytest.raises(ValueError, match="window"):
            validate_kpi_filter(_with_required_threshold({"aggregation": agg}))

    @pytest.mark.parametrize("agg", ROLLING_AGGS)
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "five"])
    def test_invalid_window_raises(self, agg, bad):
        with pytest.raises(ValueError, match="window"):
            validate_kpi_filter(_with_required_threshold({"aggregation": agg, "window": bad}))


class TestWindowNotSupported:
    @pytest.mark.parametrize("agg", NON_ROLLING_AGGS)
    def test_window_on_non_rolling_raises(self, agg):
        with pytest.raises(ValueError, match="window"):
            validate_kpi_filter(_with_required_threshold({"aggregation": agg, "window": 5}))

    @pytest.mark.parametrize("agg", NON_ROLLING_AGGS)
    def test_window_none_is_accepted(self, agg):
        validate_kpi_filter(_with_required_threshold({"aggregation": agg, "window": None}))
