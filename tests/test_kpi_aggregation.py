"""
Tests for filter_timeseries_kpi — focused on the rolling and percentile
aggregations added alongside the centralized validation helper.

The pre-existing aggregations (accumulation, count_*, top_accumulation, std, …)
are exercised indirectly through test_mrts_*/test_weather_* and have not been
duplicated here.

Numeric expectations use a 30-day linear series (values 1..30) so they can be
hand-derived from the rolling-window and percentile arithmetic.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from earthdaily.agriculture.core.api_utils import filter_timeseries_kpi

pytestmark = pytest.mark.public


@pytest.fixture
def linear_series():
    """30 daily records with values 1..30 — easy to reason about by hand."""
    return pd.DataFrame(
        {
            "date": [date(2025, 4, 1) + timedelta(days=i) for i in range(30)],
            "value": list(range(1, 31)),
        }
    )


# ===================================================================
# Rolling
# ===================================================================


class TestRollingAvg:
    def test_returns_mean_of_rolling_series(self, linear_series):
        # rolling(window=5).mean() over 1..30 yields [3, 4, ..., 28] (26 points).
        # Mean of an arithmetic sequence: (3 + 28) / 2 = 15.5.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="rolling_avg",
            window=5,
        )
        assert result["current_period"]["value"] == pytest.approx(15.5, abs=0.01)

    def test_requires_window(self, linear_series):
        with pytest.raises(ValueError, match="window"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation="rolling_avg",
            )

    def test_invalid_window_raises(self, linear_series):
        with pytest.raises(ValueError, match="window"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation="rolling_avg",
                window=0,
            )


class TestRollingAvgGt:
    def test_counts_rolling_points_above_threshold(self, linear_series):
        # Rolling values: [3, 4, …, 28]. Strictly > 10 means {11..28} → 18 points.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="rolling_avg_gt",
            window=5,
            threshold=10,
        )
        assert result["current_period"]["value"] == 18

    def test_requires_window_and_threshold(self, linear_series):
        with pytest.raises(ValueError, match="window|threshold"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation="rolling_avg_gt",
                threshold=10,
            )
        with pytest.raises(ValueError, match="window|threshold"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation="rolling_avg_gt",
                window=5,
            )


class TestRollingAvgLt:
    def test_counts_rolling_points_below_threshold(self, linear_series):
        # Rolling values: [3, 4, …, 28]. Strictly < 10 means {3..9} → 7 points.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="rolling_avg_lt",
            window=5,
            threshold=10,
        )
        assert result["current_period"]["value"] == 7


# ===================================================================
# Percentile
# ===================================================================


class TestPercentile:
    def test_returns_pth_percentile_value(self, linear_series):
        # quantile(0.9) of 1..30 with linear interpolation: position = 0.9*29 = 26.1,
        # interpolated between sorted[26]=27 and sorted[27]=28 → 27.1.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="percentile",
            threshold=90,
        )
        assert result["current_period"]["value"] == pytest.approx(27.1, abs=0.01)

    def test_zero_and_hundred_are_accepted(self, linear_series):
        r0 = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="percentile",
            threshold=0,
        )
        r100 = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="percentile",
            threshold=100,
        )
        assert r0["current_period"]["value"] == 1
        assert r100["current_period"]["value"] == 30


class TestPercentileGt:
    def test_counts_strictly_above_cutoff(self, linear_series):
        # 90th percentile cutoff ≈ 27.1; values > 27.1 in 1..30: {28, 29, 30} → 3.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="percentile_gt",
            threshold=90,
        )
        assert result["current_period"]["value"] == 3


class TestPercentileLt:
    def test_counts_strictly_below_cutoff(self, linear_series):
        # 10th percentile cutoff ≈ 3.9; values < 3.9 in 1..30: {1, 2, 3} → 3.
        result = filter_timeseries_kpi(
            linear_series,
            "2025-04-01",
            "2025-04-30",
            kpi_name="t",
            aggregation="percentile_lt",
            threshold=10,
        )
        assert result["current_period"]["value"] == 3


class TestPercentileValidation:
    @pytest.mark.parametrize("agg", ["percentile", "percentile_gt", "percentile_lt"])
    def test_missing_threshold_raises(self, linear_series, agg):
        with pytest.raises(ValueError, match="percentile"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation=agg,
            )

    @pytest.mark.parametrize("agg", ["percentile", "percentile_gt", "percentile_lt"])
    @pytest.mark.parametrize("bad", [-1, 200])
    def test_out_of_range_threshold_raises(self, linear_series, agg, bad):
        with pytest.raises(ValueError, match="percentile"):
            filter_timeseries_kpi(
                linear_series,
                "2025-04-01",
                "2025-04-30",
                kpi_name="t",
                aggregation=agg,
                threshold=bad,
            )
