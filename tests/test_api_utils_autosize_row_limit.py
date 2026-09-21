"""
Unit tests for earthdaily.agriculture.core.api_utils.autosize_row_limit — the helper that
sizes a daily-timeseries row limit ($limit) to cover an inclusive [start, end] span so long
multi-year ranges are not truncated to the earliest N days.
"""

from datetime import date, datetime

import pytest

from earthdaily.agriculture.core.api_utils import autosize_row_limit

pytestmark = pytest.mark.public


class TestAutosizeRowLimit:
    def test_single_season_hits_floor(self):
        """A ~123-day season stays at the 1000 floor (existing behavior preserved)."""
        assert autosize_row_limit("2025-06-01", "2025-10-01") == 1000

    def test_short_forecast_hits_floor(self):
        """A 10-day forecast stays at the 1000 floor."""
        assert autosize_row_limit("2025-06-01", "2025-06-10") == 1000

    def test_multi_year_span_exceeds_floor(self):
        """A 4-year span (1461 days incl. one leap day) → 1461 + 366 buffer = 1827."""
        # 2020 is a leap year: 366 + 365 + 365 + 365 = 1461 inclusive days.
        assert autosize_row_limit("2020-01-01", "2023-12-31") == 1461 + 366

    def test_matches_formula_for_reported_repro(self):
        """The verified repro range must produce a limit that covers its full span."""
        span = (date(2026, 6, 30) - date(2020, 5, 8)).days + 1
        assert autosize_row_limit("2020-05-08", "2026-06-30") == max(1000, span + 366)

    def test_accepts_date_and_datetime_objects(self):
        """date / datetime inputs behave the same as ISO strings."""
        from_str = autosize_row_limit("2020-01-01", "2023-12-31")
        from_date = autosize_row_limit(date(2020, 1, 1), date(2023, 12, 31))
        from_dt = autosize_row_limit(datetime(2020, 1, 1), datetime(2023, 12, 31, 23, 59))
        assert from_str == from_date == from_dt

    def test_reversed_range_clamps_to_floor(self):
        """end < start → span clamped to 1 day, so the floor applies (no negative limit)."""
        assert autosize_row_limit("2026-06-30", "2020-05-08") == 1000

    def test_custom_floor_and_buffer(self):
        """floor and buffer are overridable."""
        # 123-day season, floor lowered below the span+buffer so the computed value wins.
        assert autosize_row_limit("2025-06-01", "2025-10-01", floor=100, buffer=10) == 123 + 10
        # floor still wins when it is the larger of the two.
        assert autosize_row_limit("2025-06-01", "2025-10-01", floor=5000, buffer=10) == 5000
