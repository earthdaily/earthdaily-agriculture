"""
Tests for RegionalExtractor.format_regional_json():
    - Returns a tuple (observed_df, daily_avg_df)
    - observedMeasures formatting (ISO time → date)
    - dailyAverage formatting (1MMDD dayOfYear → date)
    - String JSON parsing
    - Empty / partial responses
"""

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_regional_json() — return contract
# ===================================================================


class TestFormatRegionalJsonReturnContract:
    """The formatter must always return a 2-tuple (observed_df, daily_avg_df)."""

    def test_returns_two_dataframes(self, configured_regional_extractor, sample_regional_response):
        result = configured_regional_extractor.format_regional_json(sample_regional_response)
        assert isinstance(result, tuple)
        assert len(result) == 2
        observed_df, daily_avg_df = result
        assert isinstance(observed_df, pd.DataFrame)
        assert isinstance(daily_avg_df, pd.DataFrame)


# ===================================================================
# format_regional_json() — observedMeasures
# ===================================================================


class TestFormatRegionalJsonObservedMeasures:
    """Tests for the observedMeasures section (ISO datetime → YYYY-MM-DD)."""

    def test_observed_row_count(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        assert len(observed_df) == 3

    def test_observed_columns(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        expected = {"date", "day_id", "indicator_type_id", "value"}
        assert expected.issubset(set(observed_df.columns))
        # date_obj is a helper column — should be dropped before returning
        assert "date_obj" not in observed_df.columns

    def test_observed_dates_normalized_yyyy_mm_dd(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        for d in observed_df["date"]:
            assert isinstance(d, str)
            assert len(d) == 10
            assert d[4] == "-" and d[7] == "-"

    def test_observed_sorted_by_date_ascending(self, configured_regional_extractor, sample_regional_response):
        """Fixture is intentionally unsorted — formatter must sort ascending."""
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        dates = observed_df["date"].tolist()
        assert dates == sorted(dates)
        assert dates[0] == "2025-01-15"
        assert dates[-1] == "2025-03-15"

    def test_observed_indicator_type_id_preserved(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        assert (observed_df["indicator_type_id"] == 1).all()

    def test_observed_value_preserved(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        # The first chronological record (2025-01-15) had value=0.42
        assert observed_df.iloc[0]["value"] == 0.42

    def test_observed_zulu_time_parsed(self, configured_regional_extractor):
        """Z suffix in ISO time should be parsed correctly."""
        response = {
            "observedMeasures": [
                {"time": "2025-06-15T12:30:00Z", "dayId": 5, "indicatorTypeId": 1, "value": 0.7},
            ],
            "dailyAverage": [],
        }
        observed_df, _ = configured_regional_extractor.format_regional_json(response)
        assert observed_df.iloc[0]["date"] == "2025-06-15"


# ===================================================================
# format_regional_json() — dailyAverage
# ===================================================================


class TestFormatRegionalJsonDailyAverage:
    """Tests for the dailyAverage section (1MMDD dayOfYear → date)."""

    def test_daily_avg_row_count(self, configured_regional_extractor, sample_regional_response):
        _, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response)
        assert len(daily_avg_df) == 3

    def test_daily_avg_columns(self, configured_regional_extractor, sample_regional_response):
        _, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response)
        expected = {"date", "day_of_year", "month", "day", "value"}
        assert expected.issubset(set(daily_avg_df.columns))
        assert "date_obj" not in daily_avg_df.columns

    def test_day_of_year_decoded_to_date(self, configured_regional_extractor, sample_regional_response):
        """dayOfYear=10115 → 1900-01-15 (base_year=1900)."""
        _, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response)
        # The fixture's earliest dayOfYear is 10115 → Jan 15
        first = daily_avg_df.iloc[0]
        assert first["month"] == 1
        assert first["day"] == 15
        assert first["day_of_year"] == 10115

    def test_daily_avg_sorted_by_date_ascending(self, configured_regional_extractor, sample_regional_response):
        _, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response)
        # Months should be 1, 2, 3 in order
        assert daily_avg_df["month"].tolist() == [1, 2, 3]

    def test_daily_avg_value_preserved(self, configured_regional_extractor, sample_regional_response):
        _, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response)
        # First chronological record (Jan 15) had value=0.40
        assert daily_avg_df.iloc[0]["value"] == 0.40


# ===================================================================
# format_regional_json() — input shapes
# ===================================================================


class TestFormatRegionalJsonInputShapes:
    """The formatter accepts dict and string JSON."""

    def test_string_json_parsed(self, configured_regional_extractor, sample_regional_response):
        as_str = json.dumps(sample_regional_response)
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json(as_str)
        assert len(observed_df) == 3
        assert len(daily_avg_df) == 3

    def test_dict_passthrough(self, configured_regional_extractor, sample_regional_response):
        observed_df, _ = configured_regional_extractor.format_regional_json(sample_regional_response)
        assert len(observed_df) == 3

    def test_unsupported_type_raises(self, configured_regional_extractor):
        """Non-dict/str input should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported response type"):
            configured_regional_extractor.format_regional_json(42)

    def test_list_raises(self, configured_regional_extractor):
        """Lists are not supported (regional API returns a wrapping dict)."""
        with pytest.raises(TypeError, match="Unsupported response type"):
            configured_regional_extractor.format_regional_json([])


# ===================================================================
# format_regional_json() — empty / partial responses
# ===================================================================


class TestFormatRegionalJsonEmpty:
    """Edge cases: missing keys, empty lists, fully empty dict."""

    def test_empty_dict_returns_two_empty_dfs(self, configured_regional_extractor):
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json({})
        assert observed_df.empty
        assert daily_avg_df.empty

    def test_no_observed_measures_key(self, configured_regional_extractor, sample_regional_response_daily_avg_only):
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json(
            sample_regional_response_daily_avg_only
        )
        assert observed_df.empty
        assert not daily_avg_df.empty

    def test_no_daily_average_key(self, configured_regional_extractor, sample_regional_response_observed_only):
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json(
            sample_regional_response_observed_only
        )
        assert not observed_df.empty
        assert daily_avg_df.empty

    def test_both_empty_lists(self, configured_regional_extractor, sample_regional_response_empty):
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json(sample_regional_response_empty)
        assert observed_df.empty
        assert daily_avg_df.empty

    def test_entity_id_passed_for_logging(self, configured_regional_extractor, sample_regional_response):
        """entity_id arg should not affect output rows but should not raise."""
        observed_df, daily_avg_df = configured_regional_extractor.format_regional_json(
            sample_regional_response, entity_id=2432528
        )
        assert len(observed_df) == 3
        assert len(daily_avg_df) == 3
