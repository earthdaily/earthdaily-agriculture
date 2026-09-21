"""
Tests for EmergenceExtractor.format_emergence_json():
    - INSEASON mode  → emergence_date / emergence_status / confirmation_status
    - HISTORICAL mode → 5 historical years + historical_average_emergence
    - DELAY mode      → emergence_date / average_emergence_date / emergence_delay
    - Edge cases (missing 'data' key, non-dict input, unknown emergence_type)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_emergence_json() — INSEASON
# ===================================================================


class TestFormatEmergenceJsonInseason:
    """INSEASON mode produces a single-row DataFrame with current-season fields."""

    def test_inseason_creates_one_row(self, configured_emergence_extractor, sample_emergence_inseason_response):
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_inseason_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_inseason_columns_present(self, configured_emergence_extractor, sample_emergence_inseason_response):
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_inseason_response)
        expected = {"entity_id", "emergence_date", "emergence_status", "confirmation_status"}
        assert expected.issubset(set(df.columns))

    def test_inseason_values_preserved(self, configured_emergence_extractor, sample_emergence_inseason_response):
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_inseason_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["emergence_date"] == "2025-04-18"
        assert row["emergence_status"] == "CONFIRMED"
        assert row["confirmation_status"] == "VALIDATED"

    def test_inseason_placeholder_date_becomes_none(self, configured_emergence_extractor):
        """safe_parse_date converts the '0001-01-01' API placeholder to None."""
        response = {
            "id": "ent_x",
            "data": {
                "EmergenceDate": "0001-01-01",
                "EmergenceStatus": "NOT_DETECTED",
                "ConfirmationStatus": "PENDING",
            },
        }
        df = configured_emergence_extractor.format_emergence_json(response)
        assert df.iloc[0]["emergence_date"] is None
        assert df.iloc[0]["emergence_status"] == "NOT_DETECTED"

    def test_inseason_unexpected_data_type_returns_empty(self, configured_emergence_extractor):
        """A non-dict 'data' value should produce an empty DataFrame."""
        df = configured_emergence_extractor.format_emergence_json({"id": "ent_x", "data": "not a dict"})
        assert df.empty


# ===================================================================
# format_emergence_json() — HISTORICAL
# ===================================================================


class TestFormatEmergenceJsonHistorical:
    """HISTORICAL mode flattens 5 prior years into separate columns."""

    def test_historical_creates_one_row(self, configured_emergence_extractor, sample_emergence_historical_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "HISTORICAL"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_historical_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_historical_columns_present(self, configured_emergence_extractor, sample_emergence_historical_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "HISTORICAL"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_historical_response)

        expected = {
            "entity_id",
            "emergence_year_1",
            "emergence_year_2",
            "emergence_year_3",
            "emergence_year_4",
            "emergence_year_5",
            "historical_average_emergence",
        }
        assert expected.issubset(set(df.columns))

    def test_historical_values_preserved(self, configured_emergence_extractor, sample_emergence_historical_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "HISTORICAL"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_historical_response)
        row = df.iloc[0]
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_5"] == "2020-04-18"
        # historical_average_emergence stays in MM-DD format (not parsed by safe_parse_date)
        assert row["historical_average_emergence"] == "04-17"

    def test_historical_uses_params_kwarg_override(
        self, configured_emergence_extractor, sample_emergence_historical_response
    ):
        """An explicit `params` kwarg should override the configured emergence_type."""
        # configured_emergence_extractor still has INSEASON, but the kwarg flips to HISTORICAL.
        df = configured_emergence_extractor.format_emergence_json(
            sample_emergence_historical_response,
            params={"emergence_type": "HISTORICAL"},
        )
        assert "emergence_year_1" in df.columns


# ===================================================================
# format_emergence_json() — HISTORICAL per-field matching-season filter
# ===================================================================


class TestFormatEmergenceJsonHistoricalMatchingSeasons:
    """
    HISTORICAL mode with a per-field ``historical_seasons`` input keeps emergence only
    for the matching years and emits ``avg_emergence_matching_years`` recomputed over
    the retained years — while leaving ``historical_average_emergence`` untouched.

    Fixture reference (sample_emergence_historical_response, year=2025):
        emergence_year_1 -> 2024-04-15   (calendar 2024)
        emergence_year_2 -> 2023-04-22   (calendar 2023)
        emergence_year_3 -> 2022-04-12   (calendar 2022)
        emergence_year_4 -> 2021-04-20   (calendar 2021)
        emergence_year_5 -> 2020-04-18   (calendar 2020)
        Hist_avg_emergence -> "04-17"
    """

    def _historical(self, extractor):
        extractor.emergence_params["emergence_type"] = "HISTORICAL"
        extractor.emergence_params["year"] = 2025
        return extractor

    def test_no_filter_passthrough_leaves_years_intact(
        self, configured_emergence_extractor, sample_emergence_historical_response
    ):
        """No historical_seasons -> existing behavior unchanged; new column present but null."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(sample_emergence_historical_response)
        row = df.iloc[0]
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_5"] == "2020-04-18"
        assert row["historical_average_emergence"] == "04-17"
        # New column is emitted (stable schema) but null when no season set is supplied.
        assert "avg_emergence_matching_years" in df.columns
        assert row["avg_emergence_matching_years"] is None

    def test_partial_filter_string_form(self, configured_emergence_extractor, sample_emergence_historical_response):
        """A comma-separated string keeps only matching years and recomputes the MM-DD average."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(
            sample_emergence_historical_response,
            historical_seasons="2020,2022,2024,2025",
        )
        row = df.iloc[0]
        # Kept: 2024 (year_1), 2022 (year_3), 2020 (year_5). Nulled: 2023 (year_2), 2021 (year_4).
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_2"] is None
        assert row["emergence_year_3"] == "2022-04-12"
        assert row["emergence_year_4"] is None
        assert row["emergence_year_5"] == "2020-04-18"
        # Average day-of-year of 04-15, 04-12, 04-18 -> 04-15 (day-of-year 105).
        assert row["avg_emergence_matching_years"] == "04-15"
        # Raw API average is preserved unchanged.
        assert row["historical_average_emergence"] == "04-17"

    def test_partial_filter_list_form_matches_string_form(
        self, configured_emergence_extractor, sample_emergence_historical_response
    ):
        """A list of ints yields the same result as the equivalent comma-separated string."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(
            sample_emergence_historical_response,
            historical_seasons=[2024, 2022, 2020],
        )
        row = df.iloc[0]
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_2"] is None
        assert row["emergence_year_3"] == "2022-04-12"
        assert row["emergence_year_4"] is None
        assert row["emergence_year_5"] == "2020-04-18"
        assert row["avg_emergence_matching_years"] == "04-15"
        assert row["historical_average_emergence"] == "04-17"

    def test_empty_set_keeps_none(self, configured_emergence_extractor, sample_emergence_historical_response):
        """An explicitly empty season set nulls every year and the new average."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(sample_emergence_historical_response, historical_seasons="")
        row = df.iloc[0]
        for n in range(1, 6):
            assert row[f"emergence_year_{n}"] is None
        assert row["avg_emergence_matching_years"] is None
        # Raw API average still preserved.
        assert row["historical_average_emergence"] == "04-17"

    def test_out_of_window_year_matches_nothing(
        self, configured_emergence_extractor, sample_emergence_historical_response
    ):
        """Years outside the API's 5-season window match nothing (no error)."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(
            sample_emergence_historical_response,
            historical_seasons=[2019, 2025],  # 2019 predates the window; 2025 has no year column
        )
        row = df.iloc[0]
        for n in range(1, 6):
            assert row[f"emergence_year_{n}"] is None
        assert row["avg_emergence_matching_years"] is None

    def test_whitespace_tokens_ignored(self, configured_emergence_extractor, sample_emergence_historical_response):
        """Whitespace / empty tokens in the string are ignored, not errored."""
        ext = self._historical(configured_emergence_extractor)
        df = ext.format_emergence_json(
            sample_emergence_historical_response,
            historical_seasons=" 2024 , , 2020 ",
        )
        row = df.iloc[0]
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_5"] == "2020-04-18"
        assert row["emergence_year_3"] is None
        assert row["avg_emergence_matching_years"] == "04-16"  # mean of 04-15 and 04-18

    def test_non_int_token_raises(self, configured_emergence_extractor, sample_emergence_historical_response):
        """A non-integer token yields a clear ValueError."""
        ext = self._historical(configured_emergence_extractor)
        with pytest.raises(ValueError):
            ext.format_emergence_json(
                sample_emergence_historical_response,
                historical_seasons="2020,foo",
            )

    def test_year_as_string_is_coerced(self, configured_emergence_extractor, sample_emergence_historical_response):
        """The notebook passes year='2025' as a string — the year mapping must still work."""
        ext = self._historical(configured_emergence_extractor)
        ext.emergence_params["year"] = "2025"  # string, as configured in the dev notebook
        df = ext.format_emergence_json(
            sample_emergence_historical_response,
            historical_seasons=[2024, 2020],
        )
        row = df.iloc[0]
        assert row["emergence_year_1"] == "2024-04-15"
        assert row["emergence_year_5"] == "2020-04-18"
        assert row["emergence_year_3"] is None
        # Mean of 04-15 (doy 105) and 04-18 (doy 108) -> 106.5 -> 106 (banker's) -> 04-16.
        assert row["avg_emergence_matching_years"] == "04-16"


# ===================================================================
# format_emergence_json() — DELAY
# ===================================================================


class TestFormatEmergenceJsonDelay:
    """DELAY mode reports current emergence vs historical average + delta in days."""

    def test_delay_creates_one_row(self, configured_emergence_extractor, sample_emergence_delay_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "DELAY"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_delay_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_delay_columns_present(self, configured_emergence_extractor, sample_emergence_delay_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "DELAY"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_delay_response)
        expected = {"entity_id", "emergence_date", "average_emergence_date", "emergence_delay"}
        assert expected.issubset(set(df.columns))

    def test_delay_values_preserved(self, configured_emergence_extractor, sample_emergence_delay_response):
        configured_emergence_extractor.emergence_params["emergence_type"] = "DELAY"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_delay_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["emergence_date"] == "2025-04-25"
        assert row["average_emergence_date"] == "04-17"
        assert row["emergence_delay"] == 8


# ===================================================================
# format_emergence_json() — edge cases
# ===================================================================


class TestFormatEmergenceJsonEdgeCases:
    """Edge cases: missing keys, non-dict input, unknown type."""

    def test_missing_data_key_returns_empty(self, configured_emergence_extractor):
        df = configured_emergence_extractor.format_emergence_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_none_returns_empty(self, configured_emergence_extractor):
        df = configured_emergence_extractor.format_emergence_json({"id": "ent_x", "data": None})
        assert df.empty

    def test_non_dict_response_raises(self, configured_emergence_extractor):
        """A list or string at the top level is not supported and should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_emergence_extractor.format_emergence_json([{"id": "x"}])

    def test_unknown_emergence_type_returns_empty(
        self, configured_emergence_extractor, sample_emergence_inseason_response
    ):
        """An unknown emergence_type yields an empty DataFrame (after warning)."""
        configured_emergence_extractor.emergence_params["emergence_type"] = "BOGUS"
        df = configured_emergence_extractor.format_emergence_json(sample_emergence_inseason_response)
        assert df.empty

    def test_missing_id_in_response_propagates_as_none(self, configured_emergence_extractor):
        """When the API response omits id, entity_id falls back to None."""
        response = {
            "data": {
                "EmergenceDate": "2025-04-18",
                "EmergenceStatus": "CONFIRMED",
                "ConfirmationStatus": "VALIDATED",
            },
        }
        df = configured_emergence_extractor.format_emergence_json(response)
        assert df.iloc[0]["entity_id"] is None
