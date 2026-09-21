"""
Tests for HarvestExtractor.format_harvest_json():
    - INSEASON_HARVEST → harvest_date / harvest_status
    - HISTORICAL_HARVEST → 5 historical years + historical_harvest_average
    - HARVEST_READINESS → harvest_readiness_date / is_ready (with 0001-01-01 placeholder)
    - Edge cases (missing 'data' key, non-dict input, unknown harvest_type)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_harvest_json() — INSEASON_HARVEST
# ===================================================================


class TestFormatHarvestJsonInseason:
    """INSEASON_HARVEST mode produces a single-row DataFrame with current-season fields."""

    def test_inseason_creates_one_row(self, configured_harvest_extractor, sample_harvest_inseason_response):
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_inseason_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_inseason_columns_present(self, configured_harvest_extractor, sample_harvest_inseason_response):
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_inseason_response)
        expected = {"entity_id", "harvest_date", "harvest_status"}
        assert expected.issubset(set(df.columns))

    def test_inseason_values_preserved(self, configured_harvest_extractor, sample_harvest_inseason_response):
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_inseason_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["harvest_date"] == "2025-08-22"
        assert row["harvest_status"] == "HARVESTED"

    def test_inseason_placeholder_date_becomes_none(self, configured_harvest_extractor):
        """safe_parse_date converts the '0001-01-01' API placeholder to None."""
        response = {
            "id": "ent_x",
            "data": {"HarvestDate": "0001-01-01", "HarvestStatus": "PENDING"},
        }
        df = configured_harvest_extractor.format_harvest_json(response)
        assert df.iloc[0]["harvest_date"] is None
        assert df.iloc[0]["harvest_status"] == "PENDING"

    def test_inseason_unexpected_data_type_returns_empty(self, configured_harvest_extractor):
        """A non-dict 'data' value should produce an empty DataFrame."""
        df = configured_harvest_extractor.format_harvest_json({"id": "ent_x", "data": "not a dict"})
        assert df.empty


# ===================================================================
# format_harvest_json() — HISTORICAL_HARVEST
# ===================================================================


class TestFormatHarvestJsonHistorical:
    """HISTORICAL_HARVEST mode flattens 5 prior years into separate columns."""

    def test_historical_creates_one_row(self, configured_harvest_extractor, sample_harvest_historical_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HISTORICAL_HARVEST"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_historical_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_historical_columns_present(self, configured_harvest_extractor, sample_harvest_historical_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HISTORICAL_HARVEST"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_historical_response)

        expected = {
            "entity_id",
            "harvest_year_1",
            "harvest_year_2",
            "harvest_year_3",
            "harvest_year_4",
            "harvest_year_5",
            "historical_harvest_average",
        }
        assert expected.issubset(set(df.columns))

    def test_historical_values_preserved(self, configured_harvest_extractor, sample_harvest_historical_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HISTORICAL_HARVEST"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_historical_response)
        row = df.iloc[0]
        assert row["harvest_year_1"] == "2024-08-15"
        assert row["harvest_year_5"] == "2020-08-18"
        # historical_harvest_average stays in MM-DD format (passes through, not date-parsed)
        assert row["historical_harvest_average"] == "08-17"

    def test_historical_uses_params_kwarg_override(
        self, configured_harvest_extractor, sample_harvest_historical_response
    ):
        """An explicit `params` kwarg should override the configured harvest_type."""
        # configured fixture is INSEASON_HARVEST, but kwarg flips to HISTORICAL_HARVEST.
        df = configured_harvest_extractor.format_harvest_json(
            sample_harvest_historical_response,
            params={"harvest_type": "HISTORICAL_HARVEST"},
        )
        assert "harvest_year_1" in df.columns


# ===================================================================
# format_harvest_json() — HISTORICAL_HARVEST per-field matching-season filter
# ===================================================================


class TestFormatHarvestJsonHistoricalMatchingSeasons:
    """
    HISTORICAL_HARVEST with a per-field ``historical_seasons`` input keeps harvest only for
    the matching years and emits ``avg_harvest_matching_years`` recomputed over the retained
    years — while leaving ``historical_harvest_average`` untouched.

    Fixture reference (sample_harvest_historical_response, year=2025):
        harvest_year_1 -> 2024-08-15   (calendar 2024)
        harvest_year_2 -> 2023-08-22   (calendar 2023)
        harvest_year_3 -> 2022-08-12   (calendar 2022)
        harvest_year_4 -> 2021-08-20   (calendar 2021)
        harvest_year_5 -> 2020-08-18   (calendar 2020)
        historical_harvest_average -> "08-17"
    """

    def _historical(self, extractor):
        extractor.harvest_params["harvest_type"] = "HISTORICAL_HARVEST"
        extractor.harvest_params["year"] = 2025
        return extractor

    def test_no_filter_passthrough_leaves_years_intact(
        self, configured_harvest_extractor, sample_harvest_historical_response
    ):
        """No historical_seasons -> existing behavior unchanged; new column present but null."""
        ext = self._historical(configured_harvest_extractor)
        df = ext.format_harvest_json(sample_harvest_historical_response)
        row = df.iloc[0]
        assert row["harvest_year_1"] == "2024-08-15"
        assert row["harvest_year_5"] == "2020-08-18"
        assert row["historical_harvest_average"] == "08-17"
        assert "avg_harvest_matching_years" in df.columns
        assert row["avg_harvest_matching_years"] is None

    def test_partial_filter_string_form(self, configured_harvest_extractor, sample_harvest_historical_response):
        """A comma-separated string keeps only matching years and recomputes the MM-DD average."""
        ext = self._historical(configured_harvest_extractor)
        df = ext.format_harvest_json(
            sample_harvest_historical_response,
            historical_seasons="2020,2022,2024,2025",
        )
        row = df.iloc[0]
        # Kept: 2024 (year_1), 2022 (year_3), 2020 (year_5). Nulled: 2023 (year_2), 2021 (year_4).
        assert row["harvest_year_1"] == "2024-08-15"
        assert row["harvest_year_2"] is None
        assert row["harvest_year_3"] == "2022-08-12"
        assert row["harvest_year_4"] is None
        assert row["harvest_year_5"] == "2020-08-18"
        # Average day-of-year of 08-15, 08-12, 08-18 -> 08-15 (day-of-year 227).
        assert row["avg_harvest_matching_years"] == "08-15"
        # Raw API average is preserved unchanged.
        assert row["historical_harvest_average"] == "08-17"

    def test_partial_filter_list_form_matches_string_form(
        self, configured_harvest_extractor, sample_harvest_historical_response
    ):
        """A list of ints yields the same result as the equivalent comma-separated string."""
        ext = self._historical(configured_harvest_extractor)
        df = ext.format_harvest_json(
            sample_harvest_historical_response,
            historical_seasons=[2024, 2022, 2020],
        )
        row = df.iloc[0]
        assert row["harvest_year_1"] == "2024-08-15"
        assert row["harvest_year_2"] is None
        assert row["harvest_year_3"] == "2022-08-12"
        assert row["harvest_year_5"] == "2020-08-18"
        assert row["avg_harvest_matching_years"] == "08-15"
        assert row["historical_harvest_average"] == "08-17"

    def test_empty_set_keeps_none(self, configured_harvest_extractor, sample_harvest_historical_response):
        """An explicitly empty season set nulls every year and the new average."""
        ext = self._historical(configured_harvest_extractor)
        df = ext.format_harvest_json(sample_harvest_historical_response, historical_seasons="")
        row = df.iloc[0]
        for n in range(1, 6):
            assert row[f"harvest_year_{n}"] is None
        assert row["avg_harvest_matching_years"] is None
        assert row["historical_harvest_average"] == "08-17"

    def test_out_of_window_year_matches_nothing(self, configured_harvest_extractor, sample_harvest_historical_response):
        """Years outside the API's 5-season window match nothing (no error)."""
        ext = self._historical(configured_harvest_extractor)
        df = ext.format_harvest_json(
            sample_harvest_historical_response,
            historical_seasons=[2019, 2025],  # 2019 predates the window; 2025 has no year column
        )
        row = df.iloc[0]
        for n in range(1, 6):
            assert row[f"harvest_year_{n}"] is None
        assert row["avg_harvest_matching_years"] is None

    def test_non_int_token_raises(self, configured_harvest_extractor, sample_harvest_historical_response):
        """A non-integer token yields a clear ValueError."""
        ext = self._historical(configured_harvest_extractor)
        with pytest.raises(ValueError):
            ext.format_harvest_json(
                sample_harvest_historical_response,
                historical_seasons="2020,foo",
            )

    def test_year_as_string_is_coerced(self, configured_harvest_extractor, sample_harvest_historical_response):
        """year may be configured as a string — the year mapping must still work."""
        ext = self._historical(configured_harvest_extractor)
        ext.harvest_params["year"] = "2025"  # string
        df = ext.format_harvest_json(
            sample_harvest_historical_response,
            historical_seasons=[2024, 2020],
        )
        row = df.iloc[0]
        assert row["harvest_year_1"] == "2024-08-15"
        assert row["harvest_year_5"] == "2020-08-18"
        assert row["harvest_year_3"] is None
        # Mean of 08-15 (doy 227) and 08-18 (doy 230) -> 228.5 -> 228 (banker's) -> 08-16.
        assert row["avg_harvest_matching_years"] == "08-16"


# ===================================================================
# format_harvest_json() — HARVEST_READINESS
# ===================================================================


class TestFormatHarvestJsonReadiness:
    """HARVEST_READINESS mode reports a date and an is_ready boolean."""

    def test_readiness_creates_one_row(self, configured_harvest_extractor, sample_harvest_readiness_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HARVEST_READINESS"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_readiness_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_readiness_columns_present(self, configured_harvest_extractor, sample_harvest_readiness_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HARVEST_READINESS"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_readiness_response)
        expected = {"entity_id", "harvest_readiness_date", "is_ready"}
        assert expected.issubset(set(df.columns))

    def test_readiness_ready_field_true(self, configured_harvest_extractor, sample_harvest_readiness_response):
        configured_harvest_extractor.harvest_params["harvest_type"] = "HARVEST_READINESS"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_readiness_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["harvest_readiness_date"] == "2025-08-25"
        # pandas may wrap bools as numpy.bool_ — compare with == rather than `is`.
        assert bool(row["is_ready"]) is True

    def test_readiness_placeholder_marks_not_ready(
        self, configured_harvest_extractor, sample_harvest_readiness_not_ready_response
    ):
        """The '0001-01-01' placeholder → harvest_readiness_date None, is_ready False."""
        configured_harvest_extractor.harvest_params["harvest_type"] = "HARVEST_READINESS"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_readiness_not_ready_response)
        row = df.iloc[0]
        assert row["harvest_readiness_date"] is None
        assert bool(row["is_ready"]) is False


# ===================================================================
# format_harvest_json() — edge cases
# ===================================================================


class TestFormatHarvestJsonEdgeCases:
    """Edge cases: missing keys, non-dict input, unknown harvest_type."""

    def test_missing_data_key_returns_empty(self, configured_harvest_extractor):
        df = configured_harvest_extractor.format_harvest_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_none_returns_empty(self, configured_harvest_extractor):
        df = configured_harvest_extractor.format_harvest_json({"id": "ent_x", "data": None})
        assert df.empty

    def test_non_dict_response_raises(self, configured_harvest_extractor):
        """A list or string at the top level is not supported and should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_harvest_extractor.format_harvest_json([{"id": "x"}])

    def test_unknown_harvest_type_returns_empty(self, configured_harvest_extractor, sample_harvest_inseason_response):
        """An unknown harvest_type yields an empty DataFrame (after warning)."""
        configured_harvest_extractor.harvest_params["harvest_type"] = "BOGUS"
        df = configured_harvest_extractor.format_harvest_json(sample_harvest_inseason_response)
        assert df.empty

    def test_missing_id_in_response_propagates_as_none(self, configured_harvest_extractor):
        """When the API response omits id, entity_id falls back to None."""
        response = {
            "data": {"HarvestDate": "2025-08-22", "HarvestStatus": "HARVESTED"},
        }
        df = configured_harvest_extractor.format_harvest_json(response)
        assert df.iloc[0]["entity_id"] is None
