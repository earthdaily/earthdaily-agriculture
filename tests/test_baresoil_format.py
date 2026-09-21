"""
Tests for BaresoilExtractor.format_baresoil_json():
    - Summary mode → one row per entity
    - Full mode with periods → one row per period
    - Full mode without periods → single row with null period info
    - Empty / missing / invalid responses
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_baresoil_json() — summary mode
# ===================================================================


class TestFormatBaresoilJsonSummary:
    """When filter='summary', each entity yields a single row."""

    def test_summary_creates_single_row(self, configured_baresoil_extractor, sample_baresoil_response_summary):
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_summary_contains_expected_columns(self, configured_baresoil_extractor, sample_baresoil_response_summary):
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        expected = {
            "entity_id",
            "year",
            "season_duration",
            "season_start_day",
            "season_start_month",
            "baresoil_days",
            "has_baresoil",
        }
        assert expected.issubset(set(df.columns))

    def test_summary_values_are_correct(self, configured_baresoil_extractor, sample_baresoil_response_summary):
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)
        row = df.iloc[0]

        assert row["entity_id"] == "entity_001"
        assert row["year"] == 2025
        assert row["season_duration"] == 120
        assert row["season_start_day"] == 1
        assert row["season_start_month"] == 4
        assert row["baresoil_days"] == 48
        assert bool(row["has_baresoil"]) is True

    def test_summary_mode_does_not_include_period_columns(
        self, configured_baresoil_extractor, sample_baresoil_response_summary
    ):
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        assert "period_number" not in df.columns
        assert "period_start" not in df.columns
        assert "period_end" not in df.columns

    def test_summary_has_baresoil_false_when_zero_days(
        self, configured_baresoil_extractor, sample_baresoil_response_no_baresoil
    ):
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_no_baresoil)

        assert len(df) == 1
        assert df.iloc[0]["baresoil_days"] == 0
        assert bool(df.iloc[0]["has_baresoil"]) is False


# ===================================================================
# format_baresoil_json() — full mode
# ===================================================================


class TestFormatBaresoilJsonFull:
    """When filter='full', the response yields one row per baresoil period."""

    def test_full_mode_creates_one_row_per_period(
        self, configured_baresoil_extractor, sample_baresoil_response_summary
    ):
        configured_baresoil_extractor.baresoil_params["filter"] = "full"
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        assert len(df) == 2  # two periods in the fixture

    def test_full_mode_period_columns_populated(self, configured_baresoil_extractor, sample_baresoil_response_summary):
        configured_baresoil_extractor.baresoil_params["filter"] = "full"
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        assert df.iloc[0]["period_number"] == 1
        assert df.iloc[0]["period_start"] == "2025-04-15"
        assert df.iloc[0]["period_end"] == "2025-05-10"
        assert df.iloc[0]["period_length"] == 25

        assert df.iloc[1]["period_number"] == 2
        assert df.iloc[1]["period_start"] == "2025-06-01"
        assert df.iloc[1]["period_length"] == 23

    def test_full_mode_carries_summary_fields_on_every_row(
        self, configured_baresoil_extractor, sample_baresoil_response_summary
    ):
        configured_baresoil_extractor.baresoil_params["filter"] = "full"
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_summary)

        # Every row should carry the same summary fields
        assert (df["entity_id"] == "entity_001").all()
        assert (df["year"] == 2025).all()
        assert (df["baresoil_days_total"] == 48).all()
        assert df["has_baresoil"].all()

    def test_full_mode_no_periods_returns_single_null_row(
        self, configured_baresoil_extractor, sample_baresoil_response_no_baresoil
    ):
        """When no baresoil periods are detected, a single row with null period info is returned."""
        configured_baresoil_extractor.baresoil_params["filter"] = "full"
        df = configured_baresoil_extractor.format_baresoil_json(sample_baresoil_response_no_baresoil)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["entity_id"] == "entity_002"
        assert row["baresoil_days_total"] == 0
        assert bool(row["has_baresoil"]) is False
        assert row["period_number"] is None
        assert row["period_start"] is None
        assert row["period_end"] is None
        assert row["period_length"] is None


# ===================================================================
# format_baresoil_json() — edge cases
# ===================================================================


class TestFormatBaresoilJsonEdgeCases:
    def test_missing_data_key_returns_empty_df(self, configured_baresoil_extractor):
        response = {"id": "ent_x"}  # no "data" key
        df = configured_baresoil_extractor.format_baresoil_json(response)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_data_dict_returns_empty_df(self, configured_baresoil_extractor):
        """An empty dict in 'data' is treated as no data → empty DataFrame."""
        response = {"id": "ent_x", "data": {}}
        df = configured_baresoil_extractor.format_baresoil_json(response)

        assert df.empty

    def test_none_data_returns_empty_df(self, configured_baresoil_extractor):
        response = {"id": "ent_x", "data": None}
        df = configured_baresoil_extractor.format_baresoil_json(response)

        assert df.empty

    def test_non_dict_response_returns_empty_df(self, configured_baresoil_extractor):
        """A string/None/list response should not raise — just return empty DataFrame."""
        assert configured_baresoil_extractor.format_baresoil_json("not_a_dict").empty
        assert configured_baresoil_extractor.format_baresoil_json(None).empty
        assert configured_baresoil_extractor.format_baresoil_json({}).empty

    def test_missing_id_falls_back_to_unknown(self, configured_baresoil_extractor):
        """If response has no 'id', entity_id should be 'unknown'."""
        response = {"data": {"baresoilDays": 5, "baresoilPeriods": []}}
        df = configured_baresoil_extractor.format_baresoil_json(response)

        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "unknown"

    def test_default_filter_mode_when_params_missing(
        self, configured_baresoil_extractor, sample_baresoil_response_summary
    ):
        """If params override is passed without a filter key, defaults to 'full'."""
        df = configured_baresoil_extractor.format_baresoil_json(
            sample_baresoil_response_summary, params={"entity_id": "override_id"}
        )

        # Default filter when params dict is provided without 'filter' key is 'full'
        # → we expect one row per period (2)
        assert len(df) == 2
        assert "period_number" in df.columns
