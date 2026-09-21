"""
Tests for InSeasonMonitoringExtractor.format_inseason_json():
    - Multi-row DataFrame from data list (one row per monitoring point)
    - Column mapping from API field names → snake_case
    - Edge cases (missing 'data', empty list, missing fields)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public


class TestFormatInseasonJsonHappyPath:
    """The ISM endpoint returns data=[{...}, {...}] — one row per monitoring point."""

    def test_creates_one_row_per_data_entry(self, configured_ism_extractor, sample_ism_response):
        df = configured_ism_extractor.format_inseason_json(sample_ism_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_expected_columns_present(self, configured_ism_extractor, sample_ism_response):
        df = configured_ism_extractor.format_inseason_json(sample_ism_response)
        expected = {
            "entity_id",
            "season",
            "date",
            "vegetation_index",
            "cumulative_index",
            "emergence_date",
            "emergence_status",
            "days_since_emergence",
            "cumulative_vs_avg",
            "historical_avg_cumulative",
            "delta",
        }
        assert expected.issubset(set(df.columns))

    def test_entity_id_propagated(self, configured_ism_extractor, sample_ism_response):
        df = configured_ism_extractor.format_inseason_json(sample_ism_response)
        assert (df["entity_id"] == "z361x33").all()

    def test_field_mapping_correct(self, configured_ism_extractor, sample_ism_response):
        """API field names should map to snake_case columns with values preserved."""
        df = configured_ism_extractor.format_inseason_json(sample_ism_response)
        first = df.iloc[0]
        assert first["season"] == "2025"
        assert first["date"] == "2025-04-15"
        assert first["vegetation_index"] == 0.32
        assert first["cumulative_index"] == 0.32
        assert first["emergence_date"] == "2025-04-12"
        assert first["emergence_status"] == "CONFIRMED"
        assert first["days_since_emergence"] == 3
        assert first["cumulative_vs_avg"] == "BELOW"
        assert first["historical_avg_cumulative"] == 0.45
        assert first["delta"] == -0.13

    def test_third_record_values(self, configured_ism_extractor, sample_ism_response):
        df = configured_ism_extractor.format_inseason_json(sample_ism_response)
        third = df.iloc[2]
        assert third["date"] == "2025-04-29"
        assert third["cumulative_index"] == 1.51
        assert third["cumulative_vs_avg"] == "ON_TRACK"


class TestFormatInseasonJsonEdgeCases:
    def test_empty_data_returns_empty_df(self, configured_ism_extractor, sample_ism_response_empty):
        df = configured_ism_extractor.format_inseason_json(sample_ism_response_empty)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_data_key_returns_empty_df(self, configured_ism_extractor):
        """Source uses `.get('data', [])` so a missing key falls through to an empty list."""
        df = configured_ism_extractor.format_inseason_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_fields_become_none(self, configured_ism_extractor):
        """A record missing several optional fields → those columns are None."""
        response = {
            "id": "ent_x",
            "data": [
                {"Season": "2025", "RequestDate": "2025-04-15"}  # only two fields
            ],
        }
        df = configured_ism_extractor.format_inseason_json(response)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["season"] == "2025"
        assert row["date"] == "2025-04-15"
        assert row["vegetation_index"] is None
        assert row["emergence_date"] is None
        assert row["delta"] is None

    def test_missing_id_in_response_propagates_as_none(self, configured_ism_extractor, sample_ism_response):
        """When the response omits id, entity_id falls back to None across all rows."""
        response = {"data": sample_ism_response["data"]}  # no 'id' key
        df = configured_ism_extractor.format_inseason_json(response)
        assert (df["entity_id"].isna() | (df["entity_id"].isnull())).all()
