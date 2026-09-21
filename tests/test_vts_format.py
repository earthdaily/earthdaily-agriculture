"""
Tests for VegationTsExtractor.format_vegetation_ts_json():
    - List response → DataFrame with date / value (and entity_id when entity_data provided)
    - Date sorted ascending, parsed to datetime
    - Edge cases (empty list, no entity_data, missing id)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public


class TestFormatVegetationTsJson:
    def test_list_creates_one_row_per_record(self, configured_vts_extractor, sample_vts_response, sample_vts_entity):
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=sample_vts_entity)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_expected_columns(self, configured_vts_extractor, sample_vts_response, sample_vts_entity):
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=sample_vts_entity)
        # entity_id should be first column when entity_data carries 'id'
        assert list(df.columns) == ["entity_id", "date", "value"]

    def test_entity_id_propagated(self, configured_vts_extractor, sample_vts_response, sample_vts_entity):
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=sample_vts_entity)
        assert (df["entity_id"] == "z361x33").all()

    def test_date_parsed_to_datetime(self, configured_vts_extractor, sample_vts_response, sample_vts_entity):
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=sample_vts_entity)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_dates_sorted_ascending(self, configured_vts_extractor, sample_vts_response, sample_vts_entity):
        """API returns descending order — formatter sorts ascending."""
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=sample_vts_entity)
        assert list(df["value"]) == [0.31, 0.45, 0.62, 0.78]

    def test_no_entity_data_omits_entity_id(self, configured_vts_extractor, sample_vts_response):
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response)
        assert list(df.columns) == ["date", "value"]
        assert "entity_id" not in df.columns

    def test_entity_data_without_id_omits_entity_id(self, configured_vts_extractor, sample_vts_response):
        entity_no_id = {"geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        df = configured_vts_extractor.format_vegetation_ts_json(sample_vts_response, entity_data=entity_no_id)
        assert "entity_id" not in df.columns


class TestFormatVegetationTsJsonEdgeCases:
    def test_empty_list_raises_keyerror(self, configured_vts_extractor):
        """An empty list raises KeyError because the formatter selects ['date','value']
        on an empty DataFrame that has no columns yet (line 576 in VTS_functions.py).

        This documents a real bug — callers should pre-check for empty responses or wrap
        the call. The bulk processing pipeline catches it as a generic Exception, so it
        surfaces as an error result rather than crashing the run.
        """
        with pytest.raises(KeyError):
            configured_vts_extractor.format_vegetation_ts_json([])

    def test_missing_value_field_becomes_none(self, configured_vts_extractor, sample_vts_entity):
        """A record without 'value' should produce a row with value=None."""
        response = [{"date": "2025-08-15T00:00:00Z"}]
        df = configured_vts_extractor.format_vegetation_ts_json(response, entity_data=sample_vts_entity)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["value"])

    def test_missing_date_field_kept_as_none(self, configured_vts_extractor, sample_vts_entity):
        response = [{"value": 0.5}]
        df = configured_vts_extractor.format_vegetation_ts_json(response, entity_data=sample_vts_entity)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["date"])
