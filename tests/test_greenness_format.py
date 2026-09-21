"""
Tests for GreennessExtractor.format_greenness_json():
    - Dict response → single-row DataFrame
    - List response → multi-row DataFrame
    - Edge cases (missing 'data' key, non-dict input, unexpected types)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_greenness_json() — dict response (single record per entity)
# ===================================================================


class TestFormatGreennessJsonDict:
    """When response['data'] is a dict, format produces a single-row DataFrame."""

    def test_dict_response_creates_single_row(self, configured_extractor, sample_api_response):
        df = configured_extractor.format_greenness_json(sample_api_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_all_data_keys_become_columns(self, configured_extractor, sample_api_response):
        df = configured_extractor.format_greenness_json(sample_api_response)
        # Fixture data: greenness_score / greenness_date / potential_score / status + entity_id
        expected = {"entity_id", "greenness_score", "greenness_date", "potential_score", "status"}
        assert expected.issubset(set(df.columns))

    def test_numeric_values_preserved(self, configured_extractor, sample_api_response):
        df = configured_extractor.format_greenness_json(sample_api_response)
        row = df.iloc[0]
        assert row["greenness_score"] == 0.85
        assert row["potential_score"] == 72.3

    def test_iso_date_string_parsed(self, configured_extractor, sample_api_response):
        """A YYYY-MM-DD string in the data dict is parsed via safe_parse_date."""
        df = configured_extractor.format_greenness_json(sample_api_response)
        # safe_parse_date returns the ISO string for valid YYYY-MM-DD inputs
        assert df.iloc[0]["greenness_date"] == "2025-06-15"

    def test_non_date_strings_kept_as_is(self, configured_extractor):
        """Strings that can't be parsed as dates fall through to the raw value."""
        response = {
            "id": "entity_001",
            "data": {"status": "GREEN", "label": "healthy", "score": 0.7},
        }
        df = configured_extractor.format_greenness_json(response)
        assert df.iloc[0]["status"] == "GREEN"
        assert df.iloc[0]["label"] == "healthy"


# ===================================================================
# format_greenness_json() — list response (multiple records)
# ===================================================================


class TestFormatGreennessJsonList:
    """When response['data'] is a list, format produces one row per item."""

    def test_list_creates_multi_row_dataframe(self, configured_extractor, sample_api_response_list):
        df = configured_extractor.format_greenness_json(sample_api_response_list)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_list_columns_match_dict_keys(self, configured_extractor, sample_api_response_list):
        df = configured_extractor.format_greenness_json(sample_api_response_list)
        # Fixture items: greenness_score / date / status + entity_id propagated
        expected = {"entity_id", "greenness_score", "date", "status"}
        assert expected.issubset(set(df.columns))

    def test_list_values_correct_per_row(self, configured_extractor, sample_api_response_list):
        df = configured_extractor.format_greenness_json(sample_api_response_list)
        # Both rows share entity_id but have distinct dates / scores
        assert (df["entity_id"] == "entity_001").all()
        assert df.iloc[0]["greenness_score"] == 0.85
        assert df.iloc[1]["greenness_score"] == 0.72


# ===================================================================
# format_greenness_json() — edge cases
# ===================================================================


class TestFormatGreennessJsonEdgeCases:
    """Edge cases: missing keys, non-dict input, unexpected nested types."""

    def test_missing_data_key_returns_empty(self, configured_extractor):
        df = configured_extractor.format_greenness_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_none_returns_empty(self, configured_extractor):
        df = configured_extractor.format_greenness_json({"id": "ent_x", "data": None})
        assert df.empty

    def test_unexpected_data_type_returns_empty(self, configured_extractor):
        """A 'data' value that's neither dict nor list returns an empty DataFrame."""
        df = configured_extractor.format_greenness_json({"id": "ent_x", "data": "junk"})
        assert df.empty

    def test_non_dict_response_raises(self, configured_extractor):
        """A list / string at the top level is not supported and should raise."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_extractor.format_greenness_json([{"id": "x"}])

    def test_empty_data_dict_still_creates_row(self, configured_extractor):
        """An empty {} under 'data' still yields one row carrying entity_id only."""
        df = configured_extractor.format_greenness_json({"id": "ent_x", "data": {}})
        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "ent_x"

    def test_missing_id_in_response_propagates_as_none(self, configured_extractor):
        """When the API response omits id, entity_id falls back to None."""
        response = {"data": {"greenness_score": 0.5, "status": "GREEN"}}
        df = configured_extractor.format_greenness_json(response)
        assert df.iloc[0]["entity_id"] is None
