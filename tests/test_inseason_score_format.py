"""
Tests for InseasonScoreExtractor.format_inseason_score_json():
    - 'full' / 'summary' detail levels (same shape — three score columns)
    - Edge cases (missing 'data', non-dict input, bad detail_level)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public


class TestFormatInseasonScoreJson:
    """The formatter produces a single-row DataFrame with three score columns."""

    def test_creates_one_row(self, configured_inseason_score_extractor, sample_inseason_score_response):
        df = configured_inseason_score_extractor.format_inseason_score_json(sample_inseason_score_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_expected_columns(self, configured_inseason_score_extractor, sample_inseason_score_response):
        df = configured_inseason_score_extractor.format_inseason_score_json(sample_inseason_score_response)
        expected = {
            "entity_id",
            "historical_potential_score",
            "inseason_potential_score",
            "relative_potential_score",
        }
        assert expected.issubset(set(df.columns))

    def test_values_preserved(self, configured_inseason_score_extractor, sample_inseason_score_response):
        df = configured_inseason_score_extractor.format_inseason_score_json(sample_inseason_score_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["historical_potential_score"] == 0.78
        assert row["inseason_potential_score"] == 0.65
        assert row["relative_potential_score"] == -0.13

    @pytest.mark.parametrize("dl", ["summary", "full"])
    def test_both_detail_levels_produce_same_shape(
        self, configured_inseason_score_extractor, sample_inseason_score_response, dl
    ):
        """In-season score formatter ignores detail_level (both produce the same wide row)."""
        df = configured_inseason_score_extractor.format_inseason_score_json(
            sample_inseason_score_response, detail_level=dl
        )
        assert len(df) == 1
        assert "inseason_potential_score" in df.columns


class TestFormatInseasonScoreJsonEdgeCases:
    def test_missing_data_returns_empty(self, configured_inseason_score_extractor):
        df = configured_inseason_score_extractor.format_inseason_score_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_none_returns_empty(self, configured_inseason_score_extractor):
        df = configured_inseason_score_extractor.format_inseason_score_json({"id": "ent_x", "data": None})
        assert df.empty

    def test_unexpected_data_type_returns_empty(self, configured_inseason_score_extractor):
        df = configured_inseason_score_extractor.format_inseason_score_json({"id": "ent_x", "data": "not a dict"})
        assert df.empty

    def test_invalid_detail_level_raises(self, configured_inseason_score_extractor, sample_inseason_score_response):
        with pytest.raises(ValueError, match="detail_level"):
            configured_inseason_score_extractor.format_inseason_score_json(
                sample_inseason_score_response, detail_level="bogus"
            )

    def test_missing_id_in_response_propagates_as_none(self, configured_inseason_score_extractor):
        response = {"data": {"inseason_potential_score": 0.6}}
        df = configured_inseason_score_extractor.format_inseason_score_json(response)
        assert df.iloc[0]["entity_id"] is None

    def test_missing_score_field_becomes_none(self, configured_inseason_score_extractor):
        """A response missing one of the three score fields → that column is None."""
        response = {
            "id": "ent_x",
            "data": {"inseason_potential_score": 0.6},  # only one of three fields
        }
        df = configured_inseason_score_extractor.format_inseason_score_json(response)
        row = df.iloc[0]
        assert row["inseason_potential_score"] == 0.6
        assert row["historical_potential_score"] is None
        assert row["relative_potential_score"] is None
