"""
Tests for HistoricalScoreExtractor.format_historical_score_json():
    - 'full' detail level → summary + per-season columns
    - 'summary' detail level → summary metrics only
    - JSON-encoded PotentialScores / SeasonBreaks parsing
    - Edge cases (missing 'data' key, non-dict input, bad detail_level)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_historical_score_json() — full detail
# ===================================================================


class TestFormatHistoricalScoreJsonFull:
    """detail_level='full' → summary metrics + potential_score_<season> + season_break_<season>."""

    def test_full_creates_one_row(self, configured_historical_score_extractor, sample_historical_score_response):
        df = configured_historical_score_extractor.format_historical_score_json(sample_historical_score_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_full_summary_columns_present(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(sample_historical_score_response)
        expected = {
            "entity_id",
            "average_potential_score",
            "olympic_mean_potential_score",
            "standard_deviation",
            "risk_score",
        }
        assert expected.issubset(set(df.columns))

    def test_full_summary_values_preserved(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(sample_historical_score_response)
        row = df.iloc[0]
        assert row["entity_id"] == "z361x33"
        assert row["average_potential_score"] == 0.78
        assert row["olympic_mean_potential_score"] == 0.80
        assert row["standard_deviation"] == 0.05
        assert row["risk_score"] == 0.22

    def test_full_per_season_potential_score_columns(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(sample_historical_score_response)
        for season in ("2020", "2021", "2022", "2023", "2024"):
            assert f"potential_score_{season}" in df.columns
        # Spot-check a value
        assert df.iloc[0]["potential_score_2021"] == 0.81

    def test_full_per_season_season_break_columns(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(sample_historical_score_response)
        for season in ("2020", "2021", "2022", "2023", "2024"):
            assert f"season_break_{season}" in df.columns
        assert df.iloc[0]["season_break_2024"] == "2024-04-14"


# ===================================================================
# format_historical_score_json() — summary detail
# ===================================================================


class TestFormatHistoricalScoreJsonSummary:
    """detail_level='summary' → summary columns only, no per-season columns."""

    def test_summary_creates_one_row(self, configured_historical_score_extractor, sample_historical_score_response):
        df = configured_historical_score_extractor.format_historical_score_json(
            sample_historical_score_response, detail_level="summary"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_summary_omits_per_season_columns(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(
            sample_historical_score_response, detail_level="summary"
        )
        assert not any(c.startswith("potential_score_") for c in df.columns)
        assert not any(c.startswith("season_break_") for c in df.columns)

    def test_summary_keeps_summary_columns(
        self, configured_historical_score_extractor, sample_historical_score_response
    ):
        df = configured_historical_score_extractor.format_historical_score_json(
            sample_historical_score_response, detail_level="summary"
        )
        expected = {
            "entity_id",
            "average_potential_score",
            "olympic_mean_potential_score",
            "standard_deviation",
            "risk_score",
        }
        assert expected.issubset(set(df.columns))


# ===================================================================
# format_historical_score_json() — JSON parsing edge cases
# ===================================================================


class TestFormatHistoricalScoreJsonParsing:
    """Edge cases around the JSON-encoded PotentialScores / SeasonBreaks fields."""

    def test_potential_scores_already_a_list(self, configured_historical_score_extractor):
        """When PotentialScores is already a list (not JSON-encoded), formatter should still work."""
        response = {
            "id": "ent_x",
            "data": {
                "AveragePotentialScore": 0.6,
                "PotentialScores": [{"season": "2024", "potentialScore": 0.6}],
                "SeasonBreaks": [{"season": "2024", "seasonBreak": "2024-04-14"}],
            },
        }
        df = configured_historical_score_extractor.format_historical_score_json(response)
        assert df.iloc[0]["potential_score_2024"] == 0.6
        assert df.iloc[0]["season_break_2024"] == "2024-04-14"

    def test_invalid_json_string_yields_no_per_season_columns(self, configured_historical_score_extractor):
        """Malformed JSON in PotentialScores should be caught and produce no per-season cols."""
        response = {
            "id": "ent_x",
            "data": {
                "AveragePotentialScore": 0.6,
                "PotentialScores": "{not valid json",
                "SeasonBreaks": "[]",
            },
        }
        df = configured_historical_score_extractor.format_historical_score_json(response)
        assert not any(c.startswith("potential_score_") for c in df.columns)

    def test_missing_potential_scores_field(self, configured_historical_score_extractor):
        """If PotentialScores is absent entirely, default '[]' is used and no per-season cols emerge."""
        response = {
            "id": "ent_x",
            "data": {"AveragePotentialScore": 0.6, "SeasonBreaks": "[]"},
        }
        df = configured_historical_score_extractor.format_historical_score_json(response)
        assert not any(c.startswith("potential_score_") for c in df.columns)


# ===================================================================
# format_historical_score_json() — edge cases
# ===================================================================


class TestFormatHistoricalScoreJsonEdgeCases:
    def test_missing_data_key_returns_empty(self, configured_historical_score_extractor):
        df = configured_historical_score_extractor.format_historical_score_json({"id": "ent_x"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_data_none_returns_empty(self, configured_historical_score_extractor):
        df = configured_historical_score_extractor.format_historical_score_json({"id": "ent_x", "data": None})
        assert df.empty

    def test_unexpected_data_type_returns_empty(self, configured_historical_score_extractor):
        df = configured_historical_score_extractor.format_historical_score_json({"id": "ent_x", "data": "not a dict"})
        assert df.empty

    def test_invalid_detail_level_raises(self, configured_historical_score_extractor, sample_historical_score_response):
        with pytest.raises(ValueError, match="detail_level"):
            configured_historical_score_extractor.format_historical_score_json(
                sample_historical_score_response, detail_level="bogus"
            )

    def test_non_dict_response_raises(self, configured_historical_score_extractor):
        """A list at the top level isn't supported.

        Note: the dict-type check is *after* an unconditional `.get("id", ...)` call,
        so the actual error is AttributeError, not the documented ValueError. Both
        are valid failures for the caller (any Exception subclass works), but we
        document the current behaviour rather than the intent.
        """
        with pytest.raises(AttributeError):
            configured_historical_score_extractor.format_historical_score_json([{"id": "x"}])

    def test_missing_id_in_response_propagates_as_none(self, configured_historical_score_extractor):
        response = {
            "data": {"AveragePotentialScore": 0.6, "PotentialScores": "[]", "SeasonBreaks": "[]"},
        }
        df = configured_historical_score_extractor.format_historical_score_json(response)
        assert df.iloc[0]["entity_id"] is None
