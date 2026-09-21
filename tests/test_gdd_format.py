"""
Tests for GDDExtractor.format_gdd_json():
    - Long-form DataFrame construction (one row per day)
    - Date parsing and sort
    - entity_id propagation
    - Edge cases (None response, missing/empty 'elements', missing fields)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_gdd_json() — happy path
# ===================================================================


class TestFormatGddJsonHappyPath:
    """The GDD endpoint returns {"elements": [<daily record>, ...]}."""

    def test_creates_one_row_per_element(self, configured_gdd_extractor, sample_gdd_response, sample_gdd_entity):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=sample_gdd_entity)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_expected_columns_present(self, configured_gdd_extractor, sample_gdd_response, sample_gdd_entity):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=sample_gdd_entity)
        expected = {"entity_id", "date", "minimum", "maximum", "daily_gdd", "cumulated_gdd"}
        assert expected.issubset(set(df.columns))

    def test_entity_id_propagated_to_each_row(self, configured_gdd_extractor, sample_gdd_response, sample_gdd_entity):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=sample_gdd_entity)
        assert (df["entity_id"] == "test_001").all()

    def test_numeric_values_preserved(self, configured_gdd_extractor, sample_gdd_response, sample_gdd_entity):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=sample_gdd_entity)
        first = df.iloc[0]
        assert first["minimum"] == pytest.approx(18.5)
        assert first["maximum"] == pytest.approx(28.2)
        assert first["daily_gdd"] == pytest.approx(13.35)
        assert first["cumulated_gdd"] == pytest.approx(13.35)

    def test_date_parsed_to_datetime(self, configured_gdd_extractor, sample_gdd_response, sample_gdd_entity):
        """ISO timestamps with timezone should be parsed to a datetime dtype."""
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=sample_gdd_entity)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_dates_sorted_ascending(self, configured_gdd_extractor, sample_gdd_entity):
        """Out-of-order elements on input should be sorted ascending by date."""
        unsorted_response = {
            "elements": [
                {
                    "date": "2023-01-19T00:00:00Z",
                    "minimum": 17.8,
                    "maximum": 27.6,
                    "dailyGrowingDegreeDay": 12.7,
                    "cumulatedGrowingDegreeDay": 40.3,
                },
                {
                    "date": "2023-01-17T00:00:00Z",
                    "minimum": 18.5,
                    "maximum": 28.2,
                    "dailyGrowingDegreeDay": 13.35,
                    "cumulatedGrowingDegreeDay": 13.35,
                },
                {
                    "date": "2023-01-18T00:00:00Z",
                    "minimum": 19.1,
                    "maximum": 29.4,
                    "dailyGrowingDegreeDay": 14.25,
                    "cumulatedGrowingDegreeDay": 27.6,
                },
            ]
        }
        df = configured_gdd_extractor.format_gdd_json(unsorted_response, entity_data=sample_gdd_entity)
        assert list(df["daily_gdd"]) == [13.35, 14.25, 12.7]


# ===================================================================
# format_gdd_json() — entity_data optional / missing
# ===================================================================


class TestFormatGddJsonNoEntity:
    """When entity_data is not provided, the entity_id column is omitted."""

    def test_no_entity_data_omits_entity_id_column(self, configured_gdd_extractor, sample_gdd_response):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response)
        assert "entity_id" not in df.columns
        # Other columns should still be present
        assert {"date", "minimum", "maximum", "daily_gdd", "cumulated_gdd"}.issubset(set(df.columns))

    def test_entity_data_without_id_omits_entity_id_column(self, configured_gdd_extractor, sample_gdd_response):
        """If entity dict lacks 'id' key, entity_id column is not added."""
        entity_no_id = {"geometry": "POINT (0 0)"}
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response, entity_data=entity_no_id)
        assert "entity_id" not in df.columns


# ===================================================================
# format_gdd_json() — edge cases
# ===================================================================


class TestFormatGddJsonEdgeCases:
    """Edge cases via the validate_api_response gate and unexpected shapes."""

    def test_none_response_returns_empty_df(self, configured_gdd_extractor):
        df = configured_gdd_extractor.format_gdd_json(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_dict_returns_empty_df(self, configured_gdd_extractor):
        df = configured_gdd_extractor.format_gdd_json({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_elements_returns_empty_df(self, configured_gdd_extractor, sample_gdd_response_empty_elements):
        df = configured_gdd_extractor.format_gdd_json(sample_gdd_response_empty_elements)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_elements_null_returns_empty_df(self, configured_gdd_extractor):
        """An explicit `elements: null` should be treated as empty (uses `or []`)."""
        df = configured_gdd_extractor.format_gdd_json({"elements": None})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_date_field_handled(self, configured_gdd_extractor):
        """A record with no 'date' should still produce a row (date=NaT after to_datetime)."""
        response = {
            "elements": [
                {
                    "minimum": 10.0,
                    "maximum": 20.0,
                    "dailyGrowingDegreeDay": 5.0,
                    "cumulatedGrowingDegreeDay": 5.0,
                }
            ]
        }
        df = configured_gdd_extractor.format_gdd_json(response)
        assert len(df) == 1
        # Missing date became NaT
        assert pd.isna(df.iloc[0]["date"])
        assert df.iloc[0]["minimum"] == 10.0

    def test_missing_numeric_fields_become_nan(self, configured_gdd_extractor):
        """A record missing numeric fields should still produce a row with NaN values."""
        response = {"elements": [{"date": "2023-01-17"}]}
        df = configured_gdd_extractor.format_gdd_json(response)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["minimum"])
        assert pd.isna(df.iloc[0]["maximum"])
        assert pd.isna(df.iloc[0]["daily_gdd"])
        assert pd.isna(df.iloc[0]["cumulated_gdd"])

    def test_invalid_date_string_raises(self, configured_gdd_extractor, sample_gdd_entity):
        """
        An unparseable date in any element causes format_gdd_json to raise.

        The per-row loop has a try/except that falls back to the raw string, but the
        trailing `pd.to_datetime(df["date"])` (line 367) is unguarded and propagates
        DateParseError (subclass of ValueError) when the column contains a junk value.
        Documenting this contract so callers know to pre-validate dates.
        """
        response = {
            "elements": [
                {
                    "date": "not-a-date",
                    "minimum": 10.0,
                    "maximum": 20.0,
                    "dailyGrowingDegreeDay": 5.0,
                    "cumulatedGrowingDegreeDay": 5.0,
                }
            ]
        }
        with pytest.raises(ValueError):
            configured_gdd_extractor.format_gdd_json(response, entity_data=sample_gdd_entity)
