"""
Tests for WeatherExtractor.format_weather_json():
    - Multi-row DataFrame from list of weather records (one row per day)
    - Nested-dict flattening (Temperature.standard, precipitation.cumulative, etc.)
    - Column ordering (entity_id → date → alphabetical params)
    - Date conversion + sort
    - Edge cases (empty list, None, missing fields, missing id)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_weather_json() — happy path
# ===================================================================


class TestFormatWeatherJsonHappyPath:
    """The weather endpoint returns a list of daily records with nested fields."""

    def test_creates_one_row_per_record(
        self, configured_weather_extractor, sample_weather_response, sample_weather_entity
    ):
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_nested_dicts_flattened(self, configured_weather_extractor, sample_weather_response, sample_weather_entity):
        """Temperature dict should be flattened to Temperature.standard / Temperature.standardMin / etc."""
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        assert "Temperature.standard" in df.columns
        assert "Temperature.standardMin" in df.columns
        assert "Temperature.standardMax" in df.columns
        assert "precipitation.cumulative" in df.columns
        # Original nested keys should NOT remain as raw dict columns
        assert "Temperature" not in df.columns
        assert "precipitation" not in df.columns

    def test_scalar_values_preserved(
        self, configured_weather_extractor, sample_weather_response, sample_weather_entity
    ):
        """Scalar fields (e.g. wind) appear unchanged."""
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        assert "wind" in df.columns
        # First record's wind=3.2
        assert df.iloc[0]["wind"] == 3.2

    def test_column_order_entity_id_then_date_then_alpha(
        self, configured_weather_extractor, sample_weather_response, sample_weather_entity
    ):
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        cols = list(df.columns)
        assert cols[0] == "entity_id"
        assert cols[1] == "date"
        # Remaining weather columns must be alphabetical
        weather_cols = cols[2:]
        assert weather_cols == sorted(weather_cols)

    def test_entity_id_propagated_to_each_row(
        self, configured_weather_extractor, sample_weather_response, sample_weather_entity
    ):
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        assert (df["entity_id"] == "z361x33").all()

    def test_date_converted_to_datetime_and_sorted(self, configured_weather_extractor, sample_weather_entity):
        """Out-of-order records on input should be sorted ascending by date."""
        unsorted = [
            {"date": "2025-06-03T00:00:00Z", "wind": 2.8},
            {"date": "2025-06-01T00:00:00Z", "wind": 3.2},
            {"date": "2025-06-02T00:00:00Z", "wind": 4.1},
        ]
        df = configured_weather_extractor.format_weather_json(unsorted, entity_data=sample_weather_entity)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert list(df["wind"]) == [3.2, 4.1, 2.8]

    def test_numeric_values_preserved(
        self, configured_weather_extractor, sample_weather_response, sample_weather_entity
    ):
        df = configured_weather_extractor.format_weather_json(
            sample_weather_response, entity_data=sample_weather_entity
        )
        first = df.iloc[0]
        assert first["Temperature.standard"] == 18.5
        assert first["Temperature.standardMin"] == 12.3
        assert first["Temperature.standardMax"] == 24.1
        assert first["precipitation.cumulative"] == 0.0


# ===================================================================
# format_weather_json() — entity_data optional
# ===================================================================


class TestFormatWeatherJsonNoEntity:
    def test_no_entity_data_omits_entity_id_column(self, configured_weather_extractor, sample_weather_response):
        df = configured_weather_extractor.format_weather_json(sample_weather_response)
        assert "entity_id" not in df.columns
        cols = list(df.columns)
        assert cols[0] == "date"

    def test_entity_data_without_id_omits_entity_id_column(self, configured_weather_extractor, sample_weather_response):
        entity_no_id = {"geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        df = configured_weather_extractor.format_weather_json(sample_weather_response, entity_data=entity_no_id)
        assert "entity_id" not in df.columns


# ===================================================================
# format_weather_json() — edge cases
# ===================================================================


class TestFormatWeatherJsonEdgeCases:
    def test_empty_list_returns_empty_df(self, configured_weather_extractor):
        df = configured_weather_extractor.format_weather_json([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_none_returns_empty_df(self, configured_weather_extractor):
        df = configured_weather_extractor.format_weather_json(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_field_becomes_partial_row(self, configured_weather_extractor):
        """A record missing some fields still produces a row; missing flattened cols become NaN
        once concatenated with other rows."""
        response = [
            {"date": "2025-06-01T00:00:00Z", "wind": 3.2},
            {"date": "2025-06-02T00:00:00Z", "Temperature": {"standard": 19.0}, "wind": 4.1},
        ]
        df = configured_weather_extractor.format_weather_json(response)
        assert len(df) == 2
        # First row's Temperature.standard is NaN; second row's value is preserved
        assert pd.isna(df.iloc[0]["Temperature.standard"])
        assert df.iloc[1]["Temperature.standard"] == 19.0
