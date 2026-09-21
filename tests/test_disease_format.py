"""
Tests for DiseaseExtractor.format_disease_json():
    - List response (notebook shape — daily disease records)
    - Dict wrappers ('data' / 'results' keys)
    - Nested dict flattening to dotted column names
    - Column ordering (entity_id → date → alphabetical disease cols)
    - Date conversion and sort
    - Empty / invalid responses
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_disease_json() — list response (notebook shape)
# ===================================================================


class TestFormatDiseaseJsonList:
    """The disease /launch endpoint returns a JSON list of daily records."""

    def test_list_creates_one_row_per_record(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_list, entity_data=sample_disease_entity
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_expected_columns_present(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_list, entity_data=sample_disease_entity
        )

        # All disease parameters from the fixture should become columns
        expected = {
            "entity_id",
            "date",
            "frogeye_leaf_spot",
            "giberella_ear_rot",
            "gray_leaf_spot",
            "tar_spot",
            "white_mold_dry",
            "white_mold_irr_15",
            "white_mold_irr_30",
        }
        assert expected.issubset(set(df.columns))

    def test_column_order_entity_id_then_date_then_alpha(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        """Column order is entity_id → date → alphabetically sorted disease params."""
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_list, entity_data=sample_disease_entity
        )

        cols = list(df.columns)
        assert cols[0] == "entity_id"
        assert cols[1] == "date"
        # Remaining disease columns must be alphabetical
        disease_cols = cols[2:]
        assert disease_cols == sorted(disease_cols)

    def test_entity_id_propagated_to_each_row(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_list, entity_data=sample_disease_entity
        )
        assert (df["entity_id"] == "z361x33").all()

    def test_date_converted_to_datetime_and_sorted(self, configured_disease_extractor, sample_disease_entity):
        """Records out of order on input should be sorted ascending by date."""
        unsorted_records = [
            {"date": "2025-06-03", "frogeye_leaf_spot": 0.3},
            {"date": "2025-06-01", "frogeye_leaf_spot": 0.1},
            {"date": "2025-06-02", "frogeye_leaf_spot": 0.2},
        ]
        df = configured_disease_extractor.format_disease_json(unsorted_records, entity_data=sample_disease_entity)

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert list(df["frogeye_leaf_spot"]) == [0.1, 0.2, 0.3]

    def test_numeric_values_preserved(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_list, entity_data=sample_disease_entity
        )

        first = df.iloc[0]
        assert first["frogeye_leaf_spot"] == pytest.approx(0.4658)
        assert first["white_mold_irr_30"] == pytest.approx(0.522747)


# ===================================================================
# format_disease_json() — entity_data optional / missing
# ===================================================================


class TestFormatDiseaseJsonNoEntity:
    """When entity_data is not provided, output omits the entity_id column."""

    def test_no_entity_data_omits_entity_id_column(self, configured_disease_extractor, sample_disease_response_list):
        df = configured_disease_extractor.format_disease_json(sample_disease_response_list)

        assert "entity_id" not in df.columns
        cols = list(df.columns)
        assert cols[0] == "date"

    def test_entity_data_without_id_field_omits_entity_id(
        self, configured_disease_extractor, sample_disease_response_list
    ):
        """If entity dict lacks an 'id' key, no entity_id column is added."""
        entity_no_id = {"geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        df = configured_disease_extractor.format_disease_json(sample_disease_response_list, entity_data=entity_no_id)
        assert "entity_id" not in df.columns


# ===================================================================
# format_disease_json() — dict wrappers
# ===================================================================


class TestFormatDiseaseJsonDictWrappers:
    """Formatter unwraps dict responses with 'data' or 'results' keys."""

    def test_dict_with_data_key(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        wrapped = {"data": sample_disease_response_list}
        df = configured_disease_extractor.format_disease_json(wrapped, entity_data=sample_disease_entity)
        assert len(df) == 3

    def test_dict_with_results_key(
        self, configured_disease_extractor, sample_disease_response_list, sample_disease_entity
    ):
        wrapped = {"results": sample_disease_response_list}
        df = configured_disease_extractor.format_disease_json(wrapped, entity_data=sample_disease_entity)
        assert len(df) == 3


# ===================================================================
# format_disease_json() — nested dict flattening
# ===================================================================


class TestFormatDiseaseJsonNestedFlattening:
    """Nested dicts in disease records should be flattened to dotted column names."""

    def test_nested_dicts_flattened(
        self, configured_disease_extractor, sample_disease_response_with_nested, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_with_nested, entity_data=sample_disease_entity
        )

        assert "metadata.sensor" in df.columns
        assert "metadata.confidence" in df.columns
        # Original nested key should NOT appear as a dict-typed column
        assert "metadata" not in df.columns

    def test_nested_values_preserved(
        self, configured_disease_extractor, sample_disease_response_with_nested, sample_disease_entity
    ):
        df = configured_disease_extractor.format_disease_json(
            sample_disease_response_with_nested, entity_data=sample_disease_entity
        )
        assert (df["metadata.sensor"] == "S2").all()
        assert df.iloc[0]["metadata.confidence"] == pytest.approx(0.9)


# ===================================================================
# format_disease_json() — empty / edge cases
# ===================================================================


class TestFormatDiseaseJsonEdgeCases:
    """Edge cases via the validate_api_response gate and unexpected shapes."""

    def test_empty_list_returns_empty_df(self, configured_disease_extractor):
        df = configured_disease_extractor.format_disease_json([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_none_returns_empty_df(self, configured_disease_extractor):
        df = configured_disease_extractor.format_disease_json(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_dict_returns_empty_df(self, configured_disease_extractor):
        df = configured_disease_extractor.format_disease_json({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_unexpected_type_returns_empty_df(self, configured_disease_extractor):
        """A non-list/dict response (e.g. a string) returns an empty DataFrame, not a raise."""
        df = configured_disease_extractor.format_disease_json("not a list")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_dict_with_empty_data_key_returns_empty(self, configured_disease_extractor):
        df = configured_disease_extractor.format_disease_json({"data": []})
        assert df.empty
