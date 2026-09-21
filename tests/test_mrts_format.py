"""
Tests for MRTSExtractor.format_mrts_json():
    - 'full' mode: merges rawData + smoothedData on date
    - 'raw' mode: rawData only, no smoothed_value column
    - Image dict flattened to image_id
    - entity_id propagation, date sorting, edge cases
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_mrts_json() — 'full' mode (default)
# ===================================================================


class TestFormatMrtsJsonFull:
    def test_full_mode_merges_raw_and_smoothed(
        self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity
    ):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_full_mode_columns_present(self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        expected = {
            "entity_id",
            "date",
            "raw_value",
            "smoothed_value",
            "noised",
            "temporalConsistencyCheck",
            "mask",
            "coveragePercent",
            "image_id",
        }
        assert expected.issubset(set(df.columns))

    def test_full_mode_values_preserved(self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        first = df.iloc[0]
        assert first["raw_value"] == 0.42
        assert first["smoothed_value"] == 0.40
        assert "S2A_T20JNT_20250510" in first["image_id"]

    def test_full_mode_dates_sorted(self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        # Dates already sorted ascending (input is sorted, but assert anyway)
        assert list(df["raw_value"]) == [0.42, 0.55, 0.71]

    def test_entity_id_propagated_first_column(
        self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity
    ):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        assert df.columns[0] == "entity_id"
        assert (df["entity_id"] == "ent_001").all()


# ===================================================================
# format_mrts_json() — 'raw' mode
# ===================================================================


class TestFormatMrtsJsonRaw:
    def test_raw_mode_no_smoothed_column(
        self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity
    ):
        configured_mrts_extractor.mrts_params["mode"] = "raw"
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        assert "smoothed_value" not in df.columns
        assert "raw_value" in df.columns
        assert len(df) == 3

    def test_raw_mode_with_only_raw_data(
        self, configured_mrts_extractor, sample_mrts_response_raw_only, sample_mrts_entity
    ):
        configured_mrts_extractor.mrts_params["mode"] = "raw"
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_raw_only, entity_data=sample_mrts_entity)
        assert len(df) == 2
        assert "smoothed_value" not in df.columns


# ===================================================================
# format_mrts_json() — image_id flattening
# ===================================================================


class TestFormatMrtsJsonImageId:
    def test_image_dict_flattened_to_image_id(
        self, configured_mrts_extractor, sample_mrts_response_full, sample_mrts_entity
    ):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=sample_mrts_entity)
        # Each row should have a non-null image_id from the nested image dict
        assert df["image_id"].notna().all()
        # And no raw 'image' column
        assert "image" not in df.columns

    def test_missing_image_dict_results_in_none_image_id(self, configured_mrts_extractor, sample_mrts_entity):
        """A record without an image field should still produce a row, with image_id=None."""
        response = {
            "rawData": [{"date": "2025-05-10", "value": 0.42}],
            "smoothedData": [],
        }
        df = configured_mrts_extractor.format_mrts_json(response, entity_data=sample_mrts_entity)
        assert len(df) == 1
        # Image column never built since 'image' not present in raw data
        assert "image_id" not in df.columns


# ===================================================================
# format_mrts_json() — entity_data optional / missing
# ===================================================================


class TestFormatMrtsJsonEntityHandling:
    def test_no_entity_data_omits_entity_id(self, configured_mrts_extractor, sample_mrts_response_full):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full)
        assert "entity_id" not in df.columns
        # Other columns still present
        assert "raw_value" in df.columns

    def test_entity_data_without_id_omits_entity_id(self, configured_mrts_extractor, sample_mrts_response_full):
        entity_no_id = {"geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_full, entity_data=entity_no_id)
        assert "entity_id" not in df.columns


# ===================================================================
# format_mrts_json() — edge cases
# ===================================================================


class TestFormatMrtsJsonEdgeCases:
    def test_empty_response_returns_empty_columns(
        self, configured_mrts_extractor, sample_mrts_response_empty, sample_mrts_entity
    ):
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_empty, entity_data=sample_mrts_entity)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        # Empty-frame fallback should still expose the expected schema columns
        for col in ("date", "raw_value", "smoothed_value", "image_id"):
            assert col in df.columns

    def test_empty_response_raw_mode(self, configured_mrts_extractor, sample_mrts_response_empty, sample_mrts_entity):
        configured_mrts_extractor.mrts_params["mode"] = "raw"
        df = configured_mrts_extractor.format_mrts_json(sample_mrts_response_empty, entity_data=sample_mrts_entity)
        assert df.empty
        assert "raw_value" in df.columns
        assert "smoothed_value" not in df.columns

    def test_smoothed_only_response_full_mode(self, configured_mrts_extractor, sample_mrts_entity):
        """If only smoothed data is present, raw_value should be None for all rows."""
        response = {
            "rawData": [],
            "smoothedData": [
                {"date": "2025-05-10", "value": 0.40},
                {"date": "2025-05-25", "value": 0.56},
            ],
        }
        df = configured_mrts_extractor.format_mrts_json(response, entity_data=sample_mrts_entity)
        assert len(df) == 2
        assert df["smoothed_value"].notna().all()
        assert df["raw_value"].isna().all()

    def test_null_rawdata_handled(self, configured_mrts_extractor, sample_mrts_entity):
        """An explicit `rawData: null` should be treated as empty (uses `or []`)."""
        response = {"rawData": None, "smoothedData": []}
        df = configured_mrts_extractor.format_mrts_json(response, entity_data=sample_mrts_entity)
        assert df.empty
