"""
Tests for FLMExtractor format methods (one per postprocess mode):
    - format_flm_map_stats_json()      → stats DataFrame (+ optional ranges)
    - format_flm_map_histogram_json()  → wide single-row histogram DataFrame
    - format_flm_links_json()          → links + worldFile + bbox + map_size
    - format_flm_map_file()            → saves PNG / TIFF.ZIP / SHP.ZIP to disk
"""

import os
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZipFile

import pandas as pd
import pytest

from tests.conftest import FLM_IMAGE_ID

pytestmark = pytest.mark.public

# ===================================================================
# format_flm_map_stats_json()
# ===================================================================


class TestFormatFlmMapStatsJson:
    """stats mode produces one row per entity with stat_max / stat_mean / stat_min."""

    def test_creates_single_row(self, configured_flm_extractor, sample_flm_stats_response):
        df = configured_flm_extractor.format_flm_map_stats_json(sample_flm_stats_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_stat_columns_present(self, configured_flm_extractor, sample_flm_stats_response):
        df = configured_flm_extractor.format_flm_map_stats_json(sample_flm_stats_response)
        assert {"stat_max", "stat_mean", "stat_min"}.issubset(set(df.columns))

    def test_stat_values_preserved(self, configured_flm_extractor, sample_flm_stats_response):
        df = configured_flm_extractor.format_flm_map_stats_json(sample_flm_stats_response)
        row = df.iloc[0]
        assert row["stat_max"] == 0.85
        assert row["stat_mean"] == 0.62
        assert row["stat_min"] == 0.21

    def test_range_kwarg_returns_tuple(self, configured_flm_extractor, sample_flm_stats_response):
        """When range=True, the formatter returns (stats_df, ranges_df)."""
        result = configured_flm_extractor.format_flm_map_stats_json(sample_flm_stats_response, range=True)
        assert isinstance(result, tuple)
        stats_df, ranges_df = result
        assert len(stats_df) == 1
        assert len(ranges_df) == 3
        assert {"min_value", "max_value", "num_pixels", "season_field_id"}.issubset(set(ranges_df.columns))

    def test_range_values_preserved(self, configured_flm_extractor, sample_flm_stats_response):
        _, ranges_df = configured_flm_extractor.format_flm_map_stats_json(sample_flm_stats_response, range=True)
        first = ranges_df.iloc[0]
        assert first["min_value"] == 0.0
        assert first["max_value"] == 0.3
        assert first["num_pixels"] == 120
        assert first["season_field_id"] == "test_001"

    def test_response_object_auto_parsed(self, configured_flm_extractor, sample_flm_stats_response):
        """A requests.Response-like object should be auto-parsed via .json()."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_stats_response
        df = configured_flm_extractor.format_flm_map_stats_json(mock_resp)
        assert df.iloc[0]["stat_max"] == 0.85

    def test_response_object_invalid_json_returns_empty(self, configured_flm_extractor):
        """A response that fails .json() should return an empty DataFrame, not raise."""
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not JSON")
        df = configured_flm_extractor.format_flm_map_stats_json(mock_resp)
        assert df.empty

    def test_empty_dict_returns_row_with_nones(self, configured_flm_extractor):
        """Missing legend → stats columns are None but the single row is still produced."""
        df = configured_flm_extractor.format_flm_map_stats_json({})
        assert len(df) == 1
        assert df.iloc[0]["stat_max"] is None


# ===================================================================
# format_flm_map_histogram_json()
# ===================================================================


class TestFormatFlmMapHistogramJson:
    """histogram mode produces one wide row with global stats + per-bucket fields."""

    def test_creates_single_row(self, configured_flm_extractor, sample_flm_histogram_response):
        df = configured_flm_extractor.format_flm_map_histogram_json(sample_flm_histogram_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_global_stats_present(self, configured_flm_extractor, sample_flm_histogram_response):
        df = configured_flm_extractor.format_flm_map_histogram_json(sample_flm_histogram_response)
        row = df.iloc[0]
        assert row["stat_min"] == 0.21
        assert row["stat_max"] == 0.85
        assert row["stat_mean"] == 0.62

    def test_per_bucket_columns_flattened(self, configured_flm_extractor, sample_flm_histogram_response):
        """Each bucket should produce value_min_{i} / value_max_{i} / num_pixels_{i} / area_{i}."""
        df = configured_flm_extractor.format_flm_map_histogram_json(sample_flm_histogram_response)
        for i in (1, 2, 3):
            for suffix in ("value_min", "value_max", "num_pixels", "area"):
                assert f"{suffix}_{i}" in df.columns

    def test_bucket_values_preserved(self, configured_flm_extractor, sample_flm_histogram_response):
        df = configured_flm_extractor.format_flm_map_histogram_json(sample_flm_histogram_response)
        row = df.iloc[0]
        assert row["value_min_1"] == 0.21
        assert row["value_max_1"] == 0.40
        assert row["num_pixels_1"] == 120
        assert row["area_1"] == 1200.0
        assert row["num_pixels_3"] == 540

    def test_no_buckets_returns_stats_only(self, configured_flm_extractor):
        """Empty items list still produces a row but only the global-stat columns."""
        response = {
            "seasonField": {"id": "test_001"},
            "histogram": {"min": 0.1, "max": 0.9, "mean": 0.5, "items": []},
        }
        df = configured_flm_extractor.format_flm_map_histogram_json(response)
        assert len(df) == 1
        assert df.iloc[0]["stat_min"] == 0.1
        assert not any(c.startswith("value_min_") for c in df.columns)

    def test_missing_histogram_key_returns_stats_only_row(self, configured_flm_extractor):
        """Response without a histogram block still returns a single row (all None stats)."""
        df = configured_flm_extractor.format_flm_map_histogram_json({"seasonField": {"id": "x"}})
        assert len(df) == 1
        assert df.iloc[0]["stat_min"] is None


# ===================================================================
# format_flm_links_json()
# ===================================================================


class TestFormatFlmLinksJson:
    """links mode flattens _links / worldFile / mapSize / bBox into one row."""

    def test_creates_single_row(self, configured_flm_extractor, sample_flm_links_response):
        df = configured_flm_extractor.format_flm_links_json(sample_flm_links_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_link_columns_present(self, configured_flm_extractor, sample_flm_links_response):
        df = configured_flm_extractor.format_flm_links_json(sample_flm_links_response)
        row = df.iloc[0]
        assert row["image_png_link"] == "https://api.example.com/flm/test_001.png"
        assert row["worldfile_link"] == "https://api.example.com/flm/test_001.pgw"
        assert row["thumbnail_link"] == "https://api.example.com/flm/test_001_thumb.png"

    def test_worldfile_columns_present(self, configured_flm_extractor, sample_flm_links_response):
        df = configured_flm_extractor.format_flm_links_json(sample_flm_links_response)
        row = df.iloc[0]
        assert row["worldfile_a"] == 1e-5
        assert row["worldfile_c"] == -97.7
        assert row["worldfile_f"] == 37.14

    def test_map_size_and_bbox_present(self, configured_flm_extractor, sample_flm_links_response):
        df = configured_flm_extractor.format_flm_links_json(sample_flm_links_response)
        row = df.iloc[0]
        assert row["map_width"] == 512
        assert row["map_height"] == 512
        assert row["bbox_xmin"] == -97.703
        assert row["bbox_ymax"] == 37.145

    def test_response_object_auto_parsed(self, configured_flm_extractor, sample_flm_links_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_links_response
        df = configured_flm_extractor.format_flm_links_json(mock_resp)
        assert df.iloc[0]["map_width"] == 512

    def test_response_object_invalid_json_returns_empty(self, configured_flm_extractor):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not JSON")
        df = configured_flm_extractor.format_flm_links_json(mock_resp)
        assert df.empty

    def test_missing_links_returns_row_with_nones(self, configured_flm_extractor):
        """Empty response still produces a single row with None values."""
        df = configured_flm_extractor.format_flm_links_json({})
        assert len(df) == 1
        assert df.iloc[0]["image_png_link"] is None


# ===================================================================
# format_flm_map_file() — file mode (PNG / TIFF.ZIP / SHP.ZIP)
# ===================================================================


def _make_zip_bytes(name_to_bytes: dict) -> bytes:
    """Helper: build an in-memory ZIP archive holding the given files."""
    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        for name, payload in name_to_bytes.items():
            zf.writestr(name, payload)
    return buf.getvalue()


class TestFormatFlmMapFilePng:
    """PNG: write response.content as a single .png file."""

    def test_png_save_success(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        configured_flm_extractor.flm_params["map_format"] = "png"
        mock_resp = MagicMock()
        mock_resp.content = b"PNG_BYTES"

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["map_format"] == "png"
        assert result["file_count"] == 1
        assert result["total_size_bytes"] == len(mock_resp.content)
        # The PNG file should exist on disk
        assert os.path.exists(result["saved_files"][0])
        with open(result["saved_files"][0], "rb") as f:
            assert f.read() == b"PNG_BYTES"

    def test_png_filename_uses_entity_name(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        """When entity has a 'name', filename should be name_id_imageid_index.png."""
        configured_flm_extractor.flm_params["map_format"] = "png"
        mock_resp = MagicMock()
        mock_resp.content = b"PNG"

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        saved = result["saved_files"][0]
        # entity 'name' is "Test_Field", id is "toto", index is "NDVI"
        assert "Test_Field" in saved
        assert "toto" in saved
        assert "NDVI" in saved


class TestFormatFlmMapFileTiff:
    """TIFF.ZIP: response.content is a ZIP; extract the .tif inside."""

    def test_tiff_zip_extracts_tif(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        configured_flm_extractor.flm_params["map_format"] = "tiff.zip"
        zip_bytes = _make_zip_bytes({"map.tif": b"TIFF_DATA"})
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["map_format"] == "tiff.zip"
        assert result["file_count"] == 1
        # The .tif should exist on disk
        saved = result["saved_files"][0]
        assert saved.endswith(".tif")
        with open(saved, "rb") as f:
            assert f.read() == b"TIFF_DATA"

    def test_tiff_zip_without_tif_returns_error(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        """If the ZIP doesn't contain a .tif, the formatter returns status='error'."""
        configured_flm_extractor.flm_params["map_format"] = "tiff.zip"
        zip_bytes = _make_zip_bytes({"readme.txt": b"hi"})
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "error"
        assert "No .tif file" in result["error_message"]


class TestFormatFlmMapFileShp:
    """SHP.ZIP: extract every component (.shp, .shx, .dbf, .prj, ...)."""

    def test_shp_zip_extracts_all_components(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        configured_flm_extractor.flm_params["map_format"] = "shp.zip"
        zip_bytes = _make_zip_bytes(
            {
                "shape.shp": b"SHP",
                "shape.shx": b"SHX",
                "shape.dbf": b"DBF",
                "shape.prj": b"PRJ",
            }
        )
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["map_format"] == "shp.zip"
        assert result["file_count"] == 4
        extensions = {os.path.splitext(p)[1] for p in result["saved_files"]}
        assert extensions == {".shp", ".shx", ".dbf", ".prj"}

    def test_shp_zip_empty_archive_returns_error(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        configured_flm_extractor.flm_params["map_format"] = "shp.zip"
        zip_bytes = _make_zip_bytes({})
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "error"


class TestFormatFlmMapFileEdgeCases:
    """Cross-cutting edge cases for format_flm_map_file."""

    def test_unsupported_map_format_returns_error(self, configured_flm_extractor, sample_flm_entity, tmp_path):
        configured_flm_extractor.flm_params["map_format"] = "weird"
        mock_resp = MagicMock()
        mock_resp.content = b""

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        assert result["status"] == "error"
        assert "Unsupported map_format" in result["error_message"]

    def test_no_output_path_returns_error(self, configured_flm_extractor, sample_flm_entity):
        """If neither output_path arg nor params['output_path'] is set, returns error."""
        configured_flm_extractor.flm_params["map_format"] = "png"
        configured_flm_extractor.flm_params["output_path"] = None
        configured_flm_extractor.output_path = None  # also unset BaseExtractor fallback
        mock_resp = MagicMock()
        mock_resp.content = b"PNG"

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=sample_flm_entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=None,
        )

        assert result["status"] == "error"
        assert "No output path" in result["error_message"]

    def test_filename_sanitizes_invalid_chars(self, configured_flm_extractor, tmp_path):
        """Windows-invalid characters in id/name should be replaced with underscores."""
        configured_flm_extractor.flm_params["map_format"] = "png"
        # ':' and '|' are invalid on Windows; FLM_IMAGE_ID contains '|' so this is realistic.
        entity = {"id": "ent:001", "name": "field|x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        mock_resp = MagicMock()
        mock_resp.content = b"PNG"

        result = configured_flm_extractor.format_flm_map_file(
            entity_data=entity,
            image_id=FLM_IMAGE_ID,
            response=mock_resp,
            output_path=str(tmp_path),
        )

        saved = result["saved_files"][0]
        basename = os.path.basename(saved)
        # No invalid Windows characters should remain
        for bad in ':"<>|?*':
            assert bad not in basename
