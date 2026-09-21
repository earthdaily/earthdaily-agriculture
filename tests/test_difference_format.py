"""
Tests for DifferenceExtractor format methods:
    - format_difference_stats_json() — flattens legend.stat + legend.ranges
    - format_difference_links_json() — flattens _links/worldFile/mapSize/bBox
    - format_difference_file() — saves PNG / TIFF.ZIP / SHP.ZIP responses to disk
"""

import os
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZipFile

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_difference_stats_json()
# ===================================================================


class TestFormatDifferenceStatsJson:
    """Stats mode: one row per entity with summary stats + flattened ranges."""

    def test_creates_single_row(self, configured_difference_extractor, sample_difference_stats_response):
        df = configured_difference_extractor.format_difference_stats_json(sample_difference_stats_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_stat_fields_extracted(self, configured_difference_extractor, sample_difference_stats_response):
        df = configured_difference_extractor.format_difference_stats_json(sample_difference_stats_response)
        row = df.iloc[0]
        assert row["stat_max"] == 0.42
        assert row["stat_mean"] == 0.05
        assert row["stat_min"] == -0.31

    def test_range_columns_flattened_per_bucket(
        self, configured_difference_extractor, sample_difference_stats_response
    ):
        """Each range should produce range_{i}_min/max/pixels/area/color_r/g/b columns."""
        df = configured_difference_extractor.format_difference_stats_json(sample_difference_stats_response)
        row = df.iloc[0]
        # Fixture has 3 ranges → range_1, range_2, range_3
        for i in (1, 2, 3):
            for suffix in ("min", "max", "pixels", "area", "color_r", "color_g", "color_b"):
                assert f"range_{i}_{suffix}" in df.columns
        # Spot-check first and last range values
        assert row["range_1_min"] == -0.5
        assert row["range_1_max"] == -0.2
        assert row["range_1_pixels"] == 120
        assert row["range_1_area"] == 1200.0
        assert row["range_1_color_r"] == 200
        assert row["range_3_pixels"] == 540
        assert row["range_3_color_g"] == 200

    def test_no_ranges_only_stats(self, configured_difference_extractor):
        """A response with stat but no ranges still yields a stats-only row."""
        df = configured_difference_extractor.format_difference_stats_json(
            {"legend": {"stat": {"max": 0.1, "mean": 0.0, "min": -0.1}, "ranges": []}}
        )
        assert len(df) == 1
        assert df.iloc[0]["stat_max"] == 0.1
        # No range_* columns when ranges=[]
        assert not any(c.startswith("range_") for c in df.columns)

    def test_empty_response_returns_empty_df(self, configured_difference_extractor):
        assert configured_difference_extractor.format_difference_stats_json({}).empty
        assert configured_difference_extractor.format_difference_stats_json(None).empty

    def test_response_object_is_parsed(self, configured_difference_extractor, sample_difference_stats_response):
        """A requests.Response-like object should be auto-parsed via .json()."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_difference_stats_response

        df = configured_difference_extractor.format_difference_stats_json(mock_response)
        assert len(df) == 1
        assert df.iloc[0]["stat_max"] == 0.42

    def test_response_object_with_invalid_json_returns_empty(self, configured_difference_extractor):
        """If .json() raises ValueError, we get an empty DataFrame (no exception)."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("not json")

        df = configured_difference_extractor.format_difference_stats_json(mock_response)
        assert df.empty


# ===================================================================
# format_difference_links_json()
# ===================================================================


class TestFormatDifferenceLinksJson:
    """Links mode: one row per entity flattening _links/worldFile/mapSize/bBox."""

    def test_creates_single_row_with_expected_columns(
        self, configured_difference_extractor, sample_difference_links_response
    ):
        df = configured_difference_extractor.format_difference_links_json(sample_difference_links_response)
        assert len(df) == 1

        expected = {
            "image_png_link",
            "worldfile_link",
            "thumbnail_link",
            "worldfile_a",
            "worldfile_b",
            "worldfile_c",
            "worldfile_d",
            "worldfile_e",
            "worldfile_f",
            "map_width",
            "map_height",
            "bbox_xmin",
            "bbox_xmax",
            "bbox_ymin",
            "bbox_ymax",
        }
        assert expected.issubset(set(df.columns))

    def test_link_values_preserved(self, configured_difference_extractor, sample_difference_links_response):
        df = configured_difference_extractor.format_difference_links_json(sample_difference_links_response)
        row = df.iloc[0]
        assert row["image_png_link"] == "https://api.example.com/diff/test_001.png"
        assert row["worldfile_link"] == "https://api.example.com/diff/test_001.pgw"
        assert row["thumbnail_link"] == "https://api.example.com/diff/test_001_thumb.png"

    def test_worldfile_values_preserved(self, configured_difference_extractor, sample_difference_links_response):
        df = configured_difference_extractor.format_difference_links_json(sample_difference_links_response)
        row = df.iloc[0]
        assert row["worldfile_a"] == 1e-5
        assert row["worldfile_c"] == -97.7
        assert row["worldfile_e"] == -1e-5
        assert row["worldfile_f"] == 37.14

    def test_map_size_and_bbox_preserved(self, configured_difference_extractor, sample_difference_links_response):
        df = configured_difference_extractor.format_difference_links_json(sample_difference_links_response)
        row = df.iloc[0]
        assert row["map_width"] == 512
        assert row["map_height"] == 512
        assert row["bbox_xmin"] == -97.703
        assert row["bbox_ymax"] == 37.145

    def test_missing_sections_yield_none_columns(self, configured_difference_extractor):
        """A bare response without sections still produces a row with all None values."""
        df = configured_difference_extractor.format_difference_links_json({})
        assert len(df) == 1
        assert df.iloc[0]["image_png_link"] is None
        assert df.iloc[0]["worldfile_a"] is None
        assert df.iloc[0]["map_width"] is None
        assert df.iloc[0]["bbox_xmin"] is None

    def test_response_object_is_parsed(self, configured_difference_extractor, sample_difference_links_response):
        mock_response = MagicMock()
        mock_response.json.return_value = sample_difference_links_response

        df = configured_difference_extractor.format_difference_links_json(mock_response)
        assert df.iloc[0]["image_png_link"] == "https://api.example.com/diff/test_001.png"


# ===================================================================
# format_difference_file()
# ===================================================================


class TestFormatDifferenceFile:
    """File mode: write PNG / TIFF / SHP responses to disk."""

    def test_png_save_writes_single_file(self, configured_difference_extractor, sample_difference_entity, tmp_path):
        configured_difference_extractor.difference_params["map_format"] = "png"
        mock_response = MagicMock()
        mock_response.content = b"\x89PNG\r\n\x1a\nFAKE_PNG_BYTES"

        result = configured_difference_extractor.format_difference_file(
            entity_data=sample_difference_entity,
            response=mock_response,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["map_format"] == "png"
        assert result["file_count"] == 1
        assert result["total_size_bytes"] == len(mock_response.content)
        # File written to disk with the expected content
        assert len(result["saved_files"]) == 1
        saved = result["saved_files"][0]
        assert saved.endswith(".png")
        assert os.path.exists(saved)
        with open(saved, "rb") as fh:
            assert fh.read() == mock_response.content

    def test_filename_uses_entity_name_when_present(self, configured_difference_extractor, tmp_path):
        configured_difference_extractor.difference_params["map_format"] = "png"
        entity = {
            "id": "test_001",
            "name": "Test_Field",
            "image_id_1": "a|b",
            "image_id_2": "c|d",
        }
        mock_response = MagicMock()
        mock_response.content = b"PNG"

        result = configured_difference_extractor.format_difference_file(
            entity_data=entity, response=mock_response, output_path=str(tmp_path)
        )

        saved = result["saved_files"][0]
        # Filename pattern: <name>_<id>_<product>.png
        assert "Test_Field_test_001_DIFFERENCE_NDVI.png" in saved

    def test_tiff_zip_extracts_single_tif(self, configured_difference_extractor, sample_difference_entity, tmp_path):
        configured_difference_extractor.difference_params["map_format"] = "tiff.zip"
        # Build an in-memory zip with a single .tif entry
        buf = BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("difference.tif", b"FAKE_TIFF_BYTES")
            zf.writestr("metadata.txt", b"unrelated")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()

        result = configured_difference_extractor.format_difference_file(
            entity_data=sample_difference_entity,
            response=mock_response,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["file_count"] == 1
        saved = result["saved_files"][0]
        assert saved.endswith(".tif")
        with open(saved, "rb") as fh:
            assert fh.read() == b"FAKE_TIFF_BYTES"

    def test_tiff_zip_no_tif_inside_returns_error(
        self, configured_difference_extractor, sample_difference_entity, tmp_path
    ):
        configured_difference_extractor.difference_params["map_format"] = "tiff.zip"
        buf = BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", b"no tiff here")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()

        result = configured_difference_extractor.format_difference_file(
            entity_data=sample_difference_entity,
            response=mock_response,
            output_path=str(tmp_path),
        )

        assert result["status"] == "error"
        assert "No .tif file" in result["error_message"]

    def test_shp_zip_writes_all_files(self, configured_difference_extractor, sample_difference_entity, tmp_path):
        configured_difference_extractor.difference_params["map_format"] = "shp.zip"
        buf = BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("difference.shp", b"SHP_BYTES")
            zf.writestr("difference.dbf", b"DBF_BYTES")
            zf.writestr("difference.shx", b"SHX_BYTES")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()

        result = configured_difference_extractor.format_difference_file(
            entity_data=sample_difference_entity,
            response=mock_response,
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert result["file_count"] == 3
        # All three extensions should be present in saved_files
        exts = {os.path.splitext(p)[1] for p in result["saved_files"]}
        assert exts == {".shp", ".dbf", ".shx"}

    def test_no_output_path_returns_error(self, configured_difference_extractor, sample_difference_entity):
        configured_difference_extractor.difference_params["map_format"] = "png"
        configured_difference_extractor.difference_params["output_path"] = None
        configured_difference_extractor.output_path = None
        mock_response = MagicMock()
        mock_response.content = b"PNG"

        result = configured_difference_extractor.format_difference_file(
            entity_data=sample_difference_entity, response=mock_response, output_path=None
        )

        assert result["status"] == "error"
        assert "output path" in result["error_message"].lower()
