"""
Tests for ZoningExtractor format methods (one per postprocess mode):
    - format_zoning_stats_json — single row of field + per-zone stats
    - format_zoning_stats_geo_json — one row per zone with merged geometry
    - format_zoning_links_json — single row of links + worldfile + bbox
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_zoning_stats_json (postprocess='stats')
# ===================================================================


class TestFormatZoningStatsJson:
    """Stats mode: one row per response, with field-level + per-zone columns."""

    def test_single_row_returned(self, configured_zoning_extractor, sample_zoning_stats_response):
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_field_level_stats_columns(self, configured_zoning_extractor, sample_zoning_stats_response):
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        expected = {
            "field_variability",
            "most_variable_zone",
            "highest_interzone_variability",
            "field_productivity_index",
            "field_variability_index",
        }
        assert expected.issubset(set(df.columns))

    def test_field_level_values_preserved(self, configured_zoning_extractor, sample_zoning_stats_response):
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        row = df.iloc[0]
        assert row["field_variability"] == "MEDIUM"
        assert row["most_variable_zone"] == "3"
        assert row["field_productivity_index"] == 0.62
        assert row["field_variability_index"] == 0.18

    def test_highest_interzone_variability_joined_csv(self, configured_zoning_extractor, sample_zoning_stats_response):
        """The list ['1','5'] should be joined with ',' into a string."""
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        assert df.iloc[0]["highest_interzone_variability"] == "1,5"

    def test_per_zone_columns_generated(self, configured_zoning_extractor, sample_zoning_stats_response):
        """For each zone in legend.ranges, three columns are added: area_percent / productivity_index / variability_index."""
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        for zone_name in ("1", "2", "3", "4", "5"):
            assert f"zone_{zone_name}_area_percent" in df.columns
            assert f"zone_{zone_name}_productivity_index" in df.columns
            assert f"zone_{zone_name}_variability_index" in df.columns

    def test_per_zone_values_preserved(self, configured_zoning_extractor, sample_zoning_stats_response):
        df = configured_zoning_extractor.format_zoning_stats_json(sample_zoning_stats_response)
        row = df.iloc[0]
        # Zone 3 in fixture: area=30.5, productivity=0.62, variability=0.20
        assert row["zone_3_area_percent"] == 30.5
        assert row["zone_3_productivity_index"] == 0.62
        assert row["zone_3_variability_index"] == 0.20

    def test_response_object_with_json_method(self, configured_zoning_extractor, sample_zoning_stats_response):
        """If a Response-like object is passed, .json() is called to extract the dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_zoning_stats_response
        df = configured_zoning_extractor.format_zoning_stats_json(mock_response)
        assert len(df) == 1
        mock_response.json.assert_called_once()

    def test_empty_response_returns_empty_df(self, configured_zoning_extractor):
        df = configured_zoning_extractor.format_zoning_stats_json({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_none_response_returns_empty_df(self, configured_zoning_extractor):
        df = configured_zoning_extractor.format_zoning_stats_json(None)
        assert df.empty

    def test_response_with_failed_json_parse_returns_empty(self, configured_zoning_extractor):
        """If the Response.json() raises ValueError, formatter swallows and returns empty df."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("not json")
        df = configured_zoning_extractor.format_zoning_stats_json(mock_response)
        assert df.empty


# ===================================================================
# format_zoning_stats_geo_json (postprocess='stats_geo')
# ===================================================================


class TestFormatZoningStatsGeoJson:
    """stats_geo mode: one row per zone, with geometry merged from segments."""

    def test_one_row_per_zone(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        # Fixture has 3 zones in legend.ranges
        assert len(df) == 3

    def test_zone_columns(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        expected = {
            "zone_name",
            "zone_geometry",
            "productivity_index",
            "variability_index",
            "area_percent",
            "number_of_pixels",
            "zone_mean",
            "zone_max",
            "zone_min",
            "zone_area",
        }
        assert expected.issubset(set(df.columns))

    def test_zone_geometry_from_single_segment(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        """Zone 1 has a single segment → zone_geometry should be that segment's WKT (no GEOMETRYCOLLECTION wrap)."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        zone_1 = df[df["zone_name"] == "1"].iloc[0]
        geom = zone_1["zone_geometry"]
        assert geom is not None
        assert geom.startswith("POLYGON")
        assert "GEOMETRYCOLLECTION" not in geom

    def test_zone_geometry_from_multi_segments_merged(
        self, configured_zoning_extractor, sample_zoning_stats_geo_response
    ):
        """Zone 2 has two segments → zone_geometry should be wrapped in GEOMETRYCOLLECTION."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        zone_2 = df[df["zone_name"] == "2"].iloc[0]
        geom = zone_2["zone_geometry"]
        assert geom.startswith("GEOMETRYCOLLECTION (")
        assert geom.count("POLYGON") == 2

    def test_zone_stats_values_preserved(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        """Zone-level stats (mean/max/min/area) come from zones[*].stats."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        zone_2 = df[df["zone_name"] == "2"].iloc[0]
        assert zone_2["zone_mean"] == 0.55
        assert zone_2["zone_max"] == 0.65
        assert zone_2["zone_min"] == 0.45
        assert zone_2["zone_area"] == 2000.0

    def test_range_values_preserved(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        """Per-zone productivity/variability/area_percent come from legend.ranges."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        zone_2 = df[df["zone_name"] == "2"].iloc[0]
        assert zone_2["productivity_index"] == 0.55
        assert zone_2["variability_index"] == 0.15
        assert zone_2["area_percent"] == 40.0
        assert zone_2["number_of_pixels"] == 1600

    def test_field_stats_repeated_per_row(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        """Field-level stats from legend.stat should be repeated on every zone row."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(sample_zoning_stats_geo_response)
        # All rows should share the same field-level fields
        assert (df["field_variability"] == "HIGH").all()
        assert (df["field_productivity_index"] == 0.55).all()
        assert (df["highest_interzone_variability"] == "1,3").all()

    def test_response_object_with_json_method(self, configured_zoning_extractor, sample_zoning_stats_geo_response):
        mock_response = MagicMock()
        mock_response.json.return_value = sample_zoning_stats_geo_response
        df = configured_zoning_extractor.format_zoning_stats_geo_json(mock_response)
        assert len(df) == 3

    def test_empty_response_returns_empty_df(self, configured_zoning_extractor):
        df = configured_zoning_extractor.format_zoning_stats_geo_json({})
        assert df.empty

    def test_no_ranges_returns_empty_df(self, configured_zoning_extractor):
        """If legend.ranges is missing/empty, no rows are emitted."""
        df = configured_zoning_extractor.format_zoning_stats_geo_json(
            {"legend": {"stat": {}, "ranges": []}, "zones": []}
        )
        assert df.empty

    def test_zone_without_geometry_yields_none(self, configured_zoning_extractor):
        """A zone with empty segments should produce zone_geometry=None."""
        response = {
            "legend": {
                "stat": {"fieldVariability": "LOW"},
                "ranges": [{"name": "1", "fieldAreaPercent": 100.0, "productivityIndex": 0.5}],
            },
            "zones": [{"id": 1, "segments": [], "stats": {"mean": 0.5}}],
        }
        df = configured_zoning_extractor.format_zoning_stats_geo_json(response)
        assert len(df) == 1
        assert df.iloc[0]["zone_geometry"] is None


# ===================================================================
# format_zoning_links_json (postprocess='links')
# ===================================================================


class TestFormatZoningLinksJson:
    """Links mode: single row with download URLs + worldfile + bbox + map size."""

    def test_single_row_returned(self, configured_zoning_extractor, sample_zoning_links_response):
        df = configured_zoning_extractor.format_zoning_links_json(sample_zoning_links_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_link_columns(self, configured_zoning_extractor, sample_zoning_links_response):
        df = configured_zoning_extractor.format_zoning_links_json(sample_zoning_links_response)
        row = df.iloc[0]
        assert row["image_png_link"] == "https://api.example.com/samz/test_001.png"
        assert row["worldfile_link"] == "https://api.example.com/samz/test_001.pgw"
        assert row["thumbnail_link"] == "https://api.example.com/samz/test_001_thumb.png"

    def test_world_file_columns(self, configured_zoning_extractor, sample_zoning_links_response):
        df = configured_zoning_extractor.format_zoning_links_json(sample_zoning_links_response)
        row = df.iloc[0]
        assert row["worldfile_a"] == 1e-5
        assert row["worldfile_e"] == -1e-5
        assert row["worldfile_c"] == -97.7
        assert row["worldfile_f"] == 37.14

    def test_bbox_columns(self, configured_zoning_extractor, sample_zoning_links_response):
        df = configured_zoning_extractor.format_zoning_links_json(sample_zoning_links_response)
        row = df.iloc[0]
        assert row["bbox_xmin"] == -97.703
        assert row["bbox_xmax"] == -97.699
        assert row["bbox_ymin"] == 37.140
        assert row["bbox_ymax"] == 37.145

    def test_map_size_columns(self, configured_zoning_extractor, sample_zoning_links_response):
        df = configured_zoning_extractor.format_zoning_links_json(sample_zoning_links_response)
        row = df.iloc[0]
        assert row["map_width"] == 512
        assert row["map_height"] == 512

    def test_response_object_with_json_method(self, configured_zoning_extractor, sample_zoning_links_response):
        mock_response = MagicMock()
        mock_response.json.return_value = sample_zoning_links_response
        df = configured_zoning_extractor.format_zoning_links_json(mock_response)
        assert len(df) == 1

    def test_missing_keys_yield_none(self, configured_zoning_extractor):
        """When _links/worldFile/etc. are absent, every column is filled with None."""
        df = configured_zoning_extractor.format_zoning_links_json({})
        assert len(df) == 1
        row = df.iloc[0]
        assert row["image_png_link"] is None
        assert row["worldfile_a"] is None
        assert row["map_width"] is None
        assert row["bbox_xmin"] is None
