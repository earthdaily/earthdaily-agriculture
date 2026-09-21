"""Tests for LocationBasedBorderExtractor.format_location_based_border_json:
- Flat dict with WKT geometry
- GeoJSON Feature with polygon geometry
- GeoJSON FeatureCollection
- Edge cases (missing geometry, empty response, non-dict input)
"""

import pandas as pd
import pytest

from tests.conftest import LOCATION_BORDER_POINT_WKT, LOCATION_BORDER_POLYGON_WKT

pytestmark = pytest.mark.public

# ===================================================================
# Flat dict response (WKT string)
# ===================================================================


class TestFormatLocationBasedBorderFlatDict:
    def test_flat_dict_with_wkt_creates_single_row(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "loc_001"
        assert df.iloc[0]["point_geometry"] == LOCATION_BORDER_POINT_WKT
        assert df.iloc[0]["polygon_geometry"] == LOCATION_BORDER_POLYGON_WKT

    def test_flat_dict_with_invalid_wkt_returns_none_polygon(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": "NOT A WKT",
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert df.iloc[0]["polygon_geometry"] is None

    def test_extra_scalar_fields_promoted_onto_row(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
            "area_ha": 12.5,
            "sourceId": "DIGIFARM",
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert df.iloc[0]["area_ha"] == 12.5
        assert df.iloc[0]["sourceId"] == "DIGIFARM"

    def test_dict_typed_extra_fields_skipped(self, configured_location_based_border_extractor):
        """Nested dict / list values are not promoted (single-row DataFrame keeps scalar shape)."""
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
            "metadata": {"source": "x"},
            "tags": ["a", "b"],
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert "metadata" not in df.columns
        assert "tags" not in df.columns


# ===================================================================
# GeoJSON Feature response
# ===================================================================


class TestFormatLocationBasedBorderGeoJsonFeature:
    def test_feature_with_polygon_geometry(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-93.601, 41.499],
                        [-93.599, 41.499],
                        [-93.599, 41.501],
                        [-93.601, 41.501],
                        [-93.601, 41.499],
                    ]
                ],
            },
            "properties": {"area_ha": 12.5, "fieldId": "F-12345"},
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert len(df) == 1
        polygon_wkt = df.iloc[0]["polygon_geometry"]
        # Don't lock in WKT formatting — just ensure shapely produced something polygonal.
        assert polygon_wkt.upper().startswith("POLYGON")
        # Properties should be promoted onto the row.
        assert df.iloc[0]["area_ha"] == 12.5
        assert df.iloc[0]["fieldId"] == "F-12345"


# ===================================================================
# GeoJSON FeatureCollection response
# ===================================================================


class TestFormatLocationBasedBorderFeatureCollection:
    def test_feature_collection_uses_first_feature(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-93.601, 41.499],
                                [-93.599, 41.499],
                                [-93.599, 41.501],
                                [-93.601, 41.501],
                                [-93.601, 41.499],
                            ]
                        ],
                    },
                    "properties": {"area_ha": 12.5},
                },
                # Second feature ignored.
                {"type": "Feature", "geometry": None, "properties": {}},
            ],
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert df.iloc[0]["polygon_geometry"].upper().startswith("POLYGON")
        assert df.iloc[0]["area_ha"] == 12.5

    def test_empty_feature_collection_returns_none_polygon(self, configured_location_based_border_extractor):
        response = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "type": "FeatureCollection",
            "features": [],
        }
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert df.iloc[0]["polygon_geometry"] is None


# ===================================================================
# Edge cases
# ===================================================================


class TestFormatLocationBasedBorderEdgeCases:
    def test_empty_dict_returns_empty_df(self, configured_location_based_border_extractor):
        df = configured_location_based_border_extractor.format_location_based_border_json({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_non_json_string_raises(self, configured_location_based_border_extractor):
        """A plain string that isn't JSON still raises a clear ValueError."""
        with pytest.raises(ValueError, match="dictionary"):
            configured_location_based_border_extractor.format_location_based_border_json("not a dict")

    def test_double_encoded_string_response_is_parsed(self, configured_location_based_border_extractor):
        """A JSON-encoded string (double-encoded API response) is parsed transparently."""
        import json as _json

        encoded = _json.dumps(
            {
                "id": "loc_001",
                "point_wkt": LOCATION_BORDER_POINT_WKT,
                "geometry": LOCATION_BORDER_POLYGON_WKT,
            }
        )
        df = configured_location_based_border_extractor.format_location_based_border_json(encoded)
        assert df.iloc[0]["entity_id"] == "loc_001"
        assert df.iloc[0]["polygon_geometry"] == LOCATION_BORDER_POLYGON_WKT

    def test_missing_geometry_yields_none_polygon(self, configured_location_based_border_extractor):
        response = {"id": "loc_001", "point_wkt": LOCATION_BORDER_POINT_WKT, "status": "ok"}
        df = configured_location_based_border_extractor.format_location_based_border_json(response)
        assert df.iloc[0]["polygon_geometry"] is None
        assert df.iloc[0]["entity_id"] == "loc_001"
