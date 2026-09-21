"""
Tests for CoverageExtractor API call logic:
    - get_satellite_coverage_by_geometry() URL construction, payload, validation
    - get_satellite_coverage_by_geometry_safe() error wrapping
    - recalibration variants
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
from tests.conftest import (
    COVERAGE_PARIS_WKT,
    FAKE_COVERAGE_URL,
    FAKE_TOKEN,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_satellite_coverage_by_geometry()
# ===================================================================


class TestGetSatelliteCoverageByGeometry:
    """Tests for the core coverage API call method."""

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity):
        """Happy path: valid entity returns parsed JSON list."""
        sample_response = [
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2C_T31UDQ_20260429T105026_L2A",
                    "spatialResolution": 10.0,
                    "date": "2026-04-29T10:57:29Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            }
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_url_targets_catalog_imagery_endpoint(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """URL should be {map_products_url}/season-fields/catalog-imagery?..."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert actual_url.startswith(f"{FAKE_COVERAGE_URL}/season-fields/catalog-imagery?")
        # Should NOT include the recalibration sub-path by default
        assert "/catalog-imagery/recalibration" not in actual_url

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_query_params_reflect_setup(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """Query params built into URL should match coverage_params (mirrors notebook cell 17)."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        actual_url = mock_post.call_args.args[0]
        assert "Maps.Type=NDVI" in actual_url
        assert "coveragePercent=$gte:80" in actual_url
        assert "coveragePercent=$lte:100" in actual_url
        assert "$limit=10000" in actual_url
        assert "image.date=$gte:2025-01-01" in actual_url
        assert "mask=auto" in actual_url

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_use_specific_date_query_param(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """When use_specific_date=True, URL must use image.date=<date> (no $gte/$lte)."""
        configured_coverage_extractor.coverage_params["use_specific_date"] = True
        configured_coverage_extractor.coverage_params["start_date"] = "2025-06-15"
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        actual_url = mock_post.call_args.args[0]
        assert "image.date=2025-06-15" in actual_url
        assert "$gte:2025-06-15" not in actual_url

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_date_range_when_both_dates_set(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """When both start_date and end_date are set, the URL should contain a $between clause."""
        configured_coverage_extractor.coverage_params["start_date"] = "2025-01-01"
        configured_coverage_extractor.coverage_params["end_date"] = "2025-12-31"
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        actual_url = mock_post.call_args.args[0]
        assert "image.date=$between:2025-01-01|2025-12-31" in actual_url

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_payload_contains_geometry_only_by_default(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """Payload body should contain just the geometry (no crop) when recalibration is False."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert "seasonFields" in sent
        assert sent["seasonFields"][0]["geometry"] == COVERAGE_PARIS_WKT
        assert "crop" not in sent["seasonFields"][0]

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_recalibration_changes_url_and_includes_crop(self, mock_post, mock_wkt, configured_coverage_extractor):
        """When recalibration=True, URL appends /recalibration and payload includes crop."""
        configured_coverage_extractor.coverage_params["recalibration"] = True
        entity = {"id": "ent_x", "geometry": COVERAGE_PARIS_WKT, "crop": "CORN"}
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(entity)

        actual_url = mock_post.call_args.args[0]
        assert "/season-fields/catalog-imagery/recalibration?" in actual_url
        sent = mock_post.call_args.kwargs["json"]
        assert sent["seasonFields"][0]["crop"] == "CORN"

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_per_entity_date_overrides_take_precedence(self, mock_post, mock_wkt, configured_coverage_extractor):
        """_coverage_start_date / _coverage_end_date on the entity dict override params."""
        entity = {
            "id": "ent_x",
            "geometry": COVERAGE_PARIS_WKT,
            "_coverage_start_date": "2022-07-01",
            "_coverage_end_date": "2025-07-31",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(entity)

        actual_url = mock_post.call_args.args[0]
        assert "image.date=$between:2022-07-01|2025-07-31" in actual_url

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_coverage_extractor):
        """Entity without 'geometry' field should raise ValueError before any HTTP call."""
        with pytest.raises(ValueError, match="geometry"):
            configured_coverage_extractor.get_satellite_coverage_by_geometry({"id": "x"})

    @patch(
        "earthdaily.agriculture.extractors.coverage_function.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_coverage_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_coverage_extractor.get_satellite_coverage_by_geometry({"id": "x", "geometry": "INVALID"})

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity):
        """HTTPError from requests should propagate."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_coverage_extractor, sample_coverage_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_coverage_extractor.get_satellite_coverage_by_geometry(sample_coverage_entity)


# ===================================================================
# get_satellite_coverage_by_geometry_safe()
# ===================================================================


class TestGetSatelliteCoverageByGeometrySafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(CoverageExtractor, "get_satellite_coverage_by_geometry")
    def test_success_returns_data(self, mock_api, configured_coverage_extractor, sample_coverage_entity):
        mock_api.return_value = [{"coveragePercent": 100.0}]

        result = configured_coverage_extractor.get_satellite_coverage_by_geometry_safe(sample_coverage_entity)

        assert result["success"] is True
        assert result["data"] == [{"coveragePercent": 100.0}]
        assert result["error"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(CoverageExtractor, "get_satellite_coverage_by_geometry")
    def test_http_error_returns_failure(self, mock_api, configured_coverage_extractor, sample_coverage_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_coverage_extractor.get_satellite_coverage_by_geometry_safe(sample_coverage_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(CoverageExtractor, "get_satellite_coverage_by_geometry")
    def test_generic_error_returns_failure(self, mock_api, configured_coverage_extractor, sample_coverage_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_coverage_extractor.get_satellite_coverage_by_geometry_safe(sample_coverage_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "test_001"


# ===================================================================
# Recalibration backward-compat aliases
# ===================================================================


class TestRecalibrationAliases:
    """Backward-compatibility aliases that flip recalibration on coverage_params."""

    @patch.object(CoverageExtractor, "get_satellite_coverage_by_geometry", return_value=[])
    def test_recalibration_alias_sets_flag_and_delegates(self, mock_main, configured_coverage_extractor):
        entity = {"id": "ent_x", "geometry": COVERAGE_PARIS_WKT, "crop": "CORN"}
        configured_coverage_extractor.get_satellite_coverage_by_geometry_recalibration(entity)
        assert configured_coverage_extractor.coverage_params["recalibration"] is True
        mock_main.assert_called_once_with(entity)

    @patch.object(
        CoverageExtractor,
        "get_satellite_coverage_by_geometry",
        return_value=[],
    )
    def test_safe_recalibration_alias(self, mock_main, configured_coverage_extractor):
        entity = {"id": "ent_x", "geometry": COVERAGE_PARIS_WKT, "crop": "CORN"}
        result = configured_coverage_extractor.get_satellite_coverage_by_geometry_safe_recalibration(entity)
        assert configured_coverage_extractor.coverage_params["recalibration"] is True
        assert result["success"] is True
