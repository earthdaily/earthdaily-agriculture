"""
Tests for DifferenceExtractor API call logic:
    - get_difference_map_api() URL/payload/validation, plus file-mode binary return
    - get_difference_map_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.difference_functions import DifferenceExtractor
from tests.conftest import (
    DIFFERENCE_IMG_1,
    DIFFERENCE_IMG_2,
    DIFFERENCE_WKT,
    FAKE_DIFFERENCE_URL,
    FAKE_TOKEN,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_difference_map_api()
# ===================================================================


class TestGetDifferenceMapApi:
    """Tests for the core difference map API call method."""

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_successful_api_call_returns_json(
        self,
        mock_post,
        mock_wkt,
        configured_difference_extractor,
        sample_difference_entity,
        sample_difference_stats_response,
    ):
        """Stats mode (no map_format): API returns parsed JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_difference_stats_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        assert result == sample_difference_stats_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_url_targets_difference_endpoint(
        self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        """URL should be {map_products_url}/maps/difference-map/{product}?... (no extension)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_DIFFERENCE_URL}/maps/difference-map/DIFFERENCE_NDVI?")
        assert "directLinks=false" in actual_url
        assert "$epsg=4326" in actual_url

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_file_mode_url_has_extension(
        self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        """When map_format is set, URL appends /image.<ext> and returns the raw Response."""
        configured_difference_extractor.difference_params["map_format"] = "tiff.zip"
        mock_response = MagicMock()
        mock_response.content = b"FAKE_ZIP_BYTES"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        actual_url = mock_post.call_args.args[0]
        assert "/maps/difference-map/DIFFERENCE_NDVI/image.tiff.zip?" in actual_url
        # File mode returns the raw Response object, not parsed JSON
        assert result is mock_response
        mock_response.json.assert_not_called()

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_directlinks_true_in_url(
        self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        configured_difference_extractor.difference_params["directLinks"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        actual_url = mock_post.call_args.args[0]
        assert "directLinks=true" in actual_url

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_payload_structure(self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity):
        """
        Payload should be:
            {"seasonField": {"geometry": <wkt>},
             "earliestImage": {"id": image_id_1},
             "latestImage":   {"id": image_id_2}}
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["seasonField"]["geometry"] == DIFFERENCE_WKT
        assert sent["earliestImage"]["id"] == DIFFERENCE_IMG_1
        assert sent["latestImage"]["id"] == DIFFERENCE_IMG_2

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Accept"] == "*/*"

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_custom_epsg_reflected_in_url(
        self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        configured_difference_extractor.difference_params["output_epsg"] = 3857
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        actual_url = mock_post.call_args.args[0]
        assert "$epsg=3857" in actual_url

    # --- Input validation errors ---

    def test_missing_image_id_1_raises(self, configured_difference_extractor):
        with pytest.raises(ValueError, match="image_id_1"):
            configured_difference_extractor.get_difference_map_api(
                {"id": "x", "geometry": DIFFERENCE_WKT, "image_id_2": DIFFERENCE_IMG_2}
            )

    def test_missing_image_id_2_raises(self, configured_difference_extractor):
        with pytest.raises(ValueError, match="image_id_2"):
            configured_difference_extractor.get_difference_map_api(
                {"id": "x", "geometry": DIFFERENCE_WKT, "image_id_1": DIFFERENCE_IMG_1}
            )

    def test_missing_geometry_raises(self, configured_difference_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_difference_extractor.get_difference_map_api(
                {"id": "x", "image_id_1": DIFFERENCE_IMG_1, "image_id_2": DIFFERENCE_IMG_2}
            )

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_difference_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_difference_extractor.get_difference_map_api(
                {
                    "id": "x",
                    "geometry": "INVALID",
                    "image_id_1": DIFFERENCE_IMG_1,
                    "image_id_2": DIFFERENCE_IMG_2,
                }
            )

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_difference_extractor.get_difference_map_api(sample_difference_entity)

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_difference_extractor, sample_difference_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_difference_extractor.get_difference_map_api(sample_difference_entity)


# ===================================================================
# get_difference_map_api_safe()
# ===================================================================


class TestGetDifferenceMapApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(DifferenceExtractor, "get_difference_map_api")
    def test_success_returns_data(
        self,
        mock_api,
        configured_difference_extractor,
        sample_difference_entity,
        sample_difference_stats_response,
    ):
        mock_api.return_value = sample_difference_stats_response

        result = configured_difference_extractor.get_difference_map_api_safe(sample_difference_entity)

        assert result["success"] is True
        assert result["data"] == sample_difference_stats_response
        assert result["error"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(DifferenceExtractor, "get_difference_map_api")
    def test_http_error_returns_failure(self, mock_api, configured_difference_extractor, sample_difference_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_difference_extractor.get_difference_map_api_safe(sample_difference_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(DifferenceExtractor, "get_difference_map_api")
    def test_generic_error_returns_failure(self, mock_api, configured_difference_extractor, sample_difference_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_difference_extractor.get_difference_map_api_safe(sample_difference_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "test_001"
