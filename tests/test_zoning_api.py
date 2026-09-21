"""
Tests for ZoningExtractor API call logic:
    - get_zoning_map_api() URL construction, payload, image_id normalization
    - get_zoning_map_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.zoning_functions import ZoningExtractor
from tests.conftest import (
    FAKE_TOKEN,
    FAKE_ZONING_URL,
    ZONING_IMAGE_ID,
    ZONING_IMAGE_ID_LIST,
    ZONING_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_zoning_map_api()
# ===================================================================


class TestGetZoningMapApi:
    """Tests for the core SAMZ API call method."""

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_successful_api_call(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity, sample_zoning_stats_response
    ):
        """Happy path: stats mode returns parsed JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_zoning_stats_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        assert result == sample_zoning_stats_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_url_targets_samz_endpoint(self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity):
        """In stats mode (no map_format), URL is {map_products_url}/maps/management-zones-map/SAMZ?..."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_ZONING_URL}/maps/management-zones-map/SAMZ?")
        # No file extension in stats mode
        assert "/SAMZ/image." not in actual_url

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_url_with_map_format_appends_extension(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        """When map_format is set (file mode), URL ends with /SAMZ/image.<format>?..."""
        configured_zoning_extractor.zoning_params["map_format"] = "png"
        mock_response = MagicMock()
        mock_response.content = b"\x89PNG..."
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        actual_url = mock_post.call_args.args[0]
        assert "/maps/management-zones-map/SAMZ/image.png?" in actual_url

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_query_params_include_directLinks_and_epsg(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        """URL must include directLinks=<bool> and $epsg=<int>."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        actual_url = mock_post.call_args.args[0]
        assert "directLinks=false" in actual_url
        assert "$epsg=4326" in actual_url

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_directLinks_true_serialized_lowercase(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        configured_zoning_extractor.zoning_params["directLinks"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        actual_url = mock_post.call_args.args[0]
        assert "directLinks=true" in actual_url

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_payload_structure(self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity):
        """Payload should include tags, images, seasonField.geometry, and zoneCount."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["tags"] == ["UA_MANAGEMENT_ZONE"]
        assert sent["seasonField"]["geometry"] == ZONING_WKT
        assert sent["zoneCount"] == 5
        assert isinstance(sent["images"], list)
        assert sent["images"][0] == {"id": ZONING_IMAGE_ID}

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_image_id_string_normalized_to_single_image(self, mock_post, mock_wkt, configured_zoning_extractor):
        """A single-string image_id should produce one entry in 'images'."""
        entity = {"id": "x", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID}
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["images"] == [{"id": ZONING_IMAGE_ID}]

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_image_id_list_passed_as_multiple_images(self, mock_post, mock_wkt, configured_zoning_extractor):
        """A list image_id (notebook workflow output) should produce N entries in 'images'."""
        entity = {"id": "x", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID_LIST}
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(entity)

        sent = mock_post.call_args.kwargs["json"]
        assert len(sent["images"]) == 3
        assert [img["id"] for img in sent["images"]] == ZONING_IMAGE_ID_LIST

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_zone_count_reflects_num_zones_setting(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        configured_zoning_extractor.zoning_params["num_zones"] = 7
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["zoneCount"] == 7

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Accept"] == "*/*"

    # --- Input validation ---

    def test_missing_image_id_raises(self, configured_zoning_extractor):
        """Entity without 'image_id' field should raise ValueError before any HTTP call."""
        with pytest.raises(ValueError, match="image_id"):
            configured_zoning_extractor.get_zoning_map_api({"id": "x", "geometry": ZONING_WKT})

    def test_missing_geometry_raises(self, configured_zoning_extractor):
        """Entity without 'geometry' field should raise ValueError before any HTTP call."""
        with pytest.raises(ValueError, match="geometry"):
            configured_zoning_extractor.get_zoning_map_api({"id": "x", "image_id": ZONING_IMAGE_ID})

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_zoning_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_zoning_extractor.get_zoning_map_api(
                {"id": "x", "geometry": "INVALID", "image_id": ZONING_IMAGE_ID}
            )

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_http_error_propagates(self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_timeout_propagates(self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

    # --- Return type by mode ---

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_returns_response_object_in_file_mode(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        """In file mode, get_zoning_map_api returns the raw Response (binary content)."""
        configured_zoning_extractor.zoning_params["map_format"] = "png"
        mock_response = MagicMock()
        mock_response.content = b"\x89PNG..."
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        # Returns the response directly, not parsed JSON
        assert result is mock_response

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_returns_dict_in_stats_mode(
        self, mock_post, mock_wkt, configured_zoning_extractor, sample_zoning_entity, sample_zoning_stats_response
    ):
        """In stats mode (no map_format), get_zoning_map_api returns parsed JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_zoning_stats_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_zoning_extractor.get_zoning_map_api(sample_zoning_entity)

        assert isinstance(result, dict)
        assert result == sample_zoning_stats_response


# ===================================================================
# get_zoning_map_api_safe()
# ===================================================================


class TestGetZoningMapApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(ZoningExtractor, "get_zoning_map_api")
    def test_success_returns_data(
        self, mock_api, configured_zoning_extractor, sample_zoning_entity, sample_zoning_stats_response
    ):
        mock_api.return_value = sample_zoning_stats_response

        result = configured_zoning_extractor.get_zoning_map_api_safe(sample_zoning_entity)

        assert result["success"] is True
        assert result["data"] == sample_zoning_stats_response
        assert result["error"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(ZoningExtractor, "get_zoning_map_api")
    def test_http_error_returns_failure(self, mock_api, configured_zoning_extractor, sample_zoning_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_zoning_extractor.get_zoning_map_api_safe(sample_zoning_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "test_001"

    @patch.object(ZoningExtractor, "get_zoning_map_api")
    def test_generic_error_returns_failure(self, mock_api, configured_zoning_extractor, sample_zoning_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_zoning_extractor.get_zoning_map_api_safe(sample_zoning_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "test_001"
