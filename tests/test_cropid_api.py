"""
Tests for cropidExtractor API call logic:
    - get_cropid_api() URL, payload, validation
    - get_cropid_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.cropid_functions import cropidExtractor
from tests.conftest import CROPID_WKT, FAKE_CROPID_URL, FAKE_TOKEN

pytestmark = pytest.mark.public

# ===================================================================
# get_cropid_api()
# ===================================================================


class TestGetCropidApi:
    """Tests for the core cropid API call method."""

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_successful_api_call(
        self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity, sample_cropid_response
    ):
        """Happy path: notebook-shaped response returned as parsed JSON dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_cropid_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

        assert result == sample_cropid_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_url_targets_cropid_endpoint(self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity):
        """URL should be {cropid_url} with no extra path appended."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url == FAKE_CROPID_URL

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_payload_structure_matches_spec(
        self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity
    ):
        """
        Payload should be:
            {"geometryWkt": <wkt>,
             "filters": {BeginYear, EndYear, Products: [mask_type], LimitNbCrop, CropMaskPercentMin}}
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["geometryWkt"] == CROPID_WKT
        assert sent["filters"]["BeginYear"] == 2020
        assert sent["filters"]["EndYear"] == 2025
        assert sent["filters"]["Products"] == ["EndSeason"]
        assert sent["filters"]["LimitNbCrop"] == 1
        assert sent["filters"]["CropMaskPercentMin"] == 75

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_custom_params_reflected_in_payload(
        self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity
    ):
        configured_cropid_extractor.cropid_params.update(
            {
                "begin_year": 2018,
                "end_year": 2024,
                "mask_type": "InSeason",
                "limit_nb_crop": 3,
                "crop_mask_percent": 60,
            }
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["filters"]["BeginYear"] == 2018
        assert sent["filters"]["EndYear"] == 2024
        assert sent["filters"]["Products"] == ["InSeason"]
        assert sent["filters"]["LimitNbCrop"] == 3
        assert sent["filters"]["CropMaskPercentMin"] == 60

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_cropid_extractor):
        with pytest.raises(ValueError, match="id"):
            configured_cropid_extractor.get_cropid_api({"geometry": CROPID_WKT, "crop": "SOYBEANS"})

    def test_missing_geometry_raises(self, configured_cropid_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_cropid_extractor.get_cropid_api({"id": "x", "crop": "SOYBEANS"})

    @patch(
        "earthdaily.agriculture.extractors.cropid_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_cropid_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_cropid_extractor.get_cropid_api({"id": "x", "geometry": "INVALID", "crop": "SOYBEANS"})

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_cropid_extractor.get_cropid_api(sample_cropid_entity)

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_cropid_extractor, sample_cropid_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_cropid_extractor.get_cropid_api(sample_cropid_entity)


# ===================================================================
# get_cropid_api_safe()
# ===================================================================


class TestGetCropidApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(cropidExtractor, "get_cropid_api")
    def test_success_returns_data(
        self, mock_api, configured_cropid_extractor, sample_cropid_entity, sample_cropid_response
    ):
        mock_api.return_value = sample_cropid_response

        result = configured_cropid_extractor.get_cropid_api_safe(sample_cropid_entity)

        assert result["success"] is True
        assert result["data"] == sample_cropid_response
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(cropidExtractor, "get_cropid_api")
    def test_http_error_returns_failure(self, mock_api, configured_cropid_extractor, sample_cropid_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_cropid_extractor.get_cropid_api_safe(sample_cropid_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(cropidExtractor, "get_cropid_api")
    def test_generic_error_returns_failure(self, mock_api, configured_cropid_extractor, sample_cropid_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_cropid_extractor.get_cropid_api_safe(sample_cropid_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["entity_id"] == "z361x33"
