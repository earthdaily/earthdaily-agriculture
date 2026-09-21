"""
Tests for FLMExtractor API call logic:
    - get_flm_map() URL construction (postprocess + map_format variants), payload, headers
    - get_flm_map_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.FLM_functions import FLMExtractor
from tests.conftest import (
    FAKE_FLM_URL,
    FAKE_TOKEN,
    FLM_IMAGE_ID,
    FLM_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_flm_map()
# ===================================================================


class TestGetFlmMap:
    """Tests for the core FLM API call method."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """Happy path: valid entity returns the requests.Response object."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"legend": {"stat": {"max": 0.85}}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        assert result is mock_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_url_stats_mode_no_extension(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """stats mode (map_format=None) → URL has no `.image.<ext>` suffix."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_FLM_URL}/maps/base-reference-map/NDVI?")
        assert "/image." not in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_url_file_mode_includes_extension(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """file mode (map_format='png') → URL includes `/image.png`."""
        configured_flm_extractor.flm_params["map_format"] = "png"
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert f"{FAKE_FLM_URL}/maps/base-reference-map/NDVI/image.png?" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_default(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """Default query string carries histogram=false, directLinks=false, $epsg=4326."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "histogram=false" in actual_url
        assert "directLinks=false" in actual_url
        assert "$epsg=4326" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_histogram_mode(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """postprocess='histogram' should flip histogram=true on the query string."""
        configured_flm_extractor.flm_params["postprocess"] = "histogram"
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "histogram=true" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_links_mode_directLinks_true(
        self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity
    ):
        """postprocess='links' implies directLinks=true on the query string."""
        configured_flm_extractor.flm_params["postprocess"] = "links"
        configured_flm_extractor.flm_params["directLinks"] = True
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "directLinks=true" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_number_bins_appended(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """When number_bins is set, the URL should include numberOfBins=<n>."""
        configured_flm_extractor.flm_params["number_bins"] = 15
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "numberOfBins=15" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_legend_type_appended(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """When legendType is set, the URL should include legendType=<value>."""
        configured_flm_extractor.flm_params["legendType"] = "Fixed"
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "legendType=Fixed" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_clipping_only_for_color_composition(
        self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity
    ):
        """clipping should NOT appear for non-COLORCOMPOSITION indexes."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "clipping=" not in actual_url
        assert "buffer=" not in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_query_params_clipping_color_composition(
        self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity
    ):
        """For COLORCOMPOSITION, clipping (and buffer if > 0) appear in the URL."""
        configured_flm_extractor.flm_params["vegetation_index"] = "COLORCOMPOSITION"
        configured_flm_extractor.flm_params["clipping"] = "Bbox"
        configured_flm_extractor.flm_params["buffer"] = 10
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "clipping=Bbox" in actual_url
        assert "buffer=10" in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_color_composition_no_buffer_when_zero(
        self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity
    ):
        """COLORCOMPOSITION + buffer=0 → buffer should NOT appear in the URL."""
        configured_flm_extractor.flm_params["vegetation_index"] = "COLORCOMPOSITION"
        configured_flm_extractor.flm_params["buffer"] = 0
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        actual_url = mock_post.call_args.args[0]
        assert "clipping=FieldBorder" in actual_url
        assert "buffer=" not in actual_url

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_payload_has_image_id_and_geometry(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """Payload body should include image.id and seasonField.geometry."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["image"]["id"] == FLM_IMAGE_ID
        assert sent["seasonField"]["geometry"] == FLM_WKT

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_authorization_header(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """Authorization header carries the bearer token."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_flm_extractor):
        """Entity without geometry should raise ValueError before any HTTP call."""
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_flm_extractor.get_flm_map({"id": "x"}, FLM_IMAGE_ID)

    @patch(
        "earthdaily.agriculture.extractors.FLM_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_flm_extractor):
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_flm_extractor.get_flm_map({"id": "x", "geometry": "INVALID"}, FLM_IMAGE_ID)

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_http_error_propagates(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.requests.post")
    def test_timeout_propagates(self, mock_post, mock_wkt, configured_flm_extractor, sample_flm_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_flm_extractor.get_flm_map(sample_flm_entity, FLM_IMAGE_ID)


# ===================================================================
# get_flm_map_safe()
# ===================================================================


class TestGetFlmMapSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(FLMExtractor, "get_flm_map")
    def test_success_returns_response(self, mock_api, configured_flm_extractor, sample_flm_entity):
        mock_response = MagicMock()
        mock_api.return_value = mock_response

        result = configured_flm_extractor.get_flm_map_safe(sample_flm_entity, FLM_IMAGE_ID)

        assert result["success"] is True
        assert result["data"] is mock_response
        assert result["error"] is None
        assert result["seasonfield_id"] == "toto"

    @patch.object(FLMExtractor, "get_flm_map")
    def test_http_error_returns_failure(self, mock_api, configured_flm_extractor, sample_flm_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_flm_extractor.get_flm_map_safe(sample_flm_entity, FLM_IMAGE_ID)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "toto"

    @patch.object(FLMExtractor, "get_flm_map")
    def test_generic_error_returns_failure(self, mock_api, configured_flm_extractor, sample_flm_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_flm_extractor.get_flm_map_safe(sample_flm_entity, FLM_IMAGE_ID)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "toto"
