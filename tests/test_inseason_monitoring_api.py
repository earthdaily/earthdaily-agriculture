"""
Tests for InSeasonMonitoringExtractor API call logic:
    - get_inseason_monitoring_api() URL, query params, payload, validation
    - get_inseason_monitoring_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_inseason_monitoring_functions import (
    InSeasonMonitoringExtractor,
)
from tests.conftest import FAKE_ISM_URL, FAKE_TOKEN, ISM_WKT

pytestmark = pytest.mark.public

# ===================================================================
# get_inseason_monitoring_api()
# ===================================================================


class TestGetInseasonMonitoringApi:
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        sample_response = {"id": "z361x33", "data": [{"Season": "2025"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_url_targets_launch_endpoint(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_ISM_URL}/launch?")

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_query_params_reflect_setup(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        """Query string should reflect inseason_monitoring_params + entity crop."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

        actual_url = mock_post.call_args.args[0]
        assert "seasonDuration=120" in actual_url
        assert "seasonStartDay=1" in actual_url
        assert "seasonStartMonth=4" in actual_url
        assert "year=2025" in actual_url
        assert "dataSource=LR" in actual_url
        assert "crop=OTHERS" in actual_url

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_payload_contains_id_and_geometry(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        """Payload body (sent via json=) should include id and geometry."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["id"] == "z361x33"
        assert sent["geometry"] == ISM_WKT

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_ism_extractor):
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_ism_extractor.get_inseason_monitoring_api({"id": "x", "crop": "CORN"})

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_ism_extractor):
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_ism_extractor.get_inseason_monitoring_api({"id": "x", "geometry": "INVALID", "crop": "CORN"})

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    def test_invalid_crop_raises(self, mock_wkt, configured_ism_extractor):
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="Invalid crop"):
                configured_ism_extractor.get_inseason_monitoring_api({"id": "x", "geometry": ISM_WKT, "crop": "BANANA"})

    # --- HTTP error handling ---

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_http_error_propagates(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.requests.post")
    def test_timeout_propagates(self, mock_post, mock_wkt, configured_ism_extractor, sample_ism_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_ism_extractor.get_inseason_monitoring_api(sample_ism_entity)


# ===================================================================
# get_inseason_monitoring_api_safe()
# ===================================================================


class TestGetInseasonMonitoringApiSafe:
    @patch.object(InSeasonMonitoringExtractor, "get_inseason_monitoring_api")
    def test_success_returns_data(self, mock_api, configured_ism_extractor, sample_ism_entity):
        mock_api.return_value = {"id": "z361x33", "data": [{"Season": "2025"}]}

        result = configured_ism_extractor.get_inseason_monitoring_api_safe(sample_ism_entity)

        assert result["success"] is True
        assert result["data"]["data"][0]["Season"] == "2025"
        assert result["error"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(InSeasonMonitoringExtractor, "get_inseason_monitoring_api")
    def test_http_error_returns_failure(self, mock_api, configured_ism_extractor, sample_ism_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_ism_extractor.get_inseason_monitoring_api_safe(sample_ism_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(InSeasonMonitoringExtractor, "get_inseason_monitoring_api")
    def test_generic_error_returns_failure(self, mock_api, configured_ism_extractor, sample_ism_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_ism_extractor.get_inseason_monitoring_api_safe(sample_ism_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
