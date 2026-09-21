"""
Tests for BaresoilExtractor API call logic:
    - get_baresoil_api() URL construction, payload, validation
    - get_baresoil_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_baresoil_function import BaresoilExtractor
from tests.conftest import FAKE_BARESOIL_URL, FAKE_TOKEN, VALID_WKT

pytestmark = pytest.mark.public

# ===================================================================
# get_baresoil_api()
# ===================================================================


class TestGetBaresoilApi:
    """Tests for the core baresoil API call method."""

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity):
        """Happy path: valid entity returns parsed JSON."""
        mock_wkt.return_value = VALID_WKT
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {"baresoilDays": 12}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        assert result == {"id": "entity_001", "data": {"baresoilDays": 12}}
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_url_targets_launch_endpoint_with_query_params(
        self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """URL should be {baresoil_url}/launch?duration=...&startDay=...&startMonth=...&year=..."""
        mock_wkt.return_value = VALID_WKT
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]

        assert actual_url.startswith(f"{FAKE_BARESOIL_URL}/launch?")
        assert "duration=120" in actual_url
        assert "startDay=1" in actual_url
        assert "startMonth=4" in actual_url
        assert "year=2025" in actual_url

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_query_params_reflect_custom_setup(
        self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """Custom param overrides should appear in the URL."""
        mock_wkt.return_value = VALID_WKT
        configured_baresoil_extractor.baresoil_params.update(
            {"season_duration": 200, "season_start_day": 15, "season_start_month": 10, "year": 2023}
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert "duration=200" in actual_url
        assert "startDay=15" in actual_url
        assert "startMonth=10" in actual_url
        assert "year=2023" in actual_url

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_payload_contains_geometry(
        self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """Payload body should contain the entity geometry; no id by default."""
        mock_wkt.return_value = VALID_WKT
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1].get("data"))
        assert sent_data["geometry"] == VALID_WKT
        assert "id" not in sent_data  # publish_af is False

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_payload_includes_id_when_publish_af(
        self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """When publish_af=True, an AF-formatted id should appear in the payload."""
        mock_wkt.return_value = VALID_WKT
        configured_baresoil_extractor.baresoil_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1].get("data"))
        assert sent_data["id"] == "SeasonField:entity_001@LEGACY_ID_NA"

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_wkt.return_value = VALID_WKT
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_baresoil_extractor):
        with pytest.raises(ValueError, match="id"):
            configured_baresoil_extractor.get_baresoil_api({"geometry": VALID_WKT})

    def test_missing_geometry_raises(self, configured_baresoil_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_baresoil_extractor.get_baresoil_api({"id": "x"})

    @patch(
        "earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt", side_effect=ValueError("bad wkt")
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_baresoil_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_baresoil_extractor.get_baresoil_api({"id": "x", "geometry": "INVALID"})

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity):
        """HTTPError from requests should propagate."""
        mock_wkt.return_value = VALID_WKT
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity):
        mock_wkt.return_value = VALID_WKT
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_baresoil_extractor.get_baresoil_api(sample_baresoil_entity)


# ===================================================================
# get_baresoil_api_safe()
# ===================================================================


class TestGetBaresoilApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(BaresoilExtractor, "get_baresoil_api")
    def test_success_returns_data(self, mock_api, configured_baresoil_extractor, sample_baresoil_entity):
        mock_api.return_value = {"id": "entity_001", "data": {"baresoilDays": 12}}

        result = configured_baresoil_extractor.get_baresoil_api_safe(sample_baresoil_entity)

        assert result["success"] is True
        assert result["data"] == {"id": "entity_001", "data": {"baresoilDays": 12}}
        assert result["error"] is None
        assert result["seasonfield_id"] == "entity_001"

    @patch.object(BaresoilExtractor, "get_baresoil_api")
    def test_http_error_returns_failure(self, mock_api, configured_baresoil_extractor, sample_baresoil_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_baresoil_extractor.get_baresoil_api_safe(sample_baresoil_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "entity_001"

    @patch.object(BaresoilExtractor, "get_baresoil_api")
    def test_generic_error_returns_failure(self, mock_api, configured_baresoil_extractor, sample_baresoil_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_baresoil_extractor.get_baresoil_api_safe(sample_baresoil_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "entity_001"
