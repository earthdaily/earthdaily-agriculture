"""
Tests for EmergenceExtractor API call logic:
    - get_emergence_api() URL construction, payload, validation
    - get_emergence_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_emergence_functions import EmergenceExtractor
from tests.conftest import (
    EMERGENCE_WKT,
    FAKE_EMERGENCE_URL,
    FAKE_TOKEN,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_emergence_api()
# ===================================================================


class TestGetEmergenceApi:
    """Tests for the core emergence API call method."""

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity):
        """Happy path: valid entity returns parsed JSON."""
        sample_response = {
            "id": "z361x33",
            "data": {
                "EmergenceDate": "2025-04-18",
                "EmergenceStatus": "CONFIRMED",
                "ConfirmationStatus": "VALIDATED",
            },
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            result = configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """URL should be {emergence_url}/launch?<params>"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_EMERGENCE_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_query_params_reflect_setup(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """Query string should reflect emergence_params + entity crop."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        actual_url = mock_post.call_args.args[0]
        assert "emergenceType=INSEASON" in actual_url
        assert "seasonDuration=120" in actual_url
        assert "seasonStartDay=1" in actual_url
        assert "seasonStartMonth=4" in actual_url
        assert "year=2025" in actual_url
        assert "dataSource=LR" in actual_url
        assert "crop=CORN" in actual_url

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_query_params_for_historical_mode(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """When emergence_type is HISTORICAL, the query string should reflect that."""
        configured_emergence_extractor.emergence_params["emergence_type"] = "HISTORICAL"
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        actual_url = mock_post.call_args.args[0]
        assert "emergenceType=HISTORICAL" in actual_url

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_payload_contains_geometry_only_by_default(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """Default payload should contain just the geometry (no id) when publish_af=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        # Payload is sent via `data=json.dumps(...)`, so it lands in kwargs['data'] as a string
        sent_raw = mock_post.call_args.kwargs["data"]
        import json

        sent = json.loads(sent_raw)
        assert sent["geometry"] == EMERGENCE_WKT
        assert "id" not in sent

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """When publish_af=True, payload should include id as 'SeasonField:<id>@LEGACY_ID_NA'."""
        configured_emergence_extractor.emergence_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        import json

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:z361x33@LEGACY_ID_NA"
        assert sent["geometry"] == EMERGENCE_WKT

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_emergence_extractor):
        """Entity without 'geometry' field should raise ValueError before any HTTP call."""
        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_emergence_extractor.get_emergence_api({"id": "x", "crop": "CORN"})

    def test_missing_id_raises(self, configured_emergence_extractor):
        """Entity without 'id' field should raise ValueError before any HTTP call."""
        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_emergence_extractor.get_emergence_api({"geometry": EMERGENCE_WKT, "crop": "CORN"})

    @patch(
        "earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_emergence_extractor):
        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_emergence_extractor.get_emergence_api({"id": "x", "geometry": "INVALID", "crop": "CORN"})

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_crop_raises(self, mock_wkt, configured_emergence_extractor):
        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="Invalid crop"):
                configured_emergence_extractor.get_emergence_api(
                    {"id": "x", "geometry": EMERGENCE_WKT, "crop": "BANANA"}
                )

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity):
        """HTTPError from requests should propagate."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_emergence_extractor.get_emergence_api(sample_emergence_entity)

    @patch("earthdaily.agriculture.processors.processor_emergence_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_emergence_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_emergence_extractor, sample_emergence_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_emergence_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_emergence_extractor.get_emergence_api(sample_emergence_entity)


# ===================================================================
# get_emergence_api_safe()
# ===================================================================


class TestGetEmergenceApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(EmergenceExtractor, "get_emergence_api")
    def test_success_returns_data(self, mock_api, configured_emergence_extractor, sample_emergence_entity):
        mock_api.return_value = {
            "id": "z361x33",
            "data": {"EmergenceDate": "2025-04-18", "EmergenceStatus": "CONFIRMED"},
        }

        result = configured_emergence_extractor.get_emergence_api_safe(sample_emergence_entity)

        assert result["success"] is True
        assert result["data"]["data"]["EmergenceStatus"] == "CONFIRMED"
        assert result["error"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(EmergenceExtractor, "get_emergence_api")
    def test_http_error_returns_failure(self, mock_api, configured_emergence_extractor, sample_emergence_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_emergence_extractor.get_emergence_api_safe(sample_emergence_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(EmergenceExtractor, "get_emergence_api")
    def test_generic_error_returns_failure(self, mock_api, configured_emergence_extractor, sample_emergence_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_emergence_extractor.get_emergence_api_safe(sample_emergence_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "z361x33"
