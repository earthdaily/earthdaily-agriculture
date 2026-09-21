"""
Tests for GreennessExtractor API call logic:
    - get_greenness_api() URL construction, query params, payload, validation
    - get_greenness_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_greenness_functions import GreennessExtractor
from tests.conftest import (
    FAKE_GREENNESS_URL,
    FAKE_TOKEN,
    VALID_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_greenness_api()
# ===================================================================


class TestGetGreennessApi:
    """Tests for the core greenness API call method."""

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """Happy path: valid entity returns parsed JSON."""
        sample_response = {
            "id": "entity_001",
            "data": {"greenness_score": 0.85, "status": "GREEN"},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.get_greenness_api(sample_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_url_targets_greenness_detection_endpoint(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """URL should be {greenness_url}/greenness-detection (no query string in URL — params kw)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(sample_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url == f"{FAKE_GREENNESS_URL}/greenness-detection"

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_query_params_reflect_setup(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """Query params (passed via params= kwarg) should reflect greenness_params + entity crop."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(sample_entity)

        params = mock_post.call_args.kwargs["params"]
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["year"] == 2025
        assert params["sowing_date"] == "2025-04-01"
        assert params["data_source"] == "LR"
        assert params["crop"] == "CORN"

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_entity_sowing_date_overrides_param(self, mock_post, mock_wkt, configured_extractor):
        """Per-entity sowing_date should win over the default in greenness_params."""
        entity = {
            "id": "ent_x",
            "geometry": VALID_WKT,
            "crop": "CORN",
            "sowing_date": "2025-05-15",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "ent_x", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(entity)

        assert mock_post.call_args.kwargs["params"]["sowing_date"] == "2025-05-15"

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_payload_contains_geometry_only_by_default(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """Default payload should contain just the geometry (no id) when publish_af=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(sample_entity)

        # Payload is sent via `data=json.dumps(...)`, so it lands in kwargs['data'] as a string.
        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["geometry"] == VALID_WKT
        assert "id" not in sent

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """When publish_af=True, payload should include id as 'SeasonField:<id>@LEGACY_ID_NA'."""
        configured_extractor.greenness_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(sample_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:entity_001@LEGACY_ID_NA"
        assert sent["geometry"] == VALID_WKT

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_authorization_header_uses_bearer_token(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "entity_001", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.get_greenness_api(sample_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_extractor):
        """Entity without 'geometry' field should raise ValueError before any HTTP call."""
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_extractor.get_greenness_api({"id": "x", "crop": "CORN"})

    def test_missing_id_raises(self, configured_extractor):
        """Entity without 'id' field should raise ValueError before any HTTP call."""
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_extractor.get_greenness_api({"geometry": VALID_WKT, "crop": "CORN"})

    @patch(
        "earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_extractor):
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_extractor.get_greenness_api({"id": "x", "geometry": "INVALID", "crop": "CORN"})

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_crop_raises(self, mock_wkt, configured_extractor):
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="Invalid crop"):
                configured_extractor.get_greenness_api({"id": "x", "geometry": VALID_WKT, "crop": "BANANA"})

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        """HTTPError from requests should propagate."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_extractor.get_greenness_api(sample_entity)

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_extractor, sample_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_extractor.get_greenness_api(sample_entity)


# ===================================================================
# get_greenness_api_safe()
# ===================================================================


class TestGetGreennessApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(GreennessExtractor, "get_greenness_api")
    def test_success_returns_data(self, mock_api, configured_extractor, sample_entity):
        mock_api.return_value = {"id": "entity_001", "data": {"greenness_score": 0.85}}

        result = configured_extractor.get_greenness_api_safe(sample_entity)

        assert result["success"] is True
        assert result["data"]["data"]["greenness_score"] == 0.85
        assert result["error"] is None
        assert result["seasonfield_id"] == "entity_001"

    @patch.object(GreennessExtractor, "get_greenness_api")
    def test_http_error_returns_failure(self, mock_api, configured_extractor, sample_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_extractor.get_greenness_api_safe(sample_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "entity_001"

    @patch.object(GreennessExtractor, "get_greenness_api")
    def test_generic_error_returns_failure(self, mock_api, configured_extractor, sample_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_extractor.get_greenness_api_safe(sample_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "entity_001"
