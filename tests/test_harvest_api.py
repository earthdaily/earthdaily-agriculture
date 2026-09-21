"""
Tests for HarvestExtractor API call logic:
    - get_harvest_api() URL construction, payload, validation
    - get_harvest_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_harvest_functions import HarvestExtractor
from tests.conftest import (
    FAKE_HARVEST_URL,
    FAKE_TOKEN,
    HARVEST_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_harvest_api()
# ===================================================================


class TestGetHarvestApi:
    """Tests for the core harvest API call method."""

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity):
        """Happy path: valid entity returns parsed JSON."""
        sample_response = {
            "id": "z361x33",
            "data": {"HarvestDate": "2025-08-22", "HarvestStatus": "HARVESTED"},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            result = configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity
    ):
        """URL should be {harvest_url}/launch?<params>"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_HARVEST_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_query_params_reflect_setup(self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity):
        """Query string should reflect harvest_params + entity crop."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        actual_url = mock_post.call_args.args[0]
        assert "harvestType=INSEASON_HARVEST" in actual_url
        assert "seasonDuration=120" in actual_url
        assert "seasonStartDay=1" in actual_url
        assert "seasonStartMonth=4" in actual_url
        assert "year=2025" in actual_url
        assert "dataSource=LR" in actual_url
        assert "crop=OTHERS" in actual_url

    @pytest.mark.parametrize("htype", ["INSEASON_HARVEST", "HISTORICAL_HARVEST", "HARVEST_READINESS"])
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_query_params_for_each_harvest_type(
        self, mock_post, mock_wkt, htype, configured_harvest_extractor, sample_harvest_entity
    ):
        """Each harvest_type should appear in the query string."""
        configured_harvest_extractor.harvest_params["harvest_type"] = htype
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        actual_url = mock_post.call_args.args[0]
        assert f"harvestType={htype}" in actual_url

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_payload_contains_geometry_only_by_default(
        self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity
    ):
        """Default payload should contain just the geometry (no id) when publish_af=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        # Payload is sent via `data=json.dumps(...)`, so it lands in kwargs['data'] as a string.
        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["geometry"] == HARVEST_WKT
        assert "id" not in sent

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(
        self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity
    ):
        """When publish_af=True, payload should include id as 'SeasonField:<id>@LEGACY_ID_NA'."""
        configured_harvest_extractor.harvest_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:z361x33@LEGACY_ID_NA"
        assert sent["geometry"] == HARVEST_WKT

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_harvest_extractor):
        """Entity without 'geometry' field should raise ValueError before any HTTP call."""
        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_harvest_extractor.get_harvest_api({"id": "x", "crop": "CORN"})

    def test_missing_id_raises(self, configured_harvest_extractor):
        """Entity without 'id' field should raise ValueError before any HTTP call."""
        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_harvest_extractor.get_harvest_api({"geometry": HARVEST_WKT, "crop": "CORN"})

    @patch(
        "earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_harvest_extractor):
        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_harvest_extractor.get_harvest_api({"id": "x", "geometry": "INVALID", "crop": "CORN"})

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_crop_raises(self, mock_wkt, configured_harvest_extractor):
        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="Invalid crop"):
                configured_harvest_extractor.get_harvest_api({"id": "x", "geometry": HARVEST_WKT, "crop": "BANANA"})

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity):
        """HTTPError from requests should propagate."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_harvest_extractor.get_harvest_api(sample_harvest_entity)

    @patch("earthdaily.agriculture.processors.processor_harvest_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_harvest_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_harvest_extractor, sample_harvest_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_harvest_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_harvest_extractor.get_harvest_api(sample_harvest_entity)


# ===================================================================
# get_harvest_api_safe()
# ===================================================================


class TestGetHarvestApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(HarvestExtractor, "get_harvest_api")
    def test_success_returns_data(self, mock_api, configured_harvest_extractor, sample_harvest_entity):
        mock_api.return_value = {
            "id": "z361x33",
            "data": {"HarvestDate": "2025-08-22", "HarvestStatus": "HARVESTED"},
        }

        result = configured_harvest_extractor.get_harvest_api_safe(sample_harvest_entity)

        assert result["success"] is True
        assert result["data"]["data"]["HarvestStatus"] == "HARVESTED"
        assert result["error"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(HarvestExtractor, "get_harvest_api")
    def test_http_error_returns_failure(self, mock_api, configured_harvest_extractor, sample_harvest_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_harvest_extractor.get_harvest_api_safe(sample_harvest_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(HarvestExtractor, "get_harvest_api")
    def test_generic_error_returns_failure(self, mock_api, configured_harvest_extractor, sample_harvest_entity):
        mock_api.side_effect = ValueError("something broke")

        result = configured_harvest_extractor.get_harvest_api_safe(sample_harvest_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == "z361x33"
