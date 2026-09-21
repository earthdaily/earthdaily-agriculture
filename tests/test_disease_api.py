"""
Tests for DiseaseExtractor API call logic:
    - get_disease_data() URL construction, payload, validation
    - get_disease_data_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_disease_risk_functions import DiseaseExtractor
from tests.conftest import (
    DISEASE_WKT,
    FAKE_DISEASE_URL,
    FAKE_TOKEN,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_disease_data()
# ===================================================================


class TestGetDiseaseData:
    """Tests for the core disease API call method."""

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity):
        """Happy path: valid entity returns parsed JSON."""
        sample_response = [
            {
                "date": "2025-06-01",
                "frogeye_leaf_spot": 0.46,
                "gray_leaf_spot": 0.99,
            }
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.get_disease_data(sample_disease_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """URL should be {disease_url}/launch?startDate=...&endDate=..."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            configured_disease_extractor.get_disease_data(sample_disease_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_DISEASE_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_query_params_reflect_entity_dates(
        self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """startDate / endDate query string entries should match the entity's dates."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            configured_disease_extractor.get_disease_data(sample_disease_entity)

        actual_url = mock_post.call_args.args[0]
        assert "startDate=2025-06-01" in actual_url
        assert "endDate=2025-10-01" in actual_url

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_payload_contains_id_and_geometry(
        self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """Request body should include the entity id and geometry."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            configured_disease_extractor.get_disease_data(sample_disease_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["id"] == "z361x33"
        assert sent["geometry"] == DISEASE_WKT

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            configured_disease_extractor.get_disease_data(sample_disease_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_start_date_raises(self, configured_disease_extractor):
        """Entity without start_date should raise ValueError before any HTTP call."""
        entity = {"id": "x", "geometry": DISEASE_WKT, "end_date": "2025-10-01"}
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_disease_extractor.get_disease_data(entity)

    def test_missing_end_date_raises(self, configured_disease_extractor):
        """Entity without end_date should raise ValueError before any HTTP call."""
        entity = {"id": "x", "geometry": DISEASE_WKT, "start_date": "2025-06-01"}
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_disease_extractor.get_disease_data(entity)

    def test_invalid_start_date_format_raises(self, configured_disease_extractor):
        """A non-YYYY-MM-DD start_date should raise ValueError."""
        entity = {
            "id": "x",
            "geometry": DISEASE_WKT,
            "start_date": "2025-13-45",  # invalid month/day
            "end_date": "2025-10-01",
        }
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_disease_extractor.get_disease_data(entity)

    def test_invalid_end_date_format_raises(self, configured_disease_extractor):
        entity = {
            "id": "x",
            "geometry": DISEASE_WKT,
            "start_date": "2025-06-01",
            "end_date": "10/01/2025",
        }
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_disease_extractor.get_disease_data(entity)

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_disease_extractor):
        entity = {
            "id": "x",
            "geometry": "INVALID",
            "start_date": "2025-06-01",
            "end_date": "2025-10-01",
        }
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_disease_extractor.get_disease_data(entity)

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_http_error_is_raised(self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity):
        """HTTPError from requests should propagate."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_disease_extractor.get_disease_data(sample_disease_entity)

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.requests.post")
    def test_timeout_is_raised(self, mock_post, mock_wkt, configured_disease_extractor, sample_disease_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_disease_extractor.get_disease_data(sample_disease_entity)


# ===================================================================
# get_disease_data_safe()
# ===================================================================


class TestGetDiseaseDataSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(DiseaseExtractor, "get_disease_data")
    def test_success_returns_data(self, mock_api, configured_disease_extractor, sample_disease_entity):
        mock_api.return_value = [{"date": "2025-06-01", "frogeye_leaf_spot": 0.5}]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.get_disease_data_safe(sample_disease_entity)

        assert result["success"] is True
        assert result["data"] == [{"date": "2025-06-01", "frogeye_leaf_spot": 0.5}]
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(DiseaseExtractor, "get_disease_data")
    def test_http_error_returns_failure(self, mock_api, configured_disease_extractor, sample_disease_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.get_disease_data_safe(sample_disease_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(DiseaseExtractor, "get_disease_data")
    def test_validation_error_returns_failure(self, mock_api, configured_disease_extractor, sample_disease_entity):
        """ValueError (e.g. invalid geometry) should be wrapped as 'Validation error'."""
        mock_api.side_effect = ValueError("bad geometry")

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.get_disease_data_safe(sample_disease_entity)

        assert result["success"] is False
        assert "Validation error" in result["error"]
        assert "bad geometry" in result["error"]

    @patch.object(DiseaseExtractor, "get_disease_data")
    def test_generic_exception_returns_failure(self, mock_api, configured_disease_extractor, sample_disease_entity):
        mock_api.side_effect = RuntimeError("boom")

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.get_disease_data_safe(sample_disease_entity)

        assert result["success"] is False
        assert "boom" in result["error"]
        assert result["entity_id"] == "z361x33"
