"""
Tests for HistoricalScoreExtractor API call logic:
    - get_historical_score_api() URL, query params, payload, validation
    - get_historical_score_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_score_functions import HistoricalScoreExtractor
from tests.conftest import (
    FAKE_HISTORICAL_SCORE_URL,
    FAKE_TOKEN,
    HISTORICAL_SCORE_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_historical_score_api()
# ===================================================================


class TestGetHistoricalScoreApi:
    """Tests for the core historical_score API call method."""

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_successful_api_call(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        sample_response = {
            "id": "z361x33",
            "data": {"AveragePotentialScore": 0.78, "RiskScore": 0.22},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            result = configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_HISTORICAL_SCORE_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_query_params_reflect_setup(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        actual_url = mock_post.call_args.args[0]
        assert "seasonDuration=120" in actual_url
        assert "seasonStartDay=1" in actual_url
        assert "seasonStartMonth=4" in actual_url
        assert "thresholdStart=0.7" in actual_url
        assert "year=2025" in actual_url
        assert "dataSource=LR" in actual_url

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_default_publish_af_false(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        """Default payload has id='' (empty string) when publish_af=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["geometry"] == HISTORICAL_SCORE_WKT
        assert sent["id"] == ""

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        configured_historical_score_extractor.historical_score_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:z361x33@LEGACY_ID_NA"

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_includes_historical_seasons_from_entity(
        self, mock_post, mock_wkt, configured_historical_score_extractor
    ):
        """Notebook cell 15: per-entity historical_seasons should appear in the payload."""
        entity = {
            "id": "z361x33",
            "geometry": HISTORICAL_SCORE_WKT,
            "crop": "OTHERS",
            "historical_seasons": [2024, 2023],
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["historicalSeasons"] == [2024, 2023]

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_omits_historical_seasons_when_none(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert "historicalSeasons" not in sent

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_authorization_header(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_historical_score_extractor):
        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_historical_score_extractor.get_historical_score_api({"geometry": HISTORICAL_SCORE_WKT})

    def test_missing_geometry_raises(self, configured_historical_score_extractor):
        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_historical_score_extractor.get_historical_score_api({"id": "x"})

    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_historical_score_extractor):
        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_historical_score_extractor.get_historical_score_api({"id": "x", "geometry": "INVALID"})

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_http_error_propagates(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_timeout_propagates(
        self, mock_post, mock_wkt, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_historical_score_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_historical_score_extractor.get_historical_score_api(sample_historical_score_entity)


# ===================================================================
# get_historical_score_api_safe()
# ===================================================================


class TestGetHistoricalScoreApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(HistoricalScoreExtractor, "get_historical_score_api")
    def test_success_returns_data(
        self, mock_api, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_api.return_value = {"id": "z361x33", "data": {"AveragePotentialScore": 0.78}}

        result = configured_historical_score_extractor.get_historical_score_api_safe(sample_historical_score_entity)
        assert result["success"] is True
        assert result["data"]["data"]["AveragePotentialScore"] == 0.78
        assert result["error"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(HistoricalScoreExtractor, "get_historical_score_api")
    def test_http_error_returns_failure(
        self, mock_api, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_historical_score_extractor.get_historical_score_api_safe(sample_historical_score_entity)
        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(HistoricalScoreExtractor, "get_historical_score_api")
    def test_generic_error_returns_failure(
        self, mock_api, configured_historical_score_extractor, sample_historical_score_entity
    ):
        mock_api.side_effect = ValueError("something broke")

        result = configured_historical_score_extractor.get_historical_score_api_safe(sample_historical_score_entity)
        assert result["success"] is False
        assert "something broke" in result["error"]
