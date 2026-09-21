"""
Tests for InseasonScoreExtractor API call logic:
    - get_inseason_score_api() URL, query params, payload, validation
    - get_inseason_score_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_score_functions import InseasonScoreExtractor
from tests.conftest import (
    FAKE_INSEASON_SCORE_URL,
    FAKE_TOKEN,
    INSEASON_SCORE_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_inseason_score_api()
# ===================================================================


class TestGetInseasonScoreApi:
    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_successful_api_call(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        sample_response = {
            "id": "z361x33",
            "data": {"historical_potential_score": 0.78, "inseason_potential_score": 0.65},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_INSEASON_SCORE_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_query_params_reflect_setup(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        actual_url = mock_post.call_args.args[0]
        assert "seasonDuration=120" in actual_url
        assert "seasonStartDay=1" in actual_url
        assert "seasonStartMonth=4" in actual_url
        assert "sowingDate=2025-10-25" in actual_url
        assert "numberHistoricalYears=1" in actual_url
        assert "threshold=0.7" in actual_url
        assert "dataSource=LR" in actual_url
        assert "crop=OTHERS" in actual_url

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_end_date_calculated_from_sowing_plus_duration(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        """When end_date isn't provided, it's computed as sowing_date + season_duration days."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        # sowing 2025-10-25 + 120 days = 2026-02-22
        actual_url = mock_post.call_args.args[0]
        assert "endDate=2026-02-22" in actual_url

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_explicit_end_date_used(self, mock_post, mock_wkt, configured_inseason_score_extractor):
        """When end_date IS provided, it overrides the calculated value."""
        entity = {
            "id": "z361x33",
            "geometry": INSEASON_SCORE_WKT,
            "crop": "OTHERS",
            "sowing_date": "2025-10-25",
            "end_date": "2026-04-01",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(entity)

        actual_url = mock_post.call_args.args[0]
        assert "endDate=2026-04-01" in actual_url

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_nb_historical_year_extends_to_match_seasons(
        self, mock_post, mock_wkt, configured_inseason_score_extractor
    ):
        """When historical_seasons has more entries than nb_historical_year, the latter is bumped up."""
        entity = {
            "id": "z361x33",
            "geometry": INSEASON_SCORE_WKT,
            "crop": "OTHERS",
            "sowing_date": "2025-10-25",
            "historical_seasons": [2024, 2023, 2022, 2021],
        }
        # configured fixture has nb_historical_year=1; 4 seasons should bump it to 4
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(entity)

        actual_url = mock_post.call_args.args[0]
        assert "numberHistoricalYears=4" in actual_url

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_default_publish_af_false(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["geometry"] == INSEASON_SCORE_WKT
        assert sent["id"] == ""

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        configured_inseason_score_extractor.inseason_score_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:z361x33@LEGACY_ID_NA"

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_payload_historical_seasons_from_entity(self, mock_post, mock_wkt, configured_inseason_score_extractor):
        entity = {
            "id": "z361x33",
            "geometry": INSEASON_SCORE_WKT,
            "crop": "OTHERS",
            "sowing_date": "2025-10-25",
            "historical_seasons": [2024, 2023],
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["historicalSeasons"] == [2024, 2023]

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_authorization_header(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "z361x33", "data": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_inseason_score_extractor):
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_inseason_score_extractor.get_inseason_score_api(
                    {
                        "geometry": INSEASON_SCORE_WKT,
                        "crop": "OTHERS",
                        "sowing_date": "2025-10-25",
                    }
                )

    def test_missing_geometry_raises(self, configured_inseason_score_extractor):
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_inseason_score_extractor.get_inseason_score_api(
                    {"id": "x", "crop": "OTHERS", "sowing_date": "2025-10-25"}
                )

    def test_missing_crop_raises(self, configured_inseason_score_extractor):
        with patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
                with pytest.raises(ValueError, match="crop"):
                    configured_inseason_score_extractor.get_inseason_score_api(
                        {
                            "id": "x",
                            "geometry": INSEASON_SCORE_WKT,
                            "sowing_date": "2025-10-25",
                        }
                    )

    def test_missing_sowing_date_raises(self, configured_inseason_score_extractor):
        with patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
                with pytest.raises(ValueError, match="sowing_date"):
                    configured_inseason_score_extractor.get_inseason_score_api(
                        {"id": "x", "geometry": INSEASON_SCORE_WKT, "crop": "OTHERS"}
                    )

    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_inseason_score_extractor):
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_inseason_score_extractor.get_inseason_score_api(
                    {
                        "id": "x",
                        "geometry": "INVALID",
                        "crop": "OTHERS",
                        "sowing_date": "2025-10-25",
                    }
                )

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_http_error_propagates(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.requests.post")
    def test_timeout_propagates(
        self,
        mock_post,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_inseason_score_extractor.get_inseason_score_api(sample_inseason_score_entity)


# ===================================================================
# get_inseason_score_api_safe()
# ===================================================================


class TestGetInseasonScoreApiSafe:
    @patch.object(InseasonScoreExtractor, "get_inseason_score_api")
    def test_success_returns_data(self, mock_api, configured_inseason_score_extractor, sample_inseason_score_entity):
        mock_api.return_value = {"id": "z361x33", "data": {"inseason_potential_score": 0.65}}
        result = configured_inseason_score_extractor.get_inseason_score_api_safe(sample_inseason_score_entity)
        assert result["success"] is True
        assert result["data"]["data"]["inseason_potential_score"] == 0.65
        assert result["error"] is None
        assert result["seasonfield_id"] == "z361x33"

    @patch.object(InseasonScoreExtractor, "get_inseason_score_api")
    def test_http_error_returns_failure(
        self, mock_api, configured_inseason_score_extractor, sample_inseason_score_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_inseason_score_extractor.get_inseason_score_api_safe(sample_inseason_score_entity)
        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(InseasonScoreExtractor, "get_inseason_score_api")
    def test_generic_error_returns_failure(
        self, mock_api, configured_inseason_score_extractor, sample_inseason_score_entity
    ):
        mock_api.side_effect = ValueError("something broke")
        result = configured_inseason_score_extractor.get_inseason_score_api_safe(sample_inseason_score_entity)
        assert result["success"] is False
        assert "something broke" in result["error"]
