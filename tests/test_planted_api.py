"""
Tests for PlantedExtractor API call logic:
    - get_planted_api() URL, query params (with PLANTED_AREA / CONTROL variants), payload
    - emergence_date resolution: row column overrides params
    - get_planted_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from earthdaily.agriculture.processors.processor_plantedarea_functions import PlantedExtractor
from tests.conftest import FAKE_PLANTED_URL, FAKE_TOKEN, PLANTED_WKT

pytestmark = pytest.mark.public

# ===================================================================
# get_planted_api()
# ===================================================================


class TestGetPlantedApi:
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity):
        sample_response = {"planted_area": 14523.5, "planted_percentage": 92.4}
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.get_planted_api(sample_planted_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_url_targets_launch_endpoint(
        self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_PLANTED_URL}/launch?")

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_planted_area_query_params(self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity):
        """PLANTED_AREA mode: query string has processorMode + emergenceDate + threshold (no controlThreshold)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        actual_url = mock_post.call_args.args[0]
        assert "processorMode=PLANTED_AREA" in actual_url
        assert "emergenceDate=2025-04-02" in actual_url  # row's date wins
        assert "threshold=30" in actual_url  # configured fixture threshold is 30 (notebook value)
        # PLANTED_AREA mode should NOT add controlThreshold
        assert "controlThreshold" not in actual_url

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_control_query_params_with_decimal_threshold(
        self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity
    ):
        """CONTROL mode: control_threshold (e.g. 4) is sent as decimal (0.04)."""
        configured_planted_extractor.planted_params["processor_mode"] = "CONTROL"
        configured_planted_extractor.planted_params["control_threshold"] = 4
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "difference": 0.025,
            "control_threshold": 0.04,
            "result": True,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        actual_url = mock_post.call_args.args[0]
        assert "processorMode=CONTROL" in actual_url
        # 4 / 100 = 0.04
        assert "controlThreshold=0.04" in actual_url

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_emergence_date_row_wins(self, mock_post, mock_wkt, configured_planted_extractor):
        """Per-entity emergence_date (notebook scenario A) overrides params default."""
        configured_planted_extractor.planted_params["emergence_date"] = "2025-04-01"
        entity = {
            "id": "z361x33",
            "geometry": PLANTED_WKT,
            "crop": "OTHERS",
            "emergence_date": "2025-04-15",  # row override
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(entity)

        actual_url = mock_post.call_args.args[0]
        assert "emergenceDate=2025-04-15" in actual_url

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_emergence_date_falls_back_to_params(self, mock_post, mock_wkt, configured_planted_extractor):
        """When entity has no emergence_date, params default is used."""
        entity = {"id": "z361x33", "geometry": PLANTED_WKT, "crop": "OTHERS"}
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(entity)

        actual_url = mock_post.call_args.args[0]
        assert "emergenceDate=2025-04-01" in actual_url

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_pandas_timestamp_emergence_date_normalized(self, mock_post, mock_wkt, configured_planted_extractor):
        """A pd.Timestamp emergence_date should be normalized to YYYY-MM-DD before being sent."""
        entity = {
            "id": "z361x33",
            "geometry": PLANTED_WKT,
            "crop": "OTHERS",
            "emergence_date": pd.Timestamp("2025-04-12"),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(entity)

        actual_url = mock_post.call_args.args[0]
        assert "emergenceDate=2025-04-12" in actual_url

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_payload_contains_geometry_only_by_default(
        self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["geometry"] == PLANTED_WKT
        assert "id" not in sent

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_payload_includes_id_when_publish_af_true(
        self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity
    ):
        configured_planted_extractor.planted_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["id"] == "SeasonField:z361x33@LEGACY_ID_NA"

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_authorization_header(self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = {"planted_area": 0, "planted_percentage": 0}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.get_planted_api(sample_planted_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_planted_extractor):
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_planted_extractor.get_planted_api({"geometry": PLANTED_WKT})

    def test_missing_geometry_raises(self, configured_planted_extractor):
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_planted_extractor.get_planted_api({"id": "x"})

    @patch(
        "earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_planted_extractor):
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_planted_extractor.get_planted_api({"id": "x", "geometry": "INVALID"})

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    def test_emergence_date_missing_in_both_raises(self, mock_wkt, configured_planted_extractor):
        """If neither row nor params has emergence_date, raise ValueError."""
        configured_planted_extractor.planted_params["emergence_date"] = None
        entity = {"id": "x", "geometry": PLANTED_WKT}
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="emergence_date"):
                configured_planted_extractor.get_planted_api(entity)

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_emergence_date_format_raises(self, mock_wkt, configured_planted_extractor):
        entity = {
            "id": "x",
            "geometry": PLANTED_WKT,
            "emergence_date": "01/04/2025",  # bad format
        }
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="emergence_date"):
                configured_planted_extractor.get_planted_api(entity)

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_http_error_propagates(self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_planted_extractor.get_planted_api(sample_planted_entity)

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.requests.post")
    def test_timeout_propagates(self, mock_post, mock_wkt, configured_planted_extractor, sample_planted_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_planted_extractor.get_planted_api(sample_planted_entity)


# ===================================================================
# get_planted_api_safe()
# ===================================================================


class TestGetPlantedApiSafe:
    @patch.object(PlantedExtractor, "get_planted_api")
    def test_success_returns_data(self, mock_api, configured_planted_extractor, sample_planted_entity):
        mock_api.return_value = {"planted_area": 14523.5, "planted_percentage": 92.4}
        result = configured_planted_extractor.get_planted_api_safe(sample_planted_entity)
        assert result["success"] is True
        assert result["data"]["planted_percentage"] == 92.4
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(PlantedExtractor, "get_planted_api")
    def test_http_error_returns_failure(self, mock_api, configured_planted_extractor, sample_planted_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_planted_extractor.get_planted_api_safe(sample_planted_entity)
        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(PlantedExtractor, "get_planted_api")
    def test_generic_error_returns_failure(self, mock_api, configured_planted_extractor, sample_planted_entity):
        mock_api.side_effect = ValueError("something broke")
        result = configured_planted_extractor.get_planted_api_safe(sample_planted_entity)
        assert result["success"] is False
        assert "something broke" in result["error"]
