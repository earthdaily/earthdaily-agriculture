"""
Tests for GDDExtractor API call logic:
    - get_gdd() URL construction, query parameters, payload, validation
    - get_gdd_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from earthdaily.agriculture.extractors.gdd_functions import GDDExtractor
from tests.conftest import (
    FAKE_GDD_URL,
    FAKE_TOKEN,
    GDD_POINT_WKT,
    GDD_POLYGON_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_gdd()
# ===================================================================


class TestGetGdd:
    """Tests for the core GDD API call method."""

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_successful_api_call(self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity):
        """Happy path: valid entity returns parsed JSON."""
        sample_response = {"elements": [{"date": "2023-01-17"}]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd(sample_gdd_entity)

        assert result == sample_response
        mock_get.assert_called_once()

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_url_targets_analytics_gdd_endpoint(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity
    ):
        """URL should be {weather_url}/analytics/gdd"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(sample_gdd_entity)

        actual_url = mock_get.call_args.args[0]
        assert actual_url == f"{FAKE_GDD_URL}/analytics/gdd"

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_query_params_default(self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity):
        """Query params should reflect gdd_params + entity geometry centroid."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(sample_gdd_entity)

        params = mock_get.call_args.kwargs["params"]
        # params is a list of (key, value) tuples — convert to dict for assertions
        params_dict = dict(params)
        assert params_dict["StartDate"] == "2023-01-17"
        assert params_dict["LastDate"] == "2023-03-17"
        assert params_dict["Provider"] == "GLOBAL1"
        assert params_dict["Location"] == GDD_POINT_WKT
        assert params_dict["LowerThreshold"] == 10.0
        assert params_dict["UpperThreshold"] == 30.0
        assert params_dict["ResetCumulativeEveryYear"] == "false"
        assert params_dict["ExtrapolateForecastData"] == "false"

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_upper_threshold_omitted_when_none(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity
    ):
        """UpperThreshold should NOT appear in the query string when params['upper_threshold'] is None."""
        configured_gdd_extractor.gdd_params["upper_threshold"] = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(sample_gdd_entity)

        keys = [k for k, _ in mock_get.call_args.kwargs["params"]]
        assert "UpperThreshold" not in keys
        assert "LowerThreshold" in keys

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_entity_dates_take_precedence_over_params(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor
    ):
        """Per-entity start_date/end_date columns should override params defaults."""
        entity = {
            "id": "test_001",
            "geometry": GDD_POINT_WKT,
            "start_date": "2024-05-01",
            "end_date": "2024-07-01",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(entity)

        params_dict = dict(mock_get.call_args.kwargs["params"])
        assert params_dict["StartDate"] == "2024-05-01"
        assert params_dict["LastDate"] == "2024-07-01"

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_pandas_timestamp_dates_converted_to_string(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor
    ):
        """pd.Timestamp dates should be normalised to YYYY-MM-DD strings."""
        entity = {
            "id": "test_001",
            "geometry": GDD_POINT_WKT,
            "start_date": pd.Timestamp("2024-05-01"),
            "end_date": pd.Timestamp("2024-07-01"),
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(entity)

        params_dict = dict(mock_get.call_args.kwargs["params"])
        assert params_dict["StartDate"] == "2024-05-01"
        assert params_dict["LastDate"] == "2024-07-01"

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", return_value=GDD_POINT_WKT)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_polygon_geometry_is_centroid_extracted(self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor):
        """A polygon geometry should be reduced to its centroid POINT before being sent."""
        entity = {"id": "test_001", "geometry": GDD_POLYGON_WKT}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(entity)

        # validate_wkt was called with the polygon
        mock_wkt.assert_called_once_with(GDD_POLYGON_WKT)
        # get_centroid_wkt was called and its return value (a POINT) made it into the query
        mock_centroid.assert_called_once()
        params_dict = dict(mock_get.call_args.kwargs["params"])
        assert params_dict["Location"] == GDD_POINT_WKT

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_authorization_header(self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity):
        """Authorization header carries the bearer token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"elements":[]}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            configured_gdd_extractor.get_gdd(sample_gdd_entity)

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Accept"] == "application/json"

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_204_no_content_returns_none(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity
    ):
        """HTTP 204 (or empty body) should return None, not raise."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd(sample_gdd_entity)

        assert result is None

    # --- Input validation errors ---

    def test_missing_start_date_in_both_row_and_params_raises(self, configured_gdd_extractor):
        """If neither row nor params provides start_date, ValueError is raised."""
        configured_gdd_extractor.gdd_params["start_date"] = None
        entity = {"id": "x", "geometry": GDD_POINT_WKT, "end_date": "2023-03-17"}
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_gdd_extractor.get_gdd(entity)

    def test_missing_end_date_in_both_row_and_params_raises(self, configured_gdd_extractor):
        configured_gdd_extractor.gdd_params["end_date"] = None
        entity = {"id": "x", "geometry": GDD_POINT_WKT, "start_date": "2023-01-17"}
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_gdd_extractor.get_gdd(entity)

    def test_invalid_start_date_format_raises(self, configured_gdd_extractor):
        entity = {
            "id": "x",
            "geometry": GDD_POINT_WKT,
            "start_date": "01/17/2023",
            "end_date": "2023-03-17",
        }
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_gdd_extractor.get_gdd(entity)

    def test_invalid_end_date_format_raises(self, configured_gdd_extractor):
        entity = {
            "id": "x",
            "geometry": GDD_POINT_WKT,
            "start_date": "2023-01-17",
            "end_date": "2023/03/17",
        }
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_gdd_extractor.get_gdd(entity)

    def test_start_after_end_raises(self, configured_gdd_extractor):
        entity = {
            "id": "x",
            "geometry": GDD_POINT_WKT,
            "start_date": "2023-06-01",
            "end_date": "2023-01-01",
        }
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_gdd_extractor.get_gdd(entity)

    @patch(
        "earthdaily.agriculture.extractors.gdd_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_gdd_extractor, sample_gdd_entity):
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_gdd_extractor.get_gdd(sample_gdd_entity)

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_http_error_propagates(
        self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_gdd_extractor.get_gdd(sample_gdd_entity)

    @patch("earthdaily.agriculture.extractors.gdd_functions.get_centroid_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.gdd_functions.requests.get")
    def test_timeout_propagates(self, mock_get, mock_wkt, mock_centroid, configured_gdd_extractor, sample_gdd_entity):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_gdd_extractor.get_gdd(sample_gdd_entity)


# ===================================================================
# get_gdd_safe()
# ===================================================================


class TestGetGddSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(GDDExtractor, "get_gdd")
    def test_success_returns_data(self, mock_api, configured_gdd_extractor, sample_gdd_entity):
        mock_api.return_value = {"elements": [{"date": "2023-01-17"}]}

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd_safe(sample_gdd_entity)

        assert result["success"] is True
        assert result["data"] == {"elements": [{"date": "2023-01-17"}]}
        assert result["error"] is None
        assert result["entity_id"] == "test_001"

    @patch.object(GDDExtractor, "get_gdd")
    def test_http_error_returns_failure(self, mock_api, configured_gdd_extractor, sample_gdd_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd_safe(sample_gdd_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["entity_id"] == "test_001"

    @patch.object(GDDExtractor, "get_gdd")
    def test_validation_error_returns_failure(self, mock_api, configured_gdd_extractor, sample_gdd_entity):
        """ValueError should be wrapped as 'Validation error'."""
        mock_api.side_effect = ValueError("bad date")

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd_safe(sample_gdd_entity)

        assert result["success"] is False
        assert "Validation error" in result["error"]
        assert "bad date" in result["error"]

    @patch.object(GDDExtractor, "get_gdd")
    def test_generic_exception_returns_failure(self, mock_api, configured_gdd_extractor, sample_gdd_entity):
        mock_api.side_effect = RuntimeError("boom")

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.get_gdd_safe(sample_gdd_entity)

        assert result["success"] is False
        assert "boom" in result["error"]
        assert result["entity_id"] == "test_001"
