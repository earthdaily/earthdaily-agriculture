"""
Tests for VegationTsExtractor API call logic:
    - get_vegetation_api() URL/query for 'period' and 'windows' modes
    - Date resolution: row > params
    - get_vegetation_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.VTS_functions import VegationTsExtractor
from tests.conftest import FAKE_TOKEN, FAKE_VTS_URL

pytestmark = pytest.mark.public

# ===================================================================
# get_vegetation_api()
# ===================================================================


class TestGetVegetationApi:
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_successful_api_call(self, mock_get, configured_vts_extractor, sample_vts_entity):
        sample_response = [{"date": "2025-08-15T00:00:00Z", "value": 0.78}]
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # The configured fixture has params dates so an entity without dates is OK
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        assert result == sample_response
        mock_get.assert_called_once()

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_url_targets_season_fields_values_endpoint(self, mock_get, configured_vts_extractor, sample_vts_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_VTS_URL}/season-fields/values?")

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_query_params_default_period_mode(self, mock_get, configured_vts_extractor, sample_vts_entity):
        """Default 'period' mode: $filter with isInTimeFrame + earliest historical year."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert "$offset=0" in actual_url
        assert "$limit=3000" in actual_url
        assert "$count=false" in actual_url
        assert "SeasonField.Id=z361x33" in actual_url
        assert "index=NDVI" in actual_url
        assert "$sort=-date" in actual_url
        assert "IsExtrapolted=true" in actual_url
        assert "$fields=date,value" in actual_url
        # Period mode: $filter + isInTimeFrame
        assert "isInTimeFrame" in actual_url

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_period_mode_historical_lookback(self, mock_get, configured_vts_extractor, sample_vts_entity):
        """Period mode subtracts historical_years from start_date.year for the floor."""
        # start_date=2021-01-01, historical_years=10 → earliest=2011-01-01
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert "Date>='2011-01-01'" in actual_url

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_period_mode_historical_years_list(self, mock_get, configured_vts_extractor, sample_vts_entity):
        """When historical_years is a list, lookback = start_year - min(list)."""
        # start=2021, historical_years=[2018, 2017, 2016] → min=2016, lookback=5
        configured_vts_extractor.vegetation_ts_params["historical_years"] = [2018, 2017, 2016]
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert "Date>='2016-01-01'" in actual_url

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_windows_mode_uses_between_filter(self, mock_get, configured_vts_extractor, sample_vts_entity):
        """Windows mode: Date=$between:<start>|<end>, no isInTimeFrame."""
        configured_vts_extractor.vegetation_ts_params["extraction_mode"] = "windows"
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert "Date=$between:2021-01-01|2026-01-01" in actual_url
        assert "isInTimeFrame" not in actual_url

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_entity_dates_take_precedence_over_params(self, mock_get, configured_vts_extractor):
        """Per-entity start_date/end_date columns override params defaults."""
        configured_vts_extractor.vegetation_ts_params["extraction_mode"] = "windows"
        entity = {
            "id": "z361x33",
            "start_date": "2024-06-01",
            "end_date": "2024-09-30",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(entity)

        actual_url = mock_get.call_args.args[0]
        assert "Date=$between:2024-06-01|2024-09-30" in actual_url

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_authorization_header(self, mock_get, configured_vts_extractor, sample_vts_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_is_extrapolated_lowercased(self, mock_get, configured_vts_extractor, sample_vts_entity):
        """is_extrapolated must be lowercased ('true'/'false') in the query."""
        configured_vts_extractor.vegetation_ts_params["is_extrapolated"] = False
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.get_vegetation_api(sample_vts_entity)

        actual_url = mock_get.call_args.args[0]
        assert "IsExtrapolted=false" in actual_url

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_vts_extractor):
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="'id'"):
                configured_vts_extractor.get_vegetation_api({"start_date": "2025-01-01", "end_date": "2025-12-31"})

    def test_no_dates_anywhere_raises(self, configured_vts_extractor):
        """If params has no dates and entity has none → ValueError."""
        configured_vts_extractor.vegetation_ts_params["start_date"] = None
        configured_vts_extractor.vegetation_ts_params["end_date"] = None
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="resolve start_date"):
                configured_vts_extractor.get_vegetation_api({"id": "x"})

    def test_invalid_start_date_format_raises(self, configured_vts_extractor):
        entity = {"id": "x", "start_date": "01/01/2025", "end_date": "2025-12-31"}
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_vts_extractor.get_vegetation_api(entity)

    def test_invalid_end_date_format_raises(self, configured_vts_extractor):
        entity = {"id": "x", "start_date": "2025-01-01", "end_date": "12/31/2025"}
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_vts_extractor.get_vegetation_api(entity)

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_http_error_propagates(self, mock_get, configured_vts_extractor, sample_vts_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_vts_extractor.get_vegetation_api(sample_vts_entity)

    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.get")
    def test_timeout_propagates(self, mock_get, configured_vts_extractor, sample_vts_entity):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_vts_extractor.get_vegetation_api(sample_vts_entity)


# ===================================================================
# get_vegetation_api_safe()
# ===================================================================


class TestGetVegetationApiSafe:
    @patch.object(VegationTsExtractor, "get_vegetation_api")
    def test_success_returns_data(self, mock_api, configured_vts_extractor, sample_vts_entity, sample_vts_response):
        mock_api.return_value = sample_vts_response

        result = configured_vts_extractor.get_vegetation_api_safe(sample_vts_entity)

        assert result["success"] is True
        assert result["data"] == sample_vts_response
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(VegationTsExtractor, "get_vegetation_api")
    def test_http_error_returns_failure(self, mock_api, configured_vts_extractor, sample_vts_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_vts_extractor.get_vegetation_api_safe(sample_vts_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(VegationTsExtractor, "get_vegetation_api")
    def test_value_error_wrapped_as_validation_error(self, mock_api, configured_vts_extractor, sample_vts_entity):
        mock_api.side_effect = ValueError("bad input")

        result = configured_vts_extractor.get_vegetation_api_safe(sample_vts_entity)

        assert result["success"] is False
        assert "Validation error" in result["error"]
        assert "bad input" in result["error"]

    @patch.object(VegationTsExtractor, "get_vegetation_api")
    def test_generic_error_returns_failure(self, mock_api, configured_vts_extractor, sample_vts_entity):
        mock_api.side_effect = RuntimeError("boom")

        result = configured_vts_extractor.get_vegetation_api_safe(sample_vts_entity)

        assert result["success"] is False
        assert "boom" in result["error"]
