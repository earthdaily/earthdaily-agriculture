"""
Tests for WeatherExtractor API call logic:
    - get_weather_data() URL, query params, payload, validation, centroid extraction
    - get_weather_data_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from earthdaily.agriculture.extractors.weather_functions import WeatherExtractor
from tests.conftest import FAKE_TOKEN, FAKE_WEATHER_URL, WEATHER_WKT

pytestmark = pytest.mark.public

# ===================================================================
# get_weather_data()
# ===================================================================


class TestGetWeatherData:
    @patch(
        "earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt",
        return_value="POINT (-58.93681679 -13.72531769)",
    )
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_successful_api_call(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        sample_response = [{"date": "2025-06-01T00:00:00Z", "Temperature": {"standard": 18.5}}]
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.get_weather_data(sample_weather_entity)

        assert result == sample_response
        mock_get.assert_called_once()

    @patch(
        "earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt",
        return_value="POINT (-58.93681679 -13.72531769)",
    )
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_url_targets_weather_endpoint(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        actual_url = mock_get.call_args.args[0]
        assert actual_url.startswith(f"{FAKE_WEATHER_URL}/weather?")

    @patch(
        "earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt",
        return_value="POINT (-58.93681679 -13.72531769)",
    )
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_query_params_default(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        actual_url = mock_get.call_args.args[0]
        assert "$offset=0" in actual_url
        assert "$count=false" in actual_url
        assert "$limit=1000" in actual_url
        assert "Provider=GLOBAL1" in actual_url
        assert "WeatherType=HISTORICAL_DAILY" in actual_url
        # Centroid POINT mocked above
        assert "Location=POINT (-58.93681679 -13.72531769)" in actual_url
        # Date $between range with full ISO timestamps
        assert "Date=$between:2025-06-01T00:00:00.0000000Z|2025-10-01T23:59:59.0000000Z" in actual_url

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_fields_param_includes_weather_param_and_date(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        """$fields query parameter should be the weather params joined by comma + Date."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        actual_url = mock_get.call_args.args[0]
        # Default fixture has weather_parameters="Temperature.standardmax"
        assert "$fields=Temperature.standardmax,Date" in actual_url

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_fields_with_list_of_parameters(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        configured_weather_extractor.weather_params["weather_parameters"] = [
            "Temperature.standardmax",
            "precipitation.cumulative",
            "wind",
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        actual_url = mock_get.call_args.args[0]
        assert "$fields=Temperature.standardmax,precipitation.cumulative,wind,Date" in actual_url

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_fields_none_expands_to_all_parameters(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        """The 'none' sentinel must expand to every available parameter, never a
        literal $fields=none (which the API rejects with 400)."""
        from earthdaily.agriculture.extractors.weather_functions import available_weather_parameters

        configured_weather_extractor.weather_params["weather_parameters"] = "none"
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        actual_url = mock_get.call_args.args[0]
        assert "$fields=none," not in actual_url
        assert f"$fields={','.join(sorted(available_weather_parameters))},Date" in actual_url

    @patch(
        "earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt",
        return_value="POINT (-58.93681679 -13.72531769)",
    )
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_pandas_timestamp_dates_normalized(self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor):
        """pd.Timestamp dates should be normalised to YYYY-MM-DD before API call."""
        entity = {
            "id": "z361x33",
            "geometry": WEATHER_WKT,
            "start_date": pd.Timestamp("2025-06-01"),
            "end_date": pd.Timestamp("2025-10-01"),
        }
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(entity)

        actual_url = mock_get.call_args.args[0]
        assert "Date=$between:2025-06-01T00:00:00.0000000Z|2025-10-01T23:59:59.0000000Z" in actual_url

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_polygon_geometry_centroid_extracted(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        """A polygon should be reduced to its centroid POINT before being sent."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        # validate_wkt was called with the polygon
        mock_wkt.assert_called_once_with(WEATHER_WKT)
        mock_centroid.assert_called_once()

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_authorization_header(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_invalid_start_date_format_raises(self, configured_weather_extractor):
        entity = {
            "id": "x",
            "geometry": WEATHER_WKT,
            "start_date": "06/01/2025",
            "end_date": "2025-10-01",
        }
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_weather_extractor.get_weather_data(entity)

    def test_invalid_end_date_format_raises(self, configured_weather_extractor):
        entity = {
            "id": "x",
            "geometry": WEATHER_WKT,
            "start_date": "2025-06-01",
            "end_date": "10/01/2025",
        }
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_weather_extractor.get_weather_data(entity)

    @patch(
        "earthdaily.agriculture.extractors.weather_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_weather_extractor):
        entity = {
            "id": "x",
            "geometry": "INVALID",
            "start_date": "2025-06-01",
            "end_date": "2025-10-01",
        }
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_weather_extractor.get_weather_data(entity)

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_http_error_propagates(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_weather_extractor.get_weather_data(sample_weather_entity)

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_timeout_propagates(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_weather_extractor.get_weather_data(sample_weather_entity)


# ===================================================================
# get_weather_data() — $limit auto-sizing / page_limit override
# ===================================================================


class TestWeatherRowLimit:
    """The row limit ($limit) must be auto-sized to the query span so multi-year ranges
    are not truncated, while short single-season requests keep the historic 1000."""

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_single_season_still_uses_floor_1000(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        """Default fixture entity spans 2025-06-01..2025-10-01 (~123 days) → floor 1000."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        assert "$limit=1000" in mock_get.call_args.args[0]

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_multi_year_range_autosizes_above_1000(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor
    ):
        """A multi-year range must lift $limit above 1000 to cover the whole span."""
        from datetime import date

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        entity = {"id": "x", "geometry": WEATHER_WKT, "start_date": "2020-05-08", "end_date": "2026-06-30"}
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(entity)

        span = (date(2026, 6, 30) - date(2020, 5, 8)).days + 1
        expected = max(1000, span + 366)
        assert f"$limit={expected}" in mock_get.call_args.args[0]
        assert "$limit=1000" not in mock_get.call_args.args[0]

    @patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)")
    @patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.weather_functions.requests.get")
    def test_page_limit_override_wins(
        self, mock_get, mock_wkt, mock_centroid, configured_weather_extractor, sample_weather_entity
    ):
        """An explicit page_limit overrides the auto-sized value."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_weather_extractor.weather_params["page_limit"] = 500
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.get_weather_data(sample_weather_entity)

        assert "$limit=500" in mock_get.call_args.args[0]


# ===================================================================
# get_weather_data_safe()
# ===================================================================


class TestGetWeatherDataSafe:
    @patch.object(WeatherExtractor, "get_weather_data")
    def test_success_returns_data(self, mock_api, configured_weather_extractor, sample_weather_entity):
        mock_api.return_value = [{"date": "2025-06-01", "Temperature": {"standard": 18.5}}]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.get_weather_data_safe(sample_weather_entity)

        assert result["success"] is True
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"
        assert result["data"][0]["Temperature"]["standard"] == 18.5

    @patch.object(WeatherExtractor, "get_weather_data")
    def test_http_error_returns_failure(self, mock_api, configured_weather_extractor, sample_weather_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.get_weather_data_safe(sample_weather_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(WeatherExtractor, "get_weather_data")
    def test_value_error_wrapped_as_validation_error(
        self, mock_api, configured_weather_extractor, sample_weather_entity
    ):
        mock_api.side_effect = ValueError("bad date")

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.get_weather_data_safe(sample_weather_entity)

        assert result["success"] is False
        assert "Validation error" in result["error"]
        assert "bad date" in result["error"]

    @patch.object(WeatherExtractor, "get_weather_data")
    def test_generic_error_returns_failure(self, mock_api, configured_weather_extractor, sample_weather_entity):
        mock_api.side_effect = RuntimeError("boom")

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.get_weather_data_safe(sample_weather_entity)

        assert result["success"] is False
        assert "boom" in result["error"]
