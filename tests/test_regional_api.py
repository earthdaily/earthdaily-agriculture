"""
Tests for RegionalExtractor API call logic:
    - safe_convert_amu_id() static helper (handles int/float/str/tuple-string)
    - get_regional_ts_by_id() URL construction, payload, validation
    - get_regional_ts_by_id_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.regional_ts_extractor import RegionalExtractor
from tests.conftest import FAKE_REGIONAL_URL, FAKE_TOKEN

pytestmark = pytest.mark.public

# ===================================================================
# safe_convert_amu_id() — static helper
# ===================================================================


class TestSafeConvertAmuId:
    """Static method that normalizes amu_id from many input shapes to int."""

    @pytest.mark.parametrize("val", [141, 0, 999999])
    def test_int_passthrough(self, val):
        assert RegionalExtractor.safe_convert_amu_id(val) == val

    def test_float_with_zero_decimal(self):
        assert RegionalExtractor.safe_convert_amu_id(141.0) == 141

    def test_float_with_non_zero_raises(self):
        with pytest.raises(ValueError, match="whole number"):
            RegionalExtractor.safe_convert_amu_id(141.5)

    def test_string_integer(self):
        assert RegionalExtractor.safe_convert_amu_id("141") == 141

    def test_string_with_whitespace_stripped(self):
        assert RegionalExtractor.safe_convert_amu_id("  2432528  ") == 2432528

    def test_tuple_string_extracts_first_id(self):
        """PostgreSQL tuple-string representation: '(141;"County";"Missouri")' → 141."""
        assert RegionalExtractor.safe_convert_amu_id('(141;"County";"Missouri")') == 141

    def test_semicolon_string_extracts_first(self):
        assert RegionalExtractor.safe_convert_amu_id("141;extra") == 141

    def test_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None or empty"):
            RegionalExtractor.safe_convert_amu_id(None)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be None or empty"):
            RegionalExtractor.safe_convert_amu_id("")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            RegionalExtractor.safe_convert_amu_id("not_a_number")


# ===================================================================
# get_regional_ts_by_id()
# ===================================================================


class TestGetRegionalTsById:
    """Tests for the core regional API call method."""

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_successful_api_call(
        self, mock_post, configured_regional_extractor, sample_regional_entity, sample_regional_response
    ):
        """Happy path: valid entity returns parsed JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_regional_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        assert result == sample_regional_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_url_targets_index_endpoint(self, mock_post, configured_regional_extractor, sample_regional_entity):
        """URL should be {regional_url}/api/{index}."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert actual_url == f"{FAKE_REGIONAL_URL}/api/vegetation-vigor-index"

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_url_changes_with_index_param(self, mock_post, configured_regional_extractor, sample_regional_entity):
        """Switching index in regional_params should switch the URL path segment."""
        configured_regional_extractor.regional_params["index"] = "daily-precipitation"
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url.endswith("/api/daily-precipitation")

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_payload_contains_amu_id_as_int_list(
        self, mock_post, configured_regional_extractor, sample_regional_entity
    ):
        """Payload 'amuIds' must be a list of ints (even when input is string)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["amuIds"] == [2432528]
        assert isinstance(sent["amuIds"][0], int)

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_string_amu_id_converted_to_int(self, mock_post, configured_regional_extractor):
        """A string amu_id like '2432528' should be coerced to int in the payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id({"amu_id": "2432528"})

        sent = mock_post.call_args.kwargs["json"]
        assert sent["amuIds"] == [2432528]

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_tuple_string_amu_id_extracts_first(self, mock_post, configured_regional_extractor):
        """A PG tuple-string amu_id should be parsed down to its first integer."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id({"amu_id": '(141;"County";"Missouri")'})

        sent = mock_post.call_args.kwargs["json"]
        assert sent["amuIds"] == [141]

    def test_invalid_amu_id_raises_value_error(self, configured_regional_extractor):
        """An unparseable amu_id should raise ValueError before the HTTP call."""
        with pytest.raises(ValueError, match="Invalid amu_id"):
            configured_regional_extractor.get_regional_ts_by_id({"amu_id": "not_a_number"})

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_payload_contains_idblock_idpixeltype_indicators(
        self, mock_post, configured_regional_extractor, sample_regional_entity
    ):
        """idBlock, idPixelType, and indicatorTypeIds should be included when set."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["idBlock"] == 281
        assert sent["idPixelType"] == 1
        assert sent["indicatorTypeIds"] == [1]

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_payload_includes_dates_and_fillyeargap(
        self, mock_post, configured_regional_extractor, sample_regional_entity
    ):
        """startDate, endDate, and fillYearGaps should reflect setup."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["startDate"] == "2018-01-01"
        assert sent["endDate"] == "2026-12-31"
        assert sent["fillYearGaps"] is False

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_payload_omits_optional_fields_when_none(
        self, mock_post, configured_regional_extractor, sample_regional_entity
    ):
        """When idblock/idpixeltype/indicatorTypeIds/end_date are None, they should not appear."""
        configured_regional_extractor.regional_params.update(
            {
                "idblock": None,
                "idpixeltype": None,
                "indicatorTypeIds": None,
                "end_date": None,
            }
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert "idBlock" not in sent
        assert "idPixelType" not in sent
        assert "indicatorTypeIds" not in sent
        assert "endDate" not in sent

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, configured_regional_extractor, sample_regional_entity
    ):
        """Authorization header must carry the current bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_column_mapping_for_amu_id_resolves_custom_column(self, mock_post, configured_regional_extractor):
        """When amu_id is mapped to 'region_code', the entity dict uses that key."""
        configured_regional_extractor.column_mapping["amu_id"] = "region_code"
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_regional_extractor.get_regional_ts_by_id({"region_code": 2121564})

        sent = mock_post.call_args.kwargs["json"]
        assert sent["amuIds"] == [2121564]

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_http_error_propagates(self, mock_post, configured_regional_extractor, sample_regional_entity):
        """HTTPError from requests should propagate out of get_regional_ts_by_id."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.requests.post")
    def test_timeout_propagates(self, mock_post, configured_regional_extractor, sample_regional_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_regional_extractor.get_regional_ts_by_id(sample_regional_entity)


# ===================================================================
# get_regional_ts_by_id_safe()
# ===================================================================


class TestGetRegionalTsByIdSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(RegionalExtractor, "get_regional_ts_by_id")
    def test_success_returns_data(
        self, mock_api, configured_regional_extractor, sample_regional_entity, sample_regional_response
    ):
        mock_api.return_value = sample_regional_response

        result = configured_regional_extractor.get_regional_ts_by_id_safe(sample_regional_entity)

        assert result["success"] is True
        assert result["data"] == sample_regional_response
        assert result["error"] is None
        assert result["seasonfield_id"] == 2432528

    @patch.object(RegionalExtractor, "get_regional_ts_by_id")
    def test_http_error_returns_failure(self, mock_api, configured_regional_extractor, sample_regional_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_regional_extractor.get_regional_ts_by_id_safe(sample_regional_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["seasonfield_id"] == 2432528

    @patch.object(RegionalExtractor, "get_regional_ts_by_id")
    def test_value_error_returns_failure(self, mock_api, configured_regional_extractor, sample_regional_entity):
        """ValueError (e.g. invalid amu_id) is caught by the generic except branch."""
        mock_api.side_effect = ValueError("Invalid amu_id in entity_data")

        result = configured_regional_extractor.get_regional_ts_by_id_safe(sample_regional_entity)

        assert result["success"] is False
        assert "Invalid amu_id" in result["error"]
        assert result["data"] is None

    @patch.object(RegionalExtractor, "get_regional_ts_by_id")
    def test_generic_error_returns_failure(self, mock_api, configured_regional_extractor, sample_regional_entity):
        mock_api.side_effect = RuntimeError("something broke")

        result = configured_regional_extractor.get_regional_ts_by_id_safe(sample_regional_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["seasonfield_id"] == 2432528
