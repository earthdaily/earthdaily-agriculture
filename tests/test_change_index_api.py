"""
Tests for ChangeIndexExtractor API call logic:
    - get_change_index_api() URL construction, payload, validation, response stamping
    - get_change_index_api_safe() error wrapping
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.processors.processor_change_index_functions import ChangeIndexExtractor
from tests.conftest import CHANGE_INDEX_WKT, FAKE_CHANGE_INDEX_URL, FAKE_TOKEN

pytestmark = pytest.mark.public

# ===================================================================
# get_change_index_api()
# ===================================================================


class TestGetChangeIndexApi:
    """Tests for the core change index API call method."""

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_successful_api_call_returns_stamped_response(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        """Happy path: response is stamped with id and reference_date if missing."""
        mock_response = MagicMock()
        # Simulate API response that doesn't include id/reference_date — processor stamps them
        mock_response.json.return_value = {
            "status": "Processor done and metrics pushed.",
            "data": {"ChangeIndex": 1.43},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        assert result["id"] == "z361x33"
        assert result["reference_date"] == "2025-06-15"
        assert result["status"] == "Processor done and metrics pushed."
        assert result["data"]["ChangeIndex"] == 1.43

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_url_targets_change_index_processor_endpoint(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert actual_url.startswith(f"{FAKE_CHANGE_INDEX_URL}/change-index-processor?")

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_query_params_include_all_expected_keys(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        """The URL query string must encode all configured params (URL-encoded keys with spaces)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]

        assert "Parameter%20Profile=change_index_v1" in actual_url
        assert "Start%20Date=2025-06-15" in actual_url
        assert "Maximum%20period%20for%20the%20reference%20image=7" in actual_url
        assert "Maximum%20period%20for%20the%20previous%20image=15" in actual_url
        assert "Minimum%20period%20for%20the%20previous%20image=5" in actual_url
        assert "MapType=NDVI" in actual_url
        assert "Collections=Sentinel-2" in actual_url
        assert "Same%20Sensor=False" in actual_url

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_multiple_collections_appended_as_repeated_params(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        configured_change_index_extractor.change_index_params["collections"] = ["Sentinel-2", "Landsat"]
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_args = mock_post.call_args
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert "Collections=Sentinel-2" in actual_url
        assert "Collections=Landsat" in actual_url

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_payload_shape_default_omits_field_id(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        """With publish_af=False (default), payload contains crop/feature/sowing_date but NOT field_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1].get("data"))
        assert "field_id" not in sent_data
        assert sent_data["crop"] == "OTHERS"
        assert sent_data["feature"] == CHANGE_INDEX_WKT
        assert sent_data["sowing_date"] == "2025-04-01"

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_payload_includes_field_id_when_publish_af_true(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        """With publish_af=True, payload carries field_id (mirrors emergence/greenness/etc.)."""
        configured_change_index_extractor.change_index_params["publish_af"] = True
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1].get("data"))
        assert sent_data["field_id"] == "z361x33"

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_payload_defaults_for_optional_fields(self, mock_post, mock_wkt, configured_change_index_extractor):
        """Missing crop should default to 'Unknown'; missing sowing_date should default to ''."""
        entity = {
            "id": "z361x33",
            "geometry": CHANGE_INDEX_WKT,
            "reference_date": "2025-06-15",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(entity)

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1].get("data"))
        assert sent_data["crop"] == "Unknown"
        assert sent_data["sowing_date"] == ""

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_authorization_header_uses_bearer_token(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises_when_publish_af_true(self, configured_change_index_extractor):
        """When publish_af=True the payload must carry field_id, so a missing
        entity id is a hard error."""
        configured_change_index_extractor.change_index_params["publish_af"] = True
        with pytest.raises(ValueError, match="id"):
            configured_change_index_extractor.get_change_index_api(
                {"geometry": CHANGE_INDEX_WKT, "reference_date": "2025-06-15"}
            )

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_missing_id_does_not_raise_when_publish_af_false(
        self, mock_post, mock_wkt, configured_change_index_extractor
    ):
        """With publish_af=False (default), id is only used for logging and may
        be absent — the API call should still succeed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # No `id` key on the entity — must not raise.
        configured_change_index_extractor.get_change_index_api(
            {"geometry": CHANGE_INDEX_WKT, "reference_date": "2025-06-15"}
        )
        sent_data = json.loads(mock_post.call_args.kwargs.get("data"))
        assert "field_id" not in sent_data

    def test_missing_geometry_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_change_index_extractor.get_change_index_api({"id": "x", "reference_date": "2025-06-15"})

    @patch(
        "earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_change_index_extractor):
        with pytest.raises(ValueError, match="bad wkt"):
            configured_change_index_extractor.get_change_index_api(
                {"id": "x", "geometry": "INVALID", "reference_date": "2025-06-15"}
            )

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_reference_date_raises(self, mock_wkt, configured_change_index_extractor):
        with pytest.raises(ValueError, match="reference_date missing"):
            configured_change_index_extractor.get_change_index_api({"id": "x", "geometry": CHANGE_INDEX_WKT})

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_reference_date_format_raises(self, mock_wkt, configured_change_index_extractor):
        with pytest.raises(ValueError, match="invalid reference_date"):
            configured_change_index_extractor.get_change_index_api(
                {"id": "x", "geometry": CHANGE_INDEX_WKT, "reference_date": "15/06/2025"}
            )

    # --- HTTP error handling ---

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_http_error_is_raised(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_change_index_extractor.get_change_index_api(sample_change_index_entity)

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.requests.post")
    def test_timeout_is_raised(
        self, mock_post, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(requests.exceptions.Timeout):
            configured_change_index_extractor.get_change_index_api(sample_change_index_entity)


# ===================================================================
# get_change_index_api_safe()
# ===================================================================


class TestGetChangeIndexApiSafe:
    """Tests for the safe (non-throwing) wrapper."""

    @patch.object(ChangeIndexExtractor, "get_change_index_api")
    def test_success_returns_data(self, mock_api, configured_change_index_extractor, sample_change_index_entity):
        mock_api.return_value = {"id": "z361x33", "data": {"ChangeIndex": 1.43}}

        result = configured_change_index_extractor.get_change_index_api_safe(sample_change_index_entity)

        assert result["success"] is True
        assert result["data"] == {"id": "z361x33", "data": {"ChangeIndex": 1.43}}
        assert result["error"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(ChangeIndexExtractor, "get_change_index_api")
    def test_http_error_returns_failure(self, mock_api, configured_change_index_extractor, sample_change_index_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_change_index_extractor.get_change_index_api_safe(sample_change_index_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None
        assert result["entity_id"] == "z361x33"

    @patch.object(ChangeIndexExtractor, "get_change_index_api")
    def test_generic_error_returns_failure(
        self, mock_api, configured_change_index_extractor, sample_change_index_entity
    ):
        mock_api.side_effect = ValueError("something broke")

        result = configured_change_index_extractor.get_change_index_api_safe(sample_change_index_entity)

        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["entity_id"] == "z361x33"
