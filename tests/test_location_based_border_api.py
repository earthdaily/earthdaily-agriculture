"""Tests for LocationBasedBorderExtractor API surface:
- get_location_based_border_api() URL construction, query params, auth, validation, response stamping
- get_location_based_border_api_safe() never-raises contract
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tests.conftest import (
    FAKE_LOCATION_BORDER_URL,
    FAKE_TOKEN,
    LOCATION_BORDER_POINT_WKT,
    LOCATION_BORDER_POLYGON_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# get_location_based_border_api()
# ===================================================================


class TestGetLocationBasedBorderApi:
    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_successful_api_call_returns_stamped_response(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"geometry": LOCATION_BORDER_POLYGON_WKT}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)
        # Entity id and the input point WKT are stamped onto the response by the API method.
        assert result["id"] == "loc_001"
        assert result["point_wkt"] == LOCATION_BORDER_POINT_WKT
        assert result["geometry"] == LOCATION_BORDER_POLYGON_WKT

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_double_encoded_string_response_is_parsed(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        """Some servers double-encode and return the JSON body as a quoted string.
        The API method must parse it through one more time so callers always see a dict."""
        encoded = json.dumps(
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}
        )
        mock_response = MagicMock()
        mock_response.json.return_value = encoded  # str, not dict
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)
        assert isinstance(result, dict)
        assert result["type"] == "Feature"
        assert result["id"] == "loc_001"
        assert result["point_wkt"] == LOCATION_BORDER_POINT_WKT

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_response_id_preserved_as_field_border_id(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        """The API returns its own ``id`` (field-border identifier). When we stamp the row's
        entity id onto the response, the original id must move to ``field_border_id`` so it
        isn't lost."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "Feature",
            "id": "855141_14TNM_us-1234",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {"id": "855141_14TNM_us-1234", "area_ha": 29.2},
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)
        assert result["id"] == "loc_001"  # row's entity id wins
        assert result["field_border_id"] == "855141_14TNM_us-1234"  # API id preserved

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_url_targets_automatic_boundary_endpoint(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)

        actual_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args.kwargs["url"]
        assert actual_url.startswith(FAKE_LOCATION_BORDER_URL)

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_query_params_carry_location_and_simplified_geom(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)

        actual_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args.kwargs["url"]
        # Location is "longitude,latitude" — POINT (-93.6 41.5) → "-93.6,41.5"
        assert "location=-93.6,41.5" in actual_url
        # simplified_geom is the boolean lower-cased (matches API spec).
        assert "simplified_geom=true" in actual_url

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_simplified_geom_false_is_lowercased(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        configured_location_based_border_extractor.location_based_border_params["simplified_geom"] = False
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)

        actual_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args.kwargs["url"]
        assert "simplified_geom=false" in actual_url

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_no_payload_sent(self, mock_get, configured_location_based_border_extractor, sample_location_border_entity):
        """The endpoint is GET — no body should be sent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)

        # neither `data` nor `json` body kwargs should be set
        kwargs = mock_get.call_args.kwargs
        assert "data" not in kwargs
        assert "json" not in kwargs

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_authorization_header_uses_bearer_token(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)
        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Accept"] == "application/json"

    # --- Input validation errors ---

    def test_missing_id_raises(self, configured_location_based_border_extractor):
        with pytest.raises(ValueError, match="'id'"):
            configured_location_based_border_extractor.get_location_based_border_api(
                {"geometry": LOCATION_BORDER_POINT_WKT}
            )

    def test_missing_geometry_raises(self, configured_location_based_border_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_location_based_border_extractor.get_location_based_border_api({"id": "x"})

    def test_invalid_geometry_raises(self, configured_location_based_border_extractor):
        with pytest.raises(ValueError, match="geometry"):
            configured_location_based_border_extractor.get_location_based_border_api({"id": "x", "geometry": "INVALID"})

    def test_polygon_geometry_rejected(self, configured_location_based_border_extractor):
        """The endpoint expects a Point — polygons must be rejected with a clear message."""
        polygon = "POLYGON ((-93.6 41.5, -93.59 41.5, -93.59 41.51, -93.6 41.51, -93.6 41.5))"
        with pytest.raises(ValueError, match="must be a Point"):
            configured_location_based_border_extractor.get_location_based_border_api({"id": "x", "geometry": polygon})

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_http_error_propagates(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        http_error = requests.exceptions.HTTPError("404")
        http_error.response = MagicMock(status_code=404, text="Not found")
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)

    @patch(
        "earthdaily.agriculture.extractors.location_based_border_functions.requests.get",
        side_effect=requests.exceptions.Timeout("timed out"),
    )
    def test_timeout_propagates(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        with pytest.raises(requests.exceptions.Timeout):
            configured_location_based_border_extractor.get_location_based_border_api(sample_location_border_entity)


# ===================================================================
# get_location_based_border_api_safe()
# ===================================================================


class TestGetLocationBasedBorderApiSafe:
    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_success_returns_data(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"geometry": LOCATION_BORDER_POLYGON_WKT}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = configured_location_based_border_extractor.get_location_based_border_api_safe(
            sample_location_border_entity
        )
        assert result["success"] is True
        assert result["entity_id"] == "loc_001"
        assert result["data"]["geometry"] == LOCATION_BORDER_POLYGON_WKT
        assert result["error"] is None

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.requests.get")
    def test_http_error_returns_failure(
        self, mock_get, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_response = MagicMock()
        http_error = requests.exceptions.HTTPError("500")
        http_error.response = MagicMock(status_code=500, text="server error")
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        result = configured_location_based_border_extractor.get_location_based_border_api_safe(
            sample_location_border_entity
        )
        assert result["success"] is False
        assert "500" in result["error"]
        assert result["data"] is None

    def test_validation_error_returns_failure(self, configured_location_based_border_extractor):
        # Missing geometry → safe wrapper catches the ValueError raised by the inner call.
        result = configured_location_based_border_extractor.get_location_based_border_api_safe({"id": "loc_001"})
        assert result["success"] is False
        assert "geometry" in result["error"]
