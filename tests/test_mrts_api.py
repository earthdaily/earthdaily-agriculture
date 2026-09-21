"""
Tests for MRTSExtractor API call logic:
    - get_mrts_api() URL, payload, validation, date handling, LAI vs other indices
    - get_mrts_api_safe() error wrapping
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.extractors.VTS_functions import MRTSExtractor
from tests.conftest import FAKE_MRTS_URL, FAKE_TOKEN, MRTS_WKT

pytestmark = pytest.mark.public

# ===================================================================
# get_mrts_api()
# ===================================================================


class TestGetMrtsApi:
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_successful_api_call(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        sample_response = {"rawData": [{"date": "2025-05-10", "value": 0.42}], "smoothedData": []}
        mock_response = MagicMock()
        mock_response.json.return_value = sample_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        assert result == sample_response
        mock_post.assert_called_once()

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_url_targets_time_serie_endpoint(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        actual_url = mock_post.call_args.args[0]
        assert actual_url == f"{FAKE_MRTS_URL}/time-serie"

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_payload_default_fields(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        """Payload should mirror mrts_params (with seasonfield.geometry only)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["seasonfield"]["geometry"] == MRTS_WKT
        assert "id" not in sent["seasonfield"]
        assert "crop" not in sent["seasonfield"]
        assert sent["vegetationIndex"] == "NDVI"
        assert sent["aggregation"] == "average"
        assert sent["smoothingMethod"] == "Whittaker"
        assert sent["clearCoverMin"] == 100
        assert sent["applyDenoiser"] is True
        assert sent["applyEndOfCurve"] is True
        assert sent["outputSaturation"] is True
        assert sent["extractRawDatasets"] is True
        assert sent["computeTemporalConsistency"] is True

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_payload_dates_iso_normalized(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        """YYYY-MM-DD start/end dates should be normalized to ISO with time component."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["startDate"] == "2025-05-01T00:00:00.000Z"
        assert sent["endDate"] == "2025-10-15T23:59:59.999Z"

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_entity_dates_take_precedence(self, mock_post, mock_wkt, configured_mrts_extractor):
        """Per-entity start_date/end_date should override params dates."""
        entity = {
            "id": "ent_x",
            "geometry": MRTS_WKT,
            "start_date": "2024-06-01",
            "end_date": "2024-09-30",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["startDate"] == "2024-06-01T00:00:00.000Z"
        assert sent["endDate"] == "2024-09-30T23:59:59.999Z"

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_sensors_omitted_when_none(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        """sensors=None should NOT appear in the payload (API uses all available)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert "sensors" not in sent

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_sensors_included_when_set(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        configured_mrts_extractor.mrts_params["sensors"] = ["Sentinel_2", "Landsat_8"]
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["sensors"] == ["Sentinel_2", "Landsat_8"]

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_temporal_consistency_threshold_included(
        self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity
    ):
        """The threshold dict should appear in the payload when compute_temporal_consistency=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["temporalConsistencyThreshold"] == {
            "Ndvi": 0.06,
            "Lai": 0.3,
            "S2Rep": 2.5,
        }

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_lai_index_includes_crop_in_seasonfield(self, mock_post, mock_wkt, configured_mrts_extractor):
        """LAI index requires crop on the seasonfield payload."""
        configured_mrts_extractor.mrts_params["vegetation_index"] = "LAI"
        entity = {"id": "ent_lai", "geometry": MRTS_WKT, "crop": "WINTER_OSR"}
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(entity)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["seasonfield"]["crop"] == "WINTER_OSR"
        assert sent["vegetationIndex"] == "LAI"

    def test_lai_without_crop_raises(self, configured_mrts_extractor):
        """LAI requires crop on the entity — missing crop must raise ValueError."""
        configured_mrts_extractor.mrts_params["vegetation_index"] = "LAI"
        entity = {"id": "ent_lai", "geometry": MRTS_WKT}  # no crop

        with patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_mrts_extractor, "ensure_token_valid"):
                with pytest.raises(ValueError):
                    configured_mrts_extractor.get_mrts_api(entity)

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_authorization_header(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        mock_response = MagicMock()
        mock_response.json.return_value = {"rawData": [], "smoothedData": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    # --- Input validation errors ---

    def test_missing_geometry_raises(self, configured_mrts_extractor):
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="geometry"):
                configured_mrts_extractor.get_mrts_api({"id": "x"})

    @patch(
        "earthdaily.agriculture.extractors.VTS_functions.validate_wkt",
        side_effect=ValueError("bad wkt"),
    )
    def test_invalid_geometry_raises(self, mock_wkt, configured_mrts_extractor):
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="bad wkt"):
                configured_mrts_extractor.get_mrts_api({"id": "x", "geometry": "INVALID"})

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_start_date_format_raises(self, mock_wkt, configured_mrts_extractor):
        entity = {
            "id": "x",
            "geometry": MRTS_WKT,
            "start_date": "01/05/2025",
            "end_date": "2025-10-15",
        }
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                configured_mrts_extractor.get_mrts_api(entity)

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_end_date_format_raises(self, mock_wkt, configured_mrts_extractor):
        entity = {
            "id": "x",
            "geometry": MRTS_WKT,
            "start_date": "2025-05-01",
            "end_date": "10/15/2025",
        }
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                configured_mrts_extractor.get_mrts_api(entity)

    def test_no_dates_anywhere_raises(self, configured_mrts_extractor):
        """If params has no dates and entity has no dates, raise."""
        del configured_mrts_extractor.mrts_params["start_date"]
        del configured_mrts_extractor.mrts_params["end_date"]
        entity = {"id": "x", "geometry": MRTS_WKT}
        with patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_mrts_extractor, "ensure_token_valid"):
                with pytest.raises(ValueError, match="start_date"):
                    configured_mrts_extractor.get_mrts_api(entity)

    # --- HTTP errors ---

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_http_error_propagates(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_mrts_extractor.get_mrts_api(sample_mrts_entity)

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.requests.post")
    def test_timeout_propagates(self, mock_post, mock_wkt, configured_mrts_extractor, sample_mrts_entity):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.Timeout):
                configured_mrts_extractor.get_mrts_api(sample_mrts_entity)


# ===================================================================
# get_mrts_api_safe()
# ===================================================================


class TestGetMrtsApiSafe:
    @patch.object(MRTSExtractor, "get_mrts_api")
    def test_success_returns_data(self, mock_api, configured_mrts_extractor, sample_mrts_entity):
        mock_api.return_value = {"rawData": [{"date": "2025-05-10", "value": 0.42}]}

        result = configured_mrts_extractor.get_mrts_api_safe(sample_mrts_entity)

        assert result["success"] is True
        assert result["data"]["rawData"][0]["value"] == 0.42
        assert result["error"] is None
        assert result["entity_id"] == "ent_001"

    @patch.object(MRTSExtractor, "get_mrts_api")
    def test_http_error_returns_failure(self, mock_api, configured_mrts_extractor, sample_mrts_entity):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"
        mock_api.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = configured_mrts_extractor.get_mrts_api_safe(sample_mrts_entity)

        assert result["success"] is False
        assert "422" in result["error"]
        assert result["data"] is None

    @patch.object(MRTSExtractor, "get_mrts_api")
    def test_value_error_returns_validation_failure(self, mock_api, configured_mrts_extractor, sample_mrts_entity):
        """ValueError is wrapped as 'Validation error: ...'."""
        mock_api.side_effect = ValueError("bad input")

        result = configured_mrts_extractor.get_mrts_api_safe(sample_mrts_entity)

        assert result["success"] is False
        assert "Validation error" in result["error"]
        assert "bad input" in result["error"]

    @patch.object(MRTSExtractor, "get_mrts_api")
    def test_generic_error_returns_failure(self, mock_api, configured_mrts_extractor, sample_mrts_entity):
        mock_api.side_effect = RuntimeError("boom")

        result = configured_mrts_extractor.get_mrts_api_safe(sample_mrts_entity)

        assert result["success"] is False
        assert "boom" in result["error"]
