"""`column_mapping` must reach EVERY entity field an extractor reads.

Several extractors read their entity inputs with a plain dict lookup —
``entity_data.get("image_id")`` — instead of ``get_entity_value``. A plain lookup
never consults ``column_mapping``, so a caller who mapped the column got **no
error and no effect**: the extractor reported the field missing while the value
sat in the DataFrame under the mapped name.

`ZARCExtractor` was the worst case. `emergence_date` is its one required entity
input — it raises without it — and it was both undocumented and unmappable.

Each field gets two tests, because the fix has to be additive:

- the **mapped** column resolves (the bug), and
- the **canonical** column still resolves with no mapping configured (the
  regression risk — `get_entity_value` falls back to the canonical key, so this
  must keep working for every existing caller).
"""

from unittest.mock import MagicMock, patch

import pytest

from earthdaily.agriculture.core.base_extractor import BaseExtractor
from earthdaily.agriculture.processors.processor_zarc_functions import ZARCExtractor
from tests.conftest import (
    DIFFERENCE_IMG_1,
    DIFFERENCE_IMG_2,
    DIFFERENCE_WKT,
)

pytestmark = pytest.mark.public


def _json_response(payload=None):
    response = MagicMock()
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status.return_value = None
    response.status_code = 200
    return response


# ===================================================================
# DifferenceExtractor — image_id_1 / image_id_2
# ===================================================================


class TestDifferenceImageIds:
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_mapped_image_id_columns_resolve(self, mock_post, _wkt, configured_difference_extractor):
        """Mapped columns must reach the request. Previously raised 'must include image_id_1'."""
        ext = configured_difference_extractor
        ext.column_mapping["image_id_1"] = "before_img"
        ext.column_mapping["image_id_2"] = "after_img"
        mock_post.return_value = _json_response()

        entity = {
            "id": "field-1",
            "geometry": DIFFERENCE_WKT,
            "before_img": DIFFERENCE_IMG_1,
            "after_img": DIFFERENCE_IMG_2,
        }

        ext.get_difference_map_api(entity)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["earliestImage"]["id"] == DIFFERENCE_IMG_1
        assert payload["latestImage"]["id"] == DIFFERENCE_IMG_2

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.requests.post")
    def test_canonical_columns_still_resolve_unmapped(
        self, mock_post, _wkt, configured_difference_extractor, sample_difference_entity
    ):
        """No mapping configured — the canonical names must behave exactly as before."""
        mock_post.return_value = _json_response()

        configured_difference_extractor.get_difference_map_api(sample_difference_entity)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["earliestImage"]["id"] == DIFFERENCE_IMG_1
        assert payload["latestImage"]["id"] == DIFFERENCE_IMG_2

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_still_raises(self, _wkt, configured_difference_extractor):
        """The guard must survive the refactor — a genuinely absent field still errors."""
        with pytest.raises(ValueError, match="image_id_1"):
            configured_difference_extractor.get_difference_map_api({"id": "field-1", "geometry": DIFFERENCE_WKT})


# ===================================================================
# ZoningExtractor — image_id
# ===================================================================


class TestZoningImageId:
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_mapped_image_id_resolves(self, mock_post, _wkt, configured_zoning_extractor):
        ext = configured_zoning_extractor
        ext.column_mapping["image_id"] = "scene_ref"
        mock_post.return_value = _json_response()

        ext.get_zoning_map_api({"id": "field-1", "geometry": DIFFERENCE_WKT, "scene_ref": "IMG-42"})

        payload = mock_post.call_args.kwargs["json"]
        assert "IMG-42" in str(payload)

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.requests.post")
    def test_canonical_image_id_still_resolves(self, mock_post, _wkt, configured_zoning_extractor):
        mock_post.return_value = _json_response()

        configured_zoning_extractor.get_zoning_map_api(
            {"id": "field-1", "geometry": DIFFERENCE_WKT, "image_id": "IMG-42"}
        )

        payload = mock_post.call_args.kwargs["json"]
        assert "IMG-42" in str(payload)

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_still_raises(self, _wkt, configured_zoning_extractor):
        with pytest.raises(ValueError, match="image_id"):
            configured_zoning_extractor.get_zoning_map_api({"id": "field-1", "geometry": DIFFERENCE_WKT})


# ===================================================================
# FLMExtractor — image_id
# ===================================================================


class TestFlmImageId:
    """`process_single_entity_flm` read `row.get("image_id")` directly.

    It returns an error dict rather than raising, so these assert on the real
    read site: with the column mapped, the run must get PAST the "Missing
    image_id" gate instead of stopping at it.
    """

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_mapped_image_id_resolves(self, _wkt, configured_flm_extractor):
        ext = configured_flm_extractor
        ext.column_mapping["image_id"] = "scene_ref"
        ext.get_flm_map = MagicMock(return_value={})

        result = ext.process_single_entity_flm(
            {"id": "field-1", "geometry": DIFFERENCE_WKT, "scene_ref": DIFFERENCE_IMG_1}
        )

        assert (result["error"] or {}).get("message") != "Missing image_id"
        ext.get_flm_map.assert_called_once()
        assert ext.get_flm_map.call_args[0][1] == DIFFERENCE_IMG_1

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_canonical_image_id_still_resolves(self, _wkt, configured_flm_extractor):
        ext = configured_flm_extractor
        ext.get_flm_map = MagicMock(return_value={})

        result = ext.process_single_entity_flm(
            {"id": "field-1", "geometry": DIFFERENCE_WKT, "image_id": DIFFERENCE_IMG_1}
        )

        assert (result["error"] or {}).get("message") != "Missing image_id"
        assert ext.get_flm_map.call_args[0][1] == DIFFERENCE_IMG_1

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_still_reports_the_error(self, _wkt, configured_flm_extractor):
        result = configured_flm_extractor.process_single_entity_flm({"id": "field-1", "geometry": DIFFERENCE_WKT})

        assert result["error"]["message"] == "Missing image_id"


# ===================================================================
# ZARCExtractor — emergence_date (REQUIRED), soil_type, cycle
# ===================================================================


@pytest.fixture
def zarc_extractor(mock_logger):
    """ZARCExtractor with BaseExtractor.__init__ bypassed — no fixture existed."""
    with patch.object(ZARCExtractor, "__init__", lambda self, *a, **kw: None):
        ext = ZARCExtractor.__new__(ZARCExtractor)

    ext.bearer_token = "fake-token"
    ext.token_expiration = 9_999_999_999  # far future: ensure_token_valid is a no-op
    ext.workflow_ref = None
    ext.env = "preprod"
    ext.logger = mock_logger
    ext.extractor_name = "ZARCExtractor"
    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None
    ext.use_cache = False
    ext.cache_key_columns = None
    ext.zarc_url = "http://fake-zarc.example.com"
    ext.zarc_params = {
        "crop": "SOYBEANS",
        "nb_days_sowing_emergence": 20,
        "soil_type": None,
        "cycle": None,
        "partial_frequency": 50,
    }
    return ext


class TestZarcEntityFields:
    """`emergence_date` is ZARC's one required entity input — and was unmappable."""

    @patch("earthdaily.agriculture.processors.processor_zarc_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_zarc_functions.requests.post")
    def test_mapped_emergence_date_resolves(self, mock_post, _wkt, zarc_extractor):
        """Previously raised 'Missing emergence_date' with the value present under the mapped name."""
        zarc_extractor.column_mapping["emergence_date"] = "emergencia"
        mock_post.return_value = _json_response({"id": "field-1", "data": {}})

        zarc_extractor.get_zarc_api({"id": "field-1", "geometry": DIFFERENCE_WKT, "emergencia": "2025-04-15"})

        assert "date_emergence=2025-04-15" in mock_post.call_args[0][0]

    @patch("earthdaily.agriculture.processors.processor_zarc_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_zarc_functions.requests.post")
    def test_canonical_emergence_date_still_resolves(self, mock_post, _wkt, zarc_extractor):
        mock_post.return_value = _json_response({"id": "field-1", "data": {}})

        zarc_extractor.get_zarc_api({"id": "field-1", "geometry": DIFFERENCE_WKT, "emergence_date": "2025-04-15"})

        assert "date_emergence=2025-04-15" in mock_post.call_args[0][0]

    @patch("earthdaily.agriculture.processors.processor_zarc_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_zarc_functions.requests.post")
    def test_iso_timestamp_is_normalised(self, mock_post, _wkt, zarc_extractor):
        """Routing through get_entity_value also normalises DATE_FIELDS, which a raw .get() did not."""
        mock_post.return_value = _json_response({"id": "field-1", "data": {}})

        zarc_extractor.get_zarc_api(
            {"id": "field-1", "geometry": DIFFERENCE_WKT, "emergence_date": "2025-04-15T00:00:00"}
        )

        assert "date_emergence=2025-04-15" in mock_post.call_args[0][0]

    @patch("earthdaily.agriculture.processors.processor_zarc_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_emergence_date_still_raises(self, _wkt, zarc_extractor):
        with pytest.raises(ValueError, match="emergence_date"):
            zarc_extractor.get_zarc_api({"id": "field-1", "geometry": DIFFERENCE_WKT})

    @patch("earthdaily.agriculture.processors.processor_zarc_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_zarc_functions.requests.post")
    def test_mapped_nb_days_overrides_the_setup_default(self, mock_post, _wkt, zarc_extractor):
        zarc_extractor.column_mapping["nb_days_sowing_emergence"] = "dias"
        mock_post.return_value = _json_response({"id": "field-1", "data": {}})

        zarc_extractor.get_zarc_api(
            {"id": "field-1", "geometry": DIFFERENCE_WKT, "emergence_date": "2025-04-15", "dias": 33}
        )

        assert "nb_days_sowing_emergence=33" in mock_post.call_args[0][0]
