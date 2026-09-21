"""Tests for LocationBasedBorderExtractor validation logic:
- setup_location_based_border_parameters() input validation
- require_location_based_border_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_location_based_border_parameters()
# ===================================================================


class TestSetupLocationBasedBorderParameters:
    def test_default_parameters_succeed(self, location_based_border_extractor):
        location_based_border_extractor.setup_location_based_border_parameters()
        params = location_based_border_extractor.location_based_border_params
        assert params["simplified_geom"] is True
        assert params["partial_frequency"] == 50

    def test_custom_parameters(self, location_based_border_extractor):
        location_based_border_extractor.setup_location_based_border_parameters(
            simplified_geom=False, partial_frequency=25
        )
        params = location_based_border_extractor.location_based_border_params
        assert params["simplified_geom"] is False
        assert params["partial_frequency"] == 25

    def test_setup_sets_cache_key_columns(self, location_based_border_extractor):
        location_based_border_extractor.setup_location_based_border_parameters()
        # cache key uses the column-mapped id and geometry columns
        assert location_based_border_extractor.cache_key_columns == ["id", "geometry"]

    def test_invalid_simplified_geom_string(self, location_based_border_extractor):
        with pytest.raises(ValueError, match="simplified_geom"):
            location_based_border_extractor.setup_location_based_border_parameters(simplified_geom="yes")

    def test_invalid_simplified_geom_int(self, location_based_border_extractor):
        with pytest.raises(ValueError, match="simplified_geom"):
            location_based_border_extractor.setup_location_based_border_parameters(simplified_geom=1)

    def test_negative_partial_frequency(self, location_based_border_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            location_based_border_extractor.setup_location_based_border_parameters(partial_frequency=-1)

    def test_non_int_partial_frequency(self, location_based_border_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            location_based_border_extractor.setup_location_based_border_parameters(partial_frequency="50")

    def test_bool_partial_frequency_rejected(self, location_based_border_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            location_based_border_extractor.setup_location_based_border_parameters(partial_frequency=True)

    def test_column_mapping_applied(self, location_based_border_extractor):
        location_based_border_extractor.setup_location_based_border_parameters(
            column_mapping={"id": "entity_id", "geometry": "point_wkt"},
        )
        assert location_based_border_extractor.column_mapping["id"] == "entity_id"
        assert location_based_border_extractor.column_mapping["geometry"] == "point_wkt"
        assert location_based_border_extractor.cache_key_columns == ["entity_id", "point_wkt"]


# ===================================================================
# require_location_based_border_params decorator
# ===================================================================


class TestRequireLocationBasedBorderParamsDecorator:
    def test_format_without_params_raises(self, location_based_border_extractor):
        assert location_based_border_extractor.location_based_border_params is None
        with pytest.raises(RuntimeError, match="No location-based-border parameters"):
            location_based_border_extractor.format_location_based_border_json({"id": "x"})
