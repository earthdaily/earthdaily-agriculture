"""
Tests for HarvestExtractor validation logic:
    - validate_crop() static method
    - setup_harvest_parameters() input validation
    - require_harvest_params decorator
"""

import pytest

from earthdaily.agriculture.processors.processor_harvest_functions import (
    HarvestExtractor,
    available_crops,
)

pytestmark = pytest.mark.public

# ===================================================================
# validate_crop() — static method
# ===================================================================


class TestValidateCrop:
    """Tests for the static validate_crop() helper."""

    @pytest.mark.parametrize("crop", ["CORN", "SOYBEANS", "OTHERS", "SUGARCANE"])
    def test_valid_crop_uppercase(self, crop):
        result = HarvestExtractor.validate_crop(crop, available_crops)
        assert result == crop

    def test_valid_crop_lowercase_normalized(self):
        assert HarvestExtractor.validate_crop("corn", available_crops) == "CORN"

    def test_valid_crop_mixed_case_normalized(self):
        assert HarvestExtractor.validate_crop("SoyBeans", available_crops) == "SOYBEANS"

    def test_valid_crop_with_whitespace(self):
        assert HarvestExtractor.validate_crop("  OTHERS  ", available_crops) == "OTHERS"

    def test_second_corn_with_space(self):
        """The 'SECOND CORN' multi-word value should be accepted."""
        assert HarvestExtractor.validate_crop("second corn", available_crops) == "SECOND CORN"

    def test_unknown_crop_raises(self):
        """COTTON is NOT in harvest's available_crops (unlike emergence/greenness)."""
        with pytest.raises(ValueError, match="Invalid crop"):
            HarvestExtractor.validate_crop("COTTON", available_crops)

    def test_bogus_crop_raises(self):
        with pytest.raises(ValueError, match="Invalid crop"):
            HarvestExtractor.validate_crop("BANANA", available_crops)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            HarvestExtractor.validate_crop("", available_crops)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            HarvestExtractor.validate_crop(None, available_crops)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid crop type"):
            HarvestExtractor.validate_crop(123, available_crops)

    def test_all_available_crops_accepted(self):
        """Every value in available_crops should pass validation."""
        for crop in available_crops:
            assert HarvestExtractor.validate_crop(crop, available_crops) == crop


# ===================================================================
# setup_harvest_parameters()
# ===================================================================


class TestSetupHarvestParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, harvest_extractor):
        """All defaults should pass validation."""
        harvest_extractor.setup_harvest_parameters()
        params = harvest_extractor.harvest_params
        assert params is not None
        assert params["harvest_type"] == "INSEASON_HARVEST"
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["year"] == 2025
        assert params["data_source"] == "LR"
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, harvest_extractor):
        """Mirrors the notebook's setup_harvest_parameters call (cell 13)."""
        harvest_extractor.setup_harvest_parameters(
            harvest_type="HARVEST_READINESS",
            season_duration=120,
            season_start_day=1,
            season_start_month=4,
            year=2025,
            data_source="LR",
            publish_af=False,
            partial_frequency=50,
            column_mapping={"crop": "crop.id"},
        )
        params = harvest_extractor.harvest_params
        assert params["harvest_type"] == "HARVEST_READINESS"
        assert harvest_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, harvest_extractor):
        """Setup should configure cache_key_columns to [mapped id]."""
        harvest_extractor.setup_harvest_parameters()
        assert harvest_extractor.cache_key_columns == ["id"]

    def test_setup_with_custom_id_column_mapping(self, harvest_extractor):
        """When id is remapped, the column_mapping should reflect that."""
        harvest_extractor.setup_harvest_parameters(column_mapping={"id": "entity_id"})
        assert harvest_extractor.column_mapping["id"] == "entity_id"

    # --- harvest_type validation ---

    @pytest.mark.parametrize("htype", ["INSEASON_HARVEST", "HISTORICAL_HARVEST", "HARVEST_READINESS"])
    def test_valid_harvest_types(self, harvest_extractor, htype):
        harvest_extractor.setup_harvest_parameters(harvest_type=htype)
        assert harvest_extractor.harvest_params["harvest_type"] == htype

    def test_invalid_harvest_type_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="Invalid harvest type"):
            harvest_extractor.setup_harvest_parameters(harvest_type="BOGUS")

    # --- month validation ---

    def test_month_below_one_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            harvest_extractor.setup_harvest_parameters(season_start_month=0)

    def test_month_above_twelve_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            harvest_extractor.setup_harvest_parameters(season_start_month=13)

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_valid_months(self, harvest_extractor, month):
        harvest_extractor.setup_harvest_parameters(season_start_month=month)
        assert harvest_extractor.harvest_params["season_start_month"] == month

    # --- day validation ---

    def test_day_below_one_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            harvest_extractor.setup_harvest_parameters(season_start_day=0)

    def test_day_above_thirty_one_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            harvest_extractor.setup_harvest_parameters(season_start_day=32)

    # --- data_source validation ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, harvest_extractor, ds):
        harvest_extractor.setup_harvest_parameters(data_source=ds)
        assert harvest_extractor.harvest_params["data_source"] == ds

    def test_invalid_data_source_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="data_source"):
            harvest_extractor.setup_harvest_parameters(data_source="HR")

    # --- publish_af validation ---

    def test_publish_af_non_bool_raises(self, harvest_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            harvest_extractor.setup_harvest_parameters(publish_af="yes")

    @pytest.mark.parametrize("flag", [True, False])
    def test_publish_af_bool_accepted(self, harvest_extractor, flag):
        harvest_extractor.setup_harvest_parameters(publish_af=flag)
        assert harvest_extractor.harvest_params["publish_af"] is flag


# ===================================================================
# require_harvest_params decorator
# ===================================================================


class TestRequireHarvestParamsDecorator:
    """Methods guarded by @require_harvest_params raise when params not set."""

    def test_get_harvest_api_without_params_raises(self, harvest_extractor):
        assert harvest_extractor.harvest_params is None
        with pytest.raises(RuntimeError, match="No harvest parameters found"):
            harvest_extractor.get_harvest_api({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"})

    def test_format_harvest_json_without_params_raises(self, harvest_extractor):
        assert harvest_extractor.harvest_params is None
        with pytest.raises(RuntimeError, match="No harvest parameters found"):
            harvest_extractor.format_harvest_json({"id": "x", "data": {}})

    def test_process_single_without_params_raises(self, harvest_extractor):
        assert harvest_extractor.harvest_params is None
        with pytest.raises(RuntimeError, match="No harvest parameters found"):
            harvest_extractor.process_single_entity_harvest({"id": "x"})

    def test_get_harvest_api_safe_does_not_require_params(self, harvest_extractor):
        """The safe wrapper is NOT guarded by the decorator; it returns failure instead of raising."""
        assert harvest_extractor.harvest_params is None
        result = harvest_extractor.get_harvest_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
        )
        assert result["success"] is False
        assert "No harvest parameters" in result["error"]
