"""
Tests for EmergenceExtractor validation logic:
    - validate_crop() static method
    - setup_emergence_parameters() input validation
    - require_emergence_params decorator
"""

import pytest

from earthdaily.agriculture.processors.processor_emergence_functions import (
    EmergenceExtractor,
    available_crops,
)

pytestmark = pytest.mark.public

# ===================================================================
# validate_crop() — static method
# ===================================================================


class TestValidateCrop:
    """Tests for the static validate_crop() helper."""

    @pytest.mark.parametrize("crop", ["CORN", "SOYBEANS", "COTTON", "OTHERS", "SUGARCANE"])
    def test_valid_crop_uppercase(self, crop):
        result = EmergenceExtractor.validate_crop(crop, available_crops)
        assert result == crop

    def test_valid_crop_lowercase_normalized(self):
        assert EmergenceExtractor.validate_crop("corn", available_crops) == "CORN"

    def test_valid_crop_mixed_case_normalized(self):
        assert EmergenceExtractor.validate_crop("SoyBeans", available_crops) == "SOYBEANS"

    def test_valid_crop_with_whitespace(self):
        assert EmergenceExtractor.validate_crop("  COTTON  ", available_crops) == "COTTON"

    def test_second_corn_with_space(self):
        """The 'SECOND CORN' multi-word value should be accepted."""
        assert EmergenceExtractor.validate_crop("second corn", available_crops) == "SECOND CORN"

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Invalid crop"):
            EmergenceExtractor.validate_crop("BANANA", available_crops)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            EmergenceExtractor.validate_crop("", available_crops)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            EmergenceExtractor.validate_crop(None, available_crops)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid crop type"):
            EmergenceExtractor.validate_crop(123, available_crops)

    def test_all_available_crops_accepted(self):
        """Every value in available_crops should pass validation."""
        for crop in available_crops:
            assert EmergenceExtractor.validate_crop(crop, available_crops) == crop


# ===================================================================
# setup_emergence_parameters()
# ===================================================================


class TestSetupEmergenceParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, emergence_extractor):
        """All defaults should pass validation."""
        emergence_extractor.setup_emergence_parameters()
        params = emergence_extractor.emergence_params
        assert params is not None
        assert params["emergence_type"] == "INSEASON"
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["year"] == 2025
        assert params["data_source"] == "LR"
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, emergence_extractor):
        """Mirrors the notebook's setup_emergence_parameters call (cell 13)."""
        emergence_extractor.setup_emergence_parameters(
            emergence_type="INSEASON",
            season_duration=120,
            season_start_day=1,
            season_start_month=4,
            year="2025",
            data_source="LR",
            publish_af=True,
            partial_frequency=50,
            column_mapping={"crop": "crop.id"},
        )
        params = emergence_extractor.emergence_params
        assert params["emergence_type"] == "INSEASON"
        assert params["publish_af"] is True
        assert emergence_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, emergence_extractor):
        """Setup should configure cache_key_columns to [mapped id]."""
        emergence_extractor.setup_emergence_parameters()
        assert emergence_extractor.cache_key_columns == ["id"]

    def test_setup_with_custom_id_column_mapping(self, emergence_extractor):
        """When id is remapped, the column_mapping should reflect that."""
        emergence_extractor.setup_emergence_parameters(column_mapping={"id": "entity_id"})
        assert emergence_extractor.column_mapping["id"] == "entity_id"

    # --- emergence_type validation ---

    @pytest.mark.parametrize("etype", ["INSEASON", "HISTORICAL", "DELAY"])
    def test_valid_emergence_types(self, emergence_extractor, etype):
        emergence_extractor.setup_emergence_parameters(emergence_type=etype)
        assert emergence_extractor.emergence_params["emergence_type"] == etype

    def test_invalid_emergence_type_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="Invalid emergence type"):
            emergence_extractor.setup_emergence_parameters(emergence_type="BOGUS")

    # --- month validation ---

    def test_month_below_one_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            emergence_extractor.setup_emergence_parameters(season_start_month=0)

    def test_month_above_twelve_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            emergence_extractor.setup_emergence_parameters(season_start_month=13)

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_valid_months(self, emergence_extractor, month):
        emergence_extractor.setup_emergence_parameters(season_start_month=month)
        assert emergence_extractor.emergence_params["season_start_month"] == month

    # --- day validation ---

    def test_day_below_one_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            emergence_extractor.setup_emergence_parameters(season_start_day=0)

    def test_day_above_thirty_one_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            emergence_extractor.setup_emergence_parameters(season_start_day=32)

    # --- data_source validation ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, emergence_extractor, ds):
        emergence_extractor.setup_emergence_parameters(data_source=ds)
        assert emergence_extractor.emergence_params["data_source"] == ds

    def test_invalid_data_source_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="data_source"):
            emergence_extractor.setup_emergence_parameters(data_source="HR")

    # --- publish_af validation ---

    def test_publish_af_non_bool_raises(self, emergence_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            emergence_extractor.setup_emergence_parameters(publish_af="yes")

    @pytest.mark.parametrize("flag", [True, False])
    def test_publish_af_bool_accepted(self, emergence_extractor, flag):
        emergence_extractor.setup_emergence_parameters(publish_af=flag)
        assert emergence_extractor.emergence_params["publish_af"] is flag


# ===================================================================
# require_emergence_params decorator
# ===================================================================


class TestRequireEmergenceParamsDecorator:
    """Methods guarded by @require_emergence_params raise when params not set."""

    def test_get_emergence_api_without_params_raises(self, emergence_extractor):
        assert emergence_extractor.emergence_params is None
        with pytest.raises(RuntimeError, match="No emergence parameters found"):
            emergence_extractor.get_emergence_api(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
            )

    def test_format_emergence_json_without_params_raises(self, emergence_extractor):
        assert emergence_extractor.emergence_params is None
        with pytest.raises(RuntimeError, match="No emergence parameters found"):
            emergence_extractor.format_emergence_json({"id": "x", "data": {}})

    def test_process_single_without_params_raises(self, emergence_extractor):
        assert emergence_extractor.emergence_params is None
        with pytest.raises(RuntimeError, match="No emergence parameters found"):
            emergence_extractor.process_single_entity_emergence({"id": "x"})

    def test_get_emergence_api_safe_does_not_require_params(self, emergence_extractor):
        """The safe wrapper is NOT guarded by the decorator; it returns failure instead of raising."""
        assert emergence_extractor.emergence_params is None
        result = emergence_extractor.get_emergence_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
        )
        assert result["success"] is False
        assert "No emergence parameters" in result["error"]
