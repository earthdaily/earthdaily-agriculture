"""
Tests for GreennessExtractor validation logic:
    - validate_crop() static method
    - setup_greenness_parameters() input validation
    - require_greenness_params decorator
"""

import pytest

from earthdaily.agriculture.processors.processor_greenness_functions import (
    GreennessExtractor,
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
        result = GreennessExtractor.validate_crop(crop, available_crops)
        assert result == crop

    def test_valid_crop_lowercase_normalized(self):
        assert GreennessExtractor.validate_crop("corn", available_crops) == "CORN"

    def test_valid_crop_mixed_case_normalized(self):
        assert GreennessExtractor.validate_crop("SoyBeans", available_crops) == "SOYBEANS"

    def test_valid_crop_with_whitespace(self):
        assert GreennessExtractor.validate_crop("  COTTON  ", available_crops) == "COTTON"

    def test_second_corn_with_space(self):
        """The 'SECOND CORN' multi-word value should be accepted."""
        assert GreennessExtractor.validate_crop("second corn", available_crops) == "SECOND CORN"

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Invalid crop"):
            GreennessExtractor.validate_crop("BANANA", available_crops)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            GreennessExtractor.validate_crop("", available_crops)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            GreennessExtractor.validate_crop(None, available_crops)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid crop type"):
            GreennessExtractor.validate_crop(123, available_crops)

    def test_all_available_crops_accepted(self):
        """Every value in available_crops should pass validation."""
        for crop in available_crops:
            assert GreennessExtractor.validate_crop(crop, available_crops) == crop


# ===================================================================
# setup_greenness_parameters()
# ===================================================================


class TestSetupGreennessParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, greenness_extractor):
        """All defaults should pass validation."""
        greenness_extractor.setup_greenness_parameters()
        params = greenness_extractor.greenness_params
        assert params is not None
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["year"] == 2025
        assert params["sowing_date"] == "2025-04-01"
        assert params["data_source"] == "LR"
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, greenness_extractor):
        """Mirrors the notebook's setup_greenness_parameters call (cell 13)."""
        greenness_extractor.setup_greenness_parameters(
            season_duration=120,
            season_start_day=1,
            season_start_month=4,
            year="2025",  # notebook passes a string
            sowing_date="2025-04-01",
            data_source="LR",
            publish_af=False,
            partial_frequency=50,
            column_mapping={"crop": "crop.id"},
        )
        params = greenness_extractor.greenness_params
        # year string is passed through verbatim — the API consumer accepts both
        assert params["year"] == "2025"
        assert params["data_source"] == "LR"
        assert greenness_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, greenness_extractor):
        """Setup should configure cache_key_columns to [mapped id]."""
        greenness_extractor.setup_greenness_parameters()
        assert greenness_extractor.cache_key_columns == ["id"]

    def test_setup_with_custom_id_column_mapping(self, greenness_extractor):
        """When id is remapped, the column_mapping reflects the override."""
        greenness_extractor.setup_greenness_parameters(column_mapping={"id": "entity_id"})
        assert greenness_extractor.column_mapping["id"] == "entity_id"

    # --- month validation ---

    def test_month_below_one_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            greenness_extractor.setup_greenness_parameters(season_start_month=0)

    def test_month_above_twelve_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            greenness_extractor.setup_greenness_parameters(season_start_month=13)

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_valid_months(self, greenness_extractor, month):
        greenness_extractor.setup_greenness_parameters(season_start_month=month)
        assert greenness_extractor.greenness_params["season_start_month"] == month

    # --- day validation ---

    def test_day_below_one_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            greenness_extractor.setup_greenness_parameters(season_start_day=0)

    def test_day_above_thirty_one_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            greenness_extractor.setup_greenness_parameters(season_start_day=32)

    # --- data_source validation ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, greenness_extractor, ds):
        greenness_extractor.setup_greenness_parameters(data_source=ds)
        assert greenness_extractor.greenness_params["data_source"] == ds

    def test_invalid_data_source_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="data_source"):
            greenness_extractor.setup_greenness_parameters(data_source="HR")

    # --- sowing_date validation ---

    def test_invalid_sowing_date_format_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="sowing_date"):
            greenness_extractor.setup_greenness_parameters(sowing_date="01/04/2025")

    def test_empty_sowing_date_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="sowing_date"):
            greenness_extractor.setup_greenness_parameters(sowing_date="")

    def test_non_string_sowing_date_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="sowing_date"):
            greenness_extractor.setup_greenness_parameters(sowing_date=20250401)

    # --- publish_af validation ---

    def test_publish_af_non_bool_raises(self, greenness_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            greenness_extractor.setup_greenness_parameters(publish_af="yes")

    @pytest.mark.parametrize("flag", [True, False])
    def test_publish_af_bool_accepted(self, greenness_extractor, flag):
        greenness_extractor.setup_greenness_parameters(publish_af=flag)
        assert greenness_extractor.greenness_params["publish_af"] is flag


# ===================================================================
# require_greenness_params decorator
# ===================================================================


class TestRequireGreennessParamsDecorator:
    """Methods guarded by @require_greenness_params raise when params not set."""

    def test_get_greenness_api_without_params_raises(self, greenness_extractor):
        assert greenness_extractor.greenness_params is None
        with pytest.raises(RuntimeError, match="No greenness parameters found"):
            greenness_extractor.get_greenness_api(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
            )

    def test_format_greenness_json_without_params_raises(self, greenness_extractor):
        assert greenness_extractor.greenness_params is None
        with pytest.raises(RuntimeError, match="No greenness parameters found"):
            greenness_extractor.format_greenness_json({"id": "x", "data": {}})

    def test_process_single_without_params_raises(self, greenness_extractor):
        assert greenness_extractor.greenness_params is None
        with pytest.raises(RuntimeError, match="No greenness parameters found"):
            greenness_extractor.process_single_entity_greenness({"id": "x"})

    def test_get_greenness_api_safe_does_not_require_params(self, greenness_extractor):
        """The safe wrapper is NOT guarded by the decorator; it returns failure instead of raising."""
        assert greenness_extractor.greenness_params is None
        result = greenness_extractor.get_greenness_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
        )
        assert result["success"] is False
        assert "No greenness parameters" in result["error"]
