"""
Tests for InSeasonMonitoringExtractor validation logic:
    - validate_crop() static method
    - setup_inseason_monitoring_parameters() input validation
    - require_inseason_monitoring_params decorator
"""

import pytest

from earthdaily.agriculture.processors.processor_inseason_monitoring_functions import (
    InSeasonMonitoringExtractor,
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
        assert InSeasonMonitoringExtractor.validate_crop(crop, available_crops) == crop

    def test_valid_crop_lowercase_normalized(self):
        assert InSeasonMonitoringExtractor.validate_crop("others", available_crops) == "OTHERS"

    def test_valid_crop_with_whitespace(self):
        assert InSeasonMonitoringExtractor.validate_crop("  CORN  ", available_crops) == "CORN"

    def test_second_corn_with_space(self):
        assert InSeasonMonitoringExtractor.validate_crop("second corn", available_crops) == "SECOND CORN"

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Invalid crop"):
            InSeasonMonitoringExtractor.validate_crop("BANANA", available_crops)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            InSeasonMonitoringExtractor.validate_crop("", available_crops)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            InSeasonMonitoringExtractor.validate_crop(None, available_crops)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid crop type"):
            InSeasonMonitoringExtractor.validate_crop(123, available_crops)

    def test_all_available_crops_accepted(self):
        for crop in available_crops:
            assert InSeasonMonitoringExtractor.validate_crop(crop, available_crops) == crop


# ===================================================================
# setup_inseason_monitoring_parameters()
# ===================================================================


class TestSetupInseasonMonitoringParameters:
    """Tests for parameter setup and validation."""

    def test_notebook_setup_succeeds(self, ism_extractor):
        """Mirrors notebook cell 13: data_source='LR' string."""
        ism_extractor.setup_inseason_monitoring_parameters(
            season_duration=120,
            season_start_day=1,
            season_start_month=4,
            year="2025",
            data_source="LR",
            partial_frequency=20,
            exclude_columns=[],
            column_mapping={"crop": "crop.id"},
        )
        params = ism_extractor.inseason_monitoring_params
        assert params is not None
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["year"] == "2025"
        assert params["data_source"] == "LR"
        assert params["partial_frequency"] == 20
        assert ism_extractor.column_mapping["crop"] == "crop.id"
        assert ism_extractor.exclude_columns == []

    def test_setup_sets_cache_key_columns(self, ism_extractor):
        """Setup should configure cache_key_columns to [mapped id]."""
        ism_extractor.setup_inseason_monitoring_parameters(data_source="LR")
        assert ism_extractor.cache_key_columns == ["id"]

    def test_setup_with_custom_id_column_mapping(self, ism_extractor):
        ism_extractor.setup_inseason_monitoring_parameters(data_source="LR", column_mapping={"id": "entity_id"})
        assert ism_extractor.column_mapping["id"] == "entity_id"

    # --- month / day validation ---

    def test_month_below_one_raises(self, ism_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            ism_extractor.setup_inseason_monitoring_parameters(season_start_month=0, data_source="LR")

    def test_month_above_twelve_raises(self, ism_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            ism_extractor.setup_inseason_monitoring_parameters(season_start_month=13, data_source="LR")

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_valid_months(self, ism_extractor, month):
        ism_extractor.setup_inseason_monitoring_parameters(season_start_month=month, data_source="LR")
        assert ism_extractor.inseason_monitoring_params["season_start_month"] == month

    def test_day_below_one_raises(self, ism_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            ism_extractor.setup_inseason_monitoring_parameters(season_start_day=0, data_source="LR")

    def test_day_above_thirty_one_raises(self, ism_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            ism_extractor.setup_inseason_monitoring_parameters(season_start_day=32, data_source="LR")

    # --- data_source validation ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, ism_extractor, ds):
        ism_extractor.setup_inseason_monitoring_parameters(data_source=ds)
        assert ism_extractor.inseason_monitoring_params["data_source"] == ds

    def test_invalid_data_source_string_raises(self, ism_extractor):
        with pytest.raises(ValueError, match="data_source"):
            ism_extractor.setup_inseason_monitoring_parameters(data_source="HR")

    def test_data_source_list_raises(self, ism_extractor):
        """Only a single 'LR' or 'MR' string is accepted — the API query carries one
        dataSource value. A list is rejected up front with an actionable message rather
        than falling through to `in {...}` and surfacing an unhashable-type TypeError.
        """
        with pytest.raises(ValueError, match="single source string"):
            ism_extractor.setup_inseason_monitoring_parameters(data_source=["LR", "MR"])

    def test_default_data_source_is_usable(self, ism_extractor):
        """The documented default must actually configure — it used to be ['LR', 'MR'],
        which the single-source validation rejected on the very first call.
        """
        ism_extractor.setup_inseason_monitoring_parameters()
        assert ism_extractor.inseason_monitoring_params["data_source"] == "LR"


# ===================================================================
# require_inseason_monitoring_params decorator
# ===================================================================


class TestRequireInseasonMonitoringParamsDecorator:
    """Methods guarded by @require_inseason_monitoring_params raise when params not set.

    Note: the decorator checks `hasattr(self, 'inseason_monitoring_params')`, so the
    attribute must be deleted (not just set to None) to trigger the guard.
    """

    def test_get_api_without_params_raises(self, ism_extractor):
        del ism_extractor.inseason_monitoring_params
        with pytest.raises(RuntimeError, match="No inseason monitoring parameters found"):
            ism_extractor.get_inseason_monitoring_api(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
            )

    def test_format_without_params_raises(self, ism_extractor):
        del ism_extractor.inseason_monitoring_params
        with pytest.raises(RuntimeError, match="No inseason monitoring parameters found"):
            ism_extractor.format_inseason_json({"id": "x", "data": []})

    def test_process_single_without_params_raises(self, ism_extractor):
        del ism_extractor.inseason_monitoring_params
        with pytest.raises(RuntimeError, match="No inseason monitoring parameters found"):
            ism_extractor.process_single_entity_inseason_monitoring({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, ism_extractor):
        """get_inseason_monitoring_api_safe is NOT guarded; returns failure instead of raising."""
        del ism_extractor.inseason_monitoring_params
        result = ism_extractor.get_inseason_monitoring_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
        )
        assert result["success"] is False
        assert "No inseason monitoring parameters" in result["error"]
