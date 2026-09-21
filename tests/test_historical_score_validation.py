"""
Tests for HistoricalScoreExtractor validation logic:
    - validate_crop() module-level helper
    - validate_historical_seasons() module-level helper
    - setup_historical_score_parameters() input validation
    - require_historical_score_params decorator
"""

import pytest

from earthdaily.agriculture.processors.processor_score_functions import (
    HistoricalScoreExtractor,
    available_crops,
    validate_crop,
    validate_historical_seasons,
)

pytestmark = pytest.mark.public

# ===================================================================
# validate_crop() — module-level helper
# ===================================================================


class TestValidateCrop:
    """Tests for the module-level validate_crop helper."""

    @pytest.mark.parametrize("crop", ["CORN", "SOYBEANS", "COTTON", "OTHERS", "SUGARCANE"])
    def test_valid_crop_uppercase(self, crop):
        assert validate_crop(crop, available_crops) == crop

    def test_lowercase_normalized(self):
        assert validate_crop("corn", available_crops) == "CORN"

    def test_mixed_case_normalized(self):
        assert validate_crop("SoyBeans", available_crops) == "SOYBEANS"

    def test_whitespace_stripped(self):
        assert validate_crop("  COTTON  ", available_crops) == "COTTON"

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Invalid crop"):
            validate_crop("BANANA", available_crops)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            validate_crop("", available_crops)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing crop"):
            validate_crop(None, available_crops)

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid crop type"):
            validate_crop(123, available_crops)


# ===================================================================
# validate_historical_seasons() — module-level helper
# ===================================================================


class TestValidateHistoricalSeasons:
    """Tests for validate_historical_seasons (notebook cell 25)."""

    def test_none_returns_none(self):
        assert validate_historical_seasons(None) is None

    def test_list_of_ints_passes_through(self):
        assert validate_historical_seasons([2024, 2023, 2022]) == [2024, 2023, 2022]

    def test_comma_separated_string_normalized_to_list(self):
        """Pipeline-flattened string '2024,2023,2022' should normalise back to a list of ints."""
        assert validate_historical_seasons("2024,2023,2022") == [2024, 2023, 2022]

    def test_int_lookback_rejected(self):
        """An int lookback (e.g. `5`) is valid for inseason but not for historical_seasons."""
        with pytest.raises(ValueError, match="historical_seasons"):
            validate_historical_seasons(5)

    def test_all_keyword_rejected(self):
        with pytest.raises(ValueError, match="historical_seasons"):
            validate_historical_seasons("ALL")


# ===================================================================
# setup_historical_score_parameters()
# ===================================================================


class TestSetupHistoricalScoreParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, historical_score_extractor):
        historical_score_extractor.setup_historical_score_parameters()
        params = historical_score_extractor.historical_score_params
        assert params is not None
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["threshold_start"] == 0.7
        assert params["year"] == 2025
        assert params["historical_seasons"] is None
        assert params["data_source"] == "LR"
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50
        assert params["detail_level"] == "full"

    def test_custom_parameters_from_notebook(self, historical_score_extractor):
        """Mirrors notebook cell 13."""
        historical_score_extractor.setup_historical_score_parameters(
            season_duration=120,
            season_start_day=1,
            season_start_month=4,
            year="2025",
            data_source="LR",
            publish_af=False,
            partial_frequency=50,
            column_mapping={"crop": "crop.id"},
        )
        params = historical_score_extractor.historical_score_params
        assert params["year"] == "2025"
        assert historical_score_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, historical_score_extractor):
        historical_score_extractor.setup_historical_score_parameters()
        assert historical_score_extractor.cache_key_columns == ["id"]

    # --- month / day validation ---

    def test_month_below_one_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            historical_score_extractor.setup_historical_score_parameters(season_start_month=0)

    def test_month_above_twelve_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            historical_score_extractor.setup_historical_score_parameters(season_start_month=13)

    def test_day_below_one_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            historical_score_extractor.setup_historical_score_parameters(season_start_day=0)

    def test_day_above_thirty_one_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            historical_score_extractor.setup_historical_score_parameters(season_start_day=32)

    # --- threshold_start ---

    def test_threshold_start_negative_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            historical_score_extractor.setup_historical_score_parameters(threshold_start=-0.1)

    def test_threshold_start_above_one_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            historical_score_extractor.setup_historical_score_parameters(threshold_start=1.1)

    def test_threshold_start_non_numeric_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            historical_score_extractor.setup_historical_score_parameters(threshold_start="high")

    @pytest.mark.parametrize("ts", [0.0, 0.5, 1.0])
    def test_valid_threshold_start(self, historical_score_extractor, ts):
        historical_score_extractor.setup_historical_score_parameters(threshold_start=ts)
        assert historical_score_extractor.historical_score_params["threshold_start"] == ts

    # --- historical_seasons ---

    def test_historical_seasons_list_accepted(self, historical_score_extractor):
        historical_score_extractor.setup_historical_score_parameters(historical_seasons=[2024, 2023, 2022])
        assert historical_score_extractor.historical_score_params["historical_seasons"] == [2024, 2023, 2022]

    def test_historical_seasons_int_rejected(self, historical_score_extractor):
        """historical_seasons must be a list, not an int."""
        with pytest.raises(ValueError, match="historical_seasons"):
            historical_score_extractor.setup_historical_score_parameters(historical_seasons=5)

    # --- data_source ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, historical_score_extractor, ds):
        historical_score_extractor.setup_historical_score_parameters(data_source=ds)
        assert historical_score_extractor.historical_score_params["data_source"] == ds

    def test_invalid_data_source_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="data_source"):
            historical_score_extractor.setup_historical_score_parameters(data_source="HR")

    # --- detail_level ---

    @pytest.mark.parametrize("dl", ["summary", "full"])
    def test_valid_detail_levels(self, historical_score_extractor, dl):
        historical_score_extractor.setup_historical_score_parameters(detail_level=dl)
        assert historical_score_extractor.historical_score_params["detail_level"] == dl

    def test_invalid_detail_level_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="detail_level"):
            historical_score_extractor.setup_historical_score_parameters(detail_level="bogus")

    # --- publish_af ---

    def test_publish_af_non_bool_raises(self, historical_score_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            historical_score_extractor.setup_historical_score_parameters(publish_af="yes")


# ===================================================================
# require_historical_score_params decorator
# ===================================================================


class TestRequireHistoricalScoreParamsDecorator:
    """Methods guarded by @require_historical_score_params raise when params not set.

    Note: the decorator checks `hasattr(self, 'historical_score_params')` rather than `is None`,
    so we need to delete the attribute to trigger the guard.
    """

    def test_get_api_without_params_raises(self, historical_score_extractor):
        del historical_score_extractor.historical_score_params
        with pytest.raises(RuntimeError, match="No historical_score parameters found"):
            historical_score_extractor.get_historical_score_api(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
            )

    def test_format_without_params_raises(self, historical_score_extractor):
        del historical_score_extractor.historical_score_params
        with pytest.raises(RuntimeError, match="No historical_score parameters found"):
            historical_score_extractor.format_historical_score_json({"id": "x", "data": {}})

    def test_process_single_without_params_raises(self, historical_score_extractor):
        del historical_score_extractor.historical_score_params
        with pytest.raises(RuntimeError, match="No historical_score parameters found"):
            historical_score_extractor.process_single_entity_historical_score({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, historical_score_extractor):
        """get_historical_score_api_safe is NOT guarded; returns failure instead of raising."""
        del historical_score_extractor.historical_score_params
        result = historical_score_extractor.get_historical_score_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        )
        assert result["success"] is False
        assert "No historical_score parameters" in result["error"]
