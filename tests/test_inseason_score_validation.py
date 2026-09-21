"""
Tests for InseasonScoreExtractor validation logic:
    - setup_inseason_score_parameters() input validation
    - require_inseason_score_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_inseason_score_parameters()
# ===================================================================


class TestSetupInseasonScoreParameters:
    def test_default_parameters_succeed(self, inseason_score_extractor):
        inseason_score_extractor.setup_inseason_score_parameters()
        params = inseason_score_extractor.inseason_score_params
        assert params is not None
        assert params["season_duration"] == 120
        assert params["season_start_day"] == 1
        assert params["season_start_month"] == 4
        assert params["nb_historical_year"] == 1
        assert params["threshold_start"] == 0.7
        assert params["historical_seasons"] is None
        assert params["data_source"] == "LR"
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50
        assert params["detail_level"] == "full"

    def test_custom_parameters_from_notebook(self, inseason_score_extractor):
        """Mirrors notebook cell 13 (October-start crop cycle)."""
        inseason_score_extractor.setup_inseason_score_parameters(
            season_duration=120,
            season_start_day=1,
            season_start_month=10,
            nb_historical_year=3,
            data_source="LR",
            publish_af=False,
            partial_frequency=50,
            column_mapping={"crop": "crop.id", "sowing_date": "sowingDate"},
        )
        params = inseason_score_extractor.inseason_score_params
        assert params["season_start_month"] == 10
        assert params["nb_historical_year"] == 3
        assert inseason_score_extractor.column_mapping["crop"] == "crop.id"
        assert inseason_score_extractor.column_mapping["sowing_date"] == "sowingDate"

    def test_setup_sets_cache_key_columns(self, inseason_score_extractor):
        inseason_score_extractor.setup_inseason_score_parameters()
        assert inseason_score_extractor.cache_key_columns == ["id"]

    # --- month / day validation ---

    def test_month_below_one_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            inseason_score_extractor.setup_inseason_score_parameters(season_start_month=0)

    def test_month_above_twelve_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            inseason_score_extractor.setup_inseason_score_parameters(season_start_month=13)

    def test_day_below_one_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            inseason_score_extractor.setup_inseason_score_parameters(season_start_day=0)

    def test_day_above_thirty_one_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            inseason_score_extractor.setup_inseason_score_parameters(season_start_day=32)

    # --- nb_historical_year validation ---

    def test_nb_historical_year_below_one_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="nb_historical_year"):
            inseason_score_extractor.setup_inseason_score_parameters(nb_historical_year=0)

    def test_nb_historical_year_non_int_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="nb_historical_year"):
            inseason_score_extractor.setup_inseason_score_parameters(nb_historical_year=1.5)

    @pytest.mark.parametrize("n", [1, 3, 10])
    def test_valid_nb_historical_year(self, inseason_score_extractor, n):
        inseason_score_extractor.setup_inseason_score_parameters(nb_historical_year=n)
        assert inseason_score_extractor.inseason_score_params["nb_historical_year"] == n

    # --- threshold_start ---

    def test_threshold_start_negative_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            inseason_score_extractor.setup_inseason_score_parameters(threshold_start=-0.1)

    def test_threshold_start_above_one_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            inseason_score_extractor.setup_inseason_score_parameters(threshold_start=1.1)

    def test_threshold_start_non_numeric_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="threshold_start"):
            inseason_score_extractor.setup_inseason_score_parameters(threshold_start="high")

    # --- historical_seasons ---

    def test_historical_seasons_list_accepted(self, inseason_score_extractor):
        inseason_score_extractor.setup_inseason_score_parameters(historical_seasons=[2024, 2023, 2022])
        assert inseason_score_extractor.inseason_score_params["historical_seasons"] == [
            2024,
            2023,
            2022,
        ]

    def test_historical_seasons_int_rejected(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="historical_seasons"):
            inseason_score_extractor.setup_inseason_score_parameters(historical_seasons=5)

    # --- data_source ---

    @pytest.mark.parametrize("ds", ["LR", "MR"])
    def test_valid_data_sources(self, inseason_score_extractor, ds):
        inseason_score_extractor.setup_inseason_score_parameters(data_source=ds)
        assert inseason_score_extractor.inseason_score_params["data_source"] == ds

    def test_invalid_data_source_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="data_source"):
            inseason_score_extractor.setup_inseason_score_parameters(data_source="HR")

    # --- detail_level ---

    @pytest.mark.parametrize("dl", ["summary", "full"])
    def test_valid_detail_levels(self, inseason_score_extractor, dl):
        inseason_score_extractor.setup_inseason_score_parameters(detail_level=dl)
        assert inseason_score_extractor.inseason_score_params["detail_level"] == dl

    def test_invalid_detail_level_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="detail_level"):
            inseason_score_extractor.setup_inseason_score_parameters(detail_level="bogus")

    # --- publish_af ---

    def test_publish_af_non_bool_raises(self, inseason_score_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            inseason_score_extractor.setup_inseason_score_parameters(publish_af="yes")


# ===================================================================
# require_inseason_score_params decorator
# ===================================================================


class TestRequireInseasonScoreParamsDecorator:
    """Note: the decorator checks `hasattr`, so we delete the attribute to trigger the guard."""

    def test_get_api_without_params_raises(self, inseason_score_extractor):
        del inseason_score_extractor.inseason_score_params
        with pytest.raises(RuntimeError, match="No inseason_score parameters found"):
            inseason_score_extractor.get_inseason_score_api(
                {
                    "id": "x",
                    "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                    "crop": "CORN",
                    "sowing_date": "2025-10-25",
                }
            )

    def test_format_without_params_raises(self, inseason_score_extractor):
        del inseason_score_extractor.inseason_score_params
        with pytest.raises(RuntimeError, match="No inseason_score parameters found"):
            inseason_score_extractor.format_inseason_score_json({"id": "x", "data": {}})

    def test_process_single_without_params_raises(self, inseason_score_extractor):
        del inseason_score_extractor.inseason_score_params
        with pytest.raises(RuntimeError, match="No inseason_score parameters found"):
            inseason_score_extractor.process_single_entity_inseason_score({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, inseason_score_extractor):
        """get_inseason_score_api_safe is NOT guarded; returns failure instead of raising."""
        del inseason_score_extractor.inseason_score_params
        result = inseason_score_extractor.get_inseason_score_api_safe(
            {
                "id": "x",
                "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                "crop": "CORN",
                "sowing_date": "2025-10-25",
            }
        )
        assert result["success"] is False
        assert "No inseason_score parameters" in result["error"]
