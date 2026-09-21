"""
Tests for WeatherExtractor validation logic:
    - setup_weather_parameters() input validation
    - require_weather_params decorator
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_weather_parameters()
# ===================================================================


class TestSetupWeatherParameters:
    def test_default_parameters_succeed(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters()

        params = weather_extractor.weather_params
        assert params is not None
        assert params["weather_type"] == "HISTORICAL_DAILY"
        # Default is a real, selectable parameter so the extractor works
        # out-of-the-box (the old "none" default produced a 400 from the API).
        assert params["weather_parameters"] == "precipitation.cumulative"
        assert params["historical_years"] == 0
        assert params["partial_frequency"] == 50
        assert params["kpi_filter"] is None

    def test_notebook_setup_succeeds(self, weather_extractor):
        """Mirrors notebook cell 15."""
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(
                weather_type="HISTORICAL_DAILY",
                weather_parameters="Temperature.standardmax",
                partial_frequency=50,
                exclude_columns=[],
                kpi_filter=None,
                column_mapping={"crop": "crop.id", "start_date": "sowingDate"},
            )

        params = weather_extractor.weather_params
        assert params["weather_parameters"] == "Temperature.standardmax"
        assert weather_extractor.column_mapping["crop"] == "crop.id"
        assert weather_extractor.column_mapping["start_date"] == "sowingDate"
        assert weather_extractor.exclude_columns == []

    def test_setup_sets_cache_key_columns(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters()
        assert weather_extractor.cache_key_columns == ["id", "date"]

    # --- weather_type ---

    @pytest.mark.parametrize("wt", ["HISTORICAL_DAILY", "FORECAST_DAILY", "FORECAST_HOURLY"])
    def test_valid_weather_types(self, weather_extractor, wt):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(weather_type=wt)
        assert weather_extractor.weather_params["weather_type"] == wt

    def test_invalid_weather_type_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="weather"):
                weather_extractor.setup_weather_parameters(weather_type="BOGUS")

    # --- weather_parameters ---

    def test_weather_parameters_none_keyword(self, weather_extractor):
        """'none' keyword returns all parameters (no validation)."""
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(weather_parameters="none")
        assert weather_extractor.weather_params["weather_parameters"] == "none"

    def test_single_string_parameter_accepted(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(weather_parameters="precipitation.cumulative")
        assert weather_extractor.weather_params["weather_parameters"] == "precipitation.cumulative"

    def test_list_of_parameters_accepted(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(
                weather_parameters=["Temperature.standardmax", "precipitation.cumulative", "wind"]
            )
        assert weather_extractor.weather_params["weather_parameters"] == [
            "Temperature.standardmax",
            "precipitation.cumulative",
            "wind",
        ]

    def test_invalid_parameter_in_list_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="weather parameter"):
                weather_extractor.setup_weather_parameters(
                    weather_parameters=["Temperature.standardmax", "bogus_param"]
                )

    def test_invalid_parameter_string_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="weather parameter"):
                weather_extractor.setup_weather_parameters(weather_parameters="bogus_param")

    # --- historical_years ---

    def test_historical_years_int_accepted(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(historical_years=5)
        assert weather_extractor.weather_params["historical_years"] == 5

    def test_historical_years_negative_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="historical_years"):
                weather_extractor.setup_weather_parameters(historical_years=-1)

    def test_historical_years_above_15_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="historical_years"):
                weather_extractor.setup_weather_parameters(historical_years=16)

    def test_historical_years_list_accepted(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(historical_years=[2024, 2023, 2022])
        assert weather_extractor.weather_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_string_normalized_to_list(self, weather_extractor):
        """Comma-separated string normalises to list of ints."""
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(historical_years="2024,2023,2022")
        assert weather_extractor.weather_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_empty_list_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="historical_years"):
                weather_extractor.setup_weather_parameters(historical_years=[])

    def test_historical_years_non_int_list_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="historical_years"):
                weather_extractor.setup_weather_parameters(historical_years=[2024, "bad"])

    def test_historical_years_invalid_type_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="historical_years"):
                weather_extractor.setup_weather_parameters(historical_years=2.5)


# ===================================================================
# kpi_filter validation
# ===================================================================


class TestSetupWeatherParametersKpiFilter:
    def test_kpi_missing_aggregation_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="aggregation"):
                weather_extractor.setup_weather_parameters(kpi_filter={"kpi_name": "noop"})

    def test_kpi_invalid_aggregation_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="aggregation"):
                weather_extractor.setup_weather_parameters(kpi_filter={"aggregation": "bogus_agg"})

    @pytest.mark.parametrize("agg", ["accumulation", "average", "max", "min", "std"])
    def test_simple_aggregations_accepted(self, weather_extractor, agg):
        kpi_filter = {"kpi_name": "test", "aggregation": agg}
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(kpi_filter=kpi_filter)
        assert weather_extractor.weather_params["kpi_filter"]["aggregation"] == agg

    # --- count_gt / count_lt ---

    def test_count_gt_requires_threshold(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="threshold"):
                weather_extractor.setup_weather_parameters(kpi_filter={"aggregation": "count_gt"})

    def test_count_gt_with_numeric_threshold(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(kpi_filter={"aggregation": "count_gt", "threshold": 30.0})
        assert weather_extractor.weather_params["kpi_filter"]["threshold"] == 30.0

    def test_count_lt_non_numeric_threshold_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="threshold"):
                weather_extractor.setup_weather_parameters(kpi_filter={"aggregation": "count_lt", "threshold": "high"})

    # --- count_between ---

    def test_count_between_requires_tuple(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="threshold"):
                weather_extractor.setup_weather_parameters(
                    kpi_filter={"aggregation": "count_between", "threshold": 0.5}
                )

    def test_count_between_min_must_be_below_max(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="max"):
                weather_extractor.setup_weather_parameters(
                    kpi_filter={"aggregation": "count_between", "threshold": (35, 5)}
                )

    def test_count_between_with_valid_tuple(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            weather_extractor.setup_weather_parameters(
                kpi_filter={"aggregation": "count_between", "threshold": (5, 35)}
            )
        assert weather_extractor.weather_params["kpi_filter"]["threshold"] == (5, 35)

    def test_simple_aggregation_with_unwanted_threshold_raises(self, weather_extractor):
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="threshold"):
                weather_extractor.setup_weather_parameters(kpi_filter={"aggregation": "average", "threshold": 30})


# ===================================================================
# require_weather_params decorator
# ===================================================================


class TestRequireWeatherParamsDecorator:
    """Methods guarded by @require_weather_params raise when params not set.

    Note: the decorator uses `hasattr(self, 'weather_params')`, so the attribute
    must be deleted (not just set to None) to trigger the guard.
    """

    def test_get_weather_data_without_params_raises(self, weather_extractor):
        del weather_extractor.weather_params
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No weather parameters found"):
                weather_extractor.get_weather_data(
                    {
                        "id": "x",
                        "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                        "start_date": "2025-06-01",
                        "end_date": "2025-10-01",
                    }
                )

    def test_get_weather_data_safe_without_params_raises(self, weather_extractor):
        """get_weather_data_safe IS guarded by @require_weather_params (unlike most safe wrappers)."""
        del weather_extractor.weather_params
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No weather parameters found"):
                weather_extractor.get_weather_data_safe(
                    {
                        "id": "x",
                        "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                        "start_date": "2025-06-01",
                        "end_date": "2025-10-01",
                    }
                )

    def test_process_single_without_params_raises(self, weather_extractor):
        del weather_extractor.weather_params
        with patch.object(weather_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No weather parameters found"):
                weather_extractor.process_single_entity_weather({"id": "x"})

    def test_format_weather_json_does_not_require_params(self, weather_extractor):
        """format_weather_json is NOT guarded — callable without setup."""
        del weather_extractor.weather_params
        df = weather_extractor.format_weather_json([])
        assert df is not None
