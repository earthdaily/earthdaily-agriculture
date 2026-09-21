"""
Tests for MRTSExtractor validation logic:
    - setup_mrts_parameters() input validation (vegetation_index, sensors, smoothing,
      aggregation, dates, clear_cover, historical_years, booleans, mode, kpi_filter,
      temporal_consistency_threshold)
    - require_mrts_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_mrts_parameters() — happy paths & individual fields
# ===================================================================


class TestSetupMrtsParameters:
    def test_default_parameters_succeed(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters()
        params = mrts_extractor.mrts_params
        assert params is not None
        assert params["start_date"] == "2025-05-01"
        assert params["end_date"] == "2025-10-15"
        assert params["sensors"] is None
        assert params["vegetation_index"] == "NDVI"
        assert params["aggregation"] == "average"
        assert params["smoothing_method"] == "Whittaker"
        assert params["apply_denoiser"] is True
        assert params["apply_end_of_curve"] is True
        assert params["clear_cover_min"] == 100
        assert params["mode"] == "full"
        assert params["historical_years"] == 10
        # default temporal consistency thresholds when compute_temporal_consistency=True
        assert params["temporal_consistency_threshold"] == {"Ndvi": 0.06, "Lai": 0.3, "S2Rep": 2.5}

    def test_custom_parameters_from_notebook(self, mrts_extractor):
        """Mirrors notebook cell 13 (S2REP / Whittaker / explicit dates)."""
        mrts_extractor.setup_mrts_parameters(
            start_date="2025-11-01",
            end_date="2026-02-28",
            sensors=None,
            vegetation_index="S2REP",
            aggregation="average",
            smoothing_method="Whittaker",
            apply_denoiser=True,
            apply_end_of_curve=True,
            column_mapping={"crop": "crop.id", "start_date": "sowingDate"},
        )
        params = mrts_extractor.mrts_params
        assert params["vegetation_index"] == "S2REP"
        assert params["start_date"] == "2025-11-01"
        assert params["end_date"] == "2026-02-28"
        assert mrts_extractor.column_mapping["crop"] == "crop.id"
        assert mrts_extractor.column_mapping["start_date"] == "sowingDate"

    def test_setup_sets_cache_key_columns(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters()
        assert mrts_extractor.cache_key_columns == ["id", "date"]

    # --- vegetation_index ---

    @pytest.mark.parametrize("vi", ["NDVI", "EVI", "CVI", "GNDVI", "NDWI", "LAI", "NDRE", "NDMI", "S2REP"])
    def test_valid_vegetation_indices(self, mrts_extractor, vi):
        mrts_extractor.setup_mrts_parameters(vegetation_index=vi)
        assert mrts_extractor.mrts_params["vegetation_index"] == vi

    def test_invalid_vegetation_index_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="vegetation_index"):
            mrts_extractor.setup_mrts_parameters(vegetation_index="BOGUS")

    # --- sensors ---

    def test_sensors_none_accepted(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(sensors=None)
        assert mrts_extractor.mrts_params["sensors"] is None

    def test_valid_sensor_list(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(sensors=["Sentinel_2", "Landsat_8"])
        assert mrts_extractor.mrts_params["sensors"] == ["Sentinel_2", "Landsat_8"]

    def test_invalid_sensor_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="Invalid sensors"):
            mrts_extractor.setup_mrts_parameters(sensors=["Sentinel_2", "BadSensor"])

    # --- smoothing_method ---

    @pytest.mark.parametrize("sm", ["Whittaker", "SavitzkyGolay", "None"])
    def test_valid_smoothing_methods(self, mrts_extractor, sm):
        mrts_extractor.setup_mrts_parameters(smoothing_method=sm)
        assert mrts_extractor.mrts_params["smoothing_method"] == sm

    def test_invalid_smoothing_method_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="smoothing"):
            mrts_extractor.setup_mrts_parameters(smoothing_method="bogus")

    # --- aggregation ---

    @pytest.mark.parametrize("agg", ["average", "median", "max", "min", "accumulation", "std"])
    def test_valid_aggregations(self, mrts_extractor, agg):
        mrts_extractor.setup_mrts_parameters(aggregation=agg)
        assert mrts_extractor.mrts_params["aggregation"] == agg

    def test_invalid_aggregation_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="aggregation"):
            mrts_extractor.setup_mrts_parameters(aggregation="bogus")

    # --- dates ---

    def test_invalid_start_date_format_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="date"):
            mrts_extractor.setup_mrts_parameters(start_date="01/05/2025")

    def test_invalid_end_date_format_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="date"):
            mrts_extractor.setup_mrts_parameters(end_date="2025/10/15")

    def test_start_after_end_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="start_date"):
            mrts_extractor.setup_mrts_parameters(start_date="2025-10-15", end_date="2025-05-01")

    def test_start_equal_to_end_raises(self, mrts_extractor):
        """start_date >= end_date raises (strict comparison in source)."""
        with pytest.raises(ValueError, match="start_date"):
            mrts_extractor.setup_mrts_parameters(start_date="2025-05-01", end_date="2025-05-01")

    # --- clear_cover_min ---

    def test_clear_cover_below_zero_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="clear_cover_min"):
            mrts_extractor.setup_mrts_parameters(clear_cover_min=-1)

    def test_clear_cover_above_hundred_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="clear_cover_min"):
            mrts_extractor.setup_mrts_parameters(clear_cover_min=101)

    @pytest.mark.parametrize("cc", [0, 50, 100])
    def test_valid_clear_cover(self, mrts_extractor, cc):
        mrts_extractor.setup_mrts_parameters(clear_cover_min=cc)
        assert mrts_extractor.mrts_params["clear_cover_min"] == cc

    # --- historical_years ---

    def test_historical_years_int_accepted(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(historical_years=5)
        assert mrts_extractor.mrts_params["historical_years"] == 5

    def test_historical_years_negative_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            mrts_extractor.setup_mrts_parameters(historical_years=-1)

    def test_historical_years_above_15_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            mrts_extractor.setup_mrts_parameters(historical_years=16)

    def test_historical_years_list_accepted(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(historical_years=[2024, 2023, 2022])
        assert mrts_extractor.mrts_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_string_normalized_to_list(self, mrts_extractor):
        """Pipeline-flattened string '2024,2023' should normalize back to a list of ints."""
        mrts_extractor.setup_mrts_parameters(historical_years="2024,2023,2022")
        assert mrts_extractor.mrts_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_empty_list_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            mrts_extractor.setup_mrts_parameters(historical_years=[])

    def test_historical_years_non_int_list_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            mrts_extractor.setup_mrts_parameters(historical_years=[2024, "bad", 2022])

    def test_historical_years_invalid_type_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            mrts_extractor.setup_mrts_parameters(historical_years=2.5)

    # --- partial_frequency ---

    def test_partial_frequency_negative_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            mrts_extractor.setup_mrts_parameters(partial_frequency=-1)

    def test_partial_frequency_non_int_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            mrts_extractor.setup_mrts_parameters(partial_frequency=1.5)

    # --- boolean params ---

    @pytest.mark.parametrize(
        "name",
        [
            "apply_denoiser",
            "apply_end_of_curve",
            "output_saturation",
            "extract_raw_datasets",
            "compute_temporal_consistency",
        ],
    )
    def test_boolean_params_must_be_bool(self, mrts_extractor, name):
        with pytest.raises(ValueError, match=name):
            mrts_extractor.setup_mrts_parameters(**{name: "yes"})

    # --- mode ---

    @pytest.mark.parametrize("mode", ["full", "raw"])
    def test_valid_modes(self, mrts_extractor, mode):
        mrts_extractor.setup_mrts_parameters(mode=mode)
        assert mrts_extractor.mrts_params["mode"] == mode

    def test_invalid_mode_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="mode"):
            mrts_extractor.setup_mrts_parameters(mode="bogus")

    # --- temporal_consistency_threshold ---

    def test_temporal_consistency_threshold_must_be_dict(self, mrts_extractor):
        with pytest.raises(ValueError, match="temporal_consistency_threshold"):
            mrts_extractor.setup_mrts_parameters(temporal_consistency_threshold=0.06)

    def test_default_threshold_used_when_none(self, mrts_extractor):
        """When compute_temporal_consistency=True and threshold=None, defaults are filled in."""
        mrts_extractor.setup_mrts_parameters(compute_temporal_consistency=True, temporal_consistency_threshold=None)
        assert mrts_extractor.mrts_params["temporal_consistency_threshold"] == {
            "Ndvi": 0.06,
            "Lai": 0.3,
            "S2Rep": 2.5,
        }


# ===================================================================
# kpi_filter validation
# ===================================================================


class TestSetupMrtsParametersKpiFilter:
    def test_kpi_filter_must_be_dict(self, mrts_extractor):
        with pytest.raises(ValueError, match="kpi_filter"):
            mrts_extractor.setup_mrts_parameters(kpi_filter="bogus")

    def test_kpi_filter_missing_aggregation_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="aggregation"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"kpi_name": "NDVI Acc"})

    def test_kpi_invalid_aggregation_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="aggregation"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "bogus_agg"})

    @pytest.mark.parametrize(
        "agg",
        ["accumulation", "average", "max", "min", "std"],
    )
    def test_valid_simple_aggregations(self, mrts_extractor, agg):
        """Simple aggregations don't require a threshold."""
        mrts_extractor.setup_mrts_parameters(kpi_filter={"kpi_name": "test", "aggregation": agg})
        assert mrts_extractor.mrts_params["kpi_filter"]["aggregation"] == agg

    # --- count_gt / count_lt ---

    def test_count_gt_requires_threshold(self, mrts_extractor):
        with pytest.raises(ValueError, match="threshold"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_gt"})

    def test_count_gt_with_numeric_threshold(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_gt", "threshold": 0.5})
        assert mrts_extractor.mrts_params["kpi_filter"]["threshold"] == 0.5

    def test_count_gt_non_numeric_threshold_raises(self, mrts_extractor):
        with pytest.raises(ValueError, match="threshold"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_gt", "threshold": "high"})

    # --- count_between ---

    def test_count_between_requires_tuple_threshold(self, mrts_extractor):
        with pytest.raises(ValueError, match="threshold"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_between", "threshold": 0.5})

    def test_count_between_min_must_be_below_max(self, mrts_extractor):
        with pytest.raises(ValueError, match="max"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_between", "threshold": (0.8, 0.2)})

    def test_count_between_with_valid_tuple(self, mrts_extractor):
        mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "count_between", "threshold": (0.2, 0.8)})
        assert mrts_extractor.mrts_params["kpi_filter"]["threshold"] == (0.2, 0.8)

    # --- top_accumulation ---

    def test_top_accumulation_requires_positive_int(self, mrts_extractor):
        with pytest.raises(ValueError, match="threshold"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "top_accumulation", "threshold": 0})

    def test_top_accumulation_with_valid_int(self, mrts_extractor):
        """Mirrors notebook cell 26: top_accumulation with 30 days."""
        mrts_extractor.setup_mrts_parameters(
            kpi_filter={
                "kpi_name": "NDVI Accumulation",
                "aggregation": "top_accumulation",
                "value_column": "smoothed_value",
                "threshold": 30,
            }
        )
        assert mrts_extractor.mrts_params["kpi_filter"]["threshold"] == 30

    def test_simple_aggregation_with_unwanted_threshold_raises(self, mrts_extractor):
        """Simple aggregations (e.g. average) shouldn't have a threshold."""
        with pytest.raises(ValueError, match="threshold"):
            mrts_extractor.setup_mrts_parameters(kpi_filter={"aggregation": "average", "threshold": 0.5})


# ===================================================================
# require_mrts_params decorator
# ===================================================================


class TestRequireMrtsParamsDecorator:
    """Methods guarded by @require_mrts_params raise when params not set."""

    def test_get_api_without_params_raises(self, mrts_extractor):
        assert mrts_extractor.mrts_params is None
        with pytest.raises(RuntimeError, match="No MRTS parameters found"):
            mrts_extractor.get_mrts_api({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})

    def test_format_without_params_raises(self, mrts_extractor):
        assert mrts_extractor.mrts_params is None
        with pytest.raises(RuntimeError, match="No MRTS parameters found"):
            mrts_extractor.format_mrts_json({"rawData": [], "smoothedData": []})

    def test_process_single_without_params_raises(self, mrts_extractor):
        assert mrts_extractor.mrts_params is None
        with pytest.raises(RuntimeError, match="No MRTS parameters found"):
            mrts_extractor.process_single_entity_mrts({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, mrts_extractor):
        """get_mrts_api_safe is NOT guarded; returns failure instead of raising."""
        assert mrts_extractor.mrts_params is None
        result = mrts_extractor.get_mrts_api_safe({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})
        assert result["success"] is False
        assert "No MRTS parameters" in result["error"]
