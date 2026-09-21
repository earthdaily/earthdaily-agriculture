"""
Tests for VegationTsExtractor validation logic:
    - setup_vegetation_ts_parameters() input validation
    - require_vegetation_ts_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_vegetation_ts_parameters()
# ===================================================================


class TestSetupVegetationTsParameters:
    def test_default_parameters_succeed(self, vts_extractor):
        """Defaults pass validation (start/end_date None, vegetation_index='NDVI', period mode)."""
        vts_extractor.setup_vegetation_ts_parameters()
        params = vts_extractor.vegetation_ts_params
        assert params is not None
        assert params["start_date"] is None
        assert params["end_date"] is None
        assert params["vegetation_index"] == "NDVI"
        assert params["is_extrapolated"] is True
        assert params["limit"] == 3000
        assert params["historical_years"] == 10
        assert params["extraction_mode"] == "period"
        assert params["target_dates"] is None
        assert params["kpi_filter"] is None

    def test_notebook_setup_period_mode(self, vts_extractor):
        """Mirrors notebook cell 14 (windows mode actually — empty target_dates)."""
        vts_extractor.setup_vegetation_ts_parameters(
            start_date="2021-01-01",
            end_date="2026-01-01",
            vegetation_index="NDVI",
            is_extrapolated=True,
            limit=3000,
            extraction_mode="windows",
            target_dates=[],
            historical_years=10,
            column_mapping={"crop": "crop.id", "start_date": "sowingDate"},
        )
        params = vts_extractor.vegetation_ts_params
        assert params["extraction_mode"] == "windows"
        assert vts_extractor.column_mapping["crop"] == "crop.id"
        assert vts_extractor.column_mapping["start_date"] == "sowingDate"

    def test_setup_sets_cache_key_columns(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters()
        assert vts_extractor.cache_key_columns == ["id", "date"]

    # --- vegetation_index ---

    @pytest.mark.parametrize("vi", ["NDVI", "EVI"])
    def test_valid_vegetation_indexes(self, vts_extractor, vi):
        vts_extractor.setup_vegetation_ts_parameters(vegetation_index=vi)
        assert vts_extractor.vegetation_ts_params["vegetation_index"] == vi

    def test_invalid_vegetation_index_raises(self, vts_extractor):
        """VTS only accepts NDVI and EVI (LAI not supported here)."""
        with pytest.raises(ValueError, match="vegetation_index"):
            vts_extractor.setup_vegetation_ts_parameters(vegetation_index="LAI")

    # --- limit ---

    def test_limit_zero_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="limit"):
            vts_extractor.setup_vegetation_ts_parameters(limit=0)

    def test_limit_negative_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="limit"):
            vts_extractor.setup_vegetation_ts_parameters(limit=-100)

    def test_limit_non_int_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="limit"):
            vts_extractor.setup_vegetation_ts_parameters(limit=3000.5)

    # --- is_extrapolated ---

    def test_is_extrapolated_non_bool_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="is_extrapolated"):
            vts_extractor.setup_vegetation_ts_parameters(is_extrapolated="yes")

    # --- dates ---

    def test_invalid_start_date_format_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="start_date"):
            vts_extractor.setup_vegetation_ts_parameters(start_date="01/01/2025")

    def test_invalid_end_date_format_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="end_date"):
            vts_extractor.setup_vegetation_ts_parameters(end_date="2025/01/01")

    def test_dates_optional_can_be_none(self, vts_extractor):
        """Both dates can be None (entity rows must then provide them)."""
        vts_extractor.setup_vegetation_ts_parameters(start_date=None, end_date=None)
        assert vts_extractor.vegetation_ts_params["start_date"] is None
        assert vts_extractor.vegetation_ts_params["end_date"] is None

    # --- extraction_mode ---

    @pytest.mark.parametrize("mode", ["period", "windows"])
    def test_valid_extraction_modes_no_target_dates(self, vts_extractor, mode):
        vts_extractor.setup_vegetation_ts_parameters(extraction_mode=mode)
        assert vts_extractor.vegetation_ts_params["extraction_mode"] == mode

    def test_invalid_extraction_mode_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="extraction_mode"):
            vts_extractor.setup_vegetation_ts_parameters(extraction_mode="bogus")

    def test_specific_dates_requires_target_dates(self, vts_extractor):
        """specific_dates mode without target_dates raises."""
        with pytest.raises(ValueError, match="target_dates"):
            vts_extractor.setup_vegetation_ts_parameters(extraction_mode="specific_dates")

    def test_specific_dates_with_empty_list_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="target_dates"):
            vts_extractor.setup_vegetation_ts_parameters(extraction_mode="specific_dates", target_dates=[])

    def test_specific_dates_with_valid_list(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters(
            extraction_mode="specific_dates",
            target_dates=["2025-06-15", "2025-07-15", "2025-08-15"],
        )
        # Stored sorted ascending
        assert vts_extractor.vegetation_ts_params["target_dates"] == [
            "2025-06-15",
            "2025-07-15",
            "2025-08-15",
        ]

    def test_specific_dates_filters_invalid_entries(self, vts_extractor):
        """Invalid date strings are skipped; valid ones are kept."""
        vts_extractor.setup_vegetation_ts_parameters(
            extraction_mode="specific_dates",
            target_dates=["2025-06-15", "not-a-date", "2025-07-15"],
        )
        assert vts_extractor.vegetation_ts_params["target_dates"] == [
            "2025-06-15",
            "2025-07-15",
        ]

    def test_specific_dates_all_invalid_raises(self, vts_extractor):
        """If every target date is malformed, raise."""
        with pytest.raises(ValueError, match="No valid dates"):
            vts_extractor.setup_vegetation_ts_parameters(
                extraction_mode="specific_dates",
                target_dates=["bogus", "01/01/2025"],
            )

    # --- historical_years ---

    def test_historical_years_int_accepted(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters(historical_years=5)
        assert vts_extractor.vegetation_ts_params["historical_years"] == 5

    def test_historical_years_negative_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            vts_extractor.setup_vegetation_ts_parameters(historical_years=-1)

    def test_historical_years_above_15_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            vts_extractor.setup_vegetation_ts_parameters(historical_years=16)

    def test_historical_years_list_accepted(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters(historical_years=[2024, 2023, 2022])
        assert vts_extractor.vegetation_ts_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_string_normalized_to_list(self, vts_extractor):
        """Comma-separated string normalises to list of ints (notebook cell 37)."""
        vts_extractor.setup_vegetation_ts_parameters(historical_years="2024,2023,2022")
        assert vts_extractor.vegetation_ts_params["historical_years"] == [2024, 2023, 2022]

    def test_historical_years_empty_list_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            vts_extractor.setup_vegetation_ts_parameters(historical_years=[])

    def test_historical_years_non_int_list_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            vts_extractor.setup_vegetation_ts_parameters(historical_years=[2024, "bad"])

    def test_historical_years_invalid_type_raises(self, vts_extractor):
        with pytest.raises(ValueError, match="historical_years"):
            vts_extractor.setup_vegetation_ts_parameters(historical_years=2.5)


# ===================================================================
# require_vegetation_ts_params decorator
# ===================================================================


class TestRequireVegetationTsParamsDecorator:
    """Methods guarded by @require_vegetation_ts_params raise when params not set.

    Note: the decorator checks `hasattr(self, 'vegetation_ts_params')`, so we delete
    the attribute to trigger the guard.
    """

    def test_get_api_without_params_raises(self, vts_extractor):
        del vts_extractor.vegetation_ts_params
        with pytest.raises(RuntimeError, match="No vegetation time series parameters found"):
            vts_extractor.get_vegetation_api({"id": "x", "start_date": "2025-01-01", "end_date": "2025-12-31"})

    def test_format_without_params_raises(self, vts_extractor):
        del vts_extractor.vegetation_ts_params
        with pytest.raises(RuntimeError, match="No vegetation time series parameters found"):
            vts_extractor.format_vegetation_ts_json([])

    def test_process_single_without_params_raises(self, vts_extractor):
        del vts_extractor.vegetation_ts_params
        with pytest.raises(RuntimeError, match="No vegetation time series parameters found"):
            vts_extractor.process_single_entity_vegetation_ts({"id": "x"})

    def test_process_specific_dates_without_params_raises(self, vts_extractor):
        del vts_extractor.vegetation_ts_params
        with pytest.raises(RuntimeError, match="No vegetation time series parameters found"):
            vts_extractor.process_single_entity_specific_dates({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, vts_extractor):
        """get_vegetation_api_safe is NOT guarded; returns failure instead of raising."""
        del vts_extractor.vegetation_ts_params
        result = vts_extractor.get_vegetation_api_safe(
            {"id": "x", "start_date": "2025-01-01", "end_date": "2025-12-31"}
        )
        assert result["success"] is False
        assert "No vegetation time series parameters" in result["error"]
