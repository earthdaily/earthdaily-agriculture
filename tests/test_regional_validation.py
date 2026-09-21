"""
Tests for RegionalExtractor validation logic:
    - setup_regional_parameters() input validation + defaults
    - require_regional_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_regional_parameters()
# ===================================================================


class TestSetupRegionalParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, regional_extractor):
        """All defaults should pass validation; end_date auto-set to current year-end."""
        regional_extractor.setup_regional_parameters()
        params = regional_extractor.regional_params
        assert params is not None
        assert params["index"] == "vegetation-vigor-index"
        assert params["start_date"] == "2025-01-01"
        # Default end_date is auto-set to f"{current_year}-12-31"
        assert params["end_date"] is not None
        assert params["end_date"].endswith("-12-31")
        assert params["fillyeargap"] is False
        assert params["idblock"] is None
        assert params["idpixeltype"] is None
        # VVI auto-sets indicatorTypeIds to [1]
        assert params["indicatorTypeIds"] == [1]

    def test_custom_parameters_from_notebook(self, regional_extractor):
        """Mirrors the notebook's setup_regional_parameters call (cell 8)."""
        regional_extractor.setup_regional_parameters(
            index="vegetation-vigor-index",
            start_date="2018-01-01",
            fillyeargap=False,
            idblock=281,
            idpixeltype=1,
            indicatorTypeIds=[1],
        )
        params = regional_extractor.regional_params
        assert params["index"] == "vegetation-vigor-index"
        assert params["start_date"] == "2018-01-01"
        assert params["idblock"] == 281
        assert params["idpixeltype"] == 1
        assert params["indicatorTypeIds"] == [1]

    def test_setup_sets_cache_key_columns(self, regional_extractor):
        """Setup should configure cache_key_columns to [id, 'date']."""
        regional_extractor.setup_regional_parameters()
        assert regional_extractor.cache_key_columns == ["id", "date"]

    def test_setup_with_custom_id_column_mapping(self, regional_extractor):
        """When id is remapped, cache_key_columns must reflect the mapped column name."""
        regional_extractor.setup_regional_parameters(column_mapping={"id": "entity_id"})
        assert regional_extractor.column_mapping["id"] == "entity_id"
        assert regional_extractor.cache_key_columns == ["entity_id", "date"]

    def test_column_mapping_for_amu_id_from_notebook(self, regional_extractor):
        """Mirrors notebook cell 29: maps amu_id -> region_code."""
        regional_extractor.setup_regional_parameters(
            column_mapping={"amu_id": "region_code"},
        )
        assert regional_extractor.column_mapping["amu_id"] == "region_code"

    # --- index validation ---

    def test_invalid_index_raises(self, regional_extractor):
        with pytest.raises(ValueError, match="Invalid index"):
            regional_extractor.setup_regional_parameters(index="bogus-index")

    @pytest.mark.parametrize(
        "idx",
        [
            "vegetation-vigor-index",
            "daily-precipitation",
            "soil-moisture",
            "min-temperature",
            "max-temperature",
            "average-temperature",
            "surface-temperature",
            "etp",
            "max-wind-speed",
            "p-etp",
            "relative-humidity",
            "snow-depth",
            "solar-radiation",
        ],
    )
    def test_valid_indexes(self, regional_extractor, idx):
        # Provide explicit indicatorTypeIds compatible with each index family
        if idx == "vegetation-vigor-index":
            regional_extractor.setup_regional_parameters(index=idx, indicatorTypeIds=[1])
        else:
            regional_extractor.setup_regional_parameters(index=idx, indicatorTypeIds=[2])
        assert regional_extractor.regional_params["index"] == idx

    @pytest.mark.parametrize(
        "idx",
        [
            "etp",
            "max-wind-speed",
            "p-etp",
            "relative-humidity",
            "snow-depth",
            "solar-radiation",
        ],
    )
    def test_weather_indexes_drop_vvi_indicator(self, regional_extractor, idx):
        """Weather-family indexes must not auto-include VVI (indicatorTypeId=1)."""
        regional_extractor.setup_regional_parameters(index=idx, indicatorTypeIds=[1, 2])
        assert 1 not in regional_extractor.regional_params["indicatorTypeIds"]
        assert 2 in regional_extractor.regional_params["indicatorTypeIds"]

    # --- date validation ---

    def test_invalid_start_date_format(self, regional_extractor):
        with pytest.raises(ValueError, match="start_date"):
            regional_extractor.setup_regional_parameters(start_date="01/01/2025")

    def test_invalid_end_date_format(self, regional_extractor):
        with pytest.raises(ValueError, match="end_date"):
            regional_extractor.setup_regional_parameters(end_date="2025/12/31")

    def test_end_date_none_defaults_to_current_year(self, regional_extractor):
        from datetime import datetime

        regional_extractor.setup_regional_parameters(end_date=None)
        expected = f"{datetime.now().year}-12-31"
        assert regional_extractor.regional_params["end_date"] == expected

    # --- idblock validation ---

    def test_invalid_idblock_raises(self, regional_extractor):
        with pytest.raises(ValueError, match="block"):
            regional_extractor.setup_regional_parameters(idblock=999)

    @pytest.mark.parametrize("block", [281, 141, 267, 226, 207, 115, 301])
    def test_valid_idblocks(self, regional_extractor, block):
        regional_extractor.setup_regional_parameters(idblock=block)
        assert regional_extractor.regional_params["idblock"] == block

    # --- idpixeltype validation ---

    def test_invalid_idpixeltype_raises(self, regional_extractor):
        with pytest.raises(ValueError, match="pixel type"):
            regional_extractor.setup_regional_parameters(idpixeltype=99999)

    @pytest.mark.parametrize("ptype", [1, 2, 3, 7, 8, 10, 11, 301, 400, 401, 402, 406, 407])
    def test_valid_idpixeltypes(self, regional_extractor, ptype):
        regional_extractor.setup_regional_parameters(idpixeltype=ptype)
        assert regional_extractor.regional_params["idpixeltype"] == ptype

    # --- indicatorTypeIds validation ---

    def test_indicator_type_ids_must_be_list(self, regional_extractor):
        with pytest.raises(ValueError, match="indicatorTypeIds must be a list"):
            regional_extractor.setup_regional_parameters(indicatorTypeIds=1)

    def test_invalid_indicator_type_id_raises(self, regional_extractor):
        with pytest.raises(ValueError, match="Invalid indicator ID"):
            regional_extractor.setup_regional_parameters(indicatorTypeIds=[99])

    @pytest.mark.parametrize("ind_id", [1, 2, 3, 4, 5])
    def test_valid_indicator_type_ids(self, regional_extractor, ind_id):
        # For VVI auto-injection logic, use a non-VVI index when ind_id != 1
        if ind_id == 1:
            regional_extractor.setup_regional_parameters(indicatorTypeIds=[ind_id])
        else:
            regional_extractor.setup_regional_parameters(index="daily-precipitation", indicatorTypeIds=[ind_id])
        assert ind_id in regional_extractor.regional_params["indicatorTypeIds"]

    # --- VVI auto-injection / cleanup ---

    def test_vvi_auto_sets_indicator_type_ids(self, regional_extractor):
        """vegetation-vigor-index with no indicatorTypeIds should auto-set to [1]."""
        regional_extractor.setup_regional_parameters(index="vegetation-vigor-index", indicatorTypeIds=None)
        assert regional_extractor.regional_params["indicatorTypeIds"] == [1]

    def test_vvi_auto_adds_1_when_missing(self, regional_extractor):
        """VVI with indicatorTypeIds=[2] should prepend 1 (warning logged)."""
        regional_extractor.setup_regional_parameters(index="vegetation-vigor-index", indicatorTypeIds=[2])
        assert 1 in regional_extractor.regional_params["indicatorTypeIds"]

    def test_non_vvi_removes_1(self, regional_extractor):
        """Non-VVI index with indicatorTypeIds containing 1 should drop 1."""
        regional_extractor.setup_regional_parameters(index="daily-precipitation", indicatorTypeIds=[1, 2])
        assert 1 not in regional_extractor.regional_params["indicatorTypeIds"]
        assert 2 in regional_extractor.regional_params["indicatorTypeIds"]

    # --- output formatting ---

    def test_output_mapping_applied(self, regional_extractor):
        regional_extractor.setup_regional_parameters(output_mapping={"value": "vvi"})
        assert regional_extractor.output_mapping == {"value": "vvi"}

    def test_exclude_columns_applied(self, regional_extractor):
        regional_extractor.setup_regional_parameters(exclude_columns=["dayId"])
        assert regional_extractor.exclude_columns == ["dayId"]


# ===================================================================
# require_regional_params decorator
# ===================================================================


class TestRequireRegionalParamsDecorator:
    """Methods guarded by @require_regional_params raise when params not set.

    The decorator checks `not hasattr(self, 'regional_params') or self.regional_params is None`,
    so setting the attribute to None (the default state from the fixture) triggers the guard.
    """

    def test_get_api_without_params_raises(self, regional_extractor):
        """get_regional_ts_by_id should fail if setup was never called."""
        assert regional_extractor.regional_params is None
        with pytest.raises(RuntimeError, match="No regional parameters found"):
            regional_extractor.get_regional_ts_by_id({"amu_id": 2432528})

    def test_process_single_without_params_raises(self, regional_extractor):
        """process_single_entity_regional is guarded too."""
        assert regional_extractor.regional_params is None
        with pytest.raises(RuntimeError, match="No regional parameters found"):
            regional_extractor.process_single_entity_regional({"amu_id": 2432528})

    def test_safe_wrapper_without_params_returns_failure(self, regional_extractor):
        """get_regional_ts_by_id_safe is NOT decorated itself, but it calls
        get_regional_ts_by_id which IS guarded. The safe wrapper catches the
        RuntimeError and returns success=False."""
        assert regional_extractor.regional_params is None
        result = regional_extractor.get_regional_ts_by_id_safe({"amu_id": 2432528})
        assert result["success"] is False
        assert "No regional parameters found" in result["error"]

    def test_format_does_not_require_params(self, regional_extractor):
        """format_regional_json is NOT guarded — callable without setup."""
        assert regional_extractor.regional_params is None
        observed_df, daily_df = regional_extractor.format_regional_json({})
        assert observed_df is not None
        assert daily_df is not None
