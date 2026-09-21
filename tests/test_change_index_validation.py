"""
Tests for ChangeIndexExtractor validation logic:
    - setup_change_index_parameters() input validation
    - require_change_index_params decorator
    - _get_reference_date() helper
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_change_index_parameters()
# ===================================================================


class TestSetupChangeIndexParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, change_index_extractor):
        change_index_extractor.setup_change_index_parameters()
        params = change_index_extractor.change_index_params

        assert params is not None
        assert params["map_type"] == "NDVI"
        assert params["collections"] == ["Sentinel-2"]
        assert params["max_period_reference"] == 7
        assert params["max_period_previous"] == 15
        assert params["min_period_previous"] == 5
        assert params["same_sensor"] is False
        assert params["parameter_profile"] == "change_index_v1"

    def test_custom_parameters(self, change_index_extractor):
        change_index_extractor.setup_change_index_parameters(
            map_type="EVI",
            collections=["Sentinel-2", "Landsat"],
            max_period_reference=14,
            max_period_previous=30,
            min_period_previous=10,
            same_sensor=True,
            parameter_profile="change_index_v2",
            partial_frequency=25,
        )
        params = change_index_extractor.change_index_params
        assert params["map_type"] == "EVI"
        assert params["collections"] == ["Sentinel-2", "Landsat"]
        assert params["max_period_reference"] == 14
        assert params["max_period_previous"] == 30
        assert params["min_period_previous"] == 10
        assert params["same_sensor"] is True
        assert params["parameter_profile"] == "change_index_v2"
        assert params["partial_frequency"] == 25

    def test_setup_sets_cache_key_columns(self, change_index_extractor):
        """cache_key_columns should be [<id_col>, 'reference_date']."""
        change_index_extractor.setup_change_index_parameters()
        assert change_index_extractor.cache_key_columns == ["id", "reference_date"]

    # --- map_type validation ---
    def test_invalid_map_type(self, change_index_extractor):
        with pytest.raises(ValueError, match="map_type"):
            change_index_extractor.setup_change_index_parameters(map_type="LAI")

    @pytest.mark.parametrize("valid_type", ["NDVI", "EVI", "CVI", "GNDVI", "NDWI"])
    def test_all_valid_map_types_accepted(self, change_index_extractor, valid_type):
        change_index_extractor.setup_change_index_parameters(map_type=valid_type)
        assert change_index_extractor.change_index_params["map_type"] == valid_type

    # --- collections validation ---
    def test_invalid_single_collection(self, change_index_extractor):
        with pytest.raises(ValueError, match="collections"):
            change_index_extractor.setup_change_index_parameters(collections=["Spot"])

    def test_invalid_collection_in_list(self, change_index_extractor):
        with pytest.raises(ValueError, match="collections"):
            change_index_extractor.setup_change_index_parameters(collections=["Sentinel-2", "Modis"])

    def test_empty_collections_list(self, change_index_extractor):
        with pytest.raises(ValueError, match="collections"):
            change_index_extractor.setup_change_index_parameters(collections=[])

    def test_non_list_collections(self, change_index_extractor):
        with pytest.raises(ValueError, match="collections"):
            change_index_extractor.setup_change_index_parameters(collections="Sentinel-2")

    # --- period validation ---
    def test_negative_max_period_reference(self, change_index_extractor):
        with pytest.raises(ValueError, match="max_period_reference"):
            change_index_extractor.setup_change_index_parameters(max_period_reference=-1)

    def test_negative_max_period_previous(self, change_index_extractor):
        with pytest.raises(ValueError, match="max_period_previous"):
            change_index_extractor.setup_change_index_parameters(max_period_previous=-1)

    def test_negative_min_period_previous(self, change_index_extractor):
        with pytest.raises(ValueError, match="min_period_previous"):
            change_index_extractor.setup_change_index_parameters(min_period_previous=-1)

    def test_non_int_period(self, change_index_extractor):
        with pytest.raises(ValueError, match="max_period_reference"):
            change_index_extractor.setup_change_index_parameters(max_period_reference=7.5)

    def test_min_not_strictly_less_than_max(self, change_index_extractor):
        """min_period_previous must be strictly less than max_period_previous."""
        with pytest.raises(ValueError, match="min_period_previous"):
            change_index_extractor.setup_change_index_parameters(min_period_previous=15, max_period_previous=15)

    def test_min_greater_than_max(self, change_index_extractor):
        with pytest.raises(ValueError, match="min_period_previous"):
            change_index_extractor.setup_change_index_parameters(min_period_previous=20, max_period_previous=15)

    # --- same_sensor validation ---
    def test_invalid_same_sensor_string(self, change_index_extractor):
        with pytest.raises(ValueError, match="same_sensor"):
            change_index_extractor.setup_change_index_parameters(same_sensor="yes")

    def test_invalid_same_sensor_int(self, change_index_extractor):
        with pytest.raises(ValueError, match="same_sensor"):
            change_index_extractor.setup_change_index_parameters(same_sensor=1)

    # --- publish_af validation ---
    def test_publish_af_default_false(self, change_index_extractor):
        change_index_extractor.setup_change_index_parameters()
        assert change_index_extractor.change_index_params["publish_af"] is False

    def test_publish_af_true_accepted(self, change_index_extractor):
        change_index_extractor.setup_change_index_parameters(publish_af=True)
        assert change_index_extractor.change_index_params["publish_af"] is True

    def test_invalid_publish_af_string(self, change_index_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            change_index_extractor.setup_change_index_parameters(publish_af="yes")

    def test_invalid_publish_af_int(self, change_index_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            change_index_extractor.setup_change_index_parameters(publish_af=1)

    # --- column_mapping integration ---
    def test_column_mapping_applied(self, change_index_extractor):
        """Custom column mapping is propagated when setting up."""
        change_index_extractor.setup_change_index_parameters(
            column_mapping={"id": "entity_id", "reference_date": "image_date"},
        )
        assert change_index_extractor.column_mapping["id"] == "entity_id"
        assert change_index_extractor.column_mapping["reference_date"] == "image_date"
        # cache_key_columns reflects the mapped id column
        assert change_index_extractor.cache_key_columns == ["entity_id", "reference_date"]


# ===================================================================
# require_change_index_params decorator
# ===================================================================


class TestRequireChangeIndexParamsDecorator:
    """Tests that methods guarded by @require_change_index_params raise when params are not set."""

    def test_format_without_params_raises(self, change_index_extractor):
        assert change_index_extractor.change_index_params is None
        with pytest.raises(RuntimeError, match="No change index parameters found"):
            change_index_extractor.format_change_index_json({"id": "x", "data": {}})


# ===================================================================
# _get_reference_date()
# ===================================================================


class TestGetReferenceDate:
    """Tests for the per-entity reference_date resolver."""

    def test_valid_iso_date_string(self, configured_change_index_extractor):
        result = configured_change_index_extractor._get_reference_date({"reference_date": "2025-06-15"}, "ent_x")
        assert result == "2025-06-15"

    def test_iso_datetime_string_trimmed(self, configured_change_index_extractor):
        """A YYYY-MM-DDTHH:MM:SS string should be trimmed to YYYY-MM-DD."""
        result = configured_change_index_extractor._get_reference_date(
            {"reference_date": "2025-06-15T00:00:00"}, "ent_x"
        )
        assert result == "2025-06-15"

    def test_pandas_timestamp(self, configured_change_index_extractor):
        result = configured_change_index_extractor._get_reference_date(
            {"reference_date": pd.Timestamp("2025-06-15")}, "ent_x"
        )
        assert result == "2025-06-15"

    def test_missing_reference_date_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="reference_date missing"):
            configured_change_index_extractor._get_reference_date({}, "ent_x")

    def test_none_reference_date_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="reference_date missing"):
            configured_change_index_extractor._get_reference_date({"reference_date": None}, "ent_x")

    def test_nan_reference_date_raises(self, configured_change_index_extractor):
        import math

        with pytest.raises(ValueError, match="reference_date missing"):
            configured_change_index_extractor._get_reference_date({"reference_date": math.nan}, "ent_x")

    def test_invalid_format_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="invalid reference_date"):
            configured_change_index_extractor._get_reference_date({"reference_date": "15/06/2025"}, "ent_x")

    def test_unsupported_type_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="unsupported reference_date type"):
            configured_change_index_extractor._get_reference_date({"reference_date": 20250615}, "ent_x")
