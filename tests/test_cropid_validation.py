"""
Tests for cropidExtractor validation logic:
    - setup_cropid_parameters() input validation
    - require_cropid_params decorator
"""

from datetime import datetime

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_cropid_parameters()
# ===================================================================


class TestSetupCropidParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, cropid_extractor):
        """All defaults should pass validation."""
        cropid_extractor.setup_cropid_parameters()
        params = cropid_extractor.cropid_params
        assert params is not None
        assert params["begin_year"] == 2020
        assert params["end_year"] == 2025
        assert params["mask_type"] == "EndSeason"
        assert params["limit_nb_crop"] == 1
        assert params["crop_mask_percent"] == 50
        assert params["mode"] == "history"
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, cropid_extractor):
        """Mirrors notebook cell 13."""
        cropid_extractor.setup_cropid_parameters(
            begin_year=2020,
            end_year=2025,
            mask_type="EndSeason",
            limit_nb_crop=1,
            crop_mask_percent=75,
            mode="year",
            exclude_columns=[],
            column_mapping={"crop": "crop.id"},
        )
        params = cropid_extractor.cropid_params
        assert params["crop_mask_percent"] == 75
        assert params["mode"] == "year"
        assert cropid_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, cropid_extractor):
        """Setup configures cache_key_columns to [mapped id, 'year']."""
        cropid_extractor.setup_cropid_parameters()
        assert cropid_extractor.cache_key_columns == ["id", "year"]

    # --- mask_type validation ---

    def test_invalid_mask_type(self, cropid_extractor):
        with pytest.raises(ValueError, match="mask_type"):
            cropid_extractor.setup_cropid_parameters(mask_type="bogus")

    def test_historical_is_no_longer_accepted(self, cropid_extractor):
        """'Historical' is not a crop-mask product — the three below are.

        It was accepted for years and passed straight through to the API's
        ``Products`` list, so a caller using it got whatever the service made of
        an unknown product rather than a clear error here.
        """
        with pytest.raises(ValueError, match="mask_type"):
            cropid_extractor.setup_cropid_parameters(mask_type="Historical")

    @pytest.mark.parametrize("mt", ["InSeason", "EndSeason", "PreSeason"])
    def test_valid_mask_types(self, cropid_extractor, mt):
        cropid_extractor.setup_cropid_parameters(mask_type=mt)
        assert cropid_extractor.cropid_params["mask_type"] == mt

    # --- year validation ---

    def test_begin_year_too_old(self, cropid_extractor):
        with pytest.raises(ValueError, match="begin_year"):
            cropid_extractor.setup_cropid_parameters(begin_year=1999)

    def test_end_year_too_far_future(self, cropid_extractor):
        future = datetime.now().year + 5
        with pytest.raises(ValueError, match="end_year"):
            cropid_extractor.setup_cropid_parameters(end_year=future)

    def test_end_year_before_begin_year_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="end_year.*begin_year"):
            cropid_extractor.setup_cropid_parameters(begin_year=2024, end_year=2020)

    # --- limit_nb_crop validation ---

    def test_limit_nb_crop_zero_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="limit_nb_crop"):
            cropid_extractor.setup_cropid_parameters(limit_nb_crop=0)

    def test_limit_nb_crop_negative_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="limit_nb_crop"):
            cropid_extractor.setup_cropid_parameters(limit_nb_crop=-1)

    def test_limit_nb_crop_non_int_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="limit_nb_crop"):
            cropid_extractor.setup_cropid_parameters(limit_nb_crop=2.5)

    # --- crop_mask_percent validation ---

    def test_crop_mask_percent_below_zero(self, cropid_extractor):
        with pytest.raises(ValueError, match="crop_mask_percent"):
            cropid_extractor.setup_cropid_parameters(crop_mask_percent=-1)

    def test_crop_mask_percent_above_hundred(self, cropid_extractor):
        with pytest.raises(ValueError, match="crop_mask_percent"):
            cropid_extractor.setup_cropid_parameters(crop_mask_percent=101)

    # --- mode validation ---

    def test_invalid_mode(self, cropid_extractor):
        with pytest.raises(ValueError, match="mode"):
            cropid_extractor.setup_cropid_parameters(mode="bogus")

    @pytest.mark.parametrize("mode", ["history", "year", "historical_season", "full_history"])
    def test_valid_modes(self, cropid_extractor, mode):
        cropid_extractor.setup_cropid_parameters(mode=mode)
        assert cropid_extractor.cropid_params["mode"] == mode

    # --- partial_frequency validation ---

    def test_negative_partial_frequency_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            cropid_extractor.setup_cropid_parameters(partial_frequency=-1)

    def test_non_int_partial_frequency_raises(self, cropid_extractor):
        with pytest.raises(ValueError, match="partial_frequency"):
            cropid_extractor.setup_cropid_parameters(partial_frequency=10.5)


# ===================================================================
# require_cropid_params decorator
# ===================================================================


class TestRequireCropidParamsDecorator:
    """Methods guarded by @require_cropid_params raise when params are not set."""

    def test_format_without_params_raises(self, cropid_extractor):
        """format_cropid_json should fail if setup was never called."""
        assert cropid_extractor.cropid_params is None
        with pytest.raises(RuntimeError, match="No crop id parameters found"):
            cropid_extractor.format_cropid_json({"resultsByYear": {}})

    def test_format_year_without_params_raises(self, cropid_extractor):
        assert cropid_extractor.cropid_params is None
        with pytest.raises(RuntimeError, match="No crop id parameters found"):
            cropid_extractor.format_cropid_year_json({"resultsByYear": {}}, {"crop": "SOYBEANS"})

    def test_format_historical_without_params_raises(self, cropid_extractor):
        assert cropid_extractor.cropid_params is None
        with pytest.raises(RuntimeError, match="No crop id parameters found"):
            cropid_extractor.format_cropid_historical_season_json({"resultsByYear": {}}, {"crop": "SOYBEANS"})
