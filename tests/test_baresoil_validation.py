"""
Tests for BaresoilExtractor validation logic:
    - setup_baresoil_parameters() input validation
    - require_baresoil_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_baresoil_parameters()
# ===================================================================


class TestSetupBaresoilParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, baresoil_extractor):
        """All defaults should pass validation."""
        baresoil_extractor.setup_baresoil_parameters()
        params = baresoil_extractor.baresoil_params
        assert params is not None
        assert params["season_duration"] == 120
        assert params["season_start_month"] == 4
        assert params["season_start_day"] == 1
        assert params["year"] == 2025
        assert params["filter"] == "summary"
        assert params["publish_af"] is False

    def test_custom_parameters(self, baresoil_extractor):
        baresoil_extractor.setup_baresoil_parameters(
            season_duration=200,
            season_start_day=15,
            season_start_month=10,
            year=2024,
            filter="full",
            publish_af=True,
            partial_frequency=25,
        )
        params = baresoil_extractor.baresoil_params
        assert params["season_duration"] == 200
        assert params["season_start_month"] == 10
        assert params["season_start_day"] == 15
        assert params["year"] == 2024
        assert params["filter"] == "full"
        assert params["publish_af"] is True
        assert params["partial_frequency"] == 25

    def test_setup_sets_cache_key_columns(self, baresoil_extractor):
        """Setup should configure cache_key_columns to the mapped id column."""
        baresoil_extractor.setup_baresoil_parameters()
        assert baresoil_extractor.cache_key_columns == ["id"]

    # --- Month validation ---
    def test_invalid_month_zero(self, baresoil_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            baresoil_extractor.setup_baresoil_parameters(season_start_month=0)

    def test_invalid_month_thirteen(self, baresoil_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            baresoil_extractor.setup_baresoil_parameters(season_start_month=13)

    def test_invalid_month_negative(self, baresoil_extractor):
        with pytest.raises(ValueError, match="season_start_month"):
            baresoil_extractor.setup_baresoil_parameters(season_start_month=-1)

    # --- Day validation ---
    def test_invalid_day_zero(self, baresoil_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            baresoil_extractor.setup_baresoil_parameters(season_start_day=0)

    def test_invalid_day_thirtytwo(self, baresoil_extractor):
        with pytest.raises(ValueError, match="season_start_day"):
            baresoil_extractor.setup_baresoil_parameters(season_start_day=32)

    # --- Filter validation ---
    def test_invalid_filter_value(self, baresoil_extractor):
        with pytest.raises(ValueError, match="filter"):
            baresoil_extractor.setup_baresoil_parameters(filter="raw")

    def test_filter_summary_accepted(self, baresoil_extractor):
        baresoil_extractor.setup_baresoil_parameters(filter="summary")
        assert baresoil_extractor.baresoil_params["filter"] == "summary"

    def test_filter_full_accepted(self, baresoil_extractor):
        baresoil_extractor.setup_baresoil_parameters(filter="full")
        assert baresoil_extractor.baresoil_params["filter"] == "full"

    # --- publish_af validation ---
    def test_invalid_publish_af_string(self, baresoil_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            baresoil_extractor.setup_baresoil_parameters(publish_af="yes")

    def test_invalid_publish_af_int(self, baresoil_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            baresoil_extractor.setup_baresoil_parameters(publish_af=1)


# ===================================================================
# require_baresoil_params decorator
# ===================================================================


class TestRequireBaresoilParamsDecorator:
    """Tests that methods guarded by @require_baresoil_params raise when params are not set."""

    def test_format_without_params_raises(self, baresoil_extractor):
        """format_baresoil_json should fail if setup was never called."""
        assert baresoil_extractor.baresoil_params is None
        with pytest.raises(RuntimeError, match="No baresoil parameters found"):
            baresoil_extractor.format_baresoil_json({"id": "x", "data": {}})
