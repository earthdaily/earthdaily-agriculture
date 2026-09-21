"""
Tests for GDDExtractor validation logic:
    - setup_gdd_parameters() validation rules
    - require_gdd_params decorator
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_gdd_parameters()
# ===================================================================


class TestSetupGddParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, gdd_extractor):
        """Defaults should pass validation (provider GLOBAL1, lower_threshold=10)."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters()

        params = gdd_extractor.gdd_params
        assert params is not None
        assert params["provider"] == "GLOBAL1"
        assert params["lower_threshold"] == 10.0
        assert params["upper_threshold"] is None
        assert params["start_date"] is None
        assert params["end_date"] is None
        assert params["reset_cumulative_every_year"] is False
        assert params["extrapolate_forecast_data"] is False
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, gdd_extractor):
        """Mirrors notebook cell 10: GLOBAL1, 10/30, 2023-01-17→2023-03-17."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(
                provider="GLOBAL1",
                lower_threshold=10,
                upper_threshold=30,
                start_date="2023-01-17",
                end_date="2023-03-17",
                reset_cumulative_every_year=False,
                extrapolate_forecast_data=False,
            )

        params = gdd_extractor.gdd_params
        assert params["lower_threshold"] == 10
        assert params["upper_threshold"] == 30
        assert params["start_date"] == "2023-01-17"
        assert params["end_date"] == "2023-03-17"

    def test_setup_sets_cache_key_columns(self, gdd_extractor):
        """cache_key_columns should be [mapped id, 'date']."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters()
        assert gdd_extractor.cache_key_columns == ["id", "date"]

    def test_column_mapping_applied(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(column_mapping={"id": "entity_id"})
        assert gdd_extractor.column_mapping["id"] == "entity_id"

    # --- provider validation ---

    def test_invalid_provider_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="provider"):
                gdd_extractor.setup_gdd_parameters(provider="BOGUS")

    def test_valid_provider_global1(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(provider="GLOBAL1")
        assert gdd_extractor.gdd_params["provider"] == "GLOBAL1"

    # --- threshold validation ---

    def test_lower_threshold_non_numeric_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="lower_threshold"):
                gdd_extractor.setup_gdd_parameters(lower_threshold="ten")

    def test_lower_threshold_bool_rejected(self, gdd_extractor):
        """bool is a subclass of int but should be rejected explicitly."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="lower_threshold"):
                gdd_extractor.setup_gdd_parameters(lower_threshold=True)

    def test_upper_threshold_non_numeric_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="upper_threshold"):
                gdd_extractor.setup_gdd_parameters(upper_threshold="thirty")

    def test_upper_threshold_bool_rejected(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="upper_threshold"):
                gdd_extractor.setup_gdd_parameters(upper_threshold=False)

    def test_lower_must_be_below_upper(self, gdd_extractor):
        """lower_threshold >= upper_threshold should raise."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="lower_threshold"):
                gdd_extractor.setup_gdd_parameters(lower_threshold=30, upper_threshold=10)

    def test_lower_equal_to_upper_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="lower_threshold"):
                gdd_extractor.setup_gdd_parameters(lower_threshold=20, upper_threshold=20)

    def test_upper_threshold_none_allowed(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(lower_threshold=10, upper_threshold=None)
        assert gdd_extractor.gdd_params["upper_threshold"] is None

    # --- date validation ---

    def test_invalid_start_date_format_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                gdd_extractor.setup_gdd_parameters(start_date="01/01/2023")

    def test_invalid_end_date_format_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="end_date"):
                gdd_extractor.setup_gdd_parameters(end_date="2023/03/17")

    def test_start_date_after_end_date_raises(self, gdd_extractor):
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="start_date"):
                gdd_extractor.setup_gdd_parameters(start_date="2023-06-01", end_date="2023-01-01")

    def test_start_equals_end_date_allowed(self, gdd_extractor):
        """A single-day range (start == end) is allowed."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(start_date="2023-01-17", end_date="2023-01-17")
        assert gdd_extractor.gdd_params["start_date"] == "2023-01-17"

    def test_dates_optional_can_be_none(self, gdd_extractor):
        """Dates default to None (overridden per-entity)."""
        with patch.object(gdd_extractor, "ensure_token_valid"):
            gdd_extractor.setup_gdd_parameters(start_date=None, end_date=None)
        assert gdd_extractor.gdd_params["start_date"] is None
        assert gdd_extractor.gdd_params["end_date"] is None


# ===================================================================
# require_gdd_params decorator
# ===================================================================


class TestRequireGddParamsDecorator:
    """Methods guarded by @require_gdd_params raise when params not set."""

    def test_get_gdd_without_params_raises(self, gdd_extractor):
        assert gdd_extractor.gdd_params is None
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No GDD parameters found"):
                gdd_extractor.get_gdd({"id": "x", "geometry": "POINT (0 0)"})

    def test_get_gdd_safe_without_params_raises(self, gdd_extractor):
        """Unlike most safe wrappers, get_gdd_safe IS guarded by @require_gdd_params."""
        assert gdd_extractor.gdd_params is None
        with patch.object(gdd_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No GDD parameters found"):
                gdd_extractor.get_gdd_safe({"id": "x", "geometry": "POINT (0 0)"})

    def test_format_gdd_json_does_not_require_params(self, gdd_extractor, sample_gdd_response):
        """format_gdd_json is NOT guarded — callable without setup."""
        assert gdd_extractor.gdd_params is None
        df = gdd_extractor.format_gdd_json(sample_gdd_response)
        assert df is not None
        assert len(df) == 3

    def test_process_single_does_not_require_params(self, gdd_extractor):
        """process_single_entity_gdd is NOT decorated; it tolerates missing params at call time
        but get_gdd will raise inside, surfaced as an error result."""
        assert gdd_extractor.gdd_params is None
        result = gdd_extractor.process_single_entity_gdd({"id": "x", "geometry": "POINT (0 0)"})
        assert result["data"] is None
        assert result["error"] is not None
