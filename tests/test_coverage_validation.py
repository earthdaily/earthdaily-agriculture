"""
Tests for CoverageExtractor validation logic:
    - setup_coverage_parameters() input validation
    - require_coverage_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_coverage_parameters()
# ===================================================================


class TestSetupCoverageParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, coverage_extractor):
        """All defaults should pass validation."""
        coverage_extractor.setup_coverage_parameters()
        params = coverage_extractor.coverage_params
        assert params is not None
        assert params["vegetation_index"] == "NDVI"
        assert params["start_date"] == "2025-01-01"
        assert params["end_date"] is None
        assert params["clear_cover_min"] == 90
        assert params["clear_cover_max"] == 100
        assert params["use_specific_date"] is False
        assert params["filter"] == "none"
        assert params["delay"] == 3
        assert params["mask"] == "Auto"  # normalized to the API's canonical casing
        assert params["recalibration"] is False
        assert params["historical_seasons"] is None

    def test_custom_parameters_from_notebook(self, coverage_extractor):
        """Mirrors the notebook's `setup_coverage_parameters(...)` call (cell 13)."""
        coverage_extractor.setup_coverage_parameters(
            vegetation_index="NDVI",
            start_date="2025-01-01",
            clear_cover_min=80,
            use_specific_date=False,
            filter="duplicate",
            delay=3,
            mask="auto",
            partial_frequency=20,
            exclude_columns=[],
            column_mapping={"crop": "crop.id"},
        )
        params = coverage_extractor.coverage_params
        assert params["clear_cover_min"] == 80
        assert params["filter"] == "duplicate"
        assert params["delay"] == 3
        assert params["mask"] == "Auto"  # normalized to the API's canonical casing
        assert params["partial_frequency"] == 20
        # column_mapping is applied separately
        assert coverage_extractor.column_mapping["crop"] == "crop.id"

    def test_setup_sets_cache_key_columns(self, coverage_extractor):
        """Setup should configure cache_key_columns to [id, image_id, mask]."""
        coverage_extractor.setup_coverage_parameters()
        assert coverage_extractor.cache_key_columns == ["id", "image_id", "mask"]

    # --- vegetation_index validation ---

    def test_invalid_vegetation_index(self, coverage_extractor):
        with pytest.raises(ValueError, match="vegetation_index"):
            coverage_extractor.setup_coverage_parameters(vegetation_index="BOGUS")

    @pytest.mark.parametrize("vi", ["NDVI", "EVI", "CVI", "CVIN", "GNDVI", "LAI", "NDWI", "NDMI", "S2REP"])
    def test_valid_vegetation_indexes(self, coverage_extractor, vi):
        coverage_extractor.setup_coverage_parameters(vegetation_index=vi)
        assert coverage_extractor.coverage_params["vegetation_index"] == vi

    # --- date validation ---

    def test_invalid_start_date_format(self, coverage_extractor):
        with pytest.raises(ValueError, match="start_date"):
            coverage_extractor.setup_coverage_parameters(start_date="01/01/2025")

    def test_invalid_end_date_format(self, coverage_extractor):
        with pytest.raises(ValueError, match="end_date"):
            coverage_extractor.setup_coverage_parameters(end_date="2025/01/01")

    def test_end_date_none_is_allowed(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters(end_date=None)
        assert coverage_extractor.coverage_params["end_date"] is None

    # --- clear_cover_min validation ---

    def test_clear_cover_min_below_zero(self, coverage_extractor):
        with pytest.raises(ValueError, match="clear_cover_min"):
            coverage_extractor.setup_coverage_parameters(clear_cover_min=-1)

    def test_clear_cover_min_above_hundred(self, coverage_extractor):
        with pytest.raises(ValueError, match="clear_cover_min"):
            coverage_extractor.setup_coverage_parameters(clear_cover_min=101)

    # --- filter validation ---

    def test_invalid_filter_value(self, coverage_extractor):
        with pytest.raises(ValueError, match="filter"):
            coverage_extractor.setup_coverage_parameters(filter="random")

    @pytest.mark.parametrize("flt", ["none", "duplicate"])
    def test_filter_simple_modes_accepted(self, coverage_extractor, flt):
        coverage_extractor.setup_coverage_parameters(filter=flt)
        assert coverage_extractor.coverage_params["filter"] == flt

    def test_crop_coverage_filter_requires_historical_seasons(self, coverage_extractor):
        with pytest.raises(ValueError, match="historical_seasons"):
            coverage_extractor.setup_coverage_parameters(filter="crop_coverage")

    def test_crop_coverage_filter_with_historical_seasons(self, coverage_extractor):
        """Mirrors the notebook's crop_coverage setup (cell 32)."""
        coverage_extractor.setup_coverage_parameters(
            filter="crop_coverage",
            historical_seasons=[2022, 2024, 2025],
            start_date="2025-07-01",
            end_date="2025-07-31",
        )
        assert coverage_extractor.coverage_params["filter"] == "crop_coverage"
        assert coverage_extractor.coverage_params["historical_seasons"] == [2022, 2024, 2025]

    # --- delay validation ---

    def test_negative_delay_raises(self, coverage_extractor):
        with pytest.raises(ValueError, match="delay"):
            coverage_extractor.setup_coverage_parameters(delay=-1)

    # --- mask validation ---

    def test_invalid_mask_value(self, coverage_extractor):
        with pytest.raises(ValueError, match="mask"):
            coverage_extractor.setup_coverage_parameters(mask="bogus")

    @pytest.mark.parametrize(
        ("mask", "expected"),
        [
            ("auto", "Auto"),
            ("native", "Native"),
            ("ML", "ML"),
            ("ACM", "ACM"),
            ("MLCirrus", "MLCirrus"),
            ("All", "All"),
            # case-insensitive on the way in, canonical on the way out
            ("AUTO", "Auto"),
            ("all", "All"),
            ("mlcirrus", "MLCirrus"),
        ],
    )
    def test_valid_masks(self, coverage_extractor, mask, expected):
        coverage_extractor.setup_coverage_parameters(mask=mask)
        assert coverage_extractor.coverage_params["mask"] == expected

    def test_mask_all_rejects_duplicate_filter(self, coverage_extractor):
        """'All' repeats each image per mask, which would double-count coverage when pairing."""
        with pytest.raises(ValueError, match="duplicate"):
            coverage_extractor.setup_coverage_parameters(mask="All", filter="duplicate")

    def test_mask_all_allowed_without_duplicate_filter(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters(mask="All", filter="none")
        assert coverage_extractor.coverage_params["mask"] == "All"

    # --- column_mapping ---

    def test_column_mapping_applied(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters(column_mapping={"crop": "crop.id"})
        assert coverage_extractor.column_mapping["crop"] == "crop.id"


# ===================================================================
# require_coverage_params decorator
# ===================================================================


class TestRequireCoverageParamsDecorator:
    """Methods guarded by @require_coverage_params raise when params not set."""

    def test_format_without_params_raises(self, coverage_extractor):
        """format_coverage_json should fail if setup was never called."""
        # Mimic the "not set" condition the decorator checks.
        # Note: the decorator checks `not hasattr(self, 'coverage_params')`,
        # so we must remove the attribute entirely.
        del coverage_extractor.coverage_params
        with pytest.raises(RuntimeError, match="No coverage parameters found"):
            coverage_extractor.format_coverage_json([])
