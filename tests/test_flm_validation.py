"""
Tests for FLMExtractor validation logic:
    - setup_flm_parameters() input validation
    - Cross-validation rules for postprocess='links' and postprocess='file'
    - require_flm_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_flm_parameters() — happy paths & individual field validation
# ===================================================================


class TestSetupFlmParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, flm_extractor):
        """All defaults should pass validation (postprocess='stats', map_format=None)."""
        flm_extractor.setup_flm_parameters()
        params = flm_extractor.flm_params
        assert params is not None
        assert params["vegetation_index"] == "NDVI"
        assert params["map_format"] is None
        assert params["output_epsg"] == 4326
        assert params["postprocess"] == "stats"
        assert params["directLinks"] is False
        assert params["clipping"] == "FieldBorder"
        assert params["buffer"] == 0
        assert params["number_bins"] is None
        assert params["legendType"] is None

    def test_setup_sets_cache_key_columns(self, flm_extractor):
        """Setup should configure cache_key_columns to [mapped id]."""
        flm_extractor.setup_flm_parameters()
        assert flm_extractor.cache_key_columns == ["id"]

    # --- vegetation_index validation ---

    @pytest.mark.parametrize(
        "vi", ["NDVI", "EVI", "NDRE", "CVI", "CVIN", "GNDVI", "LAI", "NDWI", "NDMI", "S2REP", "COLORCOMPOSITION"]
    )
    def test_valid_vegetation_indexes(self, flm_extractor, vi):
        flm_extractor.setup_flm_parameters(vegetation_index=vi)
        assert flm_extractor.flm_params["vegetation_index"] == vi

    def test_invalid_vegetation_index_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="vegetation_index"):
            flm_extractor.setup_flm_parameters(vegetation_index="BOGUS")

    # --- map_format validation ---

    @pytest.mark.parametrize("mf", [None, "png", "tiff.zip", "shp.zip"])
    def test_valid_map_formats(self, flm_extractor, mf):
        kwargs = {"map_format": mf}
        # map_format must be paired with postprocess='file' if not None
        if mf is not None:
            kwargs["postprocess"] = "file"
            kwargs["output_path"] = "/tmp/out"
        flm_extractor.setup_flm_parameters(**kwargs)
        assert flm_extractor.flm_params["map_format"] == mf

    def test_invalid_map_format_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="map_format"):
            flm_extractor.setup_flm_parameters(map_format="jpg")

    # --- postprocess validation ---

    @pytest.mark.parametrize("pp", ["stats", "links", "histogram"])
    def test_valid_postprocess_modes(self, flm_extractor, pp):
        flm_extractor.setup_flm_parameters(postprocess=pp)
        assert flm_extractor.flm_params["postprocess"] == pp

    def test_postprocess_file_requires_map_format_and_output_path(self, flm_extractor):
        """postprocess='file' has its own cross-validation; happy path needs both fields."""
        flm_extractor.setup_flm_parameters(postprocess="file", map_format="png", output_path="/tmp/out")
        assert flm_extractor.flm_params["postprocess"] == "file"

    def test_invalid_postprocess_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="postprocess"):
            flm_extractor.setup_flm_parameters(postprocess="bogus")

    # --- directLinks validation ---

    def test_invalid_directLinks_type_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="directLinks"):
            flm_extractor.setup_flm_parameters(directLinks="yes")

    # --- clipping validation ---

    @pytest.mark.parametrize("clip", ["FieldBorder", "Bbox"])
    def test_valid_clipping(self, flm_extractor, clip):
        flm_extractor.setup_flm_parameters(clipping=clip)
        assert flm_extractor.flm_params["clipping"] == clip

    def test_invalid_clipping_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="clipping"):
            flm_extractor.setup_flm_parameters(clipping="bogus")

    # --- buffer validation ---

    def test_negative_buffer_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="buffer"):
            flm_extractor.setup_flm_parameters(buffer=-5)

    def test_non_numeric_buffer_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="buffer"):
            flm_extractor.setup_flm_parameters(buffer="ten")

    # --- number_bins validation ---

    def test_number_bins_none_is_default(self, flm_extractor):
        flm_extractor.setup_flm_parameters(number_bins=None)
        assert flm_extractor.flm_params["number_bins"] is None

    @pytest.mark.parametrize("nb", [1, 15, 255])
    def test_valid_number_bins(self, flm_extractor, nb):
        flm_extractor.setup_flm_parameters(number_bins=nb)
        assert flm_extractor.flm_params["number_bins"] == nb

    def test_number_bins_below_one_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="number_bins"):
            flm_extractor.setup_flm_parameters(number_bins=0)

    def test_number_bins_above_255_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="number_bins"):
            flm_extractor.setup_flm_parameters(number_bins=256)

    def test_number_bins_non_int_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="number_bins"):
            flm_extractor.setup_flm_parameters(number_bins=15.5)

    def test_number_bins_bool_rejected(self, flm_extractor):
        """bool is a subclass of int but should be rejected explicitly."""
        with pytest.raises(ValueError, match="number_bins"):
            flm_extractor.setup_flm_parameters(number_bins=True)

    # --- legendType validation ---

    @pytest.mark.parametrize("lt", ["Fixed", "Dynamic", "Common"])
    def test_valid_legend_types(self, flm_extractor, lt):
        flm_extractor.setup_flm_parameters(legendType=lt)
        assert flm_extractor.flm_params["legendType"] == lt

    def test_legend_type_none_is_default(self, flm_extractor):
        flm_extractor.setup_flm_parameters(legendType=None)
        assert flm_extractor.flm_params["legendType"] is None

    def test_invalid_legend_type_raises(self, flm_extractor):
        with pytest.raises(ValueError, match="legendType"):
            flm_extractor.setup_flm_parameters(legendType="bogus")


# ===================================================================
# Cross-validation: postprocess='links'
# ===================================================================


class TestSetupFlmParametersLinksMode:
    """postprocess='links' has cross-validation rules:
    - directLinks must be True (auto-corrected if False)
    - map_format must be None
    """

    def test_links_mode_auto_corrects_directlinks(self, flm_extractor):
        """postprocess='links' with directLinks=False should auto-correct to True."""
        flm_extractor.setup_flm_parameters(postprocess="links", directLinks=False)
        assert flm_extractor.flm_params["directLinks"] is True

    def test_links_mode_keeps_directlinks_true(self, flm_extractor):
        flm_extractor.setup_flm_parameters(postprocess="links", directLinks=True)
        assert flm_extractor.flm_params["directLinks"] is True

    def test_links_mode_with_map_format_raises(self, flm_extractor):
        """postprocess='links' must not have map_format set."""
        with pytest.raises(ValueError, match="map_format"):
            flm_extractor.setup_flm_parameters(postprocess="links", map_format="png")

    def test_links_mode_map_format_none_succeeds(self, flm_extractor):
        flm_extractor.setup_flm_parameters(postprocess="links", map_format=None)
        assert flm_extractor.flm_params["map_format"] is None
        assert flm_extractor.flm_params["directLinks"] is True


# ===================================================================
# Cross-validation: postprocess='file'
# ===================================================================


class TestSetupFlmParametersFileMode:
    """postprocess='file' requires both map_format and output_path."""

    def test_file_mode_requires_map_format(self, flm_extractor):
        with pytest.raises(ValueError, match="map_format"):
            flm_extractor.setup_flm_parameters(postprocess="file", output_path="/tmp/out")

    def test_file_mode_requires_output_path(self, flm_extractor):
        with pytest.raises(ValueError, match="output_path"):
            flm_extractor.setup_flm_parameters(postprocess="file", map_format="png")

    @pytest.mark.parametrize("mf", ["png", "tiff.zip", "shp.zip"])
    def test_file_mode_with_required_fields(self, flm_extractor, mf):
        """All three valid file formats should pass when map_format and output_path are set."""
        flm_extractor.setup_flm_parameters(postprocess="file", map_format=mf, output_path="/tmp/out")
        assert flm_extractor.flm_params["postprocess"] == "file"
        assert flm_extractor.flm_params["map_format"] == mf
        assert flm_extractor.flm_params["output_path"] == "/tmp/out"


# ===================================================================
# require_flm_params decorator
# ===================================================================


class TestRequireFlmParamsDecorator:
    """Methods guarded by @require_flm_params raise when params not set."""

    def test_get_flm_map_without_params_raises(self, flm_extractor):
        assert flm_extractor.flm_params is None
        with pytest.raises(RuntimeError, match="No FLM parameters found"):
            flm_extractor.get_flm_map({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}, "img|1")

    def test_format_stats_without_params_raises(self, flm_extractor):
        assert flm_extractor.flm_params is None
        with pytest.raises(RuntimeError, match="No FLM parameters found"):
            flm_extractor.format_flm_map_stats_json({"legend": {}})

    def test_format_histogram_without_params_raises(self, flm_extractor):
        assert flm_extractor.flm_params is None
        with pytest.raises(RuntimeError, match="No FLM parameters found"):
            flm_extractor.format_flm_map_histogram_json({"histogram": {}})

    def test_format_links_without_params_raises(self, flm_extractor):
        assert flm_extractor.flm_params is None
        with pytest.raises(RuntimeError, match="No FLM parameters found"):
            flm_extractor.format_flm_links_json({"_links": {}})

    def test_process_single_without_params_raises(self, flm_extractor):
        assert flm_extractor.flm_params is None
        with pytest.raises(RuntimeError, match="No FLM parameters found"):
            flm_extractor.process_single_entity_flm({"id": "x"})

    def test_get_flm_map_safe_does_not_require_params(self, flm_extractor):
        """The safe wrapper isn't guarded — it returns failure instead of raising."""
        assert flm_extractor.flm_params is None
        result = flm_extractor.get_flm_map_safe({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}, "img|1")
        assert result["success"] is False
        assert "No FLM parameters" in result["error"]
