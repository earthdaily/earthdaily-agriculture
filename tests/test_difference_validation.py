"""
Tests for DifferenceExtractor validation logic:
    - setup_difference_parameters() input validation
    - require_difference_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_difference_parameters()
# ===================================================================


class TestSetupDifferenceParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, difference_extractor):
        """All defaults should pass validation."""
        difference_extractor.setup_difference_parameters()
        params = difference_extractor.difference_params
        assert params is not None
        assert params["product"] == "DIFFERENCE_NDVI"
        assert params["output_epsg"] == 4326
        assert params["postprocess"] == "stats"
        assert params["map_format"] is None
        assert params["output_path"] is None
        assert params["skip_existing"] is True
        assert params["directLinks"] is False
        assert params["partial_frequency"] == 50

    def test_short_product_names_resolve_to_full(self, difference_extractor):
        """Short index name should resolve to 'DIFFERENCE_<NAME>' (notebook cell 14)."""
        difference_extractor.setup_difference_parameters(product="NDVI")
        assert difference_extractor.difference_params["product"] == "DIFFERENCE_NDVI"

    def test_full_product_names_accepted(self, difference_extractor):
        """Full DIFFERENCE_<NAME> names should be accepted as-is."""
        difference_extractor.setup_difference_parameters(product="DIFFERENCE_EVI")
        assert difference_extractor.difference_params["product"] == "DIFFERENCE_EVI"

    def test_invalid_product_raises(self, difference_extractor):
        with pytest.raises(ValueError, match="product"):
            difference_extractor.setup_difference_parameters(product="BOGUS")

    @pytest.mark.parametrize("short", ["NDVI", "EVI", "GNDVI", "NDRE", "CVI", "CVIN", "LAI", "NDWI", "NDMI", "S2REP"])
    def test_all_short_product_names_resolve(self, difference_extractor, short):
        difference_extractor.setup_difference_parameters(product=short)
        assert difference_extractor.difference_params["product"] == f"DIFFERENCE_{short}"

    def test_setup_sets_cache_key_columns(self, difference_extractor):
        """Setup configures cache_key_columns to [mapped id, 'image_id_1', 'image_id_2']."""
        difference_extractor.setup_difference_parameters()
        assert difference_extractor.cache_key_columns == ["id", "image_id_1", "image_id_2"]

    def test_cache_disabled_by_default(self, difference_extractor):
        """When use_cache is None, cache must remain disabled (results depend on image IDs)."""
        difference_extractor.use_cache = True  # pretend it had been enabled
        difference_extractor.setup_difference_parameters()  # use_cache=None
        assert difference_extractor.use_cache is False

    def test_use_cache_true_keeps_cache_enabled(self, difference_extractor):
        difference_extractor.setup_difference_parameters(use_cache=True)
        assert difference_extractor.use_cache is True

    # --- postprocess validation ---

    def test_invalid_postprocess_raises(self, difference_extractor):
        with pytest.raises(ValueError, match="postprocess"):
            difference_extractor.setup_difference_parameters(postprocess="bogus")

    @pytest.mark.parametrize("pp", ["stats", "links", "file"])
    def test_valid_postprocess_modes(self, difference_extractor, pp):
        kwargs = {"postprocess": pp}
        # 'file' mode requires map_format and output_path
        if pp == "file":
            kwargs["map_format"] = "png"
            kwargs["output_path"] = "/tmp/out"
        difference_extractor.setup_difference_parameters(**kwargs)
        assert difference_extractor.difference_params["postprocess"] == pp

    # --- map_format validation ---

    def test_invalid_map_format_raises(self, difference_extractor):
        with pytest.raises(ValueError, match="map_format"):
            difference_extractor.setup_difference_parameters(map_format="jpg")

    @pytest.mark.parametrize("mf", ["png", "tiff.zip", "shp.zip"])
    def test_valid_map_formats(self, difference_extractor, mf):
        difference_extractor.setup_difference_parameters(postprocess="file", map_format=mf, output_path="/tmp/out")
        assert difference_extractor.difference_params["map_format"] == mf

    # --- postprocess='links' cross-validation ---

    def test_links_mode_auto_corrects_directlinks(self, difference_extractor):
        """postprocess='links' with directLinks=False must auto-correct to True."""
        difference_extractor.setup_difference_parameters(postprocess="links", directLinks=False)
        assert difference_extractor.difference_params["directLinks"] is True

    def test_links_mode_with_map_format_raises(self, difference_extractor):
        """postprocess='links' must not have map_format set."""
        with pytest.raises(ValueError, match="map_format"):
            difference_extractor.setup_difference_parameters(postprocess="links", directLinks=True, map_format="png")

    # --- postprocess='file' cross-validation ---

    def test_file_mode_requires_map_format(self, difference_extractor):
        with pytest.raises(ValueError, match="map_format"):
            difference_extractor.setup_difference_parameters(postprocess="file", output_path="/tmp/out")

    def test_file_mode_requires_output_path(self, difference_extractor):
        with pytest.raises(ValueError, match="output_path"):
            difference_extractor.setup_difference_parameters(postprocess="file", map_format="png")

    # --- column_mapping ---

    def test_column_mapping_applied(self, difference_extractor):
        difference_extractor.setup_difference_parameters(column_mapping={"id": "entity_id"})
        assert difference_extractor.column_mapping["id"] == "entity_id"


# ===================================================================
# require_difference_params decorator
# ===================================================================


class TestRequireDifferenceParamsDecorator:
    """Methods guarded by @require_difference_params raise when params are not set."""

    def test_format_stats_without_params_raises(self, difference_extractor):
        assert difference_extractor.difference_params is None
        with pytest.raises(RuntimeError, match="No difference parameters found"):
            difference_extractor.format_difference_stats_json({"legend": {}})

    def test_format_links_without_params_raises(self, difference_extractor):
        assert difference_extractor.difference_params is None
        with pytest.raises(RuntimeError, match="No difference parameters found"):
            difference_extractor.format_difference_links_json({"_links": {}})
