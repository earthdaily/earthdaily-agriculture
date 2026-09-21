"""
Tests for ZoningExtractor validation logic:
    - setup_zoning_parameters() input validation + cross-mode constraints
    - require_zoning_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_zoning_parameters()
# ===================================================================


class TestSetupZoningParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, zoning_extractor):
        """All defaults should pass validation."""
        zoning_extractor.setup_zoning_parameters()
        params = zoning_extractor.zoning_params
        assert params is not None
        assert params["num_zones"] == 5
        assert params["output_epsg"] == 4326
        assert params["postprocess"] == "stats"
        assert params["map_format"] is None
        assert params["output_path"] is None
        assert params["directLinks"] is False

    def test_custom_parameters_from_notebook(self, zoning_extractor):
        """Mirrors the notebook setup (cell 14): num_zones=5, postprocess='stats'."""
        zoning_extractor.setup_zoning_parameters(
            num_zones=5,
            postprocess="stats",
            output_epsg=4326,
        )
        params = zoning_extractor.zoning_params
        assert params["num_zones"] == 5
        assert params["postprocess"] == "stats"
        assert params["output_epsg"] == 4326

    def test_links_mode_setup_from_notebook(self, zoning_extractor):
        """Mirrors notebook cell 24: postprocess='links', directLinks=True."""
        zoning_extractor.setup_zoning_parameters(
            num_zones=5,
            postprocess="links",
            directLinks=True,
            output_epsg=4326,
        )
        params = zoning_extractor.zoning_params
        assert params["postprocess"] == "links"
        assert params["directLinks"] is True

    def test_file_mode_setup_from_notebook(self, zoning_extractor):
        """Mirrors notebook cell 26: postprocess='file', map_format='png', output_path set."""
        zoning_extractor.setup_zoning_parameters(
            num_zones=5,
            postprocess="file",
            map_format="png",
            output_path="/tmp/output",
            output_epsg=4326,
        )
        params = zoning_extractor.zoning_params
        assert params["postprocess"] == "file"
        assert params["map_format"] == "png"
        assert params["output_path"] == "/tmp/output"

    def test_stats_geo_mode_setup_from_notebook(self, zoning_extractor):
        """Mirrors notebook cell 28: postprocess='stats_geo'."""
        zoning_extractor.setup_zoning_parameters(
            num_zones=5,
            postprocess="stats_geo",
            output_epsg=4326,
        )
        assert zoning_extractor.zoning_params["postprocess"] == "stats_geo"

    def test_setup_sets_cache_key_columns(self, zoning_extractor):
        """Setup should configure cache_key_columns to [id, 'image_id']."""
        zoning_extractor.setup_zoning_parameters()
        assert zoning_extractor.cache_key_columns == ["id", "image_id"]

    def test_setup_with_custom_id_column_mapping(self, zoning_extractor):
        """When id is remapped, cache_key_columns reflects the mapped column name."""
        zoning_extractor.setup_zoning_parameters(column_mapping={"id": "entity_id"})
        assert zoning_extractor.column_mapping["id"] == "entity_id"
        assert zoning_extractor.cache_key_columns == ["entity_id", "image_id"]

    def test_use_cache_default_off(self, zoning_extractor):
        """Cache is disabled by default for zoning (image_id list keys are unstable)."""
        zoning_extractor.setup_zoning_parameters()
        assert zoning_extractor.use_cache is False

    def test_column_mapping_applied(self, zoning_extractor):
        zoning_extractor.setup_zoning_parameters(column_mapping={"geometry": "geom_wkt"})
        assert zoning_extractor.column_mapping["geometry"] == "geom_wkt"

    # --- num_zones validation ---

    @pytest.mark.parametrize("n", [1, 0, -3, 11, 99])
    def test_invalid_num_zones_raises(self, zoning_extractor, n):
        with pytest.raises(ValueError, match="num_zones"):
            zoning_extractor.setup_zoning_parameters(num_zones=n)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_valid_num_zones(self, zoning_extractor, n):
        zoning_extractor.setup_zoning_parameters(num_zones=n)
        assert zoning_extractor.zoning_params["num_zones"] == n

    def test_num_zones_must_be_int(self, zoning_extractor):
        with pytest.raises(ValueError, match="num_zones"):
            zoning_extractor.setup_zoning_parameters(num_zones=5.5)

    # --- postprocess validation ---

    def test_invalid_postprocess_raises(self, zoning_extractor):
        with pytest.raises(ValueError, match="postprocess"):
            zoning_extractor.setup_zoning_parameters(postprocess="bogus")

    @pytest.mark.parametrize("mode", ["stats", "stats_geo", "links", "file"])
    def test_valid_postprocess_modes(self, zoning_extractor, mode):
        # links mode requires directLinks=True (auto-corrected); file mode needs map_format + output_path
        if mode == "file":
            zoning_extractor.setup_zoning_parameters(postprocess=mode, map_format="png", output_path="/tmp")
        else:
            zoning_extractor.setup_zoning_parameters(postprocess=mode)
        assert zoning_extractor.zoning_params["postprocess"] == mode

    # --- map_format validation ---

    def test_invalid_map_format_raises(self, zoning_extractor):
        with pytest.raises(ValueError, match="map_format"):
            zoning_extractor.setup_zoning_parameters(postprocess="file", map_format="jpeg", output_path="/tmp")

    @pytest.mark.parametrize("fmt", ["png", "tiff.zip", "shp.zip"])
    def test_valid_map_formats(self, zoning_extractor, fmt):
        zoning_extractor.setup_zoning_parameters(postprocess="file", map_format=fmt, output_path="/tmp")
        assert zoning_extractor.zoning_params["map_format"] == fmt

    # --- cross-mode validation: links ---

    def test_postprocess_links_auto_sets_directLinks(self, zoning_extractor):
        """postprocess='links' should auto-correct directLinks to True with a warning."""
        zoning_extractor.setup_zoning_parameters(postprocess="links", directLinks=False)
        assert zoning_extractor.zoning_params["directLinks"] is True

    def test_postprocess_links_with_map_format_raises(self, zoning_extractor):
        """postprocess='links' must have map_format=None."""
        with pytest.raises(ValueError, match="map_format"):
            zoning_extractor.setup_zoning_parameters(postprocess="links", map_format="png")

    # --- cross-mode validation: file ---

    def test_postprocess_file_requires_map_format(self, zoning_extractor):
        with pytest.raises(ValueError, match="map_format"):
            zoning_extractor.setup_zoning_parameters(postprocess="file", map_format=None, output_path="/tmp")

    def test_postprocess_file_requires_output_path(self, zoning_extractor):
        with pytest.raises(ValueError, match="output_path"):
            zoning_extractor.setup_zoning_parameters(postprocess="file", map_format="png", output_path=None)

    # --- output formatting ---

    def test_output_mapping_applied(self, zoning_extractor):
        zoning_extractor.setup_zoning_parameters(output_mapping={"field_variability": "fv"})
        assert zoning_extractor.output_mapping == {"field_variability": "fv"}

    def test_exclude_columns_applied(self, zoning_extractor):
        zoning_extractor.setup_zoning_parameters(exclude_columns=["zone_1_area_percent"])
        assert zoning_extractor.exclude_columns == ["zone_1_area_percent"]


# ===================================================================
# require_zoning_params decorator
# ===================================================================


class TestRequireZoningParamsDecorator:
    """Methods guarded by @require_zoning_params raise when params not set.

    The decorator checks `not hasattr(self, 'zoning_params') or self.zoning_params is None`,
    so the bare fixture (zoning_params=None) triggers the guard.
    """

    def test_get_api_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.get_zoning_map_api(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "image_id": "a|b"}
            )

    def test_process_single_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.process_single_entity_zoning(
                {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "image_id": "a|b"}
            )

    def test_format_stats_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.format_zoning_stats_json({})

    def test_format_stats_geo_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.format_zoning_stats_geo_json({})

    def test_format_links_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.format_zoning_links_json({})

    def test_format_file_without_params_raises(self, zoning_extractor):
        assert zoning_extractor.zoning_params is None
        with pytest.raises(RuntimeError, match="No zoning parameters found"):
            zoning_extractor.format_zoning_file({"id": "x"}, "img|id", response=None)

    def test_safe_wrapper_returns_failure_without_params(self, zoning_extractor):
        """get_zoning_map_api_safe is NOT decorated; it catches the RuntimeError and wraps it."""
        assert zoning_extractor.zoning_params is None
        result = zoning_extractor.get_zoning_map_api_safe(
            {"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "image_id": "a|b"}
        )
        assert result["success"] is False
        assert "No zoning parameters found" in result["error"]
