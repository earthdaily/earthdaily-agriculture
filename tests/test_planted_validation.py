"""
Tests for PlantedExtractor validation logic:
    - setup_planted_parameters() input validation
    - require_planted_params decorator
"""

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_planted_parameters()
# ===================================================================


class TestSetupPlantedParameters:
    def test_default_parameters_succeed(self, planted_extractor):
        planted_extractor.setup_planted_parameters()
        params = planted_extractor.planted_params
        assert params is not None
        assert params["processor_mode"] == "PLANTED_AREA"
        assert params["emergence_date"] is None
        assert params["threshold"] == 120
        assert params["control_threshold"] == 4
        assert params["publish_af"] is False
        assert params["partial_frequency"] == 50

    def test_custom_parameters_from_notebook(self, planted_extractor):
        """Mirrors notebook cell 13 (PLANTED_AREA mode with explicit emergence_date)."""
        planted_extractor.setup_planted_parameters(
            processor_mode="PLANTED_AREA",
            emergence_date="2025-04-01",
            threshold=30,
            control_threshold=4,
        )
        params = planted_extractor.planted_params
        assert params["processor_mode"] == "PLANTED_AREA"
        assert params["emergence_date"] == "2025-04-01"
        assert params["threshold"] == 30

    def test_setup_sets_cache_key_columns(self, planted_extractor):
        planted_extractor.setup_planted_parameters()
        assert planted_extractor.cache_key_columns == ["id"]

    def test_setup_with_custom_id_column_mapping(self, planted_extractor):
        planted_extractor.setup_planted_parameters(column_mapping={"id": "entity_id"})
        assert planted_extractor.column_mapping["id"] == "entity_id"

    def test_emergence_date_optional_at_setup(self, planted_extractor):
        """Notebook cell 25: setup without emergence_date is allowed (per-entity required)."""
        planted_extractor.setup_planted_parameters(emergence_date=None)
        assert planted_extractor.planted_params["emergence_date"] is None

    # --- processor_mode ---

    @pytest.mark.parametrize("mode", ["PLANTED_AREA", "CONTROL"])
    def test_valid_processor_modes(self, planted_extractor, mode):
        planted_extractor.setup_planted_parameters(processor_mode=mode)
        assert planted_extractor.planted_params["processor_mode"] == mode

    def test_invalid_processor_mode_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="processor_mode"):
            planted_extractor.setup_planted_parameters(processor_mode="BOGUS")

    # --- emergence_date format ---

    def test_invalid_emergence_date_format_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="emergence_date"):
            planted_extractor.setup_planted_parameters(emergence_date="01/04/2025")

    def test_valid_emergence_date_format(self, planted_extractor):
        planted_extractor.setup_planted_parameters(emergence_date="2025-04-01")
        assert planted_extractor.planted_params["emergence_date"] == "2025-04-01"

    # --- threshold ---

    def test_threshold_zero_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="threshold"):
            planted_extractor.setup_planted_parameters(threshold=0)

    def test_threshold_negative_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="threshold"):
            planted_extractor.setup_planted_parameters(threshold=-5)

    def test_threshold_non_int_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="threshold"):
            planted_extractor.setup_planted_parameters(threshold=30.5)

    @pytest.mark.parametrize("t", [1, 30, 120])
    def test_valid_thresholds(self, planted_extractor, t):
        planted_extractor.setup_planted_parameters(threshold=t)
        assert planted_extractor.planted_params["threshold"] == t

    # --- control_threshold ---

    def test_control_threshold_negative_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="control_threshold"):
            planted_extractor.setup_planted_parameters(control_threshold=-1)

    def test_control_threshold_non_numeric_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="control_threshold"):
            planted_extractor.setup_planted_parameters(control_threshold="high")

    @pytest.mark.parametrize("ct", [0, 4, 4.5])
    def test_valid_control_thresholds(self, planted_extractor, ct):
        planted_extractor.setup_planted_parameters(control_threshold=ct)
        assert planted_extractor.planted_params["control_threshold"] == ct

    # --- publish_af ---

    def test_publish_af_non_bool_raises(self, planted_extractor):
        with pytest.raises(ValueError, match="publish_af"):
            planted_extractor.setup_planted_parameters(publish_af="yes")

    @pytest.mark.parametrize("flag", [True, False])
    def test_publish_af_bool_accepted(self, planted_extractor, flag):
        planted_extractor.setup_planted_parameters(publish_af=flag)
        assert planted_extractor.planted_params["publish_af"] is flag


# ===================================================================
# require_planted_params decorator
# ===================================================================


class TestRequirePlantedParamsDecorator:
    """Methods guarded by @require_planted_params raise when params not set."""

    def test_get_api_without_params_raises(self, planted_extractor):
        assert planted_extractor.planted_params is None
        with pytest.raises(RuntimeError, match="No planted area parameters found"):
            planted_extractor.get_planted_api({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})

    def test_format_without_params_raises(self, planted_extractor):
        assert planted_extractor.planted_params is None
        with pytest.raises(RuntimeError, match="No planted area parameters found"):
            planted_extractor.format_planted_json({})

    def test_process_single_without_params_raises(self, planted_extractor):
        assert planted_extractor.planted_params is None
        with pytest.raises(RuntimeError, match="No planted area parameters found"):
            planted_extractor.process_single_entity_planted({"id": "x"})

    def test_safe_wrapper_does_not_require_params(self, planted_extractor):
        """get_planted_api_safe is NOT guarded; returns failure instead of raising."""
        assert planted_extractor.planted_params is None
        result = planted_extractor.get_planted_api_safe({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})
        assert result["success"] is False
        assert "No planted area parameters" in result["error"]
