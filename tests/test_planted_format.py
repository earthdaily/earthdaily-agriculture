"""
Tests for PlantedExtractor.format_planted_json():
    - PLANTED_AREA mode: planted_area_m2 + planted_percentage
    - CONTROL mode: difference + control_threshold + control_result
    - Edge cases (missing keys, non-dict, unknown processor_mode)
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_planted_json() — PLANTED_AREA mode
# ===================================================================


class TestFormatPlantedJsonPlantedAreaMode:
    def test_planted_area_creates_one_row(self, configured_planted_extractor, sample_planted_response):
        df = configured_planted_extractor.format_planted_json(sample_planted_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_planted_area_columns_present(self, configured_planted_extractor, sample_planted_response):
        df = configured_planted_extractor.format_planted_json(sample_planted_response)
        expected = {"entity_id", "planted_area_m2", "planted_percentage"}
        assert expected.issubset(set(df.columns))

    def test_planted_area_values_preserved(self, configured_planted_extractor, sample_planted_response):
        df = configured_planted_extractor.format_planted_json(sample_planted_response)
        row = df.iloc[0]
        assert row["planted_area_m2"] == 14523.5
        assert row["planted_percentage"] == 92.4

    def test_planted_area_missing_keys_returns_empty(self, configured_planted_extractor):
        """If required keys are missing, no row is appended → empty DataFrame."""
        df = configured_planted_extractor.format_planted_json({"id": "x"})
        assert df.empty

    def test_entity_id_propagated_from_response(self, configured_planted_extractor, sample_planted_response):
        """The processor stamps id onto the response before calling format; without it,
        entity_id is None."""
        sample_planted_response["id"] = "ent_via_processor"
        df = configured_planted_extractor.format_planted_json(sample_planted_response)
        assert df.iloc[0]["entity_id"] == "ent_via_processor"


# ===================================================================
# format_planted_json() — CONTROL mode
# ===================================================================


class TestFormatPlantedJsonControlMode:
    def test_control_creates_one_row(self, configured_planted_extractor, sample_planted_control_response):
        configured_planted_extractor.planted_params["processor_mode"] = "CONTROL"
        df = configured_planted_extractor.format_planted_json(sample_planted_control_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_control_columns_present(self, configured_planted_extractor, sample_planted_control_response):
        configured_planted_extractor.planted_params["processor_mode"] = "CONTROL"
        df = configured_planted_extractor.format_planted_json(sample_planted_control_response)
        expected = {"entity_id", "difference", "control_threshold", "control_result"}
        assert expected.issubset(set(df.columns))

    def test_control_values_preserved(self, configured_planted_extractor, sample_planted_control_response):
        configured_planted_extractor.planted_params["processor_mode"] = "CONTROL"
        df = configured_planted_extractor.format_planted_json(sample_planted_control_response)
        row = df.iloc[0]
        assert row["difference"] == 0.025
        assert row["control_threshold"] == 0.04
        # control_result wraps via pandas; compare with `bool(...)` to avoid np.True_ issue
        assert bool(row["control_result"]) is True

    def test_control_missing_keys_returns_empty(self, configured_planted_extractor):
        configured_planted_extractor.planted_params["processor_mode"] = "CONTROL"
        df = configured_planted_extractor.format_planted_json({"id": "x"})
        assert df.empty

    def test_control_uses_params_kwarg_override(self, configured_planted_extractor, sample_planted_control_response):
        """The configured fixture is PLANTED_AREA — params kwarg should flip to CONTROL."""
        df = configured_planted_extractor.format_planted_json(
            sample_planted_control_response,
            params={"processor_mode": "CONTROL"},
        )
        assert "difference" in df.columns


# ===================================================================
# format_planted_json() — edge cases
# ===================================================================


class TestFormatPlantedJsonEdgeCases:
    def test_non_dict_response_raises(self, configured_planted_extractor):
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_planted_extractor.format_planted_json([{"id": "x"}])

    def test_unknown_processor_mode_returns_empty(self, configured_planted_extractor, sample_planted_response):
        configured_planted_extractor.planted_params["processor_mode"] = "BOGUS"
        df = configured_planted_extractor.format_planted_json(sample_planted_response)
        assert df.empty

    def test_missing_id_in_response_propagates_as_none(self, configured_planted_extractor):
        response = {"planted_area": 100.0, "planted_percentage": 50.0}
        df = configured_planted_extractor.format_planted_json(response)
        assert df.iloc[0]["entity_id"] is None
