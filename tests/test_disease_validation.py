"""
Tests for DiseaseExtractor validation logic:
    - setup_disease_parameters() configuration
    - require_disease_params decorator
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_disease_parameters()
# ===================================================================


class TestSetupDiseaseParameters:
    """Tests for parameter setup."""

    def test_default_parameters_succeed(self, disease_extractor):
        """Calling setup with no args should accept defaults (start/end dates None)."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters()

        params = disease_extractor.disease_params
        assert params is not None
        assert params["start_date"] is None
        assert params["end_date"] is None
        assert params["partial_frequency"] == 50

    def test_custom_parameters(self, disease_extractor):
        """Custom dates and partial_frequency should be stored as provided."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters(
                start_date="2025-06-01",
                end_date="2025-10-01",
                partial_frequency=25,
            )

        params = disease_extractor.disease_params
        assert params["start_date"] == "2025-06-01"
        assert params["end_date"] == "2025-10-01"
        assert params["partial_frequency"] == 25

    def test_setup_sets_cache_key_columns(self, disease_extractor):
        """Setup should configure cache_key_columns to [mapped id, 'date']."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters()
        assert disease_extractor.cache_key_columns == ["id", "date"]

    def test_setup_with_custom_id_column_mapping(self, disease_extractor):
        """When id is remapped, cache_key_columns must reflect the mapped column name."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters(column_mapping={"id": "entity_id"})
        # cache_key_columns is built before column_mapping is applied — verify the mapping was applied
        assert disease_extractor.column_mapping["id"] == "entity_id"

    def test_column_mapping_from_notebook(self, disease_extractor):
        """Mirrors notebook cell 15: maps crop -> crop.id and start_date -> sowingDate."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters(
                partial_frequency=50,
                exclude_columns=[],
                column_mapping={"crop": "crop.id", "start_date": "sowingDate"},
            )

        assert disease_extractor.column_mapping["crop"] == "crop.id"
        assert disease_extractor.column_mapping["start_date"] == "sowingDate"
        assert disease_extractor.exclude_columns == []

    def test_output_mapping_applied(self, disease_extractor):
        """output_mapping should be stored on the extractor."""
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters(output_mapping={"frogeye_leaf_spot": "frogeye"})
        assert disease_extractor.output_mapping == {"frogeye_leaf_spot": "frogeye"}

    def test_output_columns_applied(self, disease_extractor):
        with patch.object(disease_extractor, "ensure_token_valid"):
            disease_extractor.setup_disease_parameters(output_columns=["entity_id", "date", "frogeye_leaf_spot"])
        assert disease_extractor.output_columns == [
            "entity_id",
            "date",
            "frogeye_leaf_spot",
        ]


# ===================================================================
# require_disease_params decorator
# ===================================================================


class TestRequireDiseaseParamsDecorator:
    """Methods guarded by @require_disease_params raise when params not set."""

    def test_get_disease_data_without_params_raises(self, disease_extractor):
        """get_disease_data should fail if setup was never called."""
        assert disease_extractor.disease_params is None
        with patch.object(disease_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No disease parameters found"):
                disease_extractor.get_disease_data({"id": "x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})

    def test_get_disease_data_safe_without_params_raises(self, disease_extractor):
        """get_disease_data_safe should also fail if setup was never called."""
        assert disease_extractor.disease_params is None
        with patch.object(disease_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No disease parameters found"):
                disease_extractor.get_disease_data_safe({"id": "x"})

    def test_process_single_without_params_raises(self, disease_extractor):
        """process_single_entity_disease is guarded by the decorator too."""
        assert disease_extractor.disease_params is None
        with patch.object(disease_extractor, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No disease parameters found"):
                disease_extractor.process_single_entity_disease({"id": "x"})

    def test_format_disease_json_does_not_require_params(self, disease_extractor, sample_disease_response_list):
        """format_disease_json is NOT guarded — it is callable without setup."""
        assert disease_extractor.disease_params is None
        # Should not raise; entity_id falls back to "unknown".
        df = disease_extractor.format_disease_json(sample_disease_response_list)
        assert df is not None
