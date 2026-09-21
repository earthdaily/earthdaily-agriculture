"""
Tests for UserManager.format_user_json():
    - One-row DataFrame construction from a single user record
    - Nested object flattening (userType.code, country.code)
    - `fields` projection
    - Edge cases (None / empty response)

Note: unlike the analytics extractors, format_user_json does NOT stamp
entity_id — entity metadata is attached later by normalize_with_metadata
inside process_single_entity_user.
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_user_json() — happy path
# ===================================================================


class TestFormatUserJsonHappyPath:
    """get_user_by_id returns a single user dict (possibly with nested objects)."""

    def test_creates_one_row(self, configured_user_manager, sample_user_response):
        df = configured_user_manager.format_user_json(sample_user_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_flattens_nested_objects(self, configured_user_manager, sample_user_response):
        df = configured_user_manager.format_user_json(sample_user_response)
        assert "userType.code" in df.columns
        assert "country.code" in df.columns
        assert df.iloc[0]["userType.code"] == "GROWER"
        assert df.iloc[0]["country.code"] == "US"

    def test_scalar_fields_preserved(self, configured_user_manager, sample_user_response):
        df = configured_user_manager.format_user_json(sample_user_response)
        first = df.iloc[0]
        assert first["id"] == "usr_001"
        assert first["login"] == "jdoe"
        assert first["email"] == "jdoe@example.com"

    def test_no_entity_id_column_added(self, configured_user_manager, sample_user_response):
        """format_user_json does not stamp entity_id (done downstream)."""
        df = configured_user_manager.format_user_json(sample_user_response, entity_data={"id": "usr_001"})
        assert "entity_id" not in df.columns


# ===================================================================
# format_user_json() — fields projection
# ===================================================================


class TestFormatUserJsonFieldsProjection:
    """When user_params['fields'] is set, the output keeps only those columns."""

    def test_fields_subset_kept(self, user_manager, sample_user_response):
        user_manager.user_params = {"fields": ["id", "login", "userType.code"], "partial_frequency": 50}
        df = user_manager.format_user_json(sample_user_response)
        assert set(df.columns) == {"id", "login", "userType.code"}

    def test_unknown_fields_ignored(self, user_manager, sample_user_response):
        """Requested columns that don't exist are silently dropped from the projection."""
        user_manager.user_params = {"fields": ["id", "does_not_exist"], "partial_frequency": 50}
        df = user_manager.format_user_json(sample_user_response)
        assert list(df.columns) == ["id"]

    def test_no_fields_keeps_all_columns(self, configured_user_manager, sample_user_response):
        """fields=None keeps every flattened column from the record."""
        df = configured_user_manager.format_user_json(sample_user_response)
        assert {"id", "login", "email", "firstname", "lastname", "companyName"}.issubset(set(df.columns))


# ===================================================================
# format_user_json() — edge cases
# ===================================================================


class TestFormatUserJsonEdgeCases:
    """Empty / missing records return None (the 'no data' sentinel)."""

    def test_none_response_returns_none(self, configured_user_manager):
        assert configured_user_manager.format_user_json(None) is None

    def test_empty_dict_returns_none(self, configured_user_manager):
        assert configured_user_manager.format_user_json({}) is None

    def test_single_field_record(self, configured_user_manager):
        df = configured_user_manager.format_user_json({"id": "usr_009"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["id"] == "usr_009"
