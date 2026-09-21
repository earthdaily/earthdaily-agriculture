"""
Tests for UserManager read-lifecycle validation logic:
    - setup_user_parameters() validation rules
    - require_user_params decorator
"""

from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# setup_user_parameters()
# ===================================================================


class TestSetupUserParameters:
    """Tests for parameter setup and validation."""

    def test_default_parameters_succeed(self, user_manager):
        """Defaults should pass validation (fields=None, partial_frequency=50)."""
        with patch.object(user_manager, "ensure_token_valid"):
            user_manager.setup_user_parameters()

        params = user_manager.user_params
        assert params is not None
        assert params["fields"] is None
        assert params["partial_frequency"] == 50

    def test_fields_list_accepted(self, user_manager):
        with patch.object(user_manager, "ensure_token_valid"):
            user_manager.setup_user_parameters(fields=["id", "login", "userType.code"])
        assert user_manager.user_params["fields"] == ["id", "login", "userType.code"]

    def test_fields_non_list_raises(self, user_manager):
        with patch.object(user_manager, "ensure_token_valid"):
            with pytest.raises(ValueError, match="fields"):
                user_manager.setup_user_parameters(fields="id")

    def test_setup_sets_cache_key_columns(self, user_manager):
        """cache_key_columns should be [mapped id]."""
        with patch.object(user_manager, "ensure_token_valid"):
            user_manager.setup_user_parameters()
        assert user_manager.cache_key_columns == ["id"]

    def test_column_mapping_applied(self, user_manager):
        with patch.object(user_manager, "ensure_token_valid"):
            user_manager.setup_user_parameters(column_mapping={"id": "user_id"})
        assert user_manager.column_mapping["id"] == "user_id"

    def test_column_mapping_reflected_in_cache_key(self, user_manager):
        """When id is remapped, the cache key column should follow."""
        with patch.object(user_manager, "ensure_token_valid"):
            user_manager.setup_user_parameters(column_mapping={"id": "user_id"})
        assert user_manager.cache_key_columns == ["user_id"]


# ===================================================================
# require_user_params decorator
# ===================================================================


class TestRequireUserParamsDecorator:
    """Methods guarded by @require_user_params raise when params not set."""

    def test_inner_bulk_without_params_raises(self, user_manager):
        """The decorated inner bulk method raises until setup_user_parameters() runs."""
        assert user_manager.user_params is None
        with patch.object(user_manager, "ensure_token_valid"):
            with pytest.raises(RuntimeError, match="No user parameters found"):
                user_manager._process_entity_user_bulk_parallel_inner(entity_list=pd.DataFrame([{"id": "usr_001"}]))

    def test_format_user_json_does_not_require_params(self, user_manager, sample_user_response):
        """format_user_json is NOT guarded — callable without setup."""
        assert user_manager.user_params is None
        df = user_manager.format_user_json(sample_user_response)
        assert df is not None
        assert len(df) == 1

    def test_process_single_does_not_require_params(self, user_manager):
        """
        process_single_entity_user is NOT decorated. With no params it still runs;
        a None API result surfaces as a 'User not found' error, not a params error.
        """
        assert user_manager.user_params is None
        with patch.object(user_manager, "get_user_by_id", return_value=None):
            result = user_manager.process_single_entity_user({"id": "usr_001"})
        assert result["data"] is None
        assert "User not found" in result["error"]["message"]
