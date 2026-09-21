"""
Tests for UserManager read-processing pipeline:
    - process_single_entity_user() — single entity with retry
    - process_entity_user_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.services.user_management import UserManager

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_user()
# ===================================================================


class TestProcessSingleEntityUser:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.services.user_management.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self, mock_retry, mock_normalize, configured_user_manager, sample_user_entity, sample_user_response
    ):
        """Happy path: API returns a user record → result contains a non-empty DataFrame."""
        mock_retry.return_value = sample_user_response

        result = configured_user_manager.process_single_entity_user(sample_user_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 1

    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_user_not_found_returns_error(self, mock_retry, configured_user_manager, sample_user_entity):
        """If get_user_by_id returns None (404), processor returns 'User not found'."""
        mock_retry.return_value = None

        result = configured_user_manager.process_single_entity_user(sample_user_entity)

        assert result["data"] is None
        assert "User not found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "usr_001"

    def test_missing_user_id_returns_error(self, configured_user_manager):
        """An entity row with an empty id is rejected before any API call."""
        result = configured_user_manager.process_single_entity_user({"id": ""})
        assert result["data"] is None
        assert "No valid user id" in result["error"]["message"]

    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, configured_user_manager, sample_user_entity):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_user_manager.process_single_entity_user(sample_user_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, configured_user_manager, sample_user_entity):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = None  # forces 'not found' path

        configured_user_manager.process_single_entity_user(sample_user_entity)

        call = mock_retry.call_args
        assert call.kwargs["max_retries"] == 5
        assert call.kwargs["base_delay"] == 1.0
        assert call.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.services.user_management.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self, mock_retry, mock_normalize, configured_user_manager, sample_user_response
    ):
        """A pd.Series row must be accepted."""
        row = pd.Series({"id": "usr_001", "name": "Jane"})
        mock_retry.return_value = sample_user_response

        result = configured_user_manager.process_single_entity_user(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch("earthdaily.agriculture.services.user_management.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.services.user_management.retry_with_backoff_no_retry_on_400")
    def test_dataframe_row_supported(self, mock_retry, mock_normalize, configured_user_manager, sample_user_response):
        """A 1-row DataFrame should be accepted (only first row used)."""
        df_in = pd.DataFrame([{"id": "usr_001"}])
        mock_retry.return_value = sample_user_response

        result = configured_user_manager.process_single_entity_user(df_in)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    def test_invalid_input_type_returns_error(self, configured_user_manager):
        """A non-dict/Series/DataFrame input should produce an error result, not raise."""
        result = configured_user_manager.process_single_entity_user("not a row")
        assert result["data"] is None
        assert "Invalid input" in result["error"]["message"]


# ===================================================================
# process_entity_user_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkUserReadParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.services.user_management.export_partial_results")
    @patch.object(UserManager, "process_single_entity_user")
    @patch.object(UserManager, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(UserManager, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self, mock_merge, mock_finalize, mock_single, mock_export, configured_user_manager, sample_user_entity_list
    ):
        success_df = pd.DataFrame([{"id": "usr", "login": "x"}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.process_entity_user_bulk_parallel(
                entity_list=sample_user_entity_list, max_workers=2, skip_export=True
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.services.user_management.export_partial_results")
    @patch.object(UserManager, "process_single_entity_user")
    @patch.object(UserManager, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(UserManager, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self, mock_merge, mock_finalize, mock_single, mock_export, configured_user_manager, sample_user_entity_list
    ):
        mock_single.return_value = {"data": None, "error": {"message": "not found", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.process_entity_user_bulk_parallel(
                entity_list=sample_user_entity_list, max_workers=2, skip_export=True
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.services.user_management.export_partial_results")
    @patch.object(UserManager, "process_single_entity_user")
    @patch.object(UserManager, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(UserManager, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self, mock_merge, mock_finalize, mock_single, mock_export, configured_user_manager, sample_user_entity_list
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"id": "x"}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.process_entity_user_bulk_parallel(
                entity_list=sample_user_entity_list, max_workers=1, skip_export=True
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_user_manager, sample_user_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with patch.object(configured_user_manager, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_user_manager.process_entity_user_bulk_parallel(
                    entity_list=sample_user_entity_list, merge_existing="invalid_mode", skip_export=True
                )

    @patch("earthdaily.agriculture.services.user_management.export_partial_results")
    @patch.object(UserManager, "process_single_entity_user")
    @patch.object(UserManager, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(UserManager, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self, mock_merge, mock_finalize, mock_single, mock_export, configured_user_manager, sample_user_entity_list
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.process_entity_user_bulk_parallel(
                entity_list=sample_user_entity_list, skip_export=True
            )

        expected_keys = {
            "results_df",
            "global_errors",
            "total_entities",
            "total_calculations",
            "successful_calculations",
            "failed_calculations",
            "failed_ids",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["results_df"], pd.DataFrame)
        assert isinstance(result["global_errors"], list)
        assert isinstance(result["failed_ids"], list)

    @patch("earthdaily.agriculture.services.user_management.export_partial_results")
    @patch.object(UserManager, "process_single_entity_user")
    @patch.object(UserManager, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(UserManager, "_merge_with_skipped_entities")
    def test_bulk_filter_excludes_matching_entities(
        self, mock_merge, mock_finalize, mock_single, mock_export, configured_user_manager
    ):
        """filter_type='exclude' with a match should drop those entities."""
        entity_list = pd.DataFrame(
            [
                {"id": "usr_001", "tag": "skip"},
                {"id": "usr_002", "tag": "keep"},
                {"id": "usr_003", "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"id": "x"}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.process_entity_user_bulk_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
