"""
Tests for GDDExtractor processing pipeline:
    - process_single_entity_gdd() — single entity with retry
    - process_entity_gdd_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.gdd_functions import GDDExtractor
from tests.conftest import GDD_POINT_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_gdd()
# ===================================================================


class TestProcessSingleEntityGdd:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.extractors.gdd_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_normalize,
        configured_gdd_extractor,
        sample_gdd_entity,
        sample_gdd_response,
    ):
        """Happy path: API returns elements → result contains a non-empty DataFrame."""
        mock_retry.return_value = sample_gdd_response

        result = configured_gdd_extractor.process_single_entity_gdd(sample_gdd_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 3

    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(self, mock_retry, configured_gdd_extractor, sample_gdd_entity):
        """If API returns None or {}, processor returns 'No GDD data returned'."""
        mock_retry.return_value = None

        result = configured_gdd_extractor.process_single_entity_gdd(sample_gdd_entity)

        assert result["data"] is None
        assert "No GDD data returned" in result["error"]["message"]
        assert result["error"]["entity_id"] == "test_001"

    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_elements_returns_error(self, mock_retry, configured_gdd_extractor, sample_gdd_entity):
        """A response with elements=[] is also treated as 'No GDD data returned'."""
        mock_retry.return_value = {"elements": []}

        result = configured_gdd_extractor.process_single_entity_gdd(sample_gdd_entity)

        assert result["data"] is None
        assert "No GDD data returned" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, configured_gdd_extractor, sample_gdd_entity):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_gdd_extractor.process_single_entity_gdd(sample_gdd_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, configured_gdd_extractor, sample_gdd_entity):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = None  # forces 'no data' path

        configured_gdd_extractor.process_single_entity_gdd(sample_gdd_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.gdd_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_gdd_extractor,
        sample_gdd_response,
    ):
        """Notebook cell 20 passes a pd.Series — process_single_entity_gdd must accept it."""
        row = pd.Series({"id": "test_001", "name": "Test_Field", "geometry": GDD_POINT_WKT})
        mock_retry.return_value = sample_gdd_response

        result = configured_gdd_extractor.process_single_entity_gdd(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch("earthdaily.agriculture.extractors.gdd_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_dataframe_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_gdd_extractor,
        sample_gdd_response,
    ):
        """A 1-row DataFrame should be accepted (only first row used)."""
        df_in = pd.DataFrame([{"id": "test_001", "geometry": GDD_POINT_WKT}])
        mock_retry.return_value = sample_gdd_response

        result = configured_gdd_extractor.process_single_entity_gdd(df_in)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    def test_invalid_input_type_returns_error(self, configured_gdd_extractor):
        """A non-dict/Series/DataFrame input should produce an error result, not raise."""
        result = configured_gdd_extractor.process_single_entity_gdd("not a row")
        assert result["data"] is None
        assert "Invalid input" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.gdd_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.gdd_functions.retry_with_backoff_no_retry_on_400")
    def test_per_entity_dates_used(
        self,
        mock_retry,
        mock_normalize,
        configured_gdd_extractor,
        sample_gdd_response,
    ):
        """Notebook cell 22: entity carries start_date / end_date columns."""
        row = pd.Series(
            {
                "id": "test_override",
                "name": "Test_Field_Override",
                "geometry": GDD_POINT_WKT,
                "start_date": "2023-02-01",
                "end_date": "2023-04-01",
            }
        )
        mock_retry.return_value = sample_gdd_response

        result = configured_gdd_extractor.process_single_entity_gdd(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_entity_gdd_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.gdd_functions.export_results")
    @patch.object(GDDExtractor, "process_single_entity_gdd")
    @patch.object(GDDExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GDDExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_gdd_extractor,
        sample_gdd_entity_list,
    ):
        success_df = pd.DataFrame(
            [{"entity_id": "ent", "date": "2023-01-17", "daily_gdd": 13.35, "cumulated_gdd": 13.35}]
        )
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                entity_list=sample_gdd_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.extractors.gdd_functions.export_results")
    @patch.object(GDDExtractor, "process_single_entity_gdd")
    @patch.object(GDDExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GDDExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_gdd_extractor,
        sample_gdd_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                entity_list=sample_gdd_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.gdd_functions.export_results")
    @patch.object(GDDExtractor, "process_single_entity_gdd")
    @patch.object(GDDExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GDDExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_gdd_extractor,
        sample_gdd_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "date": "2023-01-17", "daily_gdd": 10}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                entity_list=sample_gdd_entity_list,
                max_workers=1,  # sequential for deterministic side_effect
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_gdd_extractor, sample_gdd_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                    entity_list=sample_gdd_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.extractors.gdd_functions.export_results")
    @patch.object(GDDExtractor, "process_single_entity_gdd")
    @patch.object(GDDExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GDDExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_gdd_extractor,
        sample_gdd_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                entity_list=sample_gdd_entity_list,
                skip_export=True,
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

    @patch("earthdaily.agriculture.extractors.gdd_functions.export_results")
    @patch.object(GDDExtractor, "process_single_entity_gdd")
    @patch.object(GDDExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GDDExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_excludes_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_gdd_extractor,
    ):
        """filter_type='exclude' with a match should drop those entities."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": GDD_POINT_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": GDD_POINT_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": GDD_POINT_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"daily_gdd": 5.0}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_gdd_extractor, "ensure_token_valid"):
            result = configured_gdd_extractor.process_entity_gdd_bulk_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
