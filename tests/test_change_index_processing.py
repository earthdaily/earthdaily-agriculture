"""
Tests for ChangeIndexExtractor processing pipeline:
    - process_single_entity_change_index() — single entity with retry logic
    - process_change_index_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_change_index_functions import ChangeIndexExtractor
from tests.conftest import CHANGE_INDEX_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_change_index()
# ===================================================================


class TestProcessSingleEntityChangeIndex:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.normalize_with_metadata")
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_change_index_extractor,
        sample_change_index_entity,
        sample_change_index_response,
    ):
        """Happy path: API returns notebook-shaped response → DataFrame returned."""
        mock_retry.return_value = sample_change_index_response
        mock_normalize.side_effect = lambda row, df: df

        result = configured_change_index_extractor.process_single_entity_change_index(sample_change_index_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty
        assert result["data"].iloc[0]["entity_id"] == "z361x33"
        assert result["data"].iloc[0]["ChangeIndex"] == 1.43

    @patch(
        "earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(
        self, mock_wkt, configured_change_index_extractor, sample_change_index_entity
    ):
        result = configured_change_index_extractor.process_single_entity_change_index(sample_change_index_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_missing_reference_date_returns_clean_error_without_api_call(
        self, mock_retry, mock_wkt, configured_change_index_extractor
    ):
        """A row missing reference_date should fail fast — no API call, no retries."""
        bad_row = {"id": "z361x33", "geometry": CHANGE_INDEX_WKT, "crop": "OTHERS"}

        result = configured_change_index_extractor.process_single_entity_change_index(bad_row)

        assert result["data"] is None
        assert "reference_date missing" in result["error"]["message"]
        # No API call should have been attempted
        mock_retry.assert_not_called()

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_invalid_reference_date_format_returns_clean_error(
        self, mock_retry, mock_wkt, configured_change_index_extractor
    ):
        bad_row = {
            "id": "z361x33",
            "geometry": CHANGE_INDEX_WKT,
            "crop": "OTHERS",
            "reference_date": "15/06/2025",
        }

        result = configured_change_index_extractor.process_single_entity_change_index(bad_row)

        assert result["data"] is None
        assert "invalid reference_date" in result["error"]["message"]
        mock_retry.assert_not_called()

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_change_index_extractor,
        sample_change_index_entity,
    ):
        """An empty response that flattens to an empty DF should yield an error result."""
        mock_retry.return_value = {}

        result = configured_change_index_extractor.process_single_entity_change_index(sample_change_index_entity)

        assert result["data"] is None
        assert "No change index results found" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_change_index_extractor,
        sample_change_index_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_change_index_extractor.process_single_entity_change_index(sample_change_index_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_is_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        configured_change_index_extractor,
        sample_change_index_entity,
    ):
        mock_retry.return_value = {}  # leads to error path but we only check the call

        configured_change_index_extractor.process_single_entity_change_index(sample_change_index_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0


# ===================================================================
# process_change_index_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.export_results")
    @patch.object(ChangeIndexExtractor, "process_single_entity_change_index")
    @patch.object(ChangeIndexExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ChangeIndexExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_change_index_extractor,
        sample_change_index_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "z361x33", "ChangeIndex": 1.43}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
            entity_list=sample_change_index_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.export_results")
    @patch.object(ChangeIndexExtractor, "process_single_entity_change_index")
    @patch.object(ChangeIndexExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ChangeIndexExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_change_index_extractor,
        sample_change_index_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
            entity_list=sample_change_index_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.export_results")
    @patch.object(ChangeIndexExtractor, "process_single_entity_change_index")
    @patch.object(ChangeIndexExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ChangeIndexExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_change_index_extractor,
        sample_change_index_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"entity_id": "x", "ChangeIndex": 1.0}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
            entity_list=sample_change_index_entity_list,
            max_workers=1,  # sequential for deterministic side_effect
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(
        self, configured_change_index_extractor, sample_change_index_entity_list
    ):
        with pytest.raises(ValueError, match="merge_existing"):
            configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
                entity_list=sample_change_index_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.export_results")
    @patch.object(ChangeIndexExtractor, "process_single_entity_change_index")
    @patch.object(ChangeIndexExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ChangeIndexExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_change_index_extractor,
        sample_change_index_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
            entity_list=sample_change_index_entity_list,
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


# ===================================================================
# Additional canonical patterns (parity with newer suites)
# ===================================================================


class TestProcessSingleEntityChangeIndexExtras:
    """Notebook-style row inputs and other canonical patterns."""

    @patch(
        "earthdaily.agriculture.processors.processor_change_index_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_change_index_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_change_index_extractor,
        sample_change_index_response,
    ):
        """A pd.Series row should be accepted (matches the notebook pattern)."""
        row = pd.Series(
            {
                "id": "z361x33",
                "geometry": CHANGE_INDEX_WKT,
                "crop": "OTHERS",
                "sowing_date": "2025-04-01",
                "reference_date": "2025-06-15",
            }
        )
        mock_retry.return_value = sample_change_index_response

        result = configured_change_index_extractor.process_single_entity_change_index(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


class TestBulkChangeIndexFilterType:
    """Bulk filter_type behaviour — parity with the other suites."""

    @patch("earthdaily.agriculture.processors.processor_change_index_functions.export_results")
    @patch.object(ChangeIndexExtractor, "process_single_entity_change_index")
    @patch.object(ChangeIndexExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ChangeIndexExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_change_index_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities."""
        entity_list = pd.DataFrame(
            [
                {
                    "id": "z361x33",
                    "geometry": CHANGE_INDEX_WKT,
                    "crop": "OTHERS",
                    "sowing_date": "2025-04-01",
                    "reference_date": "2025-06-15",
                    "tag": "skip",
                },
                {
                    "id": "7e5gwem",
                    "geometry": CHANGE_INDEX_WKT,
                    "crop": "OTHERS",
                    "sowing_date": "2025-04-01",
                    "reference_date": "2025-06-15",
                    "tag": "keep",
                },
                {
                    "id": "x3vbx3q",
                    "geometry": CHANGE_INDEX_WKT,
                    "crop": "OTHERS",
                    "sowing_date": "2025-04-01",
                    "reference_date": "2025-06-15",
                    "tag": "skip",
                },
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_change_index_extractor.process_change_index_bulk_extraction_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
