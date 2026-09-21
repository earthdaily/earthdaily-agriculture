"""
Tests for BaresoilExtractor processing pipeline:
    - process_single_entity_baresoil() — single entity with retry
    - process_baresoil_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_baresoil_function import BaresoilExtractor
from tests.conftest import VALID_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_baresoil()
# ===================================================================


class TestProcessSingleEntityBaresoil:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.normalize_with_metadata")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_baresoil_extractor,
        sample_baresoil_entity,
    ):
        """Happy path: API returns data → result contains a DataFrame."""
        mock_wkt.return_value = VALID_WKT
        mock_retry.return_value = {
            "id": "entity_001",
            "data": {
                "year": 2025,
                "duration": 120,
                "startDay": 1,
                "startMonth": 4,
                "baresoilDays": 12,
                "baresoilPeriods": [],
            },
        }
        # normalize_with_metadata just passes the DataFrame through for tests
        mock_normalize.side_effect = lambda row, df: df

        result = configured_baresoil_extractor.process_single_entity_baresoil(sample_baresoil_entity)

        assert result["error"] is None
        assert result["data"] is not None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_error(
        self, mock_retry, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """If API returns no data, result should carry an error message."""
        mock_wkt.return_value = VALID_WKT
        mock_retry.return_value = {"id": "entity_001"}  # no "data" key

        result = configured_baresoil_extractor.process_single_entity_baresoil(sample_baresoil_entity)

        assert result["data"] is None
        assert result["error"] is not None
        assert "No baresoil results found" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity):
        """If geometry validation fails, should return error without calling API."""
        result = configured_baresoil_extractor.process_single_entity_baresoil(sample_baresoil_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "entity_001"

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self, mock_retry, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """If retry exhausts all attempts and raises, error should be captured."""
        mock_wkt.return_value = VALID_WKT
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_baresoil_extractor.process_single_entity_baresoil(sample_baresoil_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt")
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.retry_with_backoff_no_retry_on_400")
    def test_retry_is_called_with_correct_params(
        self, mock_retry, mock_wkt, configured_baresoil_extractor, sample_baresoil_entity
    ):
        """Verify retry_with_backoff is called with expected max_retries and delays."""
        mock_wkt.return_value = VALID_WKT
        mock_retry.return_value = {"id": "entity_001", "data": {}}  # empty data → returns error

        configured_baresoil_extractor.process_single_entity_baresoil(sample_baresoil_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0


# ===================================================================
# process_baresoil_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.export_results")
    @patch.object(BaresoilExtractor, "process_single_entity_baresoil")
    @patch.object(BaresoilExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(BaresoilExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_baresoil_extractor,
        sample_baresoil_entity_list,
    ):
        """All entities succeed → successful_calculations == total."""
        success_df = pd.DataFrame([{"entity_id": "ent", "baresoil_days": 5}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
            entity_list=sample_baresoil_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.export_results")
    @patch.object(BaresoilExtractor, "process_single_entity_baresoil")
    @patch.object(BaresoilExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(BaresoilExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_baresoil_extractor,
        sample_baresoil_entity_list,
    ):
        """All entities fail → failed_calculations == total."""
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
            entity_list=sample_baresoil_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.export_results")
    @patch.object(BaresoilExtractor, "process_single_entity_baresoil")
    @patch.object(BaresoilExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(BaresoilExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_baresoil_extractor,
        sample_baresoil_entity_list,
    ):
        """Mix of successes and failures → counts are correct."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"entity_id": "x", "baresoil_days": 7}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
            entity_list=sample_baresoil_entity_list,
            max_workers=1,  # sequential for deterministic side_effect
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_baresoil_extractor, sample_baresoil_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with pytest.raises(ValueError, match="merge_existing"):
            configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
                entity_list=sample_baresoil_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.export_results")
    @patch.object(BaresoilExtractor, "process_single_entity_baresoil")
    @patch.object(BaresoilExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(BaresoilExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_baresoil_extractor,
        sample_baresoil_entity_list,
    ):
        """Return dict should have all expected keys with correct types."""
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
            entity_list=sample_baresoil_entity_list,
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


class TestProcessSingleEntityBaresoilExtras:
    """Notebook-style row inputs and other canonical patterns."""

    @patch(
        "earthdaily.agriculture.processors.processor_baresoil_function.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_baresoil_function.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_baresoil_extractor,
        sample_baresoil_response_summary,
    ):
        """A pd.Series row should be accepted (matches the notebook pattern)."""
        row = pd.Series({"id": "entity_001", "geometry": VALID_WKT})
        mock_retry.return_value = sample_baresoil_response_summary

        result = configured_baresoil_extractor.process_single_entity_baresoil(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


class TestBulkBaresoilFilterType:
    """Bulk filter_type behaviour — parity with the other suites."""

    @patch("earthdaily.agriculture.processors.processor_baresoil_function.export_results")
    @patch.object(BaresoilExtractor, "process_single_entity_baresoil")
    @patch.object(BaresoilExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(BaresoilExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_baresoil_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": VALID_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": VALID_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": VALID_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_baresoil_extractor.process_baresoil_bulk_extraction_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
