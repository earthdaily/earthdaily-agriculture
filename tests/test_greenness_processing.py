"""
Tests for GreennessExtractor processing pipeline:
    - process_single_entity_greenness() — single entity with retry
    - process_greenness_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_greenness_functions import GreennessExtractor
from tests.conftest import VALID_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_greenness()
# ===================================================================


class TestProcessSingleEntityGreenness:
    """Tests for the single-entity processing pipeline."""

    @patch(
        "earthdaily.agriculture.processors.processor_greenness_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_extractor,
        sample_entity,
    ):
        """Happy path: API returns data → result contains a non-empty DataFrame."""
        mock_retry.return_value = {
            "id": "entity_001",
            "data": {"greenness_score": 0.9, "status": "GREEN"},
        }

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(sample_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty
        assert result["data"].iloc[0]["status"] == "GREEN"

    @patch(
        "earthdaily.agriculture.processors.processor_greenness_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_response_id_filled_when_missing(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_extractor,
        sample_entity,
    ):
        """When publish_af=False the API may not echo `id`; processor backfills it."""
        mock_retry.return_value = {"data": {"greenness_score": 0.9, "status": "GREEN"}}

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(sample_entity)

        assert result["error"] is None
        assert result["data"].iloc[0]["entity_id"] == "entity_001"

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_extractor,
        sample_entity,
    ):
        """If API returns a response with no usable 'data', processor returns an error."""
        mock_retry.return_value = {"id": "entity_001"}  # no 'data' key

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(sample_entity)

        assert result["data"] is None
        assert result["error"] is not None
        assert "No greenness results found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "entity_001"

    @patch(
        "earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_extractor, sample_entity):
        """Geometry validation failure should short-circuit before any API call."""
        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(sample_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "entity_001"

    def test_invalid_crop_returns_error(self, configured_extractor):
        """Invalid crop should return error, not raise."""
        entity = {"id": "ent_bad", "geometry": VALID_WKT, "crop": "BANANA"}

        with patch(
            "earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x
        ):
            with patch.object(configured_extractor, "ensure_token_valid"):
                result = configured_extractor.process_single_entity_greenness(entity)

        assert result["data"] is None
        assert "Invalid crop" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_extractor,
        sample_entity,
    ):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(sample_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        configured_extractor,
        sample_entity,
    ):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = {"id": "entity_001"}  # no data → 'no results' path

        with patch.object(configured_extractor, "ensure_token_valid"):
            configured_extractor.process_single_entity_greenness(sample_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.processors.processor_greenness_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_greenness_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_extractor,
    ):
        """Notebook cell 23 passes a pd.Series — process_single_entity_greenness must accept it."""
        row = pd.Series({"id": "z361x33", "geometry": VALID_WKT, "crop": "OTHERS"})
        mock_retry.return_value = {
            "id": "z361x33",
            "data": {"greenness_score": 0.85, "status": "GREEN"},
        }

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_single_entity_greenness(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_greenness_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
        sample_entity_list,
    ):
        """All entities succeed → successful_calculations == total."""
        success_df = pd.DataFrame([{"entity_id": "ent", "greenness_score": 0.9, "status": "GREEN"}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=sample_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
        sample_entity_list,
    ):
        """All entities fail → failed_calculations == total."""
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=sample_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
        sample_entity_list,
    ):
        """Mix of successes and failures → counts are correct."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "greenness_score": 0.7}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=sample_entity_list,
                max_workers=1,  # sequential for deterministic side_effect ordering
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_extractor, sample_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with patch.object(configured_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_extractor.process_greenness_bulk_extraction_parallel(
                    entity_list=sample_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
        sample_entity_list,
    ):
        """Return dict should have all expected keys with correct types."""
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=sample_entity_list,
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

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_include_keeps_only_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
    ):
        """Notebook cell 25 uses filter_type='include' — only matching entities are processed."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": VALID_WKT, "crop": "OTHERS"},
                {"id": "ent_002", "geometry": VALID_WKT, "crop": "CORN"},
                {"id": "ent_003", "geometry": VALID_WKT, "crop": "OTHERS"},
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "status": "GREEN"}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="OTHERS",
                filter_type="include",
                skip_export=True,
            )

        # Two OTHERS entities included, CORN entity filtered out
        assert result["total_calculations"] == 2
        assert result["total_entities"] == 3

    @patch("earthdaily.agriculture.processors.processor_greenness_functions.export_results")
    @patch.object(GreennessExtractor, "process_single_entity_greenness")
    @patch.object(GreennessExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(GreennessExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_extractor,
    ):
        """The exclude variant — matching entities are dropped."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": VALID_WKT, "crop": "OTHERS"},
                {"id": "ent_002", "geometry": VALID_WKT, "crop": "CORN"},
                {"id": "ent_003", "geometry": VALID_WKT, "crop": "OTHERS"},
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "status": "GREEN"}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_extractor, "ensure_token_valid"):
            result = configured_extractor.process_greenness_bulk_extraction_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="OTHERS",
                filter_type="exclude",
                skip_export=True,
            )

        # Two OTHERS entities excluded, only CORN entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
