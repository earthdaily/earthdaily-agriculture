"""
Tests for InSeasonMonitoringExtractor processing pipeline:
    - process_single_entity_inseason_monitoring() — single entity with retry
    - process_inseason_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_inseason_monitoring_functions import (
    InSeasonMonitoringExtractor,
)
from tests.conftest import ISM_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_inseason_monitoring()
# ===================================================================


class TestProcessSingleEntityInseasonMonitoring:
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_ism_extractor,
        sample_ism_entity,
        sample_ism_response,
    ):
        """Happy path: API returns 3 records → result contains a 3-row DataFrame."""
        mock_retry.return_value = sample_ism_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 3
        assert result["data"].iloc[0]["emergence_status"] == "CONFIRMED"

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_response_id_filled_when_missing(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_ism_extractor,
        sample_ism_entity,
    ):
        """Processor backfills `id` if the API didn't echo it back."""
        # Response with no top-level 'id'
        mock_retry.return_value = {
            "data": [{"Season": "2025", "RequestDate": "2025-04-15", "VegetationIndexValue": 0.32}]
        }

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        assert result["error"] is None
        assert result["data"].iloc[0]["entity_id"] == "z361x33"

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_empty_data_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_ism_extractor,
        sample_ism_entity,
        sample_ism_response_empty,
    ):
        """Response with empty data list → 'No in-season monitoring results found'."""
        mock_retry.return_value = sample_ism_response_empty

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        assert result["data"] is None
        assert "No in-season monitoring results found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_ism_extractor, sample_ism_entity):
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    def test_invalid_crop_returns_error(self, configured_ism_extractor):
        entity = {"id": "ent_bad", "geometry": ISM_WKT, "crop": "BANANA"}

        with patch(
            "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
            side_effect=lambda x: x,
        ):
            with patch.object(configured_ism_extractor, "ensure_token_valid"):
                result = configured_ism_extractor.process_single_entity_inseason_monitoring(entity)

        assert result["data"] is None
        assert "Invalid crop" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_ism_extractor,
        sample_ism_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        configured_ism_extractor,
        sample_ism_entity,
    ):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = {"id": "z361x33", "data": []}

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            configured_ism_extractor.process_single_entity_inseason_monitoring(sample_ism_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.validate_wkt",
        side_effect=lambda x: x,
    )
    @patch(
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions.retry_with_backoff_no_retry_on_400"
    )
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_ism_extractor,
        sample_ism_response,
    ):
        """Notebook cell 23 passes a pd.Series — process_single_entity must accept it."""
        row = pd.Series({"id": "z361x33", "geometry": ISM_WKT, "crop": "OTHERS"})
        mock_retry.return_value = sample_ism_response

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_single_entity_inseason_monitoring(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_inseason_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.export_results")
    @patch.object(InSeasonMonitoringExtractor, "process_single_entity_inseason_monitoring")
    @patch.object(InSeasonMonitoringExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InSeasonMonitoringExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_ism_extractor,
        sample_ism_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "season": "2025", "vegetation_index": 0.5}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                entity_list=sample_ism_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.export_results")
    @patch.object(InSeasonMonitoringExtractor, "process_single_entity_inseason_monitoring")
    @patch.object(InSeasonMonitoringExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InSeasonMonitoringExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_ism_extractor,
        sample_ism_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                entity_list=sample_ism_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.export_results")
    @patch.object(InSeasonMonitoringExtractor, "process_single_entity_inseason_monitoring")
    @patch.object(InSeasonMonitoringExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InSeasonMonitoringExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_ism_extractor,
        sample_ism_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "vegetation_index": 0.5}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                entity_list=sample_ism_entity_list,
                max_workers=1,  # sequential for deterministic side_effect ordering
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_ism_extractor, sample_ism_entity_list):
        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                    entity_list=sample_ism_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.export_results")
    @patch.object(InSeasonMonitoringExtractor, "process_single_entity_inseason_monitoring")
    @patch.object(InSeasonMonitoringExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InSeasonMonitoringExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_ism_extractor,
        sample_ism_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                entity_list=sample_ism_entity_list,
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

    @patch("earthdaily.agriculture.processors.processor_inseason_monitoring_functions.export_results")
    @patch.object(InSeasonMonitoringExtractor, "process_single_entity_inseason_monitoring")
    @patch.object(InSeasonMonitoringExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InSeasonMonitoringExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_ism_extractor,
    ):
        """Notebook cell 25 uses filter_type='exclude' on crop=CORN."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": ISM_WKT, "crop": "OTHERS"},
                {"id": "ent_002", "geometry": ISM_WKT, "crop": "CORN"},
                {"id": "ent_003", "geometry": ISM_WKT, "crop": "CORN"},
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "vegetation_index": 0.5}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_ism_extractor, "ensure_token_valid"):
            result = configured_ism_extractor.process_inseason_bulk_extraction_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="CORN",
                filter_type="exclude",
                skip_export=True,
            )

        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
