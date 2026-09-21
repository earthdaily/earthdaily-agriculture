"""
Tests for PlantedExtractor processing pipeline:
    - process_single_entity_planted() — single entity with retry
    - process_planted_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_plantedarea_functions import PlantedExtractor
from tests.conftest import PLANTED_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_planted()
# ===================================================================


class TestProcessSingleEntityPlanted:
    @patch(
        "earthdaily.agriculture.processors.processor_plantedarea_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_planted_extractor,
        sample_planted_entity,
        sample_planted_response,
    ):
        mock_retry.return_value = sample_planted_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert result["data"].iloc[0]["planted_percentage"] == 92.4

    @patch(
        "earthdaily.agriculture.processors.processor_plantedarea_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_processor_stamps_id_onto_response(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_planted_extractor,
        sample_planted_entity,
        sample_planted_response,
    ):
        """The API response doesn't include `id`; processor stamps the entity id before formatting."""
        mock_retry.return_value = dict(sample_planted_response)  # no 'id' key

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        assert result["error"] is None
        assert result["data"].iloc[0]["entity_id"] == "z361x33"

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_planted_extractor,
        sample_planted_entity,
    ):
        """If the API response is missing the planted-area keys, processor returns 'No planted area results'."""
        mock_retry.return_value = {}

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        assert result["data"] is None
        assert "No planted area results found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch(
        "earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_planted_extractor, sample_planted_entity):
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_planted_extractor,
        sample_planted_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        configured_planted_extractor,
        sample_planted_entity,
    ):
        mock_retry.return_value = {}

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            configured_planted_extractor.process_single_entity_planted(sample_planted_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.processors.processor_plantedarea_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_planted_extractor,
        sample_planted_response,
    ):
        """Notebook cell 23 passes a pd.Series — process_single_entity_planted must accept it."""
        row = pd.Series(
            {
                "id": "z361x33",
                "geometry": PLANTED_WKT,
                "crop": "OTHERS",
                "emergence_date": "2025-04-02",
            }
        )
        mock_retry.return_value = sample_planted_response

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_single_entity_planted(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_planted_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.export_results")
    @patch.object(PlantedExtractor, "process_single_entity_planted")
    @patch.object(PlantedExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(PlantedExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_planted_extractor,
        sample_planted_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "planted_area_m2": 14523.5, "planted_percentage": 92.4}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_planted_bulk_extraction_parallel(
                entity_list=sample_planted_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.export_results")
    @patch.object(PlantedExtractor, "process_single_entity_planted")
    @patch.object(PlantedExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(PlantedExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_planted_extractor,
        sample_planted_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_planted_bulk_extraction_parallel(
                entity_list=sample_planted_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.export_results")
    @patch.object(PlantedExtractor, "process_single_entity_planted")
    @patch.object(PlantedExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(PlantedExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_planted_extractor,
        sample_planted_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"entity_id": "x", "planted_percentage": 90}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_planted_bulk_extraction_parallel(
                entity_list=sample_planted_entity_list,
                max_workers=1,
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_planted_extractor, sample_planted_entity_list):
        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_planted_extractor.process_planted_bulk_extraction_parallel(
                    entity_list=sample_planted_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.export_results")
    @patch.object(PlantedExtractor, "process_single_entity_planted")
    @patch.object(PlantedExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(PlantedExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_planted_extractor,
        sample_planted_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_planted_bulk_extraction_parallel(
                entity_list=sample_planted_entity_list,
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

    @patch("earthdaily.agriculture.processors.processor_plantedarea_functions.export_results")
    @patch.object(PlantedExtractor, "process_single_entity_planted")
    @patch.object(PlantedExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(PlantedExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_planted_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": PLANTED_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": PLANTED_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": PLANTED_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_planted_extractor, "ensure_token_valid"):
            result = configured_planted_extractor.process_planted_bulk_extraction_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
