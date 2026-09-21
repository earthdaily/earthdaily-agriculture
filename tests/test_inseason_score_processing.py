"""
Tests for InseasonScoreExtractor processing pipeline:
    - process_single_entity_inseason_score() — single entity with retry
    - process_inseason_score_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_score_functions import InseasonScoreExtractor
from tests.conftest import INSEASON_SCORE_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_inseason_score()
# ===================================================================


class TestProcessSingleEntityInseasonScore:
    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
        sample_inseason_score_response,
    ):
        mock_retry.return_value = sample_inseason_score_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(
                sample_inseason_score_entity
            )

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert result["data"].iloc[0]["inseason_potential_score"] == 0.65

    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_response_id_filled_when_missing(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        """When publish_af=False the API may not echo `id`; processor backfills it."""
        mock_retry.return_value = {"data": {"inseason_potential_score": 0.6}}

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(
                sample_inseason_score_entity
            )

        assert result["error"] is None
        assert result["data"].iloc[0]["entity_id"] == "z361x33"

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_data_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        """Response with no 'data' key → 'No inseason_score results found'."""
        mock_retry.return_value = {"id": "z361x33"}

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(
                sample_inseason_score_entity
            )

        assert result["data"] is None
        assert "No inseason_score results found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(
        self, mock_wkt, configured_inseason_score_extractor, sample_inseason_score_entity
    ):
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(
                sample_inseason_score_entity
            )

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]

    def test_invalid_crop_returns_error(self, configured_inseason_score_extractor):
        entity = {
            "id": "ent_bad",
            "geometry": INSEASON_SCORE_WKT,
            "crop": "BANANA",
            "sowing_date": "2025-10-25",
        }
        with patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
                result = configured_inseason_score_extractor.process_single_entity_inseason_score(entity)

        assert result["data"] is None
        assert "Invalid crop" in result["error"]["message"]

    def test_missing_sowing_date_returns_error(self, configured_inseason_score_extractor):
        """Without sowing_date the processor short-circuits with an error result."""
        entity = {"id": "ent_x", "geometry": INSEASON_SCORE_WKT, "crop": "OTHERS"}

        with patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x):
            with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
                result = configured_inseason_score_extractor.process_single_entity_inseason_score(entity)

        assert result["data"] is None
        assert "sowing_date" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(
                sample_inseason_score_entity
            )

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        configured_inseason_score_extractor,
        sample_inseason_score_entity,
    ):
        mock_retry.return_value = {"id": "z361x33"}

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            configured_inseason_score_extractor.process_single_entity_inseason_score(sample_inseason_score_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.processors.processor_score_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_score_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_score_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_with_historical_seasons(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_inseason_score_extractor,
        sample_inseason_score_response,
    ):
        """Notebook cell 23: pd.Series with historical_seasons list."""
        row = pd.Series(
            {
                "id": "z361x33",
                "geometry": INSEASON_SCORE_WKT,
                "crop": "OTHERS",
                "sowing_date": "2025-10-25",
                "historical_seasons": [2024, 2023],
            }
        )
        mock_retry.return_value = sample_inseason_score_response

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_single_entity_inseason_score(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_inseason_score_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.processors.processor_score_functions.export_results")
    @patch.object(InseasonScoreExtractor, "process_single_entity_inseason_score")
    @patch.object(InseasonScoreExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InseasonScoreExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_inseason_score_extractor,
        sample_inseason_score_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "inseason_potential_score": 0.65}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                entity_list=sample_inseason_score_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.processors.processor_score_functions.export_results")
    @patch.object(InseasonScoreExtractor, "process_single_entity_inseason_score")
    @patch.object(InseasonScoreExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InseasonScoreExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_inseason_score_extractor,
        sample_inseason_score_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                entity_list=sample_inseason_score_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_score_functions.export_results")
    @patch.object(InseasonScoreExtractor, "process_single_entity_inseason_score")
    @patch.object(InseasonScoreExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InseasonScoreExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_inseason_score_extractor,
        sample_inseason_score_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "inseason_potential_score": 0.6}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                entity_list=sample_inseason_score_entity_list,
                max_workers=1,
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(
        self, configured_inseason_score_extractor, sample_inseason_score_entity_list
    ):
        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                    entity_list=sample_inseason_score_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.processors.processor_score_functions.export_results")
    @patch.object(InseasonScoreExtractor, "process_single_entity_inseason_score")
    @patch.object(InseasonScoreExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InseasonScoreExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_inseason_score_extractor,
        sample_inseason_score_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                entity_list=sample_inseason_score_entity_list,
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

    @patch("earthdaily.agriculture.processors.processor_score_functions.export_results")
    @patch.object(InseasonScoreExtractor, "process_single_entity_inseason_score")
    @patch.object(InseasonScoreExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(InseasonScoreExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_inseason_score_extractor,
    ):
        """Notebook cell 28 uses filter_type='exclude' on crop=CORN."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": INSEASON_SCORE_WKT, "crop": "OTHERS", "sowing_date": "2025-10-25"},
                {"id": "ent_002", "geometry": INSEASON_SCORE_WKT, "crop": "CORN", "sowing_date": "2025-10-25"},
                {"id": "ent_003", "geometry": INSEASON_SCORE_WKT, "crop": "CORN", "sowing_date": "2025-10-25"},
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "inseason_potential_score": 0.6}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_inseason_score_extractor, "ensure_token_valid"):
            result = configured_inseason_score_extractor.process_inseason_score_bulk_extraction_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="CORN",
                filter_type="exclude",
                skip_export=True,
            )

        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
