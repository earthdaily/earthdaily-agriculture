"""
Tests for MRTSExtractor processing pipeline:
    - process_single_entity_mrts() — single entity with retry, KPI mode, date expansion
    - process_mrts_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.VTS_functions import MRTSExtractor
from tests.conftest import MRTS_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_mrts() — non-KPI mode
# ===================================================================


class TestProcessSingleEntityMrts:
    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_mrts_extractor,
        sample_mrts_entity,
        sample_mrts_response_full,
    ):
        mock_retry.return_value = sample_mrts_response_full

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 3
        assert result["data"].iloc[0]["raw_value"] == 0.42

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_mrts_extractor,
        sample_mrts_entity,
        sample_mrts_response_empty,
    ):
        """Empty rawData + smoothedData → 'No MRTS data found'."""
        mock_retry.return_value = sample_mrts_response_empty

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        assert result["data"] is None
        assert "No MRTS data found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "ent_001"

    def test_missing_start_date_returns_error(self, configured_mrts_extractor):
        """If params has no start_date and entity has none → error result."""
        configured_mrts_extractor.mrts_params["start_date"] = None
        entity = {"id": "ent_x", "geometry": MRTS_WKT, "end_date": "2025-10-15"}

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(entity)

        assert result["data"] is None
        assert "start_date" in result["error"]["message"]

    def test_missing_end_date_returns_error(self, configured_mrts_extractor):
        configured_mrts_extractor.mrts_params["end_date"] = None
        entity = {"id": "ent_x", "geometry": MRTS_WKT, "start_date": "2025-05-01"}

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(entity)

        assert result["data"] is None
        assert "end_date" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_mrts_extractor,
        sample_mrts_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self, mock_retry, mock_wkt, configured_mrts_extractor, sample_mrts_entity
    ):
        mock_retry.return_value = {"rawData": [], "smoothedData": []}

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_mrts_extractor,
        sample_mrts_response_full,
    ):
        """Notebook cell 23 passes a dict via .to_dict(); also accepts Series directly."""
        row = pd.Series({"id": "ent_001", "geometry": MRTS_WKT})
        mock_retry.return_value = sample_mrts_response_full

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_dataframe_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_mrts_extractor,
        sample_mrts_response_full,
    ):
        """A 1-row DataFrame should be accepted (only first row used)."""
        df_in = pd.DataFrame([{"id": "ent_001", "geometry": MRTS_WKT}])
        mock_retry.return_value = sample_mrts_response_full

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(df_in)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    def test_invalid_input_type_returns_error(self, configured_mrts_extractor):
        result = configured_mrts_extractor.process_single_entity_mrts("not a row")
        assert result["data"] is None
        assert "Invalid input type" in result["error"]["message"]


# ===================================================================
# process_single_entity_mrts() — KPI mode
# ===================================================================


class TestProcessSingleEntityMrtsKpi:
    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.filter_timeseries_kpi")
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_mode_returns_single_row(
        self,
        mock_retry,
        mock_wkt,
        mock_kpi,
        mock_normalize,
        configured_mrts_extractor,
        sample_mrts_entity,
        sample_mrts_response_full,
    ):
        """Notebook cell 26: kpi_filter=top_accumulation produces a single-row KPI DataFrame."""
        configured_mrts_extractor.mrts_params["kpi_filter"] = {
            "kpi_name": "NDVI Accumulation",
            "aggregation": "top_accumulation",
            "value_column": "smoothed_value",
            "threshold": 30,
        }
        configured_mrts_extractor.mrts_params["historical_years"] = 5

        mock_retry.return_value = sample_mrts_response_full
        mock_kpi.return_value = {
            "kpi_name": "NDVI Accumulation",
            "aggregation": "top_accumulation",
            "current_period": {"value": 21.5, "num_records": 30},
            "historical_avg": {"value": 19.8, "num_years": 4},
            "comparison": {"difference": 1.7, "percent_change": 8.6},
        }

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["kpi_name"] == "NDVI Accumulation"
        assert df.iloc[0]["aggregation"] == "top_accumulation"
        assert df.iloc[0]["current_value"] == 21.5
        assert df.iloc[0]["historical_avg"] == 19.8
        # filter_timeseries_kpi was invoked
        mock_kpi.assert_called_once()

    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch(
        "earthdaily.agriculture.extractors.VTS_functions.filter_timeseries_kpi", side_effect=RuntimeError("kpi blew up")
    )
    @patch("earthdaily.agriculture.extractors.VTS_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_failure_returns_error(
        self,
        mock_retry,
        mock_wkt,
        mock_kpi,
        mock_normalize,
        configured_mrts_extractor,
        sample_mrts_entity,
        sample_mrts_response_full,
    ):
        """KPI computation raising should produce an error result, not propagate."""
        configured_mrts_extractor.mrts_params["kpi_filter"] = {
            "kpi_name": "NDVI Acc",
            "aggregation": "accumulation",
        }
        mock_retry.return_value = sample_mrts_response_full

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_single_entity_mrts(sample_mrts_entity)

        assert result["data"] is None
        assert "KPI computation failed" in result["error"]["message"]
        assert "kpi blew up" in result["error"]["message"]


# ===================================================================
# process_mrts_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(MRTSExtractor, "process_single_entity_mrts")
    @patch.object(MRTSExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(MRTSExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_mrts_extractor,
        sample_mrts_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "date": "2025-05-10", "raw_value": 0.42}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_mrts_bulk_parallel(
                entity_list=sample_mrts_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(MRTSExtractor, "process_single_entity_mrts")
    @patch.object(MRTSExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(MRTSExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_mrts_extractor,
        sample_mrts_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_mrts_bulk_parallel(
                entity_list=sample_mrts_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(MRTSExtractor, "process_single_entity_mrts")
    @patch.object(MRTSExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(MRTSExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_mrts_extractor,
        sample_mrts_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "raw_value": 0.5}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_mrts_bulk_parallel(
                entity_list=sample_mrts_entity_list,
                max_workers=1,
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_mrts_extractor, sample_mrts_entity_list):
        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_mrts_extractor.process_mrts_bulk_parallel(
                    entity_list=sample_mrts_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(MRTSExtractor, "process_single_entity_mrts")
    @patch.object(MRTSExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(MRTSExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_mrts_extractor,
        sample_mrts_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_mrts_bulk_parallel(
                entity_list=sample_mrts_entity_list,
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

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(MRTSExtractor, "process_single_entity_mrts")
    @patch.object(MRTSExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(MRTSExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_mrts_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": MRTS_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": MRTS_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": MRTS_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_mrts_extractor, "ensure_token_valid"):
            result = configured_mrts_extractor.process_mrts_bulk_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
