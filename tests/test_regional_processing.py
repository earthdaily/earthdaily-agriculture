"""
Tests for RegionalExtractor processing pipeline:
    - process_single_entity_regional() — single entity with retry, returns 2 dataframes
    - process_entity_regional_bulk_parallel() — bulk parallel processing

NOTE: Unlike most extractors that return {data, error}, RegionalExtractor returns
{observed_data, daily_avg_data, error} for single-entity, and bulk returns
{observed_results_df, daily_avg_results_df, ...} (no flat 'results_df' key).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.regional_ts_extractor import RegionalExtractor

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_regional()
# ===================================================================


class TestProcessSingleEntityRegional:
    """Tests for the single-entity processing pipeline."""

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_both_dataframes(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_entity,
        sample_regional_response,
    ):
        """Happy path: API returns both sections → result has both DataFrames populated."""
        mock_retry.return_value = sample_regional_response

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert result["error"] is None
        assert isinstance(result["observed_data"], pd.DataFrame)
        assert not result["observed_data"].empty
        assert isinstance(result["daily_avg_data"], pd.DataFrame)
        assert not result["daily_avg_data"].empty

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_no_data_error(
        self,
        mock_retry,
        configured_regional_extractor,
        sample_regional_entity,
        sample_regional_response_empty,
    ):
        """If both observed and dailyAverage are empty, result should carry 'No data found'."""
        mock_retry.return_value = sample_regional_response_empty

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert result["observed_data"] is None
        assert result["daily_avg_data"] is None
        assert result["error"] is not None
        assert "No data found" in result["error"]["message"]
        assert result["error"]["entity_id"] == 2432528

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_observed_only_returned(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_entity,
        sample_regional_response_observed_only,
    ):
        """When dailyAverage is empty but observedMeasures has data → success with daily_avg=None."""
        mock_retry.return_value = sample_regional_response_observed_only

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert result["error"] is None
        assert isinstance(result["observed_data"], pd.DataFrame)
        assert not result["observed_data"].empty
        assert result["daily_avg_data"] is None

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_daily_avg_only_returned(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_entity,
        sample_regional_response_daily_avg_only,
    ):
        """When observedMeasures is empty but dailyAverage has data → success with observed=None."""
        mock_retry.return_value = sample_regional_response_daily_avg_only

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert result["error"] is None
        assert result["observed_data"] is None
        assert isinstance(result["daily_avg_data"], pd.DataFrame)
        assert not result["daily_avg_data"].empty

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, configured_regional_extractor, sample_regional_entity):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert result["observed_data"] is None
        assert result["daily_avg_data"] is None
        assert result["error"] is not None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, configured_regional_extractor, sample_regional_entity):
        """retry_with_backoff_no_retry_on_400 must be called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = {"observedMeasures": [], "dailyAverage": []}

        configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_response,
    ):
        """The notebook (cell 18) passes a pd.Series — must be accepted."""
        row = pd.Series({"amu_id": 2432528})
        mock_retry.return_value = sample_regional_response

        result = configured_regional_extractor.process_single_entity_regional(row)

        assert result["error"] is None
        assert isinstance(result["observed_data"], pd.DataFrame)

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_string_amu_id_in_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_response,
    ):
        """A string amu_id should flow through (conversion happens in get_regional_ts_by_id)."""
        row = {"amu_id": "2432528"}
        mock_retry.return_value = sample_regional_response

        result = configured_regional_extractor.process_single_entity_regional(row)

        assert result["error"] is None

    @patch(
        "earthdaily.agriculture.extractors.regional_ts_extractor.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.retry_with_backoff_no_retry_on_400")
    def test_returns_expected_keys(
        self,
        mock_retry,
        mock_normalize,
        configured_regional_extractor,
        sample_regional_entity,
        sample_regional_response,
    ):
        """Single-entity result must have exactly {observed_data, daily_avg_data, error}."""
        mock_retry.return_value = sample_regional_response

        result = configured_regional_extractor.process_single_entity_regional(sample_regional_entity)

        assert set(result.keys()) == {"observed_data", "daily_avg_data", "error"}


# ===================================================================
# process_entity_regional_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method.

    Note: bulk return contract differs from other extractors — it returns
    {observed_results_df, daily_avg_results_df, ...} instead of {results_df, ...}.
    """

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
        sample_regional_entity_list,
    ):
        """All entities succeed → successful_calculations == total."""
        success_observed = pd.DataFrame([{"date": "2025-01-15", "value": 0.42}])
        success_daily_avg = pd.DataFrame([{"date": "1900-01-15", "value": 0.40}])
        mock_single.return_value = {
            "observed_data": success_observed,
            "daily_avg_data": success_daily_avg,
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=sample_regional_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
        sample_regional_entity_list,
    ):
        """All entities fail → failed_calculations == total."""
        mock_single.return_value = {
            "observed_data": None,
            "daily_avg_data": None,
            "error": {"message": "API error", "entity_id": "x"},
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=sample_regional_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
        sample_regional_entity_list,
    ):
        """Mix of successes and failures → counts are correct."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "observed_data": pd.DataFrame([{"date": "2025-01-15", "value": 0.5}]),
                    "daily_avg_data": pd.DataFrame([{"date": "1900-01-15", "value": 0.4}]),
                    "error": None,
                }
            return {
                "observed_data": None,
                "daily_avg_data": None,
                "error": {"message": "fail", "entity_id": "y"},
            }

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=sample_regional_entity_list,
            max_workers=1,  # sequential for deterministic side_effect ordering
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_regional_extractor, sample_regional_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with pytest.raises(ValueError, match="merge_existing"):
            configured_regional_extractor.process_entity_regional_bulk_parallel(
                entity_list=sample_regional_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
        sample_regional_entity_list,
    ):
        """Return dict must have the regional-specific keys (observed/daily_avg)."""
        mock_single.return_value = {
            "observed_data": pd.DataFrame([{"x": 1}]),
            "daily_avg_data": pd.DataFrame([{"x": 1}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=sample_regional_entity_list,
            skip_export=True,
        )

        expected_keys = {
            "results_df",
            "observed_results_df",
            "daily_avg_results_df",
            "global_errors",
            "total_entities",
            "total_calculations",
            "successful_calculations",
            "failed_calculations",
            "failed_ids",
        }
        assert set(result.keys()) == expected_keys
        # results_df is the primary frame and must alias the observed series
        assert isinstance(result["results_df"], pd.DataFrame)
        assert result["results_df"] is result["observed_results_df"]
        assert isinstance(result["observed_results_df"], pd.DataFrame)
        assert isinstance(result["daily_avg_results_df"], pd.DataFrame)
        assert isinstance(result["global_errors"], list)
        assert isinstance(result["failed_ids"], list)

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities (mirrors notebook cell 24)."""
        entity_list = pd.DataFrame(
            [
                {"amu_id": 2432528, "tag": "skip"},
                {"amu_id": 2121564, "tag": "keep"},
                {"amu_id": 2121566, "tag": "skip"},
            ]
        )
        mock_single.return_value = {
            "observed_data": pd.DataFrame([{"date": "2025-01-15", "value": 0.5}]),
            "daily_avg_data": pd.DataFrame([{"date": "1900-01-15", "value": 0.4}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3

    @patch("earthdaily.agriculture.extractors.regional_ts_extractor.export_results")
    @patch.object(RegionalExtractor, "process_single_entity_regional")
    @patch.object(RegionalExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(RegionalExtractor, "_merge_with_skipped_entities")
    def test_bulk_no_data_marks_entity_failed(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_regional_extractor,
        sample_regional_entity_list,
    ):
        """A row that returns observed=None and daily_avg=None counts as a failure."""
        mock_single.return_value = {
            "observed_data": None,
            "daily_avg_data": None,
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_regional_extractor.process_entity_regional_bulk_parallel(
            entity_list=sample_regional_entity_list,
            max_workers=1,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3
