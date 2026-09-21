"""
Tests for VegationTsExtractor processing pipeline:
    - process_single_entity_vegetation_ts() — single entity with retry, KPI mode
    - process_single_entity_specific_dates() — punctual extraction at target dates
    - process_entity_vegetation_ts_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.VTS_functions import VegationTsExtractor
from tests.conftest import VTS_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_vegetation_ts() — non-KPI mode
# ===================================================================


class TestProcessSingleEntityVegetationTs:
    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_normalize,
        configured_vts_extractor,
        sample_vts_entity,
        sample_vts_response,
    ):
        mock_retry.return_value = sample_vts_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 4

    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(
        self,
        mock_retry,
        configured_vts_extractor,
        sample_vts_entity,
        sample_vts_response_empty,
    ):
        """Empty list response → error result.

        The intent is `'No vegetation data found'`, but `format_vegetation_ts_json([])`
        raises `KeyError` first (see the format-test docstring). The processor catches
        that as a generic exception and wraps it as an error result, so the run survives
        even though the error message reflects the underlying KeyError.
        """
        mock_retry.return_value = sample_vts_response_empty

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        assert result["data"] is None
        assert result["error"] is not None
        assert result["error"]["entity_id"] == "z361x33"

    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, configured_vts_extractor, sample_vts_entity):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, configured_vts_extractor, sample_vts_entity):
        mock_retry.return_value = []

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_vts_extractor,
        sample_vts_response,
    ):
        """Notebook cell 33: a pd.Series row with optional `years` field."""
        row = pd.Series({"id": "3a5yn53", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"})
        mock_retry.return_value = sample_vts_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_single_entity_vegetation_ts() — KPI mode
# ===================================================================


class TestProcessSingleEntityVegetationTsKpi:
    @patch("earthdaily.agriculture.extractors.VTS_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.VTS_functions.filter_timeseries_kpi")
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_mode_returns_single_row(
        self,
        mock_retry,
        mock_kpi,
        mock_normalize,
        configured_vts_extractor,
        sample_vts_entity,
        sample_vts_response,
    ):
        """Notebook cell 31: kpi_filter='accumulation' produces a single-row KPI DataFrame."""
        configured_vts_extractor.vegetation_ts_params["kpi_filter"] = {
            "kpi_name": "Summer NDVI Accumulation",
            "aggregation": "accumulation",
        }
        mock_retry.return_value = sample_vts_response
        mock_kpi.return_value = {
            "kpi_name": "Summer NDVI Accumulation",
            "aggregation": "accumulation",
            "current_period": {"value": 21.5, "num_records": 30},
            "historical_avg": {"value": 19.8, "num_years": 4},
            "comparison": {"difference": 1.7, "percent_change": 8.6},
        }

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["kpi_name"] == "Summer NDVI Accumulation"
        assert df.iloc[0]["current_value"] == 21.5
        assert df.iloc[0]["historical_avg"] == 19.8
        mock_kpi.assert_called_once()

    @patch(
        "earthdaily.agriculture.extractors.VTS_functions.filter_timeseries_kpi", side_effect=RuntimeError("kpi blew up")
    )
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_failure_returns_error(
        self,
        mock_retry,
        mock_kpi,
        configured_vts_extractor,
        sample_vts_entity,
        sample_vts_response,
    ):
        configured_vts_extractor.vegetation_ts_params["kpi_filter"] = {
            "kpi_name": "Summer NDVI",
            "aggregation": "accumulation",
        }
        mock_retry.return_value = sample_vts_response

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_vegetation_ts(sample_vts_entity)

        assert result["data"] is None
        assert "KPI computation failed" in result["error"]["message"]
        assert "kpi blew up" in result["error"]["message"]


# ===================================================================
# process_single_entity_specific_dates() — punctual extraction
# ===================================================================


class TestProcessSingleEntitySpecificDates:
    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_specific_dates_lookup(self, mock_retry, configured_vts_extractor, sample_vts_entity):
        """API records are looked up by date; missing dates yield 'No data'."""
        configured_vts_extractor.vegetation_ts_params.update(
            {
                "extraction_mode": "specific_dates",
                "target_dates": ["2025-06-15", "2025-07-15", "2025-08-15"],
            }
        )
        mock_retry.return_value = {
            "value": [
                {"date": "2025-06-15T00:00:00Z", "value": 0.45},
                {"date": "2025-08-15T00:00:00Z", "value": 0.78},
                # 2025-07-15 missing
            ]
        }
        # Add a name field for the field_name column
        sample_vts_entity["name"] = "Test_Field"

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_specific_dates(sample_vts_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 3
        # Column name reflects vegetation_index lowercased
        assert "ndvi_value" in df.columns
        # First row (2025-06-15) has the value
        assert df.iloc[0]["ndvi_value"] == 0.45
        # Missing date filled with "No data"
        assert df.iloc[1]["ndvi_value"] == "No data"
        # Field name carried through
        assert (df["field_name"] == "Test_Field").all()

    def test_specific_dates_without_target_dates_returns_error(self, configured_vts_extractor, sample_vts_entity):
        """Without target_dates configured, processor returns error result."""
        configured_vts_extractor.vegetation_ts_params["target_dates"] = []

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_specific_dates(sample_vts_entity)

        assert result["data"] is None
        assert "No target dates" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.VTS_functions.retry_with_backoff_no_retry_on_400")
    def test_specific_dates_api_failure_returns_no_data_dataframe(
        self, mock_retry, configured_vts_extractor, sample_vts_entity
    ):
        """If the API raises, processor still returns a DataFrame with 'No data' for every target."""
        configured_vts_extractor.vegetation_ts_params.update(
            {
                "extraction_mode": "specific_dates",
                "target_dates": ["2025-06-15", "2025-07-15"],
            }
        )
        mock_retry.side_effect = RuntimeError("api down")

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_single_entity_specific_dates(sample_vts_entity)

        # Error is set BUT data is also a DataFrame full of "No data" rows
        assert result["error"] is not None
        assert "api down" in result["error"]["message"]
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 2
        assert (result["data"]["ndvi_value"] == "No data").all()


# ===================================================================
# process_entity_vegetation_ts_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(VegationTsExtractor, "process_single_entity_vegetation_ts")
    @patch.object(VegationTsExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(VegationTsExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_vts_extractor,
        sample_vts_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "date": "2025-08-15", "value": 0.78}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_entity_vegetation_ts_bulk_parallel(
                entity_list=sample_vts_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(VegationTsExtractor, "process_single_entity_vegetation_ts")
    @patch.object(VegationTsExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(VegationTsExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_vts_extractor,
        sample_vts_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_entity_vegetation_ts_bulk_parallel(
                entity_list=sample_vts_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    def test_bulk_invalid_merge_existing_raises(self, configured_vts_extractor, sample_vts_entity_list):
        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_vts_extractor.process_entity_vegetation_ts_bulk_parallel(
                    entity_list=sample_vts_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.extractors.VTS_functions.export_results")
    @patch.object(VegationTsExtractor, "process_single_entity_vegetation_ts")
    @patch.object(VegationTsExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(VegationTsExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_vts_extractor,
        sample_vts_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_entity_vegetation_ts_bulk_parallel(
                entity_list=sample_vts_entity_list,
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
    @patch.object(VegationTsExtractor, "process_single_entity_vegetation_ts")
    @patch.object(VegationTsExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(VegationTsExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_vts_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": VTS_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": VTS_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": VTS_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_vts_extractor, "ensure_token_valid"):
            result = configured_vts_extractor.process_entity_vegetation_ts_bulk_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
