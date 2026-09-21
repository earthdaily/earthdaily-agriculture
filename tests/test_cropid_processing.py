"""
Tests for cropidExtractor processing pipeline:
    - process_single_entity_cropid() — history / year / historical_season modes
    - process_cropid_bulk_extraction_parallel() — bulk parallel processing
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.cropid_functions import cropidExtractor
from tests.conftest import CROPID_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_cropid()
# ===================================================================


class TestProcessSingleEntityCropid:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.extractors.cropid_functions.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_history_mode_returns_full_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_cropid_extractor,
        sample_cropid_entity,
        sample_cropid_response,
    ):
        """mode='history' (default) returns one row per (year, crop) — all 6 from fixture."""
        mock_retry.return_value = sample_cropid_response
        mock_normalize.side_effect = lambda r, df: df

        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["error"] is None
        assert len(result["data"]) == 6

    @patch("earthdaily.agriculture.extractors.cropid_functions.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_year_mode_filters_by_entity_crop(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_cropid_extractor,
        sample_cropid_entity,
        sample_cropid_response,
    ):
        """mode='year' filters to entity's crop (notebook cell 27 → 4 SOYBEANS rows)."""
        configured_cropid_extractor.cropid_params["mode"] = "year"
        mock_retry.return_value = sample_cropid_response
        mock_normalize.side_effect = lambda r, df: df

        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 4
        assert (df["eda_crop_code"] == "SOYBEANS").all()
        assert list(df["year"]) == [2020, 2022, 2024, 2025]

    @patch("earthdaily.agriculture.extractors.cropid_functions.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_historical_season_mode_returns_string_in_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_cropid_extractor,
        sample_cropid_entity,
        sample_cropid_response,
    ):
        """mode='historical_season' wraps the comma-separated years into a 1-row DataFrame."""
        configured_cropid_extractor.cropid_params["mode"] = "historical_season"
        mock_retry.return_value = sample_cropid_response
        mock_normalize.side_effect = lambda r, df: df

        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "z361x33"
        assert df.iloc[0]["historical_season"] == "2020,2022,2024,2025"

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_historical_season_mode_no_matches_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_cropid_extractor,
        sample_cropid_response,
    ):
        """mode='historical_season' with no matching crop returns an error result."""
        configured_cropid_extractor.cropid_params["mode"] = "historical_season"
        mock_retry.return_value = sample_cropid_response
        entity = {"id": "ent_x", "geometry": CROPID_WKT, "crop": "RICE"}

        result = configured_cropid_extractor.process_single_entity_cropid(entity)

        assert result["data"] is None
        assert "No historical seasons found" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_year_mode_empty_results_returns_no_results_error(
        self,
        mock_retry,
        mock_wkt,
        configured_cropid_extractor,
        sample_cropid_entity,
        sample_cropid_response_empty,
    ):
        """In 'year' mode, empty resultsByYear → 'No crop ID results found' error."""
        configured_cropid_extractor.cropid_params["mode"] = "year"
        mock_retry.return_value = sample_cropid_response_empty

        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["data"] is None
        assert "No crop ID results found" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.extractors.cropid_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_cropid_extractor, sample_cropid_entity):
        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, mock_wkt, configured_cropid_extractor, sample_cropid_entity):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.cropid_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_cropid_extractor,
        sample_cropid_entity,
        sample_cropid_response,
    ):
        """retry_with_backoff is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = sample_cropid_response

        configured_cropid_extractor.process_single_entity_cropid(sample_cropid_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.cropid_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.cropid_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.cropid_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self, mock_retry, mock_wkt, mock_normalize, configured_cropid_extractor, sample_cropid_response
    ):
        """Notebook cell 27 passes a pd.Series — must be supported."""
        configured_cropid_extractor.cropid_params["mode"] = "year"
        row = pd.Series({"id": "z361x33", "geometry": CROPID_WKT, "crop": "SOYBEANS"})
        mock_retry.return_value = sample_cropid_response

        result = configured_cropid_extractor.process_single_entity_cropid(row)

        assert result["error"] is None
        assert len(result["data"]) == 4


# ===================================================================
# process_cropid_bulk_extraction_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.cropid_functions.export_results")
    @patch.object(cropidExtractor, "process_single_entity_cropid")
    @patch.object(cropidExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(cropidExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_cropid_extractor,
        sample_cropid_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "year": 2024, "eda_crop_code": "CORN"}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
            entity_list=sample_cropid_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.extractors.cropid_functions.export_results")
    @patch.object(cropidExtractor, "process_single_entity_cropid")
    @patch.object(cropidExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(cropidExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_cropid_extractor,
        sample_cropid_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
            entity_list=sample_cropid_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.cropid_functions.export_results")
    @patch.object(cropidExtractor, "process_single_entity_cropid")
    @patch.object(cropidExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(cropidExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_cropid_extractor,
        sample_cropid_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"x": 1}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
            entity_list=sample_cropid_entity_list,
            max_workers=1,  # sequential for deterministic side_effect ordering
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_cropid_extractor, sample_cropid_entity_list):
        with pytest.raises(ValueError, match="merge_existing"):
            configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
                entity_list=sample_cropid_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.extractors.cropid_functions.export_results")
    @patch.object(cropidExtractor, "process_single_entity_cropid")
    @patch.object(cropidExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(cropidExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_cropid_extractor,
        sample_cropid_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
            entity_list=sample_cropid_entity_list,
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

    @patch("earthdaily.agriculture.extractors.cropid_functions.export_results")
    @patch.object(cropidExtractor, "process_single_entity_cropid")
    @patch.object(cropidExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(cropidExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_cropid_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": CROPID_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": CROPID_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": CROPID_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_cropid_extractor.process_cropid_bulk_extraction_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
