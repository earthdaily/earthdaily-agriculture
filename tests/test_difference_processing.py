"""
Tests for DifferenceExtractor processing pipeline:
    - process_single_entity_difference() — stats / links / file modes + image_id validation
    - process_entity_difference_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.difference_functions import DifferenceExtractor
from tests.conftest import DIFFERENCE_IMG_1, DIFFERENCE_IMG_2, DIFFERENCE_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_difference()
# ===================================================================


class TestProcessSingleEntityDifference:
    """Tests for the single-entity processing pipeline."""

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.normalize_with_metadata", side_effect=lambda r, df: df
    )
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_stats_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_difference_extractor,
        sample_difference_entity,
        sample_difference_stats_response,
    ):
        """Stats mode: API returns parsed JSON → DataFrame with stat_* and range_* cols."""
        mock_retry.return_value = sample_difference_stats_response

        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["error"] is None
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["stat_max"] == 0.42
        assert df.iloc[0]["range_1_pixels"] == 120

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.normalize_with_metadata", side_effect=lambda r, df: df
    )
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_links_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_difference_extractor,
        sample_difference_entity,
        sample_difference_links_response,
    ):
        """Links mode: API returns links payload → DataFrame with worldfile / bbox cols."""
        configured_difference_extractor.difference_params["postprocess"] = "links"
        configured_difference_extractor.difference_params["directLinks"] = True
        mock_retry.return_value = sample_difference_links_response

        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["error"] is None
        df = result["data"]
        assert df.iloc[0]["image_png_link"] == "https://api.example.com/diff/test_001.png"
        assert df.iloc[0]["map_width"] == 512

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.normalize_with_metadata", side_effect=lambda r, df: df
    )
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_file_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_difference_extractor,
        sample_difference_entity,
        tmp_path,
    ):
        """File mode (PNG): API returns a Response, file is written, summary row returned."""
        configured_difference_extractor.difference_params.update(
            {"postprocess": "file", "map_format": "png", "output_path": str(tmp_path)}
        )

        mock_response = MagicMock()
        mock_response.content = b"PNG_BYTES"
        mock_retry.return_value = mock_response

        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["error"] is None
        df = result["data"]
        assert df.iloc[0]["status"] == "downloaded"
        assert df.iloc[0]["map_format"] == "png"
        assert df.iloc[0]["file_count"] == 1
        assert df.iloc[0]["total_size_bytes"] == len(mock_response.content)

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_stats_mode_empty_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_difference_extractor,
        sample_difference_entity,
    ):
        """Empty response in stats mode → 'Empty API response' error."""
        mock_retry.return_value = {}

        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["data"] is not None  # validate_api_response returns empty df
        assert result["error"] is not None
        assert "Empty" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(
        self, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        """Geometry validation failure short-circuits before any API call."""
        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "test_001"

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_1_returns_error(self, mock_wkt, configured_difference_extractor):
        row = {"id": "test_001", "geometry": DIFFERENCE_WKT, "image_id_2": DIFFERENCE_IMG_2}
        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["data"] is None
        assert "Missing image_id_1" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_2_returns_error(self, mock_wkt, configured_difference_extractor):
        row = {"id": "test_001", "geometry": DIFFERENCE_WKT, "image_id_1": DIFFERENCE_IMG_1}
        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["data"] is None
        assert "Missing image_id_2" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_image_id_format_no_separator_returns_error(self, mock_wkt, configured_difference_extractor):
        """An image_id with no '|' separator is rejected."""
        row = {
            "id": "test_001",
            "geometry": DIFFERENCE_WKT,
            "image_id_1": "no_pipe_here",
            "image_id_2": DIFFERENCE_IMG_2,
        }
        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["data"] is None
        assert "Invalid image_id_1" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_image_id_format_too_many_separators_returns_error(self, mock_wkt, configured_difference_extractor):
        """An image_id with more than 3 '|' separators is rejected."""
        row = {
            "id": "test_001",
            "geometry": DIFFERENCE_WKT,
            "image_id_1": "a|b|c|d|e",  # 4 separators
            "image_id_2": DIFFERENCE_IMG_2,
        }
        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["data"] is None
        assert "Invalid image_id_1" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_image_id_format_empty_part_returns_error(self, mock_wkt, configured_difference_extractor):
        """An image_id with an empty segment between '|' is rejected."""
        row = {
            "id": "test_001",
            "geometry": DIFFERENCE_WKT,
            "image_id_1": "prefix||",
            "image_id_2": DIFFERENCE_IMG_2,
        }
        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["data"] is None
        assert "empty parts" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self, mock_retry, mock_wkt, configured_difference_extractor, sample_difference_entity
    ):
        """Retry exhaustion → captured error result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.normalize_with_metadata", side_effect=lambda r, df: df
    )
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_difference_extractor,
        sample_difference_entity,
        sample_difference_stats_response,
    ):
        """retry_with_backoff_no_retry_on_400(max_retries=5, base_delay=1.0, max_delay=60.0)."""
        mock_retry.return_value = sample_difference_stats_response

        configured_difference_extractor.process_single_entity_difference(sample_difference_entity)

        call_args = mock_retry.call_args
        # The retry function uses positional + kwargs
        assert call_args.kwargs["max_retries"] == 5
        assert call_args.kwargs["base_delay"] == 1.0
        assert call_args.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.extractors.difference_functions.normalize_with_metadata", side_effect=lambda r, df: df
    )
    @patch("earthdaily.agriculture.extractors.difference_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.difference_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_difference_extractor,
        sample_difference_stats_response,
    ):
        """Notebook cell 22 passes a pd.Series — must be supported."""
        row = pd.Series(
            {
                "id": "test_001",
                "name": "Test_Field",
                "geometry": DIFFERENCE_WKT,
                "image_id_1": DIFFERENCE_IMG_1,
                "image_id_2": DIFFERENCE_IMG_2,
            }
        )
        mock_retry.return_value = sample_difference_stats_response

        result = configured_difference_extractor.process_single_entity_difference(row)

        assert result["error"] is None
        assert result["data"] is not None


# ===================================================================
# process_entity_difference_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.difference_functions.export_results")
    @patch.object(DifferenceExtractor, "process_single_entity_difference")
    @patch.object(DifferenceExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DifferenceExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_difference_extractor,
        sample_difference_entity_list,
    ):
        success_df = pd.DataFrame([{"stat_max": 0.4, "stat_mean": 0.05, "stat_min": -0.3}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_difference_extractor.process_entity_difference_bulk_parallel(
            entity_list=sample_difference_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.extractors.difference_functions.export_results")
    @patch.object(DifferenceExtractor, "process_single_entity_difference")
    @patch.object(DifferenceExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DifferenceExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_difference_extractor,
        sample_difference_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_difference_extractor.process_entity_difference_bulk_parallel(
            entity_list=sample_difference_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.difference_functions.export_results")
    @patch.object(DifferenceExtractor, "process_single_entity_difference")
    @patch.object(DifferenceExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DifferenceExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_difference_extractor,
        sample_difference_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"x": 1}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_difference_extractor.process_entity_difference_bulk_parallel(
            entity_list=sample_difference_entity_list,
            max_workers=1,  # sequential for deterministic side_effect ordering
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_difference_extractor, sample_difference_entity_list):
        with pytest.raises(ValueError, match="merge_existing"):
            configured_difference_extractor.process_entity_difference_bulk_parallel(
                entity_list=sample_difference_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.extractors.difference_functions.export_results")
    @patch.object(DifferenceExtractor, "process_single_entity_difference")
    @patch.object(DifferenceExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DifferenceExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_difference_extractor,
        sample_difference_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_difference_extractor.process_entity_difference_bulk_parallel(
            entity_list=sample_difference_entity_list,
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

    @patch("earthdaily.agriculture.extractors.difference_functions.export_results")
    @patch.object(DifferenceExtractor, "process_single_entity_difference")
    @patch.object(DifferenceExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DifferenceExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_difference_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": DIFFERENCE_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": DIFFERENCE_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": DIFFERENCE_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_difference_extractor.process_entity_difference_bulk_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
