"""
Tests for FLMExtractor processing pipeline:
    - process_single_entity_flm() — all four postprocess modes
    - process_entity_flm_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.FLM_functions import FLMExtractor
from tests.conftest import FLM_IMAGE_ID, FLM_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_flm()
# ===================================================================


class TestProcessSingleEntityFlmStats:
    """stats mode — extracts legend.stat into a single-row DataFrame."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_stats_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_flm_extractor,
        sample_flm_entity,
        sample_flm_stats_response,
    ):
        """API returns stats response → DataFrame with stat_max/stat_mean/stat_min."""
        # The API returns a Response object; format_flm_map_stats_json handles parsing.
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_stats_response
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["error"] is None
        df = result["data"]
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["stat_max"] == 0.85
        assert df.iloc[0]["stat_mean"] == 0.62

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_stats_mode_empty_response_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_flm_extractor,
        sample_flm_entity,
    ):
        """Stats formatter returns a row with None stats; pipeline treats that as success unless empty."""
        # An empty dict still produces a row with all-None values, so the stats path
        # treats "no stats" only when the DataFrame is empty. Force an explicitly empty
        # DataFrame by mocking the format method.
        mock_resp = MagicMock()
        mock_retry.return_value = mock_resp
        with patch.object(configured_flm_extractor, "format_flm_map_stats_json", return_value=pd.DataFrame()):
            with patch.object(configured_flm_extractor, "ensure_token_valid"):
                result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["data"] is None
        assert "No map stats found" in result["error"]["message"]


class TestProcessSingleEntityFlmLinks:
    """links mode — extracts _links / worldFile / mapSize / bBox."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_links_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_flm_extractor,
        sample_flm_entity,
        sample_flm_links_response,
    ):
        """API returns links payload → DataFrame with image_png_link / map_width / bbox cols."""
        configured_flm_extractor.flm_params["postprocess"] = "links"
        configured_flm_extractor.flm_params["directLinks"] = True

        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_links_response
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["error"] is None
        df = result["data"]
        assert df.iloc[0]["image_png_link"] == "https://api.example.com/flm/test_001.png"
        assert df.iloc[0]["map_width"] == 512


class TestProcessSingleEntityFlmHistogram:
    """histogram mode — global stats + per-bucket columns."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_histogram_mode_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_flm_extractor,
        sample_flm_entity,
        sample_flm_histogram_response,
    ):
        configured_flm_extractor.flm_params["postprocess"] = "histogram"

        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_histogram_response
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["error"] is None
        df = result["data"]
        assert df.iloc[0]["stat_max"] == 0.85
        assert df.iloc[0]["num_pixels_1"] == 120
        assert df.iloc[0]["num_pixels_3"] == 540


class TestProcessSingleEntityFlmFile:
    """file mode — saves PNG / TIFF.ZIP / SHP.ZIP and returns a summary row."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_file_mode_png_success(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_flm_extractor,
        sample_flm_entity,
        tmp_path,
    ):
        configured_flm_extractor.flm_params.update(
            {"postprocess": "file", "map_format": "png", "output_path": str(tmp_path)}
        )

        mock_resp = MagicMock()
        mock_resp.content = b"PNG_BYTES"
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["error"] is None
        df = result["data"]
        assert df.iloc[0]["status"] == "downloaded"
        assert df.iloc[0]["map_format"] == "png"
        assert df.iloc[0]["file_count"] == 1
        assert df.iloc[0]["total_size_bytes"] == len(mock_resp.content)

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_file_mode_save_failure_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_flm_extractor,
        sample_flm_entity,
    ):
        """If format_flm_map_file returns status='error', the processor returns an error result."""
        configured_flm_extractor.flm_params.update(
            {"postprocess": "file", "map_format": "weird", "output_path": "/tmp"}
        )
        mock_resp = MagicMock()
        mock_resp.content = b""
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["data"] is None
        assert "Unsupported map_format" in result["error"]["message"]


class TestProcessSingleEntityFlmInputValidation:
    """Common input-validation paths for process_single_entity_flm."""

    @patch(
        "earthdaily.agriculture.extractors.FLM_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_geometry_validation_failure_returns_error(self, mock_wkt, configured_flm_extractor, sample_flm_entity):
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_missing_image_id_returns_error(self, mock_wkt, configured_flm_extractor):
        """Entity without image_id should return error before any API call."""
        entity = {"id": "x", "geometry": FLM_WKT}  # no image_id
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(entity)

        assert result["data"] is None
        assert "Missing image_id" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_invalid_image_id_format_returns_error(self, mock_wkt, configured_flm_extractor):
        """An image_id with no '|' separator is invalid."""
        entity = {"id": "x", "geometry": FLM_WKT, "image_id": "no-separator"}
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(entity)

        assert result["data"] is None
        assert "Invalid image_id format" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    def test_image_id_with_empty_part_returns_error(self, mock_wkt, configured_flm_extractor):
        """An image_id with empty parts (e.g. 'a||b') is invalid."""
        entity = {"id": "x", "geometry": FLM_WKT, "image_id": "a||b"}
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(entity)

        assert result["data"] is None
        assert "empty parts" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, mock_wkt, configured_flm_extractor, sample_flm_entity):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, mock_wkt, configured_flm_extractor, sample_flm_entity):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            configured_flm_extractor.process_single_entity_flm(sample_flm_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.FLM_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.FLM_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.FLM_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_flm_extractor,
        sample_flm_stats_response,
    ):
        """The notebook (cell 26) passes a pd.Series — process_single_entity_flm must accept it."""
        row = pd.Series({"id": "toto", "geometry": FLM_WKT, "image_id": FLM_IMAGE_ID})
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_flm_stats_response
        mock_retry.return_value = mock_resp

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_single_entity_flm(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_entity_flm_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.FLM_functions.export_results")
    @patch.object(FLMExtractor, "process_single_entity_flm")
    @patch.object(FLMExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(FLMExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_flm_extractor,
        sample_flm_entity_list,
    ):
        success_df = pd.DataFrame([{"stat_max": 0.85, "stat_mean": 0.6, "stat_min": 0.2}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_entity_flm_bulk_parallel(
                entity_list=sample_flm_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.extractors.FLM_functions.export_results")
    @patch.object(FLMExtractor, "process_single_entity_flm")
    @patch.object(FLMExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(FLMExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_flm_extractor,
        sample_flm_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_entity_flm_bulk_parallel(
                entity_list=sample_flm_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.FLM_functions.export_results")
    @patch.object(FLMExtractor, "process_single_entity_flm")
    @patch.object(FLMExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(FLMExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_flm_extractor,
        sample_flm_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"stat_max": 0.8}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_entity_flm_bulk_parallel(
                entity_list=sample_flm_entity_list,
                max_workers=1,  # sequential for deterministic side_effect ordering
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_flm_extractor, sample_flm_entity_list):
        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_flm_extractor.process_entity_flm_bulk_parallel(
                    entity_list=sample_flm_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.extractors.FLM_functions.export_results")
    @patch.object(FLMExtractor, "process_single_entity_flm")
    @patch.object(FLMExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(FLMExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_flm_extractor,
        sample_flm_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_entity_flm_bulk_parallel(
                entity_list=sample_flm_entity_list,
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

    @patch("earthdaily.agriculture.extractors.FLM_functions.export_results")
    @patch.object(FLMExtractor, "process_single_entity_flm")
    @patch.object(FLMExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(FLMExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_flm_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": FLM_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": FLM_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": FLM_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_flm_extractor, "ensure_token_valid"):
            result = configured_flm_extractor.process_entity_flm_bulk_parallel(
                entity_list=entity_list,
                filter_column="tag",
                filter_value="skip",
                filter_type="exclude",
                skip_export=True,
            )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
