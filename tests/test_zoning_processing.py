"""
Tests for ZoningExtractor processing pipeline:
    - process_single_entity_zoning() — routes to stats / stats_geo / links / file
    - process_entity_zoning_bulk_parallel() — bulk parallel processing
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.zoning_functions import ZoningExtractor
from tests.conftest import ZONING_IMAGE_ID, ZONING_IMAGE_ID_LIST, ZONING_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_zoning() — stats mode (default)
# ===================================================================


class TestProcessSingleEntityZoningStats:
    """Stats mode is the default (notebook cell 14, 22)."""

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_zoning_extractor,
        sample_zoning_entity,
        sample_zoning_stats_response,
    ):
        """Happy path: API returns stats response → result has a DataFrame, no error."""
        mock_retry.return_value = sample_zoning_stats_response

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_image_id_string_supported(
        self, mock_retry, mock_wkt, mock_normalize, configured_zoning_extractor, sample_zoning_stats_response
    ):
        """A single-string image_id should be accepted."""
        row = {"id": "ent_x", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID}
        mock_retry.return_value = sample_zoning_stats_response

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_image_id_list_supported(
        self, mock_retry, mock_wkt, mock_normalize, configured_zoning_extractor, sample_zoning_stats_response
    ):
        """A list image_id (workflow transform output) must be accepted (notebook cell 36)."""
        row = {"id": "ent_x", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID_LIST}
        mock_retry.return_value = sample_zoning_stats_response

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    def test_missing_image_id_returns_error(self, configured_zoning_extractor):
        """No image_id field on the row → error before any API call."""
        row = {"id": "ent_x", "geometry": ZONING_WKT}

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["data"] is None
        assert "Missing image_id" in result["error"]["message"]
        assert result["error"]["entity_id"] == "ent_x"

    def test_invalid_image_id_no_separator_returns_error(self, configured_zoning_extractor):
        """An image_id without any '|' separator should be rejected."""
        row = {"id": "ent_x", "geometry": ZONING_WKT, "image_id": "no_separator_here"}

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["data"] is None
        assert "Invalid image_id" in result["error"]["message"]

    def test_invalid_image_id_too_many_separators_returns_error(self, configured_zoning_extractor):
        """An image_id with more than 3 '|' separators should be rejected."""
        row = {"id": "ent_x", "geometry": ZONING_WKT, "image_id": "a|b|c|d|e"}

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["data"] is None
        assert "Invalid image_id" in result["error"]["message"]

    def test_invalid_image_id_empty_parts_returns_error(self, configured_zoning_extractor):
        """An image_id with empty parts between '|' should be rejected."""
        row = {"id": "ent_x", "geometry": ZONING_WKT, "image_id": "a||c"}

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["data"] is None
        assert "empty parts" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_geometry_validation_failure_returns_error(self, mock_wkt, configured_zoning_extractor):
        """Geometry validation failure should short-circuit before any API call."""
        row = {"id": "ent_001", "geometry": "INVALID_WKT", "image_id": ZONING_IMAGE_ID}

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "ent_001"

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, mock_wkt, configured_zoning_extractor, sample_zoning_entity):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self, mock_retry, mock_wkt, configured_zoning_extractor, sample_zoning_entity
    ):
        """retry_with_backoff_no_retry_on_400 must be called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = {}  # empty response → "Empty API response" error path

        configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self, mock_retry, mock_wkt, mock_normalize, configured_zoning_extractor, sample_zoning_stats_response
    ):
        """The notebook (cell 22) passes a pd.Series — must be accepted."""
        row = pd.Series(
            {
                "id": "test_001",
                "name": "Test_Field",
                "geometry": ZONING_WKT,
                "image_id": ZONING_IMAGE_ID_LIST,
            }
        )
        mock_retry.return_value = sample_zoning_stats_response

        result = configured_zoning_extractor.process_single_entity_zoning(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_single_entity_zoning() — stats_geo mode
# ===================================================================


class TestProcessSingleEntityZoningStatsGeo:
    """stats_geo mode emits one row per zone (notebook cell 28)."""

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_stats_geo_mode_returns_one_row_per_zone(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_zoning_extractor,
        sample_zoning_entity,
        sample_zoning_stats_geo_response,
    ):
        configured_zoning_extractor.zoning_params["postprocess"] = "stats_geo"
        mock_retry.return_value = sample_zoning_stats_geo_response

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 3
        assert "zone_geometry" in df.columns
        assert "zone_name" in df.columns


# ===================================================================
# process_single_entity_zoning() — links mode
# ===================================================================


class TestProcessSingleEntityZoningLinks:
    """Links mode emits a single row of URLs/metadata (notebook cell 24)."""

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_links_mode_returns_single_row(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_zoning_extractor,
        sample_zoning_entity,
        sample_zoning_links_response,
    ):
        configured_zoning_extractor.zoning_params["postprocess"] = "links"
        configured_zoning_extractor.zoning_params["directLinks"] = True
        mock_retry.return_value = sample_zoning_links_response

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["image_png_link"].endswith(".png")


# ===================================================================
# process_single_entity_zoning() — file mode
# ===================================================================


class TestProcessSingleEntityZoningFile:
    """File mode saves the raw response and returns a summary DataFrame (notebook cell 26)."""

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_file_mode_success_returns_summary(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_zoning_extractor,
        sample_zoning_entity,
        tmp_path,
    ):
        """A successful file save should return a 1-row summary DataFrame."""
        configured_zoning_extractor.zoning_params["postprocess"] = "file"
        configured_zoning_extractor.zoning_params["map_format"] = "png"
        configured_zoning_extractor.zoning_params["output_path"] = str(tmp_path)

        # Simulate a Response object with binary PNG content
        mock_response = MagicMock()
        mock_response.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_retry.return_value = mock_response

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["status"] == "downloaded"
        assert df.iloc[0]["map_format"] == "png"
        assert df.iloc[0]["file_count"] == 1

    @patch(
        "earthdaily.agriculture.extractors.zoning_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.extractors.zoning_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.zoning_functions.retry_with_backoff_no_retry_on_400")
    def test_file_mode_save_failure_returns_error(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_zoning_extractor,
        sample_zoning_entity,
    ):
        """File save with no output_path on params or extractor should produce a save error."""
        configured_zoning_extractor.zoning_params["postprocess"] = "file"
        configured_zoning_extractor.zoning_params["map_format"] = "png"
        configured_zoning_extractor.zoning_params["output_path"] = None
        configured_zoning_extractor.output_path = None

        mock_response = MagicMock()
        mock_response.content = b"\x89PNG..."
        mock_retry.return_value = mock_response

        result = configured_zoning_extractor.process_single_entity_zoning(sample_zoning_entity)

        assert result["data"] is None
        assert "Failed to save file" in result["error"]["message"]


# ===================================================================
# process_entity_zoning_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.zoning_functions.export_results")
    @patch.object(ZoningExtractor, "process_single_entity_zoning")
    @patch.object(ZoningExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ZoningExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_zoning_extractor,
        sample_zoning_entity_list,
    ):
        success_df = pd.DataFrame([{"field_variability": "MEDIUM", "field_productivity_index": 0.6}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_zoning_extractor.process_entity_zoning_bulk_parallel(
            entity_list=sample_zoning_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.extractors.zoning_functions.export_results")
    @patch.object(ZoningExtractor, "process_single_entity_zoning")
    @patch.object(ZoningExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ZoningExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_zoning_extractor,
        sample_zoning_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_zoning_extractor.process_entity_zoning_bulk_parallel(
            entity_list=sample_zoning_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.zoning_functions.export_results")
    @patch.object(ZoningExtractor, "process_single_entity_zoning")
    @patch.object(ZoningExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ZoningExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_zoning_extractor,
        sample_zoning_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"x": 1}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_zoning_extractor.process_entity_zoning_bulk_parallel(
            entity_list=sample_zoning_entity_list,
            max_workers=1,  # sequential for deterministic side_effect ordering
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_zoning_extractor, sample_zoning_entity_list):
        with pytest.raises(ValueError, match="merge_existing"):
            configured_zoning_extractor.process_entity_zoning_bulk_parallel(
                entity_list=sample_zoning_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.extractors.zoning_functions.export_results")
    @patch.object(ZoningExtractor, "process_single_entity_zoning")
    @patch.object(ZoningExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ZoningExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_zoning_extractor,
        sample_zoning_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_zoning_extractor.process_entity_zoning_bulk_parallel(
            entity_list=sample_zoning_entity_list,
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

    @patch("earthdaily.agriculture.extractors.zoning_functions.export_results")
    @patch.object(ZoningExtractor, "process_single_entity_zoning")
    @patch.object(ZoningExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(ZoningExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_zoning_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID, "tag": "skip"},
                {"id": "ent_002", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID, "tag": "keep"},
                {"id": "ent_003", "geometry": ZONING_WKT, "image_id": ZONING_IMAGE_ID, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_zoning_extractor.process_entity_zoning_bulk_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
