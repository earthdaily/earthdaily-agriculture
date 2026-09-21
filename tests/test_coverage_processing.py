"""
Tests for CoverageExtractor processing pipeline:
    - process_single_entity_coverage() — single entity with retry
    - process_entity_coverage_bulk_parallel() — bulk parallel processing
    - crop_coverage filter mode (date expansion + per-year post-filter)
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
from tests.conftest import COVERAGE_BRAZIL_WKT, COVERAGE_PARIS_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_coverage()
# ===================================================================


class TestProcessSingleEntityCoverage:
    """Tests for the single-entity processing pipeline."""

    @patch("earthdaily.agriculture.extractors.coverage_function.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_coverage_extractor,
    ):
        """Happy path: API returns records → result contains a non-empty DataFrame.

        Entity matches the notebook's process_single_entity_coverage row (cell 23).
        """
        row = {"id": "z361x33", "geometry": COVERAGE_BRAZIL_WKT}
        mock_retry.return_value = [
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2C_T31UDQ_20260429",
                    "spatialResolution": 10.0,
                    "date": "2026-04-29T10:57:29Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            }
        ]
        mock_normalize.side_effect = lambda r, df: df

        # Disable filter so we don't go through _filter_duplicates (which would
        # drop the single-record DataFrame).
        configured_coverage_extractor.coverage_params["filter"] = "none"
        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_no_imagery_error(self, mock_retry, mock_wkt, configured_coverage_extractor):
        """If API returns an empty list, result should carry a 'No imagery found' error."""
        row = {"id": "z361x33", "geometry": COVERAGE_BRAZIL_WKT}
        mock_retry.return_value = []

        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["data"] is None
        assert result["error"] is not None
        assert "No imagery found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    @patch(
        "earthdaily.agriculture.extractors.coverage_function.validate_wkt",
        side_effect=ValueError("invalid geometry"),
    )
    def test_validation_failure_returns_error(self, mock_wkt, configured_coverage_extractor):
        """Geometry validation failure should short-circuit before any API call."""
        row = {"id": "ent_001", "geometry": "INVALID_WKT"}

        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["data"] is None
        assert "invalid geometry" in result["error"]["message"]
        assert result["error"]["entity_id"] == "ent_001"

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(self, mock_retry, mock_wkt, configured_coverage_extractor):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        row = {"id": "ent_001", "geometry": COVERAGE_PARIS_WKT}
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(self, mock_retry, mock_wkt, configured_coverage_extractor):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        row = {"id": "ent_001", "geometry": COVERAGE_PARIS_WKT}
        mock_retry.return_value = []  # empty → no imagery found error

        configured_coverage_extractor.process_single_entity_coverage(row)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.coverage_function.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(self, mock_retry, mock_wkt, mock_normalize, configured_coverage_extractor):
        """The notebook passes a pd.Series — process_single_entity_coverage must accept it."""
        row = pd.Series({"id": "z361x33", "geometry": COVERAGE_BRAZIL_WKT})
        mock_retry.return_value = [
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2C",
                    "spatialResolution": 10.0,
                    "date": "2026-04-29T10:57:29Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            }
        ]
        mock_normalize.side_effect = lambda r, df: df

        configured_coverage_extractor.coverage_params["filter"] = "none"
        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_single_entity_coverage() — crop_coverage mode
# ===================================================================


class TestProcessSingleEntityCropCoverage:
    """When filter='crop_coverage', date expansion and post-filtering apply."""

    @patch("earthdaily.agriculture.extractors.coverage_function.normalize_with_metadata")
    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_crop_coverage_post_filter_keeps_only_in_window_dates(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_coverage_extractor,
    ):
        """
        Mirrors notebook cell 34: filter='crop_coverage' with historical_seasons
        keeps only images within [start_date, end_date] for each historical year.
        """
        configured_coverage_extractor.coverage_params.update(
            {
                "filter": "crop_coverage",
                "start_date": "2025-07-01",
                "end_date": "2025-07-31",
                "historical_seasons": [2022, 2024, 2025],
                "clear_cover_min": 0,  # don't lose records to coverage threshold
            }
        )

        # API returns records spanning multiple years; only July dates per year
        # in [2022, 2024, 2025] should survive the post-filter.
        mock_retry.return_value = [
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2A_2022",
                    "spatialResolution": 10.0,
                    "date": "2022-07-15T10:00:00Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            },
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2A_2023",
                    "spatialResolution": 10.0,
                    "date": "2023-07-15T10:00:00Z",  # 2023 not in historical_seasons
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            },
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2A_2024",
                    "spatialResolution": 10.0,
                    "date": "2024-07-20T10:00:00Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            },
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2A_2025_aug",
                    "spatialResolution": 10.0,
                    "date": "2025-08-05T10:00:00Z",  # outside July window
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            },
        ]
        mock_normalize.side_effect = lambda r, df: df

        row = {"id": "ent_x", "geometry": COVERAGE_BRAZIL_WKT}
        result = configured_coverage_extractor.process_single_entity_coverage(row)

        assert result["error"] is None
        df = result["data"]
        # Should keep only the 2022 and 2024 July records (2025 record is in August;
        # 2023 not in historical_seasons).
        kept = set(df["date"])
        assert kept == {"2022-07-15", "2024-07-20"}

    @patch("earthdaily.agriculture.extractors.coverage_function.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.extractors.coverage_function.retry_with_backoff_no_retry_on_400")
    def test_crop_coverage_embeds_expanded_date_range_into_row(
        self,
        mock_retry,
        mock_wkt,
        configured_coverage_extractor,
    ):
        """
        process_single_entity_coverage should compute expanded crop_coverage dates
        (min year start → max year end) and embed them into the row passed downstream.
        """
        configured_coverage_extractor.coverage_params.update(
            {
                "filter": "crop_coverage",
                "start_date": "2025-07-01",
                "end_date": "2025-07-31",
                "historical_seasons": [2022, 2024, 2025],
            }
        )

        # Make the API call return empty so we exit before post-filter.
        # The retry callable executes our internal `_call_api`, which calls
        # get_satellite_coverage_by_geometry with the (now-expanded) row.
        captured = {}

        def fake_retry(fn, **kwargs):
            # Trigger the inner _call_api so we can inspect the patched API call
            return fn()

        mock_retry.side_effect = fake_retry

        with patch.object(CoverageExtractor, "get_satellite_coverage_by_geometry", autospec=True) as mock_api:

            def capture(self, entity):
                captured["row"] = entity
                return []

            mock_api.side_effect = capture
            row = {"id": "ent_x", "geometry": COVERAGE_BRAZIL_WKT}
            configured_coverage_extractor.process_single_entity_coverage(row)

        # Min year is 2022, max year is 2025
        assert captured["row"]["_coverage_start_date"] == "2022-07-01"
        assert captured["row"]["_coverage_end_date"] == "2025-07-31"

    def test_crop_coverage_per_entity_historical_seasons_override(self, configured_coverage_extractor):
        """Mirrors notebook cell 36: per-entity historical_seasons overrides global."""
        configured_coverage_extractor.coverage_params.update(
            {
                "filter": "crop_coverage",
                "start_date": "2025-07-01",
                "end_date": "2025-07-31",
                "historical_seasons": [2022, 2024, 2025],
            }
        )

        row = {
            "id": "ent_custom",
            "geometry": COVERAGE_BRAZIL_WKT,
            "historical_seasons": [2023, 2024],
        }
        start, end = configured_coverage_extractor._compute_crop_coverage_dates(row)
        assert start == "2023-07-01"
        assert end == "2024-07-31"


# ===================================================================
# process_entity_coverage_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
        sample_coverage_entity_list,
    ):
        """All entities succeed → successful_calculations == total."""
        success_df = pd.DataFrame([{"image_id": "img1", "coverage_percent": 100}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=sample_coverage_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
        sample_coverage_entity_list,
    ):
        """All entities fail → failed_calculations == total."""
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=sample_coverage_entity_list,
            max_workers=2,
            skip_export=True,
        )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
        sample_coverage_entity_list,
    ):
        """Mix of successes and failures → counts are correct."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"image_id": "x", "coverage_percent": 90}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=sample_coverage_entity_list,
            max_workers=1,  # sequential for deterministic side_effect ordering
            skip_export=True,
        )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_coverage_extractor, sample_coverage_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with pytest.raises(ValueError, match="merge_existing"):
            configured_coverage_extractor.process_entity_coverage_bulk_parallel(
                entity_list=sample_coverage_entity_list,
                merge_existing="invalid_mode",
                skip_export=True,
            )

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
        sample_coverage_entity_list,
    ):
        """Return dict should have all expected keys with correct types."""
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=sample_coverage_entity_list,
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

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_recalibration_flag_propagated(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
        sample_coverage_entity_list,
    ):
        """Calling bulk with recalibration=True should set coverage_params['recalibration']."""
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=sample_coverage_entity_list,
            skip_export=True,
            recalibration=True,
        )

        assert configured_coverage_extractor.coverage_params["recalibration"] is True

    @patch("earthdaily.agriculture.extractors.coverage_function.export_results")
    @patch.object(CoverageExtractor, "process_single_entity_coverage")
    @patch.object(CoverageExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(CoverageExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude_drops_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_coverage_extractor,
    ):
        """filter_type='exclude' on a column drops matching entities; remaining entities
        flow through process_single_entity. The other entity fields don't matter here
        because process_single_entity is mocked."""
        entity_list = pd.DataFrame(
            [
                {"id": "ent_001", "geometry": COVERAGE_PARIS_WKT, "tag": "skip"},
                {"id": "ent_002", "geometry": COVERAGE_PARIS_WKT, "tag": "keep"},
                {"id": "ent_003", "geometry": COVERAGE_PARIS_WKT, "tag": "skip"},
            ]
        )
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        result = configured_coverage_extractor.process_entity_coverage_bulk_parallel(
            entity_list=entity_list,
            filter_column="tag",
            filter_value="skip",
            filter_type="exclude",
            skip_export=True,
        )

        # Two 'skip' entities excluded → only 1 entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
