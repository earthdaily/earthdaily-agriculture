"""Tests for LocationBasedBorderExtractor processing surface:
- process_single_entity_location_based_border
- process_location_based_border_bulk_extraction_parallel (smoke + filter)
"""

from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import (
    LOCATION_BORDER_POINT_WKT,
    LOCATION_BORDER_POLYGON_WKT,
)

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_location_based_border
# ===================================================================


class TestProcessSingleEntityLocationBasedBorder:
    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_retry.return_value = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(
            sample_location_border_entity
        )
        assert result["error"] is None
        assert result["data"] is not None
        df = result["data"]
        assert df.iloc[0]["polygon_geometry"] == LOCATION_BORDER_POLYGON_WKT

    def test_validation_failure_polygon_returns_error(self, configured_location_based_border_extractor):
        polygon = "POLYGON ((-93.6 41.5, -93.59 41.5, -93.59 41.51, -93.6 41.51, -93.6 41.5))"
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(
            {"id": "loc_001", "geometry": polygon}
        )
        assert result["data"] is None
        assert "Point" in result["error"]["message"]

    def test_invalid_geometry_returns_error(self, configured_location_based_border_extractor):
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(
            {"id": "loc_001", "geometry": "INVALID"}
        )
        assert result["data"] is None
        assert result["error"]["entity_id"] == "loc_001"

    @patch(
        "earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400",
        side_effect=Exception("API blew up"),
    )
    def test_api_exception_returns_error(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity
    ):
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(
            sample_location_border_entity
        )
        assert result["data"] is None
        assert "API blew up" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity
    ):
        # validate_api_response treats falsy responses as "no data" → empty DataFrame
        mock_retry.return_value = {}
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(
            sample_location_border_entity
        )
        assert result["data"] is None
        assert result["error"]["entity_id"] == "loc_001"

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity
    ):
        mock_retry.return_value = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        series = pd.Series(sample_location_border_entity)
        result = configured_location_based_border_extractor.process_single_entity_location_based_border(series)
        assert result["error"] is None


# ===================================================================
# Bulk processing
# ===================================================================


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_bulk_all_success(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity_list
    ):
        mock_retry.return_value = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        result = configured_location_based_border_extractor.process_location_based_border_bulk_extraction_parallel(
            entity_list=sample_location_border_entity_list,
            max_workers=2,
            skip_export=True,
        )
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["results_df"]) == 3

    @patch(
        "earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400",
        side_effect=Exception("network error"),
    )
    def test_bulk_all_fail(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity_list
    ):
        result = configured_location_based_border_extractor.process_location_based_border_bulk_extraction_parallel(
            entity_list=sample_location_border_entity_list,
            max_workers=2,
            skip_export=True,
        )
        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_bulk_returns_expected_keys(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity_list
    ):
        mock_retry.return_value = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        result = configured_location_based_border_extractor.process_location_based_border_bulk_extraction_parallel(
            entity_list=sample_location_border_entity_list,
            max_workers=2,
            skip_export=True,
        )
        for key in (
            "results_df",
            "global_errors",
            "total_entities",
            "total_calculations",
            "successful_calculations",
            "failed_calculations",
            "failed_ids",
        ):
            assert key in result

    def test_bulk_invalid_merge_existing_raises(
        self, configured_location_based_border_extractor, sample_location_border_entity_list
    ):
        with pytest.raises(ValueError, match="Invalid merge_existing"):
            configured_location_based_border_extractor.process_location_based_border_bulk_extraction_parallel(
                entity_list=sample_location_border_entity_list,
                merge_existing="bogus",
            )

    @patch("earthdaily.agriculture.extractors.location_based_border_functions.retry_with_backoff_no_retry_on_400")
    def test_bulk_filter_exclude_drops_matching_entities(
        self, mock_retry, configured_location_based_border_extractor, sample_location_border_entity_list
    ):
        mock_retry.return_value = {
            "id": "loc_001",
            "point_wkt": LOCATION_BORDER_POINT_WKT,
            "geometry": LOCATION_BORDER_POLYGON_WKT,
        }
        df = sample_location_border_entity_list.copy()
        df["region"] = ["IA", "IA", "MN"]
        result = configured_location_based_border_extractor.process_location_based_border_bulk_extraction_parallel(
            entity_list=df,
            filter_column="region",
            filter_value="MN",
            filter_type="exclude",
            max_workers=2,
            skip_export=True,
        )
        # 2 of 3 entities processed (MN excluded).
        assert result["total_calculations"] == 2
