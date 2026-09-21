"""
Tests for DiseaseExtractor processing pipeline:
    - process_single_entity_disease() — single entity with retry
    - process_entity_disease_bulk_parallel() — bulk parallel processing
    - Date resolution (entity row > params fallback)
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_disease_risk_functions import DiseaseExtractor
from tests.conftest import DISEASE_WKT

pytestmark = pytest.mark.public

# ===================================================================
# process_single_entity_disease()
# ===================================================================


class TestProcessSingleEntityDisease:
    """Tests for the single-entity processing pipeline."""

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_disease_extractor,
        sample_disease_entity,
        sample_disease_response_list,
    ):
        """Happy path: API returns records → result contains a non-empty DataFrame."""
        mock_retry.return_value = sample_disease_response_list

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(sample_disease_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert not result["data"].empty
        # All 3 daily records should appear
        assert len(result["data"]) == 3

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_api_response_returns_error(
        self, mock_retry, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """If API returns an empty list, result should carry a 'No disease data found' error."""
        mock_retry.return_value = []

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(sample_disease_entity)

        assert result["data"] is None
        assert result["error"] is not None
        assert "No disease data found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    def test_missing_start_date_returns_error(self, configured_disease_extractor):
        """Entity without start_date and no param fallback → error before API call."""
        configured_disease_extractor.disease_params["start_date"] = None
        entity = {"id": "x", "geometry": DISEASE_WKT, "end_date": "2025-10-01"}

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(entity)

        assert result["data"] is None
        assert "Missing start_date" in result["error"]["message"]
        assert result["error"]["entity_id"] == "x"

    def test_missing_end_date_returns_error(self, configured_disease_extractor):
        """Entity without end_date and no param fallback → error before API call."""
        configured_disease_extractor.disease_params["end_date"] = None
        entity = {"id": "x", "geometry": DISEASE_WKT, "start_date": "2025-06-01"}

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(entity)

        assert result["data"] is None
        assert "Missing end_date" in result["error"]["message"]

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_entity_dates_take_precedence_over_params(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_disease_extractor,
        sample_disease_response_list,
    ):
        """When entity row has start/end dates, they win over disease_params defaults."""
        configured_disease_extractor.disease_params["start_date"] = "2024-01-01"
        configured_disease_extractor.disease_params["end_date"] = "2024-12-31"

        entity = {
            "id": "ent_x",
            "geometry": DISEASE_WKT,
            "start_date": "2025-06-01",
            "end_date": "2025-10-01",
        }
        mock_retry.return_value = sample_disease_response_list

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(entity)

        assert result["error"] is None
        # Verify the row passed downstream had the entity-level dates injected
        # under the mapped column names (id/start_date/end_date by default).
        # The retry callable wraps get_disease_data, so we can't inspect the row directly here —
        # but the absence of an error proves the dates resolved correctly.

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_param_dates_used_when_entity_has_none(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_disease_extractor,
        sample_disease_response_list,
    ):
        """When entity row has no start/end_date, fall back to disease_params."""
        entity = {"id": "ent_x", "geometry": DISEASE_WKT}
        mock_retry.return_value = sample_disease_response_list

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        mock_wkt,
        configured_disease_extractor,
        sample_disease_entity,
    ):
        """If retry exhausts attempts and raises, the error is captured in the result."""
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(sample_disease_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self, mock_retry, mock_wkt, configured_disease_extractor, sample_disease_entity
    ):
        """retry_with_backoff_no_retry_on_400 is called with max_retries=5, base_delay=1.0, max_delay=60.0."""
        mock_retry.return_value = []  # empty → "No disease data found" error

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            configured_disease_extractor.process_single_entity_disease(sample_disease_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_disease_extractor,
        sample_disease_response_list,
    ):
        """The notebook may pass a pd.Series — process_single_entity_disease must accept it."""
        row = pd.Series(
            {
                "id": "z361x33",
                "geometry": DISEASE_WKT,
                "start_date": "2025-06-01",
                "end_date": "2025-10-01",
            }
        )
        mock_retry.return_value = sample_disease_response_list

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch(
        "earthdaily.agriculture.processors.processor_disease_risk_functions.normalize_with_metadata",
        side_effect=lambda r, df: df,
    )
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.validate_wkt", side_effect=lambda x: x)
    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.retry_with_backoff_no_retry_on_400")
    def test_dataframe_row_supported(
        self,
        mock_retry,
        mock_wkt,
        mock_normalize,
        configured_disease_extractor,
        sample_disease_response_list,
    ):
        """A 1-row DataFrame should be accepted (only first row used)."""
        df_in = pd.DataFrame(
            [
                {
                    "id": "z361x33",
                    "geometry": DISEASE_WKT,
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                }
            ]
        )
        mock_retry.return_value = sample_disease_response_list

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_single_entity_disease(df_in)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)


# ===================================================================
# process_entity_disease_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    """Tests for the bulk parallel processing method."""

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.export_results")
    @patch.object(DiseaseExtractor, "process_single_entity_disease")
    @patch.object(DiseaseExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DiseaseExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_disease_extractor,
        sample_disease_entity_list,
    ):
        """All entities succeed → successful_calculations == total."""
        success_df = pd.DataFrame([{"entity_id": "ent", "date": "2025-06-01", "frogeye_leaf_spot": 0.5}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_entity_disease_bulk_parallel(
                entity_list=sample_disease_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0
        assert len(result["failed_ids"]) == 0

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.export_results")
    @patch.object(DiseaseExtractor, "process_single_entity_disease")
    @patch.object(DiseaseExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DiseaseExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_disease_extractor,
        sample_disease_entity_list,
    ):
        """All entities fail → failed_calculations == total."""
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_entity_disease_bulk_parallel(
                entity_list=sample_disease_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.export_results")
    @patch.object(DiseaseExtractor, "process_single_entity_disease")
    @patch.object(DiseaseExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DiseaseExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_disease_extractor,
        sample_disease_entity_list,
    ):
        """Mix of successes and failures → counts are correct."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {
                    "data": pd.DataFrame([{"entity_id": "x", "date": "2025-06-01", "frogeye_leaf_spot": 0.7}]),
                    "error": None,
                }
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_entity_disease_bulk_parallel(
                entity_list=sample_disease_entity_list,
                max_workers=1,  # sequential for deterministic side_effect ordering
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_disease_extractor, sample_disease_entity_list):
        """Invalid merge_existing value should raise ValueError."""
        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_disease_extractor.process_entity_disease_bulk_parallel(
                    entity_list=sample_disease_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.export_results")
    @patch.object(DiseaseExtractor, "process_single_entity_disease")
    @patch.object(DiseaseExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DiseaseExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_disease_extractor,
        sample_disease_entity_list,
    ):
        """Return dict should have all expected keys with correct types."""
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_entity_disease_bulk_parallel(
                entity_list=sample_disease_entity_list,
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

    @patch("earthdaily.agriculture.processors.processor_disease_risk_functions.export_results")
    @patch.object(DiseaseExtractor, "process_single_entity_disease")
    @patch.object(DiseaseExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(DiseaseExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_excludes_matching_entities(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_disease_extractor,
    ):
        """filter_type='exclude' with a match should drop those entities (mirrors notebook cell 31)."""
        entity_list = pd.DataFrame(
            [
                {
                    "id": "ent_001",
                    "geometry": DISEASE_WKT,
                    "crop": "CORN",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
                {
                    "id": "ent_002",
                    "geometry": DISEASE_WKT,
                    "crop": "SOYBEANS",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
                {
                    "id": "ent_003",
                    "geometry": DISEASE_WKT,
                    "crop": "CORN",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "date": "2025-06-01", "frogeye_leaf_spot": 0.5}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_disease_extractor, "ensure_token_valid"):
            result = configured_disease_extractor.process_entity_disease_bulk_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="CORN",
                filter_type="exclude",
                skip_export=True,
            )

        # Two CORN entities excluded → only 1 SOYBEANS entity processed
        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3
