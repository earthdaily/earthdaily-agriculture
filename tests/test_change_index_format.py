"""
Tests for ChangeIndexExtractor.format_change_index_json():
    - Real notebook response → flattened single-row DataFrame
    - Edge cases: empty/None responses, non-dict input, unexpected data types
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_change_index_json() — happy path with real notebook response
# ===================================================================


class TestFormatChangeIndexJsonNotebookResponse:
    """Validates flattening of the real API response captured from the dev notebook."""

    def test_returns_single_row_dataframe(self, configured_change_index_extractor, sample_change_index_response):
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_top_level_fields_set(self, configured_change_index_extractor, sample_change_index_response):
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)
        row = df.iloc[0]

        assert row["entity_id"] == "z361x33"
        assert row["reference_date"] == "2025-06-15"
        assert row["status"] == "Processor done and metrics pushed."

    def test_metric_columns_flattened_from_data(self, configured_change_index_extractor, sample_change_index_response):
        """All metric keys nested under 'data' should appear as top-level columns."""
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)

        expected_metrics = {
            "CurrentMapAverage",
            "CurrentMapMinimum",
            "CurrentMapMaximum",
            "CurrentMapStddev",
            "CurrentMapVariance",
            "ReferenceMapAverage",
            "ReferenceMapMinimum",
            "ReferenceMapMaximum",
            "ReferenceMapStddev",
            "ReferenceMapVariance",
            "SPAEFIndex",
            "ChangeIndex",
            "SPAEFIndexV2",
            "ChangeIndexV2",
            "PearsonCoefficient",
            "Covariance",
            "HistogramMatching",
            "date_ref",
            "sensor_ref",
            "date_current",
            "sensor_current",
        }
        assert expected_metrics.issubset(set(df.columns))

    def test_metric_values_preserved(self, configured_change_index_extractor, sample_change_index_response):
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)
        row = df.iloc[0]

        assert row["SPAEFIndex"] == 0.71
        assert row["ChangeIndex"] == 1.43
        assert row["SPAEFIndexV2"] == 0.52
        assert row["ChangeIndexV2"] == 2.41
        assert row["Covariance"] == 1.0
        assert row["sensor_ref"] == "SENTINEL_2"
        assert row["sensor_current"] == "SENTINEL_2"
        assert row["date_ref"] == "2025-06-12T14:16:31Z"
        assert row["date_current"] == "2025-06-07T14:16:54Z"

    def test_inner_status_does_not_clobber_outer_status(
        self, configured_change_index_extractor, sample_change_index_response
    ):
        """The 'status' key inside 'data' is intentionally skipped during flattening."""
        # Tamper the inner status to make sure the outer one wins.
        sample_change_index_response["data"]["status"] = "INNER_STATUS_SHOULD_BE_IGNORED"
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)

        assert df.iloc[0]["status"] == "Processor done and metrics pushed."

    def test_total_columns_match_notebook_count(self, configured_change_index_extractor, sample_change_index_response):
        """The notebook reports 24 columns: 3 top-level + 21 metric columns."""
        df = configured_change_index_extractor.format_change_index_json(sample_change_index_response)
        assert df.shape[1] == 24


# ===================================================================
# format_change_index_json() — edge cases
# ===================================================================


class TestFormatChangeIndexJsonEdgeCases:
    def test_empty_dict_returns_empty_df(self, configured_change_index_extractor):
        df = configured_change_index_extractor.format_change_index_json({})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_none_response_raises(self, configured_change_index_extractor):
        """None is rejected as non-dict input."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_change_index_extractor.format_change_index_json(None)

    def test_non_dict_input_raises(self, configured_change_index_extractor):
        with pytest.raises(ValueError, match="must be a dictionary"):
            configured_change_index_extractor.format_change_index_json("not_a_dict")

    def test_data_field_missing_returns_only_top_level(self, configured_change_index_extractor):
        """If 'data' is missing, the row still gets entity_id, reference_date, and status."""
        response = {"id": "z361x33", "reference_date": "2025-06-15", "status": "ok"}
        df = configured_change_index_extractor.format_change_index_json(response)

        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "z361x33"
        assert df.iloc[0]["status"] == "ok"
        # No metric columns added
        assert "ChangeIndex" not in df.columns

    def test_data_field_not_dict_logs_warning_and_returns_top_level_only(self, configured_change_index_extractor):
        """If 'data' is a list or string, no flattening occurs but the row is still returned."""
        response = {
            "id": "z361x33",
            "reference_date": "2025-06-15",
            "status": "ok",
            "data": ["unexpected", "list"],
        }
        df = configured_change_index_extractor.format_change_index_json(response)

        assert len(df) == 1
        # Logger warning was issued
        configured_change_index_extractor.logger.bind.return_value.warning.assert_called()

    def test_unexpected_top_level_keys_promoted(self, configured_change_index_extractor):
        """Unknown top-level keys should be promoted onto the row (future-proofing)."""
        response = {
            "id": "z361x33",
            "reference_date": "2025-06-15",
            "status": "ok",
            "data": {"ChangeIndex": 1.0},
            "extra_field": "future_value",
        }
        df = configured_change_index_extractor.format_change_index_json(response)

        assert df.iloc[0]["extra_field"] == "future_value"
        assert df.iloc[0]["ChangeIndex"] == 1.0
