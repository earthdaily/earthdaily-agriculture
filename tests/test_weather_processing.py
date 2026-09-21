"""
Tests for WeatherExtractor processing pipeline:
    - process_single_entity_weather() — single entity with retry, KPI mode, date expansion
    - process_entity_weather_bulk_parallel() — bulk parallel processing
"""

import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.weather_functions import WeatherExtractor
from tests.conftest import WEATHER_WKT

pytestmark = pytest.mark.public


def _weather_server_sim(value=1.0):
    """Return a requests.get side_effect that mimics the Weather API: one row per day in
    ascending date order over the URL's ``$between`` range, truncated to the URL's ``$limit``
    (the server keeps the *earliest* N rows). Lets tests reproduce silent truncation without
    a live call — a too-small $limit drops the most recent days."""

    def _server(url, *args, **kwargs):
        limit = int(re.search(r"\$limit=(\d+)", url).group(1))
        between = re.search(r"Date=\$between:(\d{4}-\d{2}-\d{2})T[^|]*\|(\d{4}-\d{2}-\d{2})T", url)
        start = datetime.strptime(between.group(1), "%Y-%m-%d").date()
        end = datetime.strptime(between.group(2), "%Y-%m-%d").date()

        rows = []
        day = start
        while day <= end:
            rows.append({"date": day.strftime("%Y-%m-%dT00:00:00Z"), "wind": value})
            day += timedelta(days=1)

        resp = MagicMock()
        resp.json.return_value = rows[:limit]  # server truncates to the earliest `limit` rows
        resp.raise_for_status.return_value = None
        return resp

    return _server


# ===================================================================
# process_single_entity_weather() — non-KPI mode
# ===================================================================


class TestProcessSingleEntityWeather:
    @patch("earthdaily.agriculture.extractors.weather_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_successful_entity_returns_dataframe(
        self,
        mock_retry,
        mock_normalize,
        configured_weather_extractor,
        sample_weather_entity,
        sample_weather_response,
    ):
        mock_retry.return_value = sample_weather_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 3

    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_empty_response_returns_error(
        self,
        mock_retry,
        configured_weather_extractor,
        sample_weather_entity,
        sample_weather_response_empty,
    ):
        """Empty list → 'No weather data found'."""
        mock_retry.return_value = sample_weather_response_empty

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        assert result["data"] is None
        assert "No weather data found" in result["error"]["message"]
        assert result["error"]["entity_id"] == "z361x33"

    def test_missing_start_date_returns_error(self, configured_weather_extractor):
        """If neither row nor params has start_date, processor returns error result."""
        entity = {"id": "x", "geometry": WEATHER_WKT, "end_date": "2025-10-01"}

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(entity)

        assert result["data"] is None
        assert "Missing start_date" in result["error"]["message"]

    def test_missing_end_date_returns_error(self, configured_weather_extractor):
        entity = {"id": "x", "geometry": WEATHER_WKT, "start_date": "2025-06-01"}

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(entity)

        assert result["data"] is None
        assert "Missing end_date" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_api_exception_returns_error(
        self,
        mock_retry,
        configured_weather_extractor,
        sample_weather_entity,
    ):
        mock_retry.side_effect = RuntimeError("max retries exceeded")

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        assert result["data"] is None
        assert "max retries exceeded" in result["error"]["message"]

    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_retry_called_with_correct_params(
        self,
        mock_retry,
        configured_weather_extractor,
        sample_weather_entity,
    ):
        mock_retry.return_value = []

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["max_retries"] == 5
        assert call_kwargs.kwargs["base_delay"] == 1.0
        assert call_kwargs.kwargs["max_delay"] == 60.0

    @patch("earthdaily.agriculture.extractors.weather_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_pandas_series_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_weather_extractor,
        sample_weather_response,
    ):
        """Notebook cell 25 passes a pd.Series — process_single_entity_weather must accept it."""
        row = pd.Series(
            {
                "id": "z361x33",
                "geometry": WEATHER_WKT,
                "crop": "SOYBEANS",
                "start_date": "2025-06-01",
                "end_date": "2025-10-01",
                "years": [2024, 2023, 2022],
            }
        )
        mock_retry.return_value = sample_weather_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(row)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    @patch("earthdaily.agriculture.extractors.weather_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_dataframe_row_supported(
        self,
        mock_retry,
        mock_normalize,
        configured_weather_extractor,
        sample_weather_response,
    ):
        """A 1-row DataFrame is accepted (only first row used)."""
        df_in = pd.DataFrame(
            [
                {
                    "id": "z361x33",
                    "geometry": WEATHER_WKT,
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                }
            ]
        )
        mock_retry.return_value = sample_weather_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(df_in)

        assert result["error"] is None
        assert isinstance(result["data"], pd.DataFrame)

    def test_invalid_input_type_returns_error(self, configured_weather_extractor):
        """A non-dict/Series/DataFrame input should produce an error result."""
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather("not a row")
        assert result["data"] is None
        assert "Invalid input" in result["error"]["message"]


# ===================================================================
# process_single_entity_weather() — KPI mode
# ===================================================================


class TestProcessSingleEntityWeatherKpi:
    @patch("earthdaily.agriculture.extractors.weather_functions.normalize_with_metadata", side_effect=lambda r, df: df)
    @patch("earthdaily.agriculture.extractors.weather_functions.filter_timeseries_kpi")
    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_mode_returns_single_row(
        self,
        mock_retry,
        mock_kpi,
        mock_normalize,
        configured_weather_extractor,
        sample_weather_entity,
        sample_weather_response,
    ):
        """KPI mode: kpi_filter='accumulation' produces a single-row KPI DataFrame."""
        configured_weather_extractor.weather_params["kpi_filter"] = {
            "kpi_name": "Summer Rainfall",
            "aggregation": "accumulation",
            "value_column": "precipitation.cumulative",
        }
        configured_weather_extractor.weather_params["historical_years"] = 5

        mock_retry.return_value = sample_weather_response
        mock_kpi.return_value = {
            "kpi_name": "Summer Rainfall",
            "aggregation": "accumulation",
            "current_period": {"value": 215.0, "num_records": 90},
            "historical_avg": {"value": 198.0, "num_years": 4},
            "comparison": {"difference": 17.0, "percent_change": 8.6},
        }

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        assert result["error"] is None
        df = result["data"]
        assert len(df) == 1
        assert df.iloc[0]["kpi_name"] == "Summer Rainfall"
        assert df.iloc[0]["kpi_column"] == "precipitation.cumulative"
        assert df.iloc[0]["current_value"] == 215.0
        assert df.iloc[0]["historical_avg"] == 198.0
        mock_kpi.assert_called_once()

    @patch(
        "earthdaily.agriculture.extractors.weather_functions.filter_timeseries_kpi",
        side_effect=RuntimeError("kpi blew up"),
    )
    @patch("earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400")
    def test_kpi_failure_returns_error(
        self,
        mock_retry,
        mock_kpi,
        configured_weather_extractor,
        sample_weather_entity,
        sample_weather_response,
    ):
        configured_weather_extractor.weather_params["kpi_filter"] = {
            "kpi_name": "test",
            "aggregation": "accumulation",
            "value_column": "Temperature.standardmax",
        }
        mock_retry.return_value = sample_weather_response

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_single_entity_weather(sample_weather_entity)

        assert result["data"] is None
        assert "KPI computation failed" in result["error"]["message"]
        assert "kpi blew up" in result["error"]["message"]


# ===================================================================
# process_entity_weather_bulk_parallel()
# ===================================================================


_FINALIZE_RETURN = (
    pd.DataFrame(),
    {"exported": True, "export_path": "/tmp", "failed_ids_saved": False, "partials_cleaned": False},
)


class TestBulkExtractionParallel:
    @patch("earthdaily.agriculture.extractors.weather_functions.export_results")
    @patch.object(WeatherExtractor, "process_single_entity_weather")
    @patch.object(WeatherExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(WeatherExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_success(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_weather_extractor,
        sample_weather_entity_list,
    ):
        success_df = pd.DataFrame([{"entity_id": "ent", "date": "2025-06-01", "Temperature.standardmax": 24.1}])
        mock_single.return_value = {"data": success_df, "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_entity_weather_bulk_parallel(
                entity_list=sample_weather_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["total_entities"] == 3
        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 3
        assert result["failed_calculations"] == 0

    @patch("earthdaily.agriculture.extractors.weather_functions.export_results")
    @patch.object(WeatherExtractor, "process_single_entity_weather")
    @patch.object(WeatherExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(WeatherExtractor, "_merge_with_skipped_entities")
    def test_bulk_all_fail(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_weather_extractor,
        sample_weather_entity_list,
    ):
        mock_single.return_value = {"data": None, "error": {"message": "API error", "entity_id": "x"}}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_entity_weather_bulk_parallel(
                entity_list=sample_weather_entity_list,
                max_workers=2,
                skip_export=True,
            )

        assert result["successful_calculations"] == 0
        assert result["failed_calculations"] == 3
        assert len(result["failed_ids"]) == 3

    @patch("earthdaily.agriculture.extractors.weather_functions.export_results")
    @patch.object(WeatherExtractor, "process_single_entity_weather")
    @patch.object(WeatherExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(WeatherExtractor, "_merge_with_skipped_entities")
    def test_bulk_mixed_results(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_weather_extractor,
        sample_weather_entity_list,
    ):
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"data": pd.DataFrame([{"entity_id": "x", "wind": 3.2}]), "error": None}
            return {"data": None, "error": {"message": "fail", "entity_id": "y"}}

        mock_single.side_effect = side_effect
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_entity_weather_bulk_parallel(
                entity_list=sample_weather_entity_list,
                max_workers=1,
                skip_export=True,
            )

        assert result["total_calculations"] == 3
        assert result["successful_calculations"] == 2
        assert result["failed_calculations"] == 1

    def test_bulk_invalid_merge_existing_raises(self, configured_weather_extractor, sample_weather_entity_list):
        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            with pytest.raises(ValueError, match="merge_existing"):
                configured_weather_extractor.process_entity_weather_bulk_parallel(
                    entity_list=sample_weather_entity_list,
                    merge_existing="invalid_mode",
                    skip_export=True,
                )

    @patch("earthdaily.agriculture.extractors.weather_functions.export_results")
    @patch.object(WeatherExtractor, "process_single_entity_weather")
    @patch.object(WeatherExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(WeatherExtractor, "_merge_with_skipped_entities")
    def test_bulk_returns_expected_keys(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_weather_extractor,
        sample_weather_entity_list,
    ):
        mock_single.return_value = {"data": pd.DataFrame([{"x": 1}]), "error": None}
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_entity_weather_bulk_parallel(
                entity_list=sample_weather_entity_list,
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

    @patch("earthdaily.agriculture.extractors.weather_functions.export_results")
    @patch.object(WeatherExtractor, "process_single_entity_weather")
    @patch.object(WeatherExtractor, "_finalize_extraction", return_value=_FINALIZE_RETURN)
    @patch.object(WeatherExtractor, "_merge_with_skipped_entities")
    def test_bulk_filter_exclude(
        self,
        mock_merge,
        mock_finalize,
        mock_single,
        mock_export,
        configured_weather_extractor,
    ):
        """Notebook cell 29 uses filter_type='exclude' on crop=CORN."""
        entity_list = pd.DataFrame(
            [
                {
                    "id": "ent_001",
                    "geometry": WEATHER_WKT,
                    "crop": "SOYBEANS",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
                {
                    "id": "ent_002",
                    "geometry": WEATHER_WKT,
                    "crop": "CORN",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
                {
                    "id": "ent_003",
                    "geometry": WEATHER_WKT,
                    "crop": "CORN",
                    "start_date": "2025-06-01",
                    "end_date": "2025-10-01",
                },
            ]
        )
        mock_single.return_value = {
            "data": pd.DataFrame([{"entity_id": "x", "wind": 3.2}]),
            "error": None,
        }
        mock_merge.side_effect = lambda **kw: kw["new_results_df"]

        with patch.object(configured_weather_extractor, "ensure_token_valid"):
            result = configured_weather_extractor.process_entity_weather_bulk_parallel(
                entity_list=entity_list,
                filter_column="crop",
                filter_value="CORN",
                filter_type="exclude",
                skip_export=True,
            )

        assert result["total_calculations"] == 1
        assert result["total_entities"] == 3


# ===================================================================
# process_single_entity_weather() — multi-year KPI, no truncation
# ===================================================================


class TestWeatherKpiNoTruncation:
    """With the $limit auto-sized to the (expanded) KPI query span, a multi-year request
    keeps every requested historical year AND the current-period window — the truncation
    bug (fixed 1000-row cap) dropped the most recent days."""

    # current season 2026-06-01..2026-06-30; expanded lookback reaches 2021 → ~1856-day span,
    # well past the old 1000-row cap, so the recent year (2025) and the 2026 window were lost.
    ENTITY = {
        "id": "kpi1",
        "geometry": WEATHER_WKT,
        "start_date": "2026-06-01",
        "end_date": "2026-06-30",
    }
    KPI_PARAMS = {
        "weather_type": "HISTORICAL_DAILY",
        "weather_parameters": "wind",
        "historical_years": [2021, 2023, 2025],  # three years, all before current 2026
        "partial_frequency": 50,
        "kpi_filter": {"aggregation": "average", "value_column": "wind", "kpi_name": "wind avg"},
        "page_limit": None,
    }

    def _run(self, ext):
        ext.weather_params = dict(self.KPI_PARAMS)
        with (
            patch("earthdaily.agriculture.extractors.weather_functions.get_centroid_wkt", return_value="POINT (0 0)"),
            patch("earthdaily.agriculture.extractors.weather_functions.validate_wkt", side_effect=lambda x: x),
            patch(
                "earthdaily.agriculture.extractors.weather_functions.retry_with_backoff_no_retry_on_400",
                side_effect=lambda func, *a, **k: func(),
            ),
            patch(
                "earthdaily.agriculture.extractors.weather_functions.requests.get", side_effect=_weather_server_sim()
            ),
            patch.object(ext, "ensure_token_valid"),
        ):
            return ext.process_single_entity_weather(dict(self.ENTITY))

    def test_all_requested_historical_years_counted(self, configured_weather_extractor):
        """num_years == the 3 requested years — not fewer due to the recent year being truncated."""
        result = self._run(configured_weather_extractor)
        assert result["error"] is None
        assert result["data"]["historical_num_years"].iloc[0] == 3

    def test_current_period_window_present(self, configured_weather_extractor):
        """The most-recent (current-period) window is present and its KPI is non-empty."""
        result = self._run(configured_weather_extractor)
        row = result["data"].iloc[0]
        assert row["current_num_records"] == 30  # inclusive 2026-06-01..2026-06-30
        assert pd.notna(row["current_value"])
