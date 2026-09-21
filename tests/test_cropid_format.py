"""
Tests for cropidExtractor format methods:
    - _matching_years() — pure helper, sorted/deduped years where edaCropCode matches
    - format_cropid_json() — flattens resultsByYear into one row per (year, crop)
    - format_cropid_year_json() — filters to entity's crop, returns (entity_id, year, eda_crop_code)
    - format_cropid_historical_season_json() — returns comma-separated string of matching years
    - format_cropid_full_history_json() — wide-form pivot, one row per entity, one column per year
"""

import pandas as pd
import pytest

from earthdaily.agriculture.extractors.cropid_functions import cropidExtractor

pytestmark = pytest.mark.public

# ===================================================================
# format_cropid_json() — history mode
# ===================================================================


class TestFormatCropidJson:
    """Without entity_data, returns one row per (year, crop) without entity_id."""

    def test_creates_one_row_per_year_crop(self, configured_cropid_extractor, sample_cropid_response):
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response)

        # 6 years, 1 crop per year → 6 rows (matches notebook cell 21 output)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6

    def test_columns_without_entity_data(self, configured_cropid_extractor, sample_cropid_response):
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response)
        assert list(df.columns) == [
            "year",
            "eda_crop_code",
            "crop_name",
            "raw_crop_name",
            "crop_mask_percent",
        ]

    def test_columns_with_entity_data(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        """When entity_data is provided, entity_id is included as the first column."""
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response, sample_cropid_entity)
        assert list(df.columns) == [
            "entity_id",
            "year",
            "eda_crop_code",
            "crop_name",
            "raw_crop_name",
            "crop_mask_percent",
        ]
        assert (df["entity_id"] == "z361x33").all()

    def test_year_is_int(self, configured_cropid_extractor, sample_cropid_response):
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response)
        # Years come from the JSON response as string keys; should be cast to int
        assert df["year"].dtype.kind in {"i", "u"}
        assert set(df["year"]) == {2020, 2021, 2022, 2023, 2024, 2025}

    def test_sorted_by_year(self, configured_cropid_extractor, sample_cropid_response):
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response)
        assert list(df["year"]) == sorted(df["year"])

    def test_crop_fields_preserved(self, configured_cropid_extractor, sample_cropid_response):
        df = configured_cropid_extractor.format_cropid_json(sample_cropid_response)
        soybean_row = df[df["year"] == 2020].iloc[0]
        assert soybean_row["eda_crop_code"] == "SOYBEANS"
        assert soybean_row["crop_name"] == "Soybean"
        assert soybean_row["raw_crop_name"] == "soybeans"
        assert soybean_row["crop_mask_percent"] == 97


# ===================================================================
# format_cropid_year_json() — year mode (filtered)
# ===================================================================


class TestFormatCropidYearJson:
    """Filter resultsByYear to only years matching the entity's crop."""

    def test_filters_to_entity_crop(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        """Notebook cell 23: entity crop SOYBEANS → 4 matching years (2020, 2022, 2024, 2025)."""
        df = configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, sample_cropid_entity)

        assert len(df) == 4
        assert list(df["year"]) == [2020, 2022, 2024, 2025]
        assert (df["eda_crop_code"] == "SOYBEANS").all()
        assert (df["entity_id"] == "z361x33").all()

    def test_columns(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        df = configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, sample_cropid_entity)
        assert list(df.columns) == ["entity_id", "year", "eda_crop_code"]

    def test_filters_to_cotton(self, configured_cropid_extractor, sample_cropid_response):
        """A different crop should pick up only its years."""
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "COTTON"}
        df = configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, entity)

        assert list(df["year"]) == [2021, 2023]
        assert (df["eda_crop_code"] == "COTTON").all()

    def test_no_matches_returns_empty_df(self, configured_cropid_extractor, sample_cropid_response):
        """A crop not present in the response returns an empty DataFrame with the schema columns."""
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "RICE"}
        df = configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, entity)

        assert df.empty
        assert list(df.columns) == ["entity_id", "year", "eda_crop_code"]

    def test_missing_crop_raises(self, configured_cropid_extractor, sample_cropid_response):
        """Entity with no 'crop' field → ValueError."""
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        with pytest.raises(ValueError, match="crop"):
            configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, entity)

    def test_empty_crop_raises(self, configured_cropid_extractor, sample_cropid_response):
        """Entity with empty crop string → ValueError."""
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": ""}
        with pytest.raises(ValueError, match="crop"):
            configured_cropid_extractor.format_cropid_year_json(sample_cropid_response, entity)


# ===================================================================
# format_cropid_historical_season_json() — historical_season mode
# ===================================================================


class TestFormatCropidHistoricalSeasonJson:
    """Returns a comma-separated string of years where the crop matches the entity."""

    def test_returns_comma_separated_years(
        self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity
    ):
        """Notebook cell 25: entity crop SOYBEANS → '2020,2022,2024,2025'."""
        result = configured_cropid_extractor.format_cropid_historical_season_json(
            sample_cropid_response, sample_cropid_entity
        )
        assert result == "2020,2022,2024,2025"

    def test_returns_string_type(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        result = configured_cropid_extractor.format_cropid_historical_season_json(
            sample_cropid_response, sample_cropid_entity
        )
        assert isinstance(result, str)

    def test_no_matches_returns_empty_string(self, configured_cropid_extractor, sample_cropid_response):
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "RICE"}
        result = configured_cropid_extractor.format_cropid_historical_season_json(sample_cropid_response, entity)
        assert result == ""

    def test_missing_crop_raises(self, configured_cropid_extractor, sample_cropid_response):
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))"}
        with pytest.raises(ValueError, match="crop"):
            configured_cropid_extractor.format_cropid_historical_season_json(sample_cropid_response, entity)

    def test_empty_crop_raises(self, configured_cropid_extractor, sample_cropid_response):
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": ""}
        with pytest.raises(ValueError, match="crop"):
            configured_cropid_extractor.format_cropid_historical_season_json(sample_cropid_response, entity)

    def test_years_are_sorted(self, configured_cropid_extractor):
        """Years in the comma-separated string must be sorted ascending."""
        # Build a response with years in non-sorted order
        response = {
            "resultsByYear": {
                "2025": {"crops": [{"edaCropCode": "CORN"}]},
                "2020": {"crops": [{"edaCropCode": "CORN"}]},
                "2022": {"crops": [{"edaCropCode": "CORN"}]},
            }
        }
        entity = {"id": "ent_x", "geometry": "POLYGON((0 0,1 0,1 1,0 1,0 0))", "crop": "CORN"}
        result = configured_cropid_extractor.format_cropid_historical_season_json(response, entity)
        assert result == "2020,2022,2025"


# ===================================================================
# _matching_years() — shared helper, pure function
# ===================================================================


class TestMatchingYears:
    """The private helper that both year-mode and historical_season-mode delegate to."""

    def test_empty_results_by_year(self):
        assert cropidExtractor._matching_years({"resultsByYear": {}}, "SOYBEANS") == []

    def test_missing_results_by_year_key(self):
        """A response with no resultsByYear key behaves like an empty one."""
        assert cropidExtractor._matching_years({}, "SOYBEANS") == []

    def test_no_matches(self, sample_cropid_response):
        assert cropidExtractor._matching_years(sample_cropid_response, "RICE") == []

    def test_single_match(self, sample_cropid_response):
        # Notebook fixture has COTTON in 2021 and 2023 only.
        assert cropidExtractor._matching_years(sample_cropid_response, "COTTON") == [2021, 2023]

    def test_multiple_matches_sorted(self, sample_cropid_response):
        # SOYBEANS appears in 2020, 2022, 2024, 2025 — out-of-key-order in source
        # dicts; helper must return ascending.
        assert cropidExtractor._matching_years(sample_cropid_response, "SOYBEANS") == [2020, 2022, 2024, 2025]

    def test_dedupes_repeated_codes_within_a_year(self):
        """If the API lists the same edaCropCode twice in one year (limit_nb_crop > 1
        edge case), the year is reported once — matching the prior `break` semantics."""
        response = {
            "resultsByYear": {
                "2022": {
                    "crops": [
                        {"edaCropCode": "CORN", "cropMaskPercent": 80},
                        {"edaCropCode": "CORN", "cropMaskPercent": 5},
                    ]
                }
            }
        }
        assert cropidExtractor._matching_years(response, "CORN") == [2022]

    def test_returns_int_years(self, sample_cropid_response):
        years = cropidExtractor._matching_years(sample_cropid_response, "SOYBEANS")
        assert all(isinstance(y, int) for y in years)


# ===================================================================
# format_cropid_full_history_json() — full_history mode (wide-form pivot)
# ===================================================================


class TestFormatCropidFullHistoryJson:
    """One row per entity, one column per year in begin_year..end_year inclusive."""

    def test_one_row_with_year_columns(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        """Notebook fixture covers 2020-2025 → 6 year columns + entity_id."""
        df = configured_cropid_extractor.format_cropid_full_history_json(sample_cropid_response, sample_cropid_entity)
        assert len(df) == 1
        assert list(df.columns) == ["entity_id", "2020", "2021", "2022", "2023", "2024", "2025"]
        assert df.loc[0, "entity_id"] == "z361x33"

    def test_full_coverage_picks_each_years_crop(
        self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity
    ):
        """Every year has exactly one crop in the fixture — those values land verbatim."""
        df = configured_cropid_extractor.format_cropid_full_history_json(sample_cropid_response, sample_cropid_entity)
        row = df.iloc[0]
        assert row["2020"] == "SOYBEANS"
        assert row["2021"] == "COTTON"
        assert row["2022"] == "SOYBEANS"
        assert row["2023"] == "COTTON"
        assert row["2024"] == "SOYBEANS"
        assert row["2025"] == "SOYBEANS"

    def test_partial_coverage_fills_nan(self, configured_cropid_extractor, sample_cropid_entity):
        """Years absent from the response become NaN in the wide frame."""
        partial_response = {
            "resultsByYear": {
                "2021": {"crops": [{"edaCropCode": "CORN", "cropMaskPercent": 90}]},
                "2024": {"crops": [{"edaCropCode": "CORN", "cropMaskPercent": 88}]},
            }
        }
        df = configured_cropid_extractor.format_cropid_full_history_json(partial_response, sample_cropid_entity)
        row = df.iloc[0]
        assert pd.isna(row["2020"])
        assert row["2021"] == "CORN"
        assert pd.isna(row["2022"])
        assert pd.isna(row["2023"])
        assert row["2024"] == "CORN"
        assert pd.isna(row["2025"])

    def test_empty_results_by_year_all_nan(
        self, configured_cropid_extractor, sample_cropid_response_empty, sample_cropid_entity
    ):
        """Empty response still yields a 1-row frame with the expected year columns, all NaN."""
        df = configured_cropid_extractor.format_cropid_full_history_json(
            sample_cropid_response_empty, sample_cropid_entity
        )
        assert len(df) == 1
        # All year columns NaN; entity_id still populated.
        year_cols = ["2020", "2021", "2022", "2023", "2024", "2025"]
        assert df.loc[0, "entity_id"] == "z361x33"
        assert df[year_cols].isna().all(axis=1).iloc[0]

    def test_multi_crop_year_picks_top_by_mask_percent(self, configured_cropid_extractor, sample_cropid_entity):
        """When a year lists several crops, the highest cropMaskPercent wins."""
        response = {
            "resultsByYear": {
                "2023": {
                    "crops": [
                        {"edaCropCode": "SOYBEANS", "cropMaskPercent": 40},
                        {"edaCropCode": "CORN", "cropMaskPercent": 55},
                        {"edaCropCode": "COTTON", "cropMaskPercent": 5},
                    ]
                }
            }
        }
        df = configured_cropid_extractor.format_cropid_full_history_json(response, sample_cropid_entity)
        assert df.loc[0, "2023"] == "CORN"

    def test_year_columns_are_strings(self, configured_cropid_extractor, sample_cropid_response, sample_cropid_entity):
        """Year columns are named with strings so the wide frame stays parquet-friendly."""
        df = configured_cropid_extractor.format_cropid_full_history_json(sample_cropid_response, sample_cropid_entity)
        for year in range(2020, 2026):
            assert str(year) in df.columns
            assert year not in df.columns  # not int-keyed

    def test_window_width_matches_begin_end_year(self, cropid_extractor, sample_cropid_response, sample_cropid_entity):
        """A 3-year window yields exactly 3 year columns regardless of how many years have data."""
        cropid_extractor.setup_cropid_parameters(begin_year=2022, end_year=2024, mode="full_history")
        df = cropid_extractor.format_cropid_full_history_json(sample_cropid_response, sample_cropid_entity)
        assert list(df.columns) == ["entity_id", "2022", "2023", "2024"]

    def test_bulk_concat_consistent_columns(self, configured_cropid_extractor, sample_cropid_entity):
        """Across entities with different year coverage, columns line up after concat — the
        whole point of the wide-form mode."""
        ent_a = {**sample_cropid_entity, "id": "ent_a"}
        ent_b = {**sample_cropid_entity, "id": "ent_b"}
        resp_a = {"resultsByYear": {"2021": {"crops": [{"edaCropCode": "CORN", "cropMaskPercent": 90}]}}}
        resp_b = {"resultsByYear": {"2024": {"crops": [{"edaCropCode": "SOYBEANS", "cropMaskPercent": 91}]}}}

        df_a = configured_cropid_extractor.format_cropid_full_history_json(resp_a, ent_a)
        df_b = configured_cropid_extractor.format_cropid_full_history_json(resp_b, ent_b)
        bulk = pd.concat([df_a, df_b], ignore_index=True)

        assert list(bulk.columns) == ["entity_id", "2020", "2021", "2022", "2023", "2024", "2025"]
        assert len(bulk) == 2
        assert bulk.loc[bulk["entity_id"] == "ent_a", "2021"].iloc[0] == "CORN"
        assert bulk.loc[bulk["entity_id"] == "ent_b", "2024"].iloc[0] == "SOYBEANS"
