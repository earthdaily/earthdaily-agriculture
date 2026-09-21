"""
Tests for CoverageExtractor.format_coverage_json():
    - List response (typical catalog-imagery shape)
    - Dict with 'items' / 'results' wrappers
    - String JSON parsing
    - Sensor inference from image_id when 'sensor' missing
    - clear_cover_min/max strict filtering
    - Duplicate filter mode
    - Empty / invalid responses
"""

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.public

# ===================================================================
# format_coverage_json() — list response (notebook shape)
# ===================================================================


class TestFormatCoverageJsonList:
    """The real catalog-imagery endpoint returns a JSON list (cell 17 output)."""

    def test_list_creates_one_row_per_record(self, configured_coverage_extractor, sample_coverage_response_list):
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        # Default clear_cover_min=80; the 75% record should be filtered out
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_expected_columns_present(self, configured_coverage_extractor, sample_coverage_response_list):
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        expected = {"image_id", "coverage_percent", "date", "mask", "sensor", "spatial_resolution"}
        assert expected.issubset(set(df.columns))
        # date_obj is a helper column — should be dropped before returning
        assert "date_obj" not in df.columns

    def test_date_normalized_to_yyyy_mm_dd(self, configured_coverage_extractor, sample_coverage_response_list):
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        # All dates should be YYYY-MM-DD strings (10 chars)
        for d in df["date"]:
            assert isinstance(d, str)
            assert len(d) == 10
            assert d[4] == "-" and d[7] == "-"

    def test_image_id_and_coverage_preserved(self, configured_coverage_extractor, sample_coverage_response_list):
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        # Records should preserve coverage_percent and image_id fidelity
        s2_row = df[df["sensor"] == "SENTINEL_2"].iloc[0]
        assert s2_row["coverage_percent"] == 100.0
        assert "S2C_T31UDQ_20260429T105026_L2A" in s2_row["image_id"]
        assert s2_row["spatial_resolution"] == 10.0

    def test_clear_cover_filtering_excludes_below_min(
        self, configured_coverage_extractor, sample_coverage_response_list
    ):
        """Strict coverage_percent filtering — values below clear_cover_min are dropped."""
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        # The fixture has a record at 75.0; default clear_cover_min=80.
        assert (df["coverage_percent"] >= 80).all()

    def test_strict_filtering_with_higher_min(self, configured_coverage_extractor, sample_coverage_response_list):
        """Raising clear_cover_min should drop more records."""
        # Fixture coverages: [100, 95, 88, 75]. With min=96, only the 100% record survives.
        configured_coverage_extractor.coverage_params["clear_cover_min"] = 96
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)
        assert (df["coverage_percent"] >= 96).all()
        assert len(df) == 1


# ===================================================================
# format_coverage_json() — dict wrappers and string input
# ===================================================================


class TestFormatCoverageJsonInputShapes:
    """The formatter accepts list, dict-with-items, dict-with-results, and string JSON."""

    def test_dict_with_items_key(self, configured_coverage_extractor, sample_coverage_response_list):
        wrapped = {"items": sample_coverage_response_list}
        df = configured_coverage_extractor.format_coverage_json(wrapped)
        assert len(df) == 3

    def test_dict_with_results_key(self, configured_coverage_extractor, sample_coverage_response_list):
        wrapped = {"results": sample_coverage_response_list}
        df = configured_coverage_extractor.format_coverage_json(wrapped)
        assert len(df) == 3

    def test_string_json_parsed(self, configured_coverage_extractor, sample_coverage_response_list):
        as_str = json.dumps(sample_coverage_response_list)
        df = configured_coverage_extractor.format_coverage_json(as_str)
        assert len(df) == 3

    def test_unknown_dict_returns_empty(self, configured_coverage_extractor):
        """Dict without 'items' or 'results' should be treated as empty data."""
        df = configured_coverage_extractor.format_coverage_json({"foo": "bar"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_unsupported_type_raises(self, configured_coverage_extractor):
        """Numbers, bools, etc. are not supported and should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported response type"):
            configured_coverage_extractor.format_coverage_json(42)


# ===================================================================
# format_coverage_json() — sensor inference
# ===================================================================


class TestFormatCoverageJsonSensorInference:
    """When the API record has no 'sensor' field, infer from image_id."""

    def test_sensor_inferred_from_image_id(self, configured_coverage_extractor, sample_coverage_response_no_sensor):
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_no_sensor)

        # Map image_id substring -> expected sensor
        by_id = {row["image_id"]: row["sensor"] for _, row in df.iterrows()}

        s2_id = next(k for k in by_id if "sentinel-2" in k.lower())
        assert by_id[s2_id] == "SENTINEL_2"

        lc09_id = next(k for k in by_id if "LC09" in k)
        assert by_id[lc09_id] == "LANDSAT_9"

        lc08_id = next(k for k in by_id if "LC08" in k)
        assert by_id[lc08_id] == "LANDSAT_8"

        unk_id = next(k for k in by_id if "unknown-sensor" in k)
        assert by_id[unk_id] == "UNKNOWN"


# ===================================================================
# format_coverage_json() — duplicate filter mode
# ===================================================================


class TestFormatCoverageJsonDuplicateFilter:
    """When filter='duplicate', the formatter calls _filter_duplicates."""

    def test_duplicate_filter_is_invoked(
        self, configured_coverage_extractor, sample_coverage_response_list, monkeypatch
    ):
        """The filter='duplicate' code path should call _filter_duplicates."""
        configured_coverage_extractor.coverage_params["filter"] = "duplicate"

        called = {"count": 0}

        def fake_filter(df_in):
            called["count"] += 1
            return df_in.iloc[:1].copy()  # keep just the first row

        monkeypatch.setattr(configured_coverage_extractor, "_filter_duplicates", fake_filter)

        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list)

        assert called["count"] == 1
        assert len(df) == 1

    def test_duplicate_filter_picks_cross_sensor_pairs(self, configured_coverage_extractor):
        """
        End-to-end check: the real _filter_duplicates keeps cross-sensor pairs taken
        within the configured delay window.
        """
        configured_coverage_extractor.coverage_params["filter"] = "duplicate"
        configured_coverage_extractor.coverage_params["delay"] = 3

        # Two sensors imaged within 1 day → a valid pair; a third Landsat far away → dropped
        records = [
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "sentinel-2-c1-l2a|S2C_T31UDQ_20260429",
                    "spatialResolution": 10.0,
                    "date": "2026-04-29T10:00:00Z",
                    "sensor": "SENTINEL_2",
                },
                "mask": "ML",
            },
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "landsat-c2l2-sr|LC09_20260430",
                    "spatialResolution": 30.0,
                    "date": "2026-04-30T10:00:00Z",
                    "sensor": "LANDSAT_9",
                },
                "mask": "ML",
            },
            {
                "coveragePercent": 100.0,
                "image": {
                    "id": "landsat-c2l2-sr|LC08_20260601",
                    "spatialResolution": 30.0,
                    "date": "2026-06-01T10:00:00Z",
                    "sensor": "LANDSAT_8",
                },
                "mask": "ML",
            },
        ]

        df = configured_coverage_extractor.format_coverage_json(records)
        # The 2 close-in-time cross-sensor records form a valid pair, the lone Landsat-8 is dropped
        assert len(df) == 2
        assert set(df["sensor"]) == {"SENTINEL_2", "LANDSAT_9"}


# ===================================================================
# format_coverage_json() — empty / edge cases
# ===================================================================


class TestFormatCoverageJsonEdgeCases:
    """Edge cases via the validate_api_response gate."""

    def test_empty_list_returns_empty_df(self, configured_coverage_extractor):
        df = configured_coverage_extractor.format_coverage_json([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_dict_with_empty_items_returns_empty_df(self, configured_coverage_extractor):
        df = configured_coverage_extractor.format_coverage_json({"items": []})
        assert df.empty

    def test_empty_dict_returns_empty_df(self, configured_coverage_extractor):
        df = configured_coverage_extractor.format_coverage_json({})
        assert df.empty

    def test_entity_id_passed_for_logging(self, configured_coverage_extractor, sample_coverage_response_list):
        """entity_id arg should not affect output rows but should not raise."""
        df = configured_coverage_extractor.format_coverage_json(sample_coverage_response_list, entity_id="some_entity")
        assert len(df) == 3
