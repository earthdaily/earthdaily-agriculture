"""
Unit tests for geohash-based spatial grouping / dedup of point-based extractors.

Covers the three BaseExtractor pieces that make Weather / GDD / GDD-offset call
the API once per (geohash × request-signature) group and broadcast the result to
every member field:

    - core.geometry.centroid_geohash
    - BaseExtractor._add_spatial_group_key
    - BaseExtractor._broadcast_grouped_results
    - BaseExtractor._run_bulk_spatially_grouped (end-to-end with a fake fetch)

No network: the "representative fetch" is a stub that returns synthetic per-group
results, and _finalize_extraction is monkeypatched to the identity so export/report
are not exercised here.
"""

import pandas as pd
import pytest

from earthdaily.agriculture.core.api_utils import normalize_with_metadata
from earthdaily.agriculture.core.geometry import centroid_geohash

pytestmark = pytest.mark.public

# Two polygons ~50 m apart (same coarse geohash cell) and one far away.
GEOM_A = "POLYGON((-49.500 -14.500, -49.499 -14.500, -49.499 -14.499, -49.500 -14.499, -49.500 -14.500))"
GEOM_B = "POLYGON((-49.4995 -14.4995, -49.4985 -14.4995, -49.4985 -14.4985, -49.4995 -14.4985, -49.4995 -14.4995))"
GEOM_FAR = "POLYGON((-40.000 -10.000, -39.999 -10.000, -39.999 -9.999, -40.000 -9.999, -40.000 -10.000))"


# ---------------------------------------------------------------------------
# centroid_geohash
# ---------------------------------------------------------------------------


def test_centroid_geohash_length_and_bucketing():
    gh_a = centroid_geohash(GEOM_A, precision=5)
    gh_b = centroid_geohash(GEOM_B, precision=5)
    gh_far = centroid_geohash(GEOM_FAR, precision=5)

    assert len(gh_a) == 5
    assert gh_a == gh_b, "polygons ~50 m apart must share a geohash-5 cell"
    assert gh_a != gh_far, "far-away polygon must land in a different cell"


def test_centroid_geohash_precision_refines():
    # A finer precision may split what a coarse cell merged; it never merges more.
    assert len(centroid_geohash(GEOM_A, precision=7)) == 7


# ---------------------------------------------------------------------------
# _add_spatial_group_key
# ---------------------------------------------------------------------------


def _gdd_entities():
    """f1 & f2 co-located same window; f3 far; f4 co-located but different start_date."""
    return pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-06-01"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-04-01", "end_date": "2025-06-01"},
            {"id": "f3", "geometry": GEOM_FAR, "start_date": "2025-04-01", "end_date": "2025-06-01"},
            {"id": "f4", "geometry": GEOM_A, "start_date": "2025-05-01", "end_date": "2025-06-01"},
        ]
    )


def test_add_spatial_group_key_groups_by_cell_and_window(gdd_extractor):
    keyed = gdd_extractor._add_spatial_group_key(
        _gdd_entities(), signature_columns=["start_date", "end_date", "provider"], precision=5
    )
    keys = dict(zip(keyed["id"], keyed["_spatial_group_key"]))

    # f1 and f2: same cell + same window → same group.
    assert keys["f1"] == keys["f2"]
    # f3 (far) and f4 (different start_date) each get their own group.
    assert keys["f3"] != keys["f1"]
    assert keys["f4"] != keys["f1"]
    assert keyed["_spatial_group_key"].nunique() == 3


def test_add_spatial_group_key_isolates_bad_geometry(gdd_extractor):
    df = _gdd_entities()
    df.loc[len(df)] = {"id": "bad", "geometry": "NOT_WKT", "start_date": "2025-04-01", "end_date": "2025-06-01"}
    keyed = gdd_extractor._add_spatial_group_key(df, signature_columns=["start_date"], precision=5)

    bad_key = keyed.loc[keyed["id"] == "bad", "_spatial_group_key"].iloc[0]
    assert bad_key.startswith("__nogeo_")
    # It must not collide with any real group.
    assert (keyed["_spatial_group_key"] == bad_key).sum() == 1


# ---------------------------------------------------------------------------
# _run_bulk_spatially_grouped  (end-to-end, stubbed fetch)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_export(monkeypatch):
    """Make _finalize_extraction a pass-through so no files are written."""

    def _identity(self, *, results_df, **kwargs):
        return results_df, {}

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    monkeypatch.setattr(BaseExtractor, "_finalize_extraction", _identity)


def test_gdd_style_broadcast(gdd_extractor, no_export):
    entities = _gdd_entities()
    calls = {}

    def fake_fetch(reps):
        # Record how many API "calls" (representative rows) we were asked to make.
        calls["n"] = len(reps)
        # Distinct daily GDD per representative so we can trace broadcast provenance.
        base = {"f1": 10, "f3": 30, "f4": 40}
        frames = []
        for _, row in reps.iterrows():
            v = base[row["id"]]
            gdd_df = pd.DataFrame(
                {"date": ["2025-04-01", "2025-04-02"], "daily_gdd": [v, v + 1], "cumulated_gdd": [v, 2 * v + 1]}
            )
            frames.append(normalize_with_metadata(row.to_dict(), gdd_df))
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    result = gdd_extractor._run_bulk_spatially_grouped(
        entity_list=entities,
        run_representatives=fake_fetch,
        signature_columns=["start_date", "end_date", "provider"],
        precision=5,
        prefix="gdd",
    )

    # 4 fields collapsed to 3 API calls.
    assert calls["n"] == 3
    assert result["representative_calls"] == 3
    assert result["grouped_from"] == 4
    assert result["successful_calculations"] == 4
    assert result["failed_calculations"] == 0

    out = result["results_df"]
    # Row-count invariant: every member gets the representative's 2 daily rows.
    assert len(out) == 8
    assert set(out["id"]) == {"f1", "f2", "f3", "f4"}

    # f2 was broadcast from f1's group: f1's GDD values, but f2's own geometry/id.
    f2 = out[out["id"] == "f2"]
    assert sorted(f2["daily_gdd"]) == [10, 11]
    assert set(f2["geometry"]) == {GEOM_B}
    assert set(f2["start_date"]) == {"2025-04-01"}
    # f4 (own group, different window) keeps its own values.
    assert sorted(out[out["id"] == "f4"]["daily_gdd"]) == [40, 41]


def test_weather_style_broadcast_entity_id_schema(weather_extractor, no_export):
    """Weather output uses an 'entity_id' column and no id/geometry — broadcast must
    re-stamp entity_id per member and not invent metadata columns."""
    entities = pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-06-01"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-04-01", "end_date": "2025-06-01"},
        ]
    )

    def fake_fetch(reps):
        frames = []
        for _, row in reps.iterrows():
            frames.append(
                pd.DataFrame({"entity_id": row["id"], "date": ["2025-04-01", "2025-04-02"], "precip": [5.0, 6.0]})
            )
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    result = weather_extractor._run_bulk_spatially_grouped(
        entity_list=entities,
        run_representatives=fake_fetch,
        signature_columns=["start_date", "end_date"],
        precision=5,
        prefix="weather",
    )

    out = result["results_df"]
    assert result["representative_calls"] == 1  # both fields share the cell + window
    assert len(out) == 4  # 2 members × 2 days
    assert set(out.columns) == {"entity_id", "date", "precip"}
    assert set(out["entity_id"]) == {"f1", "f2"}
    # No metadata columns leaked in from the entity_list.
    assert "geometry" not in out.columns and "id" not in out.columns


def test_failed_representative_marks_group_members_failed(gdd_extractor, no_export):
    entities = _gdd_entities()

    def fake_fetch(reps):
        # Drop f3's group entirely (simulate that bucket's call failing).
        frames = []
        for _, row in reps.iterrows():
            if row["id"] == "f3":
                continue
            gdd_df = pd.DataFrame({"date": ["2025-04-01"], "daily_gdd": [10], "cumulated_gdd": [10]})
            frames.append(normalize_with_metadata(row.to_dict(), gdd_df))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return {"results_df": out, "global_errors": [{"entity_id": "f3", "message": "boom"}]}

    result = gdd_extractor._run_bulk_spatially_grouped(
        entity_list=entities,
        run_representatives=fake_fetch,
        signature_columns=["start_date", "end_date", "provider"],
        precision=5,
        prefix="gdd",
    )

    assert result["failed_ids"] == ["f3"]
    assert result["successful_calculations"] == 3
    assert set(result["results_df"]["id"]) == {"f1", "f2", "f4"}


# ---------------------------------------------------------------------------
# Window widening: dates leave the group key, so a cell is called once over the
# union of its members' windows.
# ---------------------------------------------------------------------------

WINDOW_COLS = ("start_date", "end_date")
CUM_COL = "cumulated_value"


def _staggered_entities():
    """f1 & f2 co-located with overlapping-but-different windows; f3 far away."""
    return pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-06-01"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-05-01", "end_date": "2025-07-01"},
            {"id": "f3", "geometry": GEOM_FAR, "start_date": "2025-04-01", "end_date": "2025-06-01"},
        ]
    )


def _resolve(extractor, df, cap=400):
    keyed = extractor._add_spatial_group_key(df, signature_columns=[], precision=5)
    return extractor._resolve_group_windows(keyed, WINDOW_COLS, max_window_days=cap)


def test_staggered_windows_collapse_into_one_union_call(weather_extractor):
    keyed = _resolve(weather_extractor, _staggered_entities())
    keys = dict(zip(keyed["id"], keyed["_spatial_group_key"]))

    # Differing windows no longer split the cell — that is the whole point.
    assert keys["f1"] == keys["f2"]
    assert keys["f3"] != keys["f1"]

    win = keyed.set_index("id")
    assert win.loc["f1", "_win_start"] == pd.Timestamp("2025-04-01")
    assert win.loc["f1", "_win_end"] == pd.Timestamp("2025-07-01")
    assert win.loc["f2", "_win_start"] == pd.Timestamp("2025-04-01")


def test_long_window_outlier_gets_its_own_call(weather_extractor):
    df = _staggered_entities()
    df.loc[len(df)] = {"id": "f4", "geometry": GEOM_A, "start_date": "2015-01-01", "end_date": "2025-12-31"}
    keyed = _resolve(weather_extractor, df, cap=400)
    keys = dict(zip(keyed["id"], keyed["_spatial_group_key"]))

    # The decade-long field is split out instead of dragging its cell back to 2015.
    assert keys["f4"] != keys["f1"]
    assert keys["f1"] == keys["f2"]
    assert keyed.set_index("id").loc["f1", "_win_start"] == pd.Timestamp("2025-04-01")


def test_representative_requests_the_union_window(weather_extractor):
    keyed = _resolve(weather_extractor, _staggered_entities())
    reps = keyed.drop_duplicates("_spatial_group_key", keep="first")
    reps = weather_extractor._stamp_representative_windows(reps, WINDOW_COLS)

    cell_rep = reps[reps["id"] == "f1"].iloc[0]
    assert cell_rep["start_date"] == "2025-04-01"
    assert cell_rep["end_date"] == "2025-07-01"  # widened to cover f2


def test_unusable_window_is_never_merged(weather_extractor):
    df = _staggered_entities()
    df.loc[len(df)] = {"id": "bad", "geometry": GEOM_A, "start_date": None, "end_date": None}
    keyed = _resolve(weather_extractor, df)
    keys = dict(zip(keyed["id"], keyed["_spatial_group_key"]))
    assert keys["bad"] != keys["f1"]


# ---------------------------------------------------------------------------
# Clipping + cumulative rebase: a grouped run must equal an ungrouped one.
# ---------------------------------------------------------------------------


def _daily_fetch(reps):
    """Stand-in for the API: 1 unit/day, accumulating from the *requested* start."""
    frames = []
    for _, row in reps.iterrows():
        dates = pd.date_range(row["start_date"], row["end_date"], freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "entity_id": row["id"],
                    "date": dates.strftime("%Y-%m-%d"),
                    "daily": 1.0,
                    CUM_COL: range(1, len(dates) + 1),
                }
            )
        )
    return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}


def _two_members_one_cell():
    return pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-04-10"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-04-05", "end_date": "2025-04-10"},
        ]
    )


def _days(first, last):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, last, freq="D")]


def _grouped_run(extractor, entities, fetch=_daily_fetch, use_cache=False, **kw):
    # Rebasing is explicit per extractor, never guessed from the column name, so the
    # synthetic cumulative column has to be declared for these tests.
    kw.setdefault("cumulative_columns", [CUM_COL])
    return extractor._run_bulk_spatially_grouped(
        entity_list=entities,
        run_representatives=fetch,
        signature_columns=[],
        window_columns=WINDOW_COLS,
        max_window_days=400,
        precision=5,
        prefix="weather",
        use_cache=use_cache,
        cache_params={},
        **kw,
    )


def test_member_is_clipped_to_its_own_window(weather_extractor, no_export):
    result = _grouped_run(weather_extractor, _two_members_one_cell())

    assert result["representative_calls"] == 1  # one call serves both fields
    out = result["results_df"]
    # f2 asked for 04-05 to 04-10 and must not receive the group's earlier days.
    f2 = out[out["entity_id"] == "f2"].sort_values("date")
    assert list(f2["date"]) == _days("2025-04-05", "2025-04-10")
    assert len(out[out["entity_id"] == "f1"]) == 10


def test_cumulative_column_is_rebased_to_the_member_start(weather_extractor, no_export):
    out = _grouped_run(weather_extractor, _two_members_one_cell())["results_df"]

    # The group pulled from 04-01, so the raw cumulative on 04-05 is 5. Rebased to
    # f2's own start it must read 1 — what an ungrouped f2-only call would return.
    f2 = out[out["entity_id"] == "f2"].sort_values("date")
    assert list(f2[CUM_COL]) == [1, 2, 3, 4, 5, 6]
    f1 = out[out["entity_id"] == "f1"].sort_values("date")
    assert list(f1[CUM_COL]) == list(range(1, 11))
    # Per-day columns are window-independent and pass through untouched.
    assert set(out["daily"]) == {1.0}


def test_grouped_output_equals_ungrouped_output(weather_extractor, no_export):
    """The invariant that makes grouping safe to switch on: identical results."""
    entities = _two_members_one_cell()

    grouped = _grouped_run(weather_extractor, entities)["results_df"]
    # Ungrouped baseline: one call per field, each over its own window.
    ungrouped = _daily_fetch(entities)["results_df"]

    key = ["entity_id", "date"]
    cols = key + [CUM_COL, "daily"]
    g = grouped[cols].sort_values(key).reset_index(drop=True)
    u = ungrouped[cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(g, u, check_dtype=False)


def test_yearly_reset_rebases_only_within_the_start_year(weather_extractor, no_export):
    """With reset_cumulative_every_year the counter restarts each January, so a
    baseline must never be carried across the boundary."""
    entities = pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2024-12-30", "end_date": "2025-01-03"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-01-01", "end_date": "2025-01-03"},
        ]
    )

    def yearly_fetch(reps):
        frames = []
        for _, row in reps.iterrows():
            dates = pd.date_range(row["start_date"], row["end_date"], freq="D")
            running, cum = {}, []
            for d in dates:  # accumulation restarts on 1 January
                running[d.year] = running.get(d.year, 0) + 1
                cum.append(running[d.year])
            frames.append(pd.DataFrame({"entity_id": row["id"], "date": dates.strftime("%Y-%m-%d"), CUM_COL: cum}))
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    out = _grouped_run(weather_extractor, entities, fetch=yearly_fetch, reset_yearly=True)["results_df"]

    # f2's window sits wholly inside 2025, where the group's counter already
    # restarted: no baseline may be subtracted.
    f2 = out[out["entity_id"] == "f2"].sort_values("date")
    assert list(f2[CUM_COL]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Geohash-keyed spatial cache
# ---------------------------------------------------------------------------


@pytest.fixture
def cached_weather(weather_extractor, tmp_path):
    weather_extractor.cache_dir = tmp_path
    weather_extractor.use_cache = True
    weather_extractor.cache_ttl_days = 30
    return weather_extractor


def _counting_fetch():
    calls = []

    def fetch(reps):
        calls.append(len(reps))
        return _daily_fetch(reps)

    return calls, fetch


def test_second_run_is_served_entirely_from_the_spatial_cache(cached_weather, no_export):
    entities = _two_members_one_cell()
    calls, fetch = _counting_fetch()

    first = _grouped_run(cached_weather, entities, fetch=fetch, use_cache=True)
    second = _grouped_run(cached_weather, entities, fetch=fetch, use_cache=True)

    assert calls == [1], "the second run must not reach the API at all"
    assert second["representative_calls"] == 0
    assert second["cache_hit"] == 1

    key = ["entity_id", "date"]
    cols = key + [CUM_COL]
    a = first["results_df"][cols].sort_values(key).reset_index(drop=True)
    b = second["results_df"][cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_cache_serves_a_different_field_set_in_the_same_cell(cached_weather, no_export):
    """The reason the cache is keyed on the cell rather than the entity id."""
    calls, fetch = _counting_fetch()
    _grouped_run(cached_weather, _two_members_one_cell(), fetch=fetch, use_cache=True)

    # A field never seen before, in an already-cached cell, asking for a sub-window.
    newcomer = pd.DataFrame([{"id": "z9", "geometry": GEOM_A, "start_date": "2025-04-02", "end_date": "2025-04-06"}])
    out = _grouped_run(cached_weather, newcomer, fetch=fetch, use_cache=True)["results_df"]

    assert calls == [1], "a new field in a cached cell must not trigger a call"
    z9 = out.sort_values("date")
    assert list(z9["date"]) == _days("2025-04-02", "2025-04-06")
    # Rebased to z9's own start, not to the cached window's start.
    assert list(z9[CUM_COL]) == [1, 2, 3, 4, 5]


def test_window_not_covered_by_cache_triggers_a_refetch(cached_weather, no_export):
    calls, fetch = _counting_fetch()
    _grouped_run(cached_weather, _two_members_one_cell(), fetch=fetch, use_cache=True)

    # Same cell, but reaching past what the cache holds.
    later = pd.DataFrame([{"id": "f9", "geometry": GEOM_A, "start_date": "2025-04-05", "end_date": "2025-04-20"}])
    result = _grouped_run(cached_weather, later, fetch=fetch, use_cache=True)

    assert calls == [1, 1], "partial coverage must refetch"
    assert result["representative_calls"] == 1
    assert len(result["results_df"]) == 16


def test_spatial_cache_info_reports_cells(cached_weather, no_export):
    _grouped_run(cached_weather, _two_members_one_cell(), use_cache=True)
    info = cached_weather.spatial_cache_info({})

    assert info["exists"] is True
    assert info["cells"] == 1
    assert info["records"] == 10


# ---------------------------------------------------------------------------
# Public bulk method: the wiring an extractor actually exposes.
# ---------------------------------------------------------------------------


def test_public_weather_bulk_groups_widens_and_caches(configured_weather_extractor, no_export, tmp_path, monkeypatch):
    """End-to-end through process_entity_weather_bulk_parallel: staggered windows in
    one cell collapse to a single widened call, and a rerun costs nothing."""
    ext = configured_weather_extractor
    ext.cache_dir = tmp_path
    ext.use_cache = True
    ext.cache_ttl_days = 30

    entities = pd.DataFrame(
        [
            {"id": "f1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-04-10"},
            {"id": "f2", "geometry": GEOM_B, "start_date": "2025-04-05", "end_date": "2025-04-12"},
        ]
    )

    requested = []

    def fake_inner(entity_list, **kwargs):
        requested.append(entity_list[["id", "start_date", "end_date"]].to_dict("records"))
        return _daily_fetch(entity_list)

    monkeypatch.setattr(ext, "_process_entity_weather_bulk_parallel_inner", fake_inner)

    first = ext.process_entity_weather_bulk_parallel(
        entity_list=entities, skip_export=True, spatial_grouping=True, use_cache=True
    )

    # One call, over the union of both windows.
    assert len(requested) == 1
    assert requested[0] == [{"id": "f1", "start_date": "2025-04-01", "end_date": "2025-04-12"}]
    assert first["representative_calls"] == 1

    out = first["results_df"]
    f2 = out[out["entity_id"] == "f2"].sort_values("date")
    assert list(f2["date"]) == _days("2025-04-05", "2025-04-12")
    # WeatherExtractor declares no cumulative columns — every weather parameter is a
    # per-day value, including precipitation.cumulative — so rows are sliced but never
    # adjusted: f2 sees the group's series verbatim.
    assert list(f2[CUM_COL]) == [5, 6, 7, 8, 9, 10, 11, 12]

    # Second identical run: fully cached, no further calls.
    second = ext.process_entity_weather_bulk_parallel(
        entity_list=entities, skip_export=True, spatial_grouping=True, use_cache=True
    )
    assert len(requested) == 1
    assert second["representative_calls"] == 0
    assert second["cache_hit"] == 1


def test_public_gdd_bulk_rebases_cumulated_gdd(gdd_extractor, no_export, monkeypatch):
    """GDD's cumulated_gdd is detected and re-zeroed the same way weather's is."""
    ext = gdd_extractor
    ext.gdd_params = {"provider": "GLOBAL1", "reset_cumulative_every_year": False}
    ext.use_cache = False

    entities = pd.DataFrame(
        [
            {"id": "g1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-04-06"},
            {"id": "g2", "geometry": GEOM_B, "start_date": "2025-04-04", "end_date": "2025-04-06"},
        ]
    )

    def fake_inner(entity_list, **kwargs):
        frames = []
        for _, row in entity_list.iterrows():
            dates = pd.date_range(row["start_date"], row["end_date"], freq="D")
            gdd_df = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "daily_gdd": 2.0,
                    "cumulated_gdd": [2.0 * (i + 1) for i in range(len(dates))],
                }
            )
            frames.append(normalize_with_metadata(row.to_dict(), gdd_df))
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    monkeypatch.setattr(ext, "_process_entity_gdd_bulk_parallel_inner", fake_inner)

    result = ext.process_entity_gdd_bulk_parallel(
        entity_list=entities, skip_export=True, spatial_grouping=True, use_cache=False
    )

    assert result["representative_calls"] == 1
    out = result["results_df"]
    g2 = out[out["id"] == "g2"].sort_values("date")
    # The cell was pulled from 04-01 (raw cumulated 8, 10, 12 on 04-04..04-06);
    # rebased to g2's own start it must read 2, 4, 6.
    assert list(g2["cumulated_gdd"]) == [2.0, 4.0, 6.0]
    assert list(g2["daily_gdd"]) == [2.0, 2.0, 2.0]


def test_timezone_aware_api_dates_are_handled(weather_extractor, no_export):
    """The live weather API returns tz-aware dates (datetime64[us, UTC]) while entity
    windows are plain strings. Window maths must not blow up on the mix."""

    def tz_aware_fetch(reps):
        frames = []
        for _, row in reps.iterrows():
            dates = pd.date_range(row["start_date"], row["end_date"], freq="D", tz="UTC")
            frames.append(
                pd.DataFrame(
                    {
                        "entity_id": row["id"],
                        "date": dates,  # tz-aware, exactly as the API delivers it
                        CUM_COL: range(1, len(dates) + 1),
                    }
                )
            )
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    result = _grouped_run(weather_extractor, _two_members_one_cell(), fetch=tz_aware_fetch)

    assert result["representative_calls"] == 1
    out = result["results_df"]
    f2 = out[out["entity_id"] == "f2"]
    assert len(f2) == 6  # clipped to 04-05..04-10 despite the tz
    assert list(f2.sort_values("date")[CUM_COL]) == [1, 2, 3, 4, 5, 6]  # and rebased


def test_weather_never_rebases_precipitation_cumulative(configured_weather_extractor, no_export, monkeypatch):
    """Measured against the live API: precipitation.cumulative is a per-day total, and
    overlapping windows return identical values. Rebasing it would corrupt the data, so
    a widened group must hand the values through untouched.

    (Probe: same field over 2025-05-01..08-31 and 2025-07-01..08-31 returned 62/62
    identical values on the overlap, max abs diff 0.0.)
    """
    ext = configured_weather_extractor
    ext.use_cache = False
    real_col = "precipitation.cumulative"

    entities = pd.DataFrame(
        [
            {"id": "w1", "geometry": GEOM_A, "start_date": "2025-04-01", "end_date": "2025-04-06"},
            {"id": "w2", "geometry": GEOM_B, "start_date": "2025-04-04", "end_date": "2025-04-06"},
        ]
    )

    # Per-day values keyed to the calendar date, so any two requests agree on overlap —
    # exactly how the real endpoint behaves.
    per_day = {
        "2025-04-01": 3.0,
        "2025-04-02": 0.0,
        "2025-04-03": 11.1,
        "2025-04-04": 5.0,
        "2025-04-05": 0.1,
        "2025-04-06": 0.5,
    }

    def fake_inner(entity_list, **kwargs):
        frames = []
        for _, row in entity_list.iterrows():
            dates = pd.date_range(row["start_date"], row["end_date"], freq="D").strftime("%Y-%m-%d")
            frames.append(pd.DataFrame({"entity_id": row["id"], "date": dates, real_col: [per_day[d] for d in dates]}))
        return {"results_df": pd.concat(frames, ignore_index=True), "global_errors": []}

    monkeypatch.setattr(ext, "_process_entity_weather_bulk_parallel_inner", fake_inner)

    grouped = ext.process_entity_weather_bulk_parallel(
        entity_list=entities, skip_export=True, spatial_grouping=True, use_cache=False
    )["results_df"]

    # w2 asked for 04-04..04-06 and must see those days' own values, unshifted.
    w2 = grouped[grouped["entity_id"] == "w2"].sort_values("date")
    assert list(w2["date"]) == ["2025-04-04", "2025-04-05", "2025-04-06"]
    assert list(w2[real_col]) == [5.0, 0.1, 0.5]

    # And the whole grouped frame equals what per-field calls would have produced.
    ungrouped = fake_inner(entities)["results_df"]
    key = ["entity_id", "date"]
    cols = key + [real_col]
    pd.testing.assert_frame_equal(
        grouped[cols].sort_values(key).reset_index(drop=True),
        ungrouped[cols].sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Regressions: the two ways spatial grouping silently did nothing in 2.5.7.
#
# Both bugs produced CORRECT output — they just extracted it the expensive way,
# so every test asserting "the numbers are right" passed while the feature was
# inert. These assert the *dedup*, not the values.
# ---------------------------------------------------------------------------


LONG = ("2018-01-01", "2026-04-30")  # ~3040 days, far longer than the 400d default cap


def _cell_members(n, window, geoms=(GEOM_A, GEOM_B)):
    """n fields in ONE geohash cell, all requesting the same window."""
    start, end = window
    return pd.DataFrame(
        [{"id": f"f{i}", "geometry": geoms[i % len(geoms)], "start_date": start, "end_date": end} for i in range(n)]
    )


class TestCapBoundsWideningNotLength:
    """`spatial_max_window_days` must bound how far the union WIDENS a member.

    Comparing the union's absolute length against the cap split a cell whose members
    all wanted the SAME long window into one bucket per field: nobody's window got
    shorter, the sharing was just destroyed, and dedup fell to 1.0x.
    """

    def test_identical_long_windows_stay_in_one_bucket(self, weather_extractor):
        entities = _cell_members(6, LONG)
        keyed = _resolve(weather_extractor, entities, cap=400)

        assert keyed["_spatial_group_key"].nunique() == 1, (
            "members all requesting the same window must share a bucket however long "
            "that window is — merging them widens nobody"
        )
        assert keyed["_win_start"].nunique() == 1 and keyed["_win_end"].nunique() == 1

    def test_long_member_does_not_widen_short_neighbours(self, weather_extractor):
        """The cap's original purpose: one long-history field must not drag the rest."""
        entities = _cell_members(3, ("2025-04-01", "2026-01-25"))  # ~300d each
        entities.loc[len(entities)] = {
            "id": "long",
            "geometry": GEOM_A,
            "start_date": "2018-01-01",
            "end_date": "2026-04-30",
        }
        keyed = _resolve(weather_extractor, entities, cap=400).set_index("id")

        assert keyed.loc["long", "_spatial_group_key"] != keyed.loc["f0", "_spatial_group_key"]
        # The short members keep their own ~300d window, not the 3040d union.
        short_span = keyed.loc["f0", "_win_end"] - keyed.loc["f0", "_win_start"]
        assert short_span <= pd.Timedelta(days=400), f"short members widened to {short_span.days}d"

    def test_long_member_arriving_first_still_does_not_widen_the_short_ones(self, weather_extractor):
        """Order independence: the binding constraint is the SMALLEST own-window in the
        bucket, so it cannot matter whether the long member is seen first."""
        entities = pd.DataFrame(
            [
                {"id": "long", "geometry": GEOM_A, "start_date": "2018-01-01", "end_date": "2026-04-30"},
                {"id": "short", "geometry": GEOM_B, "start_date": "2025-04-01", "end_date": "2026-01-25"},
            ]
        )
        keyed = _resolve(weather_extractor, entities, cap=400).set_index("id")

        assert keyed.loc["long", "_spatial_group_key"] != keyed.loc["short", "_spatial_group_key"]
        short_span = keyed.loc["short", "_win_end"] - keyed.loc["short", "_win_start"]
        assert short_span <= pd.Timedelta(days=400)

    def test_ordinary_staggered_windows_still_merge(self, weather_extractor):
        """The fix must not loosen the cap for the case it was written for."""
        keyed = _resolve(weather_extractor, _staggered_entities(), cap=400)
        keys = dict(zip(keyed["id"], keyed["_spatial_group_key"]))
        assert keys["f1"] == keys["f2"]

    def test_collapse_to_no_dedup_is_logged_not_silent(self, weather_extractor):
        """A cap that buys nothing must say so, naming the knob and the window."""
        # Two members whose union genuinely exceeds every allowance.
        entities = pd.DataFrame(
            [
                {"id": "a", "geometry": GEOM_A, "start_date": "2020-01-01", "end_date": "2020-03-01"},
                {"id": "b", "geometry": GEOM_B, "start_date": "2026-01-01", "end_date": "2026-03-01"},
            ]
        )
        _resolve(weather_extractor, entities, cap=400)

        warnings = [str(c) for c in weather_extractor.logger.warning.call_args_list]
        assert any(
            "spatial_max_window_days" in w for w in warnings
        ), f"expected a warning naming the knob, got: {warnings}"


class TestGroupedEqualsUngroupedOnLongWindows:
    """Criterion 5 — grouping changes cost, never values, including under the cap fix."""

    def test_long_identical_windows_match_the_ungrouped_baseline(self, weather_extractor, no_export):
        entities = _cell_members(4, ("2025-01-01", "2025-04-11"))  # 100 days

        grouped = _grouped_run(weather_extractor, entities)["results_df"]
        ungrouped = _daily_fetch(entities)["results_df"]

        key = ["entity_id", "date"]
        cols = key + [CUM_COL, "daily"]
        g = grouped[cols].sort_values(key).reset_index(drop=True)
        u = ungrouped[cols].sort_values(key).reset_index(drop=True)

        pd.testing.assert_frame_equal(g, u, check_dtype=False)
        for col in (CUM_COL, "daily"):
            max_abs_diff = (pd.to_numeric(g[col]) - pd.to_numeric(u[col])).abs().max()
            assert max_abs_diff == 0, f"{col} drifted by {max_abs_diff}"
