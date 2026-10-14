"""core.sharding — split, plan, find reports and merge for sharded stateless runs (local paths).

The same functions run on s3:// / az:// through core._fs; the remote round trip is in
tests/test_cloud_writers_integration.py (moto).
"""

import json

import pandas as pd
import pytest

from earthdaily.agriculture.core import sharding
from earthdaily.agriculture.core.api_utils import export_results

pytestmark = pytest.mark.public


def _entities(n=10):
    return pd.DataFrame({"id": [f"e{i}" for i in range(n)], "geometry": [f"POINT ({i} 0)" for i in range(n)]})


def _write_final(results_dir, rows, prefix="cov"):
    path, _ = export_results(pd.DataFrame(rows), [], results_dir, prefix=prefix, verbose=False)
    return path


class TestSelectShard:
    def test_is_positional_with_a_string_index(self):
        df = _entities(5).set_index("id", drop=False)  # label % n would raise
        parts = [sharding.select_shard(df, i, 2) for i in range(2)]
        assert [list(p["id"]) for p in parts] == [["e0", "e2", "e4"], ["e1", "e3"]]

    def test_shards_cover_every_entity_exactly_once(self):
        df = _entities(11)
        ids = [i for k in range(4) for i in sharding.select_shard(df, k, 4)["id"]]
        assert sorted(ids) == sorted(df["id"])

    def test_rejects_an_out_of_range_shard(self):
        with pytest.raises(ValueError):
            sharding.select_shard(_entities(), 3, 3)


class TestPlanNumShards:
    def test_sized_from_a_time_budget(self):
        # 960/min x (15 min - 5 s) x 0.8 ~= 11 456 per shard
        assert sharding.plan_num_shards(100_000, 960, target_minutes=15) == 9

    def test_slow_extractors_get_more_shards(self):
        assert sharding.plan_num_shards(10_000, 89) > sharding.plan_num_shards(10_000, 960)

    def test_at_least_one_shard(self):
        assert sharding.plan_num_shards(0, 100) == 1
        assert sharding.plan_num_shards(3, 1000) == 1


class TestSplitEntities:
    def test_writes_one_file_per_shard_and_a_manifest(self, tmp_path):
        base = str(tmp_path / "run")
        manifest = sharding.split_entities(_entities(10), base, 3)

        assert [s["n_entities"] for s in manifest["shards"]] == [4, 3, 3]
        assert json.loads((tmp_path / "run" / "shards.json").read_text()) == manifest
        back = pd.concat(sharding.read_table(s["input"]) for s in manifest["shards"])
        assert sorted(back["id"]) == sorted(_entities(10)["id"])
        assert manifest["shards"][1]["output_prefix"].endswith("shard-1")

    def test_empty_shards_are_not_listed(self, tmp_path):
        manifest = sharding.split_entities(_entities(2), str(tmp_path / "run"), 5)
        assert [s["shard"] for s in manifest["shards"]] == [0, 1]

    def test_accepts_a_file_uri(self, tmp_path):
        src = tmp_path / "entities.csv"
        _entities(4).to_csv(src, index=False)
        manifest = sharding.split_entities(str(src), str(tmp_path / "run"), 2)
        assert manifest["n_entities"] == 4


class TestReports:
    def test_find_report_is_the_newest_final(self, tmp_path):
        d = str(tmp_path / "results")
        _write_final(d, {"entity_id": ["a"], "v": [1]})
        newest = _write_final(d, {"entity_id": ["b"], "v": [2]})
        assert sharding.find_report(d, "cov") == newest

    def test_find_report_sees_parquet(self, tmp_path):
        d = str(tmp_path / "results")
        path, _ = export_results(
            pd.DataFrame({"entity_id": ["a"]}), [], d, prefix="cov", verbose=False, export_format="parquet"
        )
        assert sharding.find_report(d) == path

    def test_no_report(self, tmp_path):
        assert sharding.find_report(str(tmp_path)) is None


class TestMerge:
    def _run(self, tmp_path, num_shards=2):
        base = str(tmp_path / "run")
        sharding.split_entities(_entities(4), base, num_shards)
        return base

    def test_merges_every_shard(self, tmp_path):
        base = self._run(tmp_path)
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e0", "e2"], "v": [0, 2]})
        _write_final(f"{base}/shard-1/results", {"entity_id": ["e1", "e3"], "v": [1, 3]})

        out = sharding.merge_shards(base, prefix="cov")
        assert out["rows"] == 4 and out["missing"] == [] and out["shards_merged"] == 2
        assert sorted(sharding.read_table(out["report_path"])["entity_id"]) == ["e0", "e1", "e2", "e3"]

    def test_a_retry_pass_adds_what_the_first_pass_lacked(self, tmp_path):
        base = self._run(tmp_path, num_shards=1)
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e0", "e1"], "v": [0, 1]})  # e2, e3 failed
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e2", "e3"], "v": [2, 3]})  # retry pass

        merged = sharding.read_table(sharding.merge_shards(base, prefix="cov")["report_path"])
        assert sorted(merged["entity_id"]) == ["e0", "e1", "e2", "e3"]

    def test_a_rerun_shard_replaces_rows_instead_of_doubling_them(self, tmp_path):
        base = self._run(tmp_path, num_shards=1)
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e0", "e0", "e1"], "v": [9, 9, 9]})  # crashed run
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e0", "e0", "e1"], "v": [1, 2, 3]})  # re-run

        merged = sharding.read_table(sharding.merge_shards(base, prefix="cov")["report_path"])
        assert len(merged) == 3 and sorted(merged["v"]) == [1, 2, 3]  # time series keep all their rows

    def test_missing_shards_are_reported_or_refused(self, tmp_path):
        base = self._run(tmp_path)
        _write_final(f"{base}/shard-0/results", {"entity_id": ["e0"], "v": [0]})

        assert sharding.merge_shards(base, prefix="cov")["missing"] == [1]
        with pytest.raises(RuntimeError, match="no results"):
            sharding.merge_shards(base, prefix="cov", require_all=True)

    def test_nothing_to_merge_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="nothing to merge"):
            sharding.merge_shards(self._run(tmp_path), prefix="cov")
