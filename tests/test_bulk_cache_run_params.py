"""
The bulk cache key must include the run-level ``params`` override.

Every extractor routes its bulk method through ``BaseExtractor._bulk_with_cache``
with ``params=<setup params>`` as the cache key and the run-level override as
``params_kw``. The key used to ignore ``params_kw``, so with caching on, a second run
with a different override (e.g. a workflow's ``run.params.params``) was served the
first run's cached rows and never called the API.
"""

import time

import pandas as pd
import pytest

from earthdaily.agriculture.core.base_extractor import BaseExtractor

pytestmark = pytest.mark.public

SETUP = {"crop": "SOYBEANS", "season_duration": 120}


class _Recorder(BaseExtractor):
    """Minimal extractor whose bulk method records the params it was run with."""

    def __init__(self, cache_dir):
        super().__init__("fake-token", time.time() + 3600, {"env": "prod", "cache_dir": str(cache_dir)})
        self.use_cache = True
        self.cache_key_columns = ["id"]
        self.calls: list = []

    def _bulk(self, entity_list, params_kw=None):
        self.calls.append(params_kw)
        tag = (params_kw or {}).get("crop", SETUP["crop"])
        return {
            "results_df": pd.DataFrame({"id": entity_list["id"], "value": tag}),
            "global_errors": [],
            "failed_ids": [],
        }

    def run(self, entities, params_kw=None):
        return self._bulk_with_cache(entity_list=entities, bulk_method=self._bulk, params=SETUP, params_kw=params_kw)


@pytest.fixture
def ex(tmp_path):
    return _Recorder(tmp_path / "cache")


@pytest.fixture
def entities():
    return pd.DataFrame({"id": ["a", "b"]})


def test_different_run_override_is_not_served_from_cache(ex, entities):
    ex.run(entities, {"crop": "CORN"})
    second = ex.run(entities, {"crop": "COTTON"})

    assert len(ex.calls) == 2, "the second override was served the first run's cache"
    assert set(second["results_df"]["value"]) == {"COTTON"}


def test_same_run_override_is_a_cache_hit(ex, entities):
    ex.run(entities, {"crop": "CORN"})
    second = ex.run(entities, {"crop": "CORN"})

    assert len(ex.calls) == 1
    assert set(second["results_df"]["value"]) == {"CORN"}


def test_override_key_ignores_dict_order(ex, entities):
    ex.run(entities, {"crop": "CORN", "season_duration": 90})
    ex.run(entities, {"season_duration": 90, "crop": "CORN"})

    assert len(ex.calls) == 1


def test_no_override_keeps_the_existing_cache_file(ex, entities):
    """Runs without an override must keep hitting caches written before this fix."""
    ex.run(entities)
    assert ex._cache_path(SETUP).exists()

    ex.run(entities, None)
    ex.run(entities, {})
    assert len(ex.calls) == 1


def test_override_and_setup_only_runs_do_not_share_a_cache(ex, entities):
    ex.run(entities)
    ex.run(entities, {"crop": "CORN"})

    assert len(ex.calls) == 2


def test_normalize_date_handles_plain_date_objects():
    """YAML and parquet date32 give datetime.date, not datetime / Timestamp."""
    from datetime import date, datetime

    assert BaseExtractor.normalize_date(date(2025, 10, 1)) == "2025-10-01"
    assert BaseExtractor.normalize_date(datetime(2025, 10, 1, 12, 30)) == "2025-10-01"
    assert BaseExtractor.normalize_date("2025-10-01T00:00:00") == "2025-10-01"


def test_every_cached_path_forwards_the_run_level_params():
    """With use_cache=True, 16 extractors called _bulk_with_cache without params_kw, so the
    run-level `params` override was silently dropped on the cached path only (the
    uncached path passed it). Guard: any _bulk_with_cache call whose bulk_method accepts
    params_kw must pass it.
    """
    import ast
    import pathlib

    import earthdaily.agriculture as pkg

    missing = []
    for f in pathlib.Path(pkg.__path__[0]).rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "_bulk_with_cache"):
                continue
            kws = {k.arg for k in call.keywords}
            target = next((ast.unparse(k.value) for k in call.keywords if k.arg == "bulk_method"), "")
            inner = fns.get(target.split(".")[-1])
            if inner is not None and "params_kw" in {a.arg for a in inner.args.args} and "params_kw" not in kws:
                missing.append(f"{f.name}:{call.lineno}")
    assert not missing, f"_bulk_with_cache without params_kw: {sorted(missing)}"
