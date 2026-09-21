"""
Tests for the CSV / Parquet results-export switch (``export_format``).

Covers the four layers the feature touches:
  1. ``api_utils.resolve_export_format`` — precedence + safe fallback.
  2. ``api_utils.export_results`` — writes csv/parquet; errors always stay csv;
     inline partial flushes (no kwarg) honour ``EDAGRO_EXPORT_FORMAT``.
  3. ``BaseExtractor`` — resolves ``self.export_format`` from config/env and
     ``_finalize_extraction`` writes the chosen format + cleans parquet partials.
  4. ``WorkflowManager`` — ``settings.export_format`` (and step-level override)
     propagate to the instantiated extractor.

Strategy mirrors test_workflow_manager.py: bypass heavy ``__init__`` paths and
use a module-level fake extractor so no real API calls are made.
"""

from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from earthdaily.agriculture.core.api_utils import (
    SUPPORTED_EXPORT_FORMATS,
    export_results,
    resolve_export_format,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public


# ---------------------------------------------------------------------------
# Module-level fake extractor — importable by the workflow YAML via
# importlib.import_module("tests.test_export_format").
# ---------------------------------------------------------------------------


class _FakeExtractor:
    """Records the export_format set on it; returns the entity_list unchanged."""

    instances: list["_FakeExtractor"] = []

    def __init__(self, bearer_token, token_expiration, config=None):
        self.column_mapping = {}
        self.export_format = None  # default; WorkflowManager may overwrite
        _FakeExtractor.instances.append(self)

    def setup_fake_parameters(self, **kwargs):
        self.setup_kwargs = kwargs

    def process_fake_bulk_parallel(
        self,
        entity_list,
        params=None,
        max_workers=10,
        output_path=None,
        fail_safe=False,
        prefix="fake",
        generate_report=False,
        skip_export=True,
    ):
        return {"results_df": entity_list, "global_errors": [], "failed_ids": []}


@pytest.fixture(autouse=True)
def _clear_instances():
    _FakeExtractor.instances.clear()
    yield
    _FakeExtractor.instances.clear()


# ---------------------------------------------------------------------------
# 1. resolve_export_format
# ---------------------------------------------------------------------------


def test_resolve_default_csv(monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    assert resolve_export_format() == "csv"
    assert resolve_export_format(None) == "csv"


def test_resolve_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("EDAGRO_EXPORT_FORMAT", "csv")
    assert resolve_export_format("parquet") == "parquet"


def test_resolve_env_var(monkeypatch):
    monkeypatch.setenv("EDAGRO_EXPORT_FORMAT", "parquet")
    assert resolve_export_format() == "parquet"


def test_resolve_case_insensitive():
    assert resolve_export_format("PARQUET") == "parquet"


def test_resolve_unknown_falls_back_to_csv_without_raising(monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    assert resolve_export_format("xlsx") == "csv"
    assert set(SUPPORTED_EXPORT_FORMATS) == {"csv", "parquet"}


# ---------------------------------------------------------------------------
# 2. export_results writer
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame({"entity_id": [1, 2], "value": [0.3, 0.5]})


def test_export_results_default_csv(sample_df, tmp_path, monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    export_results(sample_df, [], str(tmp_path), prefix="cov", verbose=False)
    assert list(tmp_path.glob("cov_results_*_final.csv"))
    assert not list(tmp_path.glob("*.parquet"))


def test_export_results_parquet_results_errors_stay_csv(sample_df, tmp_path):
    export_results(
        sample_df,
        [{"id": 3, "error": "boom"}],
        str(tmp_path),
        prefix="cov",
        verbose=False,
        export_format="parquet",
    )
    pq = list(tmp_path.glob("cov_results_*_final.parquet"))
    assert pq, "results should be parquet"
    assert list(tmp_path.glob("cov_errors_*_final.csv")), "errors must stay csv"
    assert not list(tmp_path.glob("*errors*.parquet"))
    back = pd.read_parquet(pq[0])
    assert list(back.columns) == ["entity_id", "value"]
    assert len(back) == 2


def test_export_results_partial_honors_env(sample_df, tmp_path, monkeypatch):
    # Inline partial flushes call export_results() without the kwarg — they must
    # still pick up the env var so partials match the final format.
    monkeypatch.setenv("EDAGRO_EXPORT_FORMAT", "parquet")
    export_results(sample_df, [], str(tmp_path), prefix="cov", partial=True, verbose=False)
    assert list(tmp_path.glob("cov_results_*_partial.parquet"))


def test_export_results_empty_returns_none(tmp_path):
    assert export_results(pd.DataFrame(), [], str(tmp_path), verbose=False) == (None, None)


# ---------------------------------------------------------------------------
# 3. BaseExtractor wiring + _finalize_extraction
# ---------------------------------------------------------------------------


def _make_extractor(config):
    ext = BaseExtractor.__new__(BaseExtractor)
    BaseExtractor.__init__(ext, "tok", 9_999_999_999, config)
    return ext


def test_base_extractor_export_format_from_config(monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    assert _make_extractor({"export_format": "parquet"}).export_format == "parquet"


def test_base_extractor_export_format_from_env(monkeypatch):
    monkeypatch.setenv("EDAGRO_EXPORT_FORMAT", "parquet")
    assert _make_extractor({}).export_format == "parquet"


def test_base_extractor_config_beats_env(monkeypatch):
    monkeypatch.setenv("EDAGRO_EXPORT_FORMAT", "parquet")
    assert _make_extractor({"export_format": "csv"}).export_format == "csv"


def test_base_extractor_default_none(monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    # None -> resolved to csv lazily at write time.
    assert _make_extractor({}).export_format is None


def test_finalize_extraction_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    ext = _make_extractor(
        {
            "output_result_dir": str(tmp_path),
            "partial_result_dir": str(tmp_path),
            "export_format": "parquet",
        }
    )
    df = pd.DataFrame({"entity_id": [1], "value": [0.4]})
    _, status = ext._finalize_extraction(df, [], [], prefix="cov", verbose=False)
    assert status["exported"] is True
    assert list(tmp_path.glob("cov_results_*_final.parquet"))


def test_finalize_extraction_cleans_parquet_partials(tmp_path, monkeypatch):
    monkeypatch.delenv("EDAGRO_EXPORT_FORMAT", raising=False)
    ext = _make_extractor(
        {
            "output_result_dir": str(tmp_path),
            "partial_result_dir": str(tmp_path),
            "export_format": "parquet",
        }
    )
    # A leftover parquet partial from an earlier flush must be cleaned on success.
    (tmp_path / "cov_20250101_000000_000000_partial.parquet").write_text("stub")
    ext._finalize_extraction(pd.DataFrame({"entity_id": [1]}), [], [], prefix="cov", verbose=False)
    assert not list(tmp_path.glob("*_partial.parquet")), "parquet partials should be cleaned up"


# ---------------------------------------------------------------------------
# 4. WorkflowManager settings propagation
# ---------------------------------------------------------------------------


def _bare_manager(tmp_path):
    with patch.object(WorkflowManager, "__init__", lambda self, *a, **kw: None):
        wm = WorkflowManager()
    wm.workflow_cfg = None
    wm.workflow_steps = None
    wm.step_map = None
    wm.workflow_results = {}
    wm.config = {}
    wm.bearer_token = "tok"
    wm.token_expiration = None
    wm.output_result_dir = str(tmp_path)
    wm.partial_result_dir = str(tmp_path)
    wm.cache_dir = str(tmp_path)
    wm.sfd_list = None
    wm._run_prefix = None
    return wm


def _fake_step(**extra):
    return {
        "name": "s1",
        "extractor": "_FakeExtractor",
        "module": "tests.test_export_format",
        "setup": {"method": "setup_fake_parameters", "params": {}},
        "run": {"method": "process_fake_bulk_parallel", "params": {"skip_export": True}},
        **extra,
    }


def test_workflow_settings_export_format_propagates(tmp_path):
    wm = _bare_manager(tmp_path)
    cfg = {"name": "pq", "settings": {"export_format": "parquet"}, "steps": [_fake_step()]}
    (tmp_path / "wf.yml").write_text(yaml.dump({"workflow": cfg}))
    wm.load_workflow(str(tmp_path / "wf.yml"))
    wm.run_workflow(entity_list=pd.DataFrame({"id": ["a"]}))
    assert _FakeExtractor.instances, "extractor was not instantiated"
    assert _FakeExtractor.instances[0].export_format == "parquet"


def test_workflow_step_level_export_format_overrides_settings(tmp_path):
    wm = _bare_manager(tmp_path)
    cfg = {
        "name": "pq",
        "settings": {"export_format": "parquet"},
        "steps": [_fake_step(export_format="csv")],  # step-level wins
    }
    (tmp_path / "wf.yml").write_text(yaml.dump({"workflow": cfg}))
    wm.load_workflow(str(tmp_path / "wf.yml"))
    wm.run_workflow(entity_list=pd.DataFrame({"id": ["a"]}))
    assert _FakeExtractor.instances[0].export_format == "csv"


def test_workflow_no_export_format_leaves_default(tmp_path):
    wm = _bare_manager(tmp_path)
    cfg = {"name": "pq", "settings": {}, "steps": [_fake_step()]}
    (tmp_path / "wf.yml").write_text(yaml.dump({"workflow": cfg}))
    wm.load_workflow(str(tmp_path / "wf.yml"))
    wm.run_workflow(entity_list=pd.DataFrame({"id": ["a"]}))
    # Untouched — BaseExtractor would resolve None -> csv at write time.
    assert _FakeExtractor.instances[0].export_format is None
