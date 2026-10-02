"""
Where postprocess="file" maps go — FLM, Difference, Zoning (stateless containers).

File mode used to REQUIRE an explicit output_path, so rasters never followed
EDAGRO_OUTPUT_PREFIX: a stateless container had to pass a path itself, and the notebook habit (a
local folder) put the maps on the pod's disk — gone with the pod, while saved_files
still listed them. Now the default is <results dir>/maps, which follows the prefix, and
a local path under a remote prefix with no output_uri mirror is warned about.
"""

import time

import pytest
from loguru import logger

from earthdaily.agriculture.extractors.difference_functions import DifferenceExtractor
from earthdaily.agriculture.extractors.FLM_functions import FLMExtractor
from earthdaily.agriculture.extractors.zoning_functions import ZoningExtractor

pytestmark = pytest.mark.public

PREFIX = "s3://bucket/runs/2026-10-02/shard-0"

SETUPS = {
    "flm": (FLMExtractor, "setup_flm_parameters", "flm_params"),
    "difference": (DifferenceExtractor, "setup_difference_parameters", "difference_params"),
    "zoning": (ZoningExtractor, "setup_zoning_parameters", "zoning_params"),
}


def _setup(kind, config, **kwargs):
    cls, setup, attr = SETUPS[kind]
    ex = cls("fake-token", time.time() + 3600, config)
    getattr(ex, setup)(postprocess="file", map_format="png", **kwargs)
    return ex, getattr(ex, attr)["output_path"]


@pytest.fixture
def warnings():
    messages: list = []
    sink = logger.add(messages.append, level="WARNING")
    yield messages
    logger.remove(sink)


@pytest.mark.parametrize("kind", sorted(SETUPS))
def test_maps_follow_the_prefix_on_a_pod(kind, monkeypatch, warnings):
    """A pod that only sets EDAGRO_OUTPUT_PREFIX gets its maps on S3 — no extra parameter."""
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX)
    _, path = _setup(kind, {"env": "prod"})
    assert path == f"{PREFIX}/results/maps"
    assert not any("LOCAL path" in str(m) for m in warnings)


@pytest.mark.parametrize("kind", sorted(SETUPS))
def test_explicit_output_path_still_wins(kind, monkeypatch):
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX)
    _, path = _setup(kind, {"env": "prod"}, output_path="s3://elsewhere/maps")
    assert path == "s3://elsewhere/maps"


@pytest.mark.parametrize("kind", sorted(SETUPS))
def test_local_maps_under_a_remote_prefix_are_warned(kind, monkeypatch, warnings):
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX)
    _, path = _setup(kind, {"env": "prod"}, output_path="results/maps")
    assert path == "results/maps"
    assert any("LOCAL path" in str(m) for m in warnings), "the lost-with-the-pod case must be loud"


def test_no_warning_when_output_uri_mirrors_the_maps(monkeypatch, warnings):
    """Local working copy + durable output_uri copy is a legitimate design."""
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX)
    monkeypatch.setenv("EDAGRO_OUTPUT_URI", "s3://bucket/maps")
    _setup("flm", {"env": "prod"}, output_path="results/maps")
    assert not any("LOCAL path" in str(m) for m in warnings)


def test_no_warning_for_plain_local_use(monkeypatch, warnings, tmp_path):
    """Notebook use, no prefix: local maps are the point, nothing to warn about."""
    monkeypatch.delenv("EDAGRO_OUTPUT_PREFIX", raising=False)
    _, path = _setup("flm", {"env": "prod", "output_result_dir": str(tmp_path)})
    assert path.startswith(str(tmp_path))
    assert not any("LOCAL path" in str(m) for m in warnings)


def test_no_warning_when_results_are_local_despite_a_prefix_in_the_shell(monkeypatch, warnings, tmp_path):
    """Explicit local results (or storage="local") with a stray remote prefix: the default
    maps path is local on purpose, so neither a warning nor "omit output_path" advice."""
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX)
    _, path = _setup("flm", {"env": "prod", "output_result_dir": str(tmp_path)})
    assert path.startswith(str(tmp_path))
    assert not any("LOCAL path" in str(m) for m in warnings)
