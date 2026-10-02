"""
EDAGRO_OUTPUT_PREFIX without WorkflowManager — the stateless-container path.

A stateless container may build an extractor directly (no WorkflowManager).
The prefix used to be resolved only inside ``WorkflowManager.__init__``, so a
directly-built extractor ignored it and wrote results / partials to the pod's local
disk, which is deleted with the pod. It now lives in ``core._fs.apply_output_prefix``,
shared by ``setup_environment()``, ``WorkflowManager`` and ``BaseExtractor``.

Also: ``BaseExtractor`` defaulted ``env`` to ``"production"``, which is not a key of
any URL table, so a config without ``"env"`` crashed on the first URL lookup.
"""

import time

import pytest

from earthdaily.agriculture.core._fs import apply_output_prefix
from earthdaily.agriculture.core.base_extractor import BaseExtractor
from earthdaily.agriculture.core.functions_enhanced import setup_environment

pytestmark = pytest.mark.public

PREFIX = "s3://bucket/runs/2026-10-01"
ROUTED = {
    "output_result_dir": f"{PREFIX}/results",
    "partial_result_dir": f"{PREFIX}/partials",
    "cache_dir": f"{PREFIX}/cache",
}


class _Plain(BaseExtractor):
    """A BaseExtractor with nothing of its own — exercises __init__ only."""


def _extractor(config):
    return _Plain("fake-token", time.time() + 3600, config)


@pytest.fixture
def prefix(monkeypatch):
    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", PREFIX + "/")  # trailing slash is stripped
    return PREFIX


@pytest.fixture
def no_prefix(monkeypatch):
    monkeypatch.delenv("EDAGRO_OUTPUT_PREFIX", raising=False)


class TestApplyOutputPrefix:
    def test_routes_the_three_writer_paths(self, prefix):
        assert apply_output_prefix({"env": "prod"}) == {"env": "prod", **ROUTED}

    def test_local_storage_ignores_the_prefix(self, prefix):
        assert apply_output_prefix({"output_result_dir": "/proj/results"}, "local") == {
            "output_result_dir": "/proj/results"
        }

    def test_no_prefix_leaves_config_untouched(self, no_prefix):
        assert apply_output_prefix({"output_result_dir": "/proj/results"}) == {"output_result_dir": "/proj/results"}

    def test_is_idempotent(self, prefix):
        once = apply_output_prefix({})
        assert apply_output_prefix(dict(once)) == once

    def test_rejects_an_unknown_storage_mode(self):
        with pytest.raises(ValueError, match="Invalid storage"):
            apply_output_prefix({}, "gcs")


class TestSetupEnvironment:
    @pytest.fixture(autouse=True)
    def _credentials(self, monkeypatch):
        for var in ("API_USERNAME", "API_PASSWORD", "API_CLIENT_ID", "API_CLIENT_SECRET"):
            monkeypatch.setenv(f"PROD_{var}", "x")

    def test_honours_the_prefix(self, prefix, tmp_path):
        config = setup_environment(env="prod", project_root=str(tmp_path))
        assert {k: config[k] for k in ROUTED} == ROUTED
        assert config["project_root"] == str(tmp_path.resolve())

    def test_local_storage_keeps_local_folders(self, prefix, tmp_path):
        config = setup_environment(env="prod", project_root=str(tmp_path), storage="local")
        assert config["output_result_dir"] == str(tmp_path.resolve() / "results")


class TestDirectlyBuiltExtractor:
    def test_config_from_setup_environment_writes_under_the_prefix(self, prefix, tmp_path, monkeypatch):
        for var in ("API_USERNAME", "API_PASSWORD", "API_CLIENT_ID", "API_CLIENT_SECRET"):
            monkeypatch.setenv(f"PROD_{var}", "x")
        ex = _extractor(setup_environment(env="prod", project_root=str(tmp_path)))
        assert ex.output_path == ROUTED["output_result_dir"]
        assert ex.partial_path == ROUTED["partial_result_dir"]

    def test_hand_built_config_falls_back_to_the_prefix(self, prefix):
        """A container may pass a minimal config — the pod's disk must not be the default."""
        ex = _extractor({"env": "prod"})
        assert ex.output_path == ROUTED["output_result_dir"]
        assert ex.partial_path == ROUTED["partial_result_dir"]
        assert ex.cache_dir == ROUTED["cache_dir"]
        assert ex.use_cache is False  # a remote cache_dir is force-disabled by design

    def test_explicit_config_paths_still_win(self, prefix):
        ex = _extractor({"env": "prod", "output_result_dir": "/explicit/results"})
        assert ex.output_path == "/explicit/results"

    def test_a_partial_config_is_not_split_across_local_and_remote(self, prefix):
        """Any explicit path makes the config complete: the prefix must not fill in the rest.

        Filling only the missing keys gave local results but partials / failed IDs on S3,
        so a retry looked for its failed IDs in the wrong place.
        """
        ex = _extractor({"env": "prod", "output_result_dir": "/explicit/results"})
        assert ex.partial_path is None
        assert not str(ex.cache_dir).startswith("s3://")

    def test_the_fallback_is_announced_once(self, prefix):
        from loguru import logger

        messages: list = []
        sink = logger.add(messages.append, level="INFO")
        try:
            _extractor({"env": "prod"})
            _extractor({"env": "prod", "output_result_dir": "/explicit/results"})
        finally:
            logger.remove(sink)
        announced = [m for m in messages if "No output paths in config" in str(m)]
        assert len(announced) == 1, "logged for the path-less config only"

    def test_without_a_prefix_nothing_changes(self, no_prefix):
        ex = _extractor({"env": "prod"})
        assert ex.output_path is None
        assert ex.partial_path is None


def test_env_defaults_to_a_real_url_key(no_prefix):
    """config without "env" used to give "production" -> KeyError on any URL lookup."""
    from earthdaily.agriculture.processors.processor_zarc_functions import ZARCExtractor

    ex = ZARCExtractor("fake-token", time.time() + 3600, {})
    assert ex.env == "prod"
    assert ex.zarc_url


def test_storage_s3_rejects_a_local_prefix(monkeypatch):
    """storage='s3' means "must be remote"; a local EDAGRO_OUTPUT_PREFIX used to pass."""
    from unittest.mock import MagicMock, patch

    from earthdaily.agriculture.services.workflow_manager import WorkflowManager

    monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", "/tmp/not-remote")
    local = {
        "env": "prod",
        "project_root": "/proj",
        "output_result_dir": "/proj/results",
        "partial_result_dir": "/proj/partials",
        "cache_dir": "/proj/cache",
    }
    with (
        patch("earthdaily.agriculture.services.workflow_manager.setup_environment", return_value=local),
        patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator", return_value=MagicMock()),
        patch("earthdaily.agriculture.services.workflow_manager.setup_logging"),
        pytest.raises(ValueError, match="remote URI"),
    ):
        WorkflowManager(env="prod", storage="s3")
