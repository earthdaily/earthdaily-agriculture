"""Tests for the cache-disable-on-remote-cache-dir behaviour introduced
on the feat/s3-fsspec-writers branch.

The local-only atomic-rename in BaseExtractor._update_cache cannot run on
object stores (no atomic rename in S3, no shared filesystem between pods),
so when ``config["cache_dir"]`` is a remote URI we force-disable caching at
construction time and refuse to flip it back on via apply_cache_setting.

Tests run against a thin BaseExtractor subclass with __init__ exercised
fully — that's the codepath we care about.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.public


@pytest.fixture
def fake_config_local(tmp_path):
    """Minimal config with a LOCAL cache_dir; everything else stubbed."""
    return {
        "env": "preprod",
        "project_root": str(tmp_path),
        "output_result_dir": str(tmp_path / "results"),
        "partial_result_dir": str(tmp_path / "partials"),
        "cache_dir": str(tmp_path / "cache"),
        "use_cache": True,
    }


@pytest.fixture
def fake_config_remote(tmp_path):
    """Minimal config with an S3 cache_dir."""
    return {
        "env": "preprod",
        "project_root": str(tmp_path),
        "output_result_dir": str(tmp_path / "results"),
        "partial_result_dir": str(tmp_path / "partials"),
        "cache_dir": "s3://my-bucket/edagro-cache",
        "use_cache": True,
    }


def _make_extractor(config):
    """Construct a real BaseExtractor with __init__ exercised fully."""
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    return BaseExtractor(
        bearer_token="fake-token",
        token_expiration=None,
        config=config,
    )


class TestCacheDirDetection:
    def test_local_cache_dir_keeps_use_cache_true(self, fake_config_local):
        ext = _make_extractor(fake_config_local)
        assert ext._cache_dir_is_remote is False
        assert ext.use_cache is True

    @pytest.mark.parametrize(
        "remote_uri",
        ["s3://bucket/cache", "gs://bucket/cache", "az://container/cache"],
    )
    def test_remote_cache_dir_forces_use_cache_false(self, fake_config_remote, remote_uri):
        cfg = dict(fake_config_remote)
        cfg["cache_dir"] = remote_uri
        ext = _make_extractor(cfg)
        assert ext._cache_dir_is_remote is True
        assert ext.use_cache is False

    def test_remote_cache_dir_kept_as_string_not_path(self, fake_config_remote):
        ext = _make_extractor(fake_config_remote)
        # Path("s3://...") would mangle the URI on Windows — must remain a plain str.
        assert isinstance(ext.cache_dir, str)
        assert ext.cache_dir == "s3://my-bucket/edagro-cache"

    def test_local_cache_dir_wrapped_as_pathlib_path(self, fake_config_local):
        from pathlib import Path

        ext = _make_extractor(fake_config_local)
        assert isinstance(ext.cache_dir, Path)


class TestApplyCacheSettingRespectsRemoteDir:
    def test_apply_cache_setting_true_on_remote_is_silently_ignored(self, fake_config_remote):
        ext = _make_extractor(fake_config_remote)
        assert ext.use_cache is False
        # Even when an extractor's setup_*_parameters() explicitly asks to enable
        # caching, the remote cache_dir gate keeps it off.
        ext.apply_cache_setting(use_cache=True)
        assert ext.use_cache is False

    def test_apply_cache_setting_false_on_remote_still_works(self, fake_config_remote):
        ext = _make_extractor(fake_config_remote)
        # Disabling is always honoured.
        ext.apply_cache_setting(use_cache=False)
        assert ext.use_cache is False

    def test_apply_cache_setting_none_on_remote_is_noop(self, fake_config_remote):
        ext = _make_extractor(fake_config_remote)
        ext.apply_cache_setting(use_cache=None)
        assert ext.use_cache is False

    def test_apply_cache_setting_true_on_local_works(self, fake_config_local):
        ext = _make_extractor(fake_config_local)
        # Start by disabling.
        ext.use_cache = False
        # Local cache_dir → re-enabling via apply_cache_setting works.
        ext.apply_cache_setting(use_cache=True)
        assert ext.use_cache is True


class TestCacheMethodsAreNoopWhenRemote:
    """Sanity-check: with use_cache=False, the cache lookup/store paths bail
    early without touching ``self.cache_dir`` (which would error on a Path
    operation against an s3:// string)."""

    def test_lookup_returns_empty_df_when_disabled(self, fake_config_remote):
        import pandas as pd

        ext = _make_extractor(fake_config_remote)
        # cache_key_columns required for the cache code path; setting it to a
        # truthy value lets us prove the lookup short-circuits on use_cache=False
        # rather than because cache_key_columns is missing.
        ext.cache_key_columns = ["id"]
        result = ext._lookup_entity_cache("any-entity-id", params={"x": 1})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_store_is_noop_when_disabled(self, fake_config_remote):
        import pandas as pd

        ext = _make_extractor(fake_config_remote)
        ext.cache_key_columns = ["id"]
        # If the gate didn't hold, _store_entity_cache would call _update_cache
        # which would try to use self.cache_dir (an s3:// string) and crash.
        # Reaching the end of this method without raising IS the assertion.
        ext._store_entity_cache(pd.DataFrame({"id": ["a"], "value": [1]}), params={"x": 1})


class TestWarningEmittedOnRemoteWhenCacheRequested:
    def test_warning_logged_when_user_asked_for_cache_with_remote_dir(self, fake_config_remote, caplog):
        import logging as stdlib_logging

        # Loguru pipes WARNING-level messages to the root logger via
        # InterceptHandler when configured to; here we just attach caplog at
        # WARNING level and inspect any record forwarded to it.
        caplog.set_level(stdlib_logging.WARNING)

        # Verify by side-effect: even without intercepting loguru's formatter
        # the use_cache gate is still applied; the warning text is asserted by
        # capturing loguru directly via a sink.
        from io import StringIO

        from loguru import logger as loguru_logger

        sink = StringIO()
        handler_id = loguru_logger.add(sink, level="WARNING")
        try:
            _make_extractor(fake_config_remote)
        finally:
            loguru_logger.remove(handler_id)

        captured = sink.getvalue()
        assert "Cache disabled" in captured
        assert "s3://my-bucket/edagro-cache" in captured

    def test_no_warning_when_cache_already_off_with_remote_dir(self, fake_config_remote):
        from io import StringIO

        from loguru import logger as loguru_logger

        cfg = dict(fake_config_remote)
        cfg["use_cache"] = False  # user wasn't asking for cache → no warning needed

        sink = StringIO()
        handler_id = loguru_logger.add(sink, level="WARNING")
        try:
            _make_extractor(cfg)
        finally:
            loguru_logger.remove(handler_id)

        # The use_cache=False / remote case is uninteresting — no warning fires.
        assert "Cache disabled" not in sink.getvalue()
