"""Unit tests for the console-only logging toggle introduced for the
dockerization story.

Loguru is a global singleton; tests reset it between runs so handler counts
are deterministic regardless of which test ran first.
"""

from __future__ import annotations

import pytest
from loguru import logger

from earthdaily.agriculture.core import logging_setup
from earthdaily.agriculture.core.logging_setup import setup_logging

pytestmark = pytest.mark.public


@pytest.fixture(autouse=True)
def _reset_loguru():
    """Wipe loguru handlers before AND after each test so global state doesn't leak."""
    logger.remove()
    yield
    logger.remove()


def _handler_count() -> int:
    return len(logger._core.handlers)


# ---------------------------------------------------------------------------
# Default behaviour: file + console
# ---------------------------------------------------------------------------


class TestSetupLoggingDefault:
    def test_default_kwargs_install_console_and_file_sinks(self, tmp_path, monkeypatch):
        # Make sure no stray env var from the host process flips us into console-only.
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "logs"))
        # 2 sinks: stderr + the rotating file sink.
        assert _handler_count() == 2
        # log_dir was created.
        assert (tmp_path / "logs").is_dir()


# ---------------------------------------------------------------------------
# log_to_console_only kwarg
# ---------------------------------------------------------------------------


class TestLogToConsoleOnlyKwarg:
    def test_skips_file_sink_and_does_not_create_log_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        log_dir = tmp_path / "should_not_be_created"
        setup_logging(log_dir=str(log_dir), log_to_console_only=True)
        # Only the stderr sink remains.
        assert _handler_count() == 1
        # mkdir is skipped — directory must NOT have been created.
        assert not log_dir.exists()

    def test_console_off_plus_console_only_yields_no_handlers(self, tmp_path, monkeypatch):
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "logs"), log_to_console=False, log_to_console_only=True)
        # No file sink (console_only) and no console sink (log_to_console=False).
        assert _handler_count() == 0


# ---------------------------------------------------------------------------
# Env-var override
# ---------------------------------------------------------------------------


class TestEnvVarOverride:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "True", "Yes"])
    def test_truthy_env_skips_file_sink_even_with_kwarg_false(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv(logging_setup._CONSOLE_ONLY_ENV, value)
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir), log_to_console_only=False)
        assert _handler_count() == 1  # console only
        assert not log_dir.exists()  # mkdir skipped

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_env_lets_kwarg_decide(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv(logging_setup._CONSOLE_ONLY_ENV, value)
        # kwarg default is False → file sink should be installed.
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert _handler_count() == 2

    def test_unset_env_lets_kwarg_decide(self, tmp_path, monkeypatch):
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "logs"), log_to_console_only=True)
        assert _handler_count() == 1


# ---------------------------------------------------------------------------
# Idempotency / re-init
# ---------------------------------------------------------------------------


class TestReinitIsClean:
    def test_calling_twice_does_not_double_up_sinks(self, tmp_path, monkeypatch):
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "logs"))
        first = _handler_count()
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert _handler_count() == first  # 2 in both calls

    def test_switch_to_console_only_drops_file_sink(self, tmp_path, monkeypatch):
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "logs"))
        assert _handler_count() == 2
        setup_logging(log_dir=str(tmp_path / "logs"), log_to_console_only=True)
        assert _handler_count() == 1


# ---------------------------------------------------------------------------
# Module reloading sanity (env var read at call time, not import time)
# ---------------------------------------------------------------------------


class TestEnvReadAtCallTime:
    def test_env_change_after_import_takes_effect_on_next_call(self, tmp_path, monkeypatch):
        # Demonstrates the env var is consulted on every setup_logging call,
        # not cached at import time — important for test fixtures and dynamic
        # toggling between subsequent runs in the same process.
        monkeypatch.delenv(logging_setup._CONSOLE_ONLY_ENV, raising=False)
        setup_logging(log_dir=str(tmp_path / "a"))
        assert _handler_count() == 2

        monkeypatch.setenv(logging_setup._CONSOLE_ONLY_ENV, "1")
        setup_logging(log_dir=str(tmp_path / "b"))
        assert _handler_count() == 1
        # Importantly, the second log_dir was NOT created (mkdir skipped).
        assert not (tmp_path / "b").exists()
