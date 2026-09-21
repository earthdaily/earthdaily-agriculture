# utils/logging_config.py
import os
import sys
from pathlib import Path

from loguru import logger

# Env var for the dockerization-friendly "console-only" toggle. When set to
# any of {"1", "true", "yes", "on"} (case-insensitive), setup_logging skips
# the file sink entirely — useful for containerized runs where stdout/stderr
# is the only log surface that survives the pod's lifetime. The env var
# overrides the kwarg so containers can flip it without code changes.
_CONSOLE_ONLY_ENV = "EDAGRO_LOG_CONSOLE_ONLY"
_TRUTHY = {"1", "true", "yes", "on"}


def _env_console_only() -> bool:
    """Read EDAGRO_LOG_CONSOLE_ONLY from the environment, tolerating common shapes."""
    return os.environ.get(_CONSOLE_ONLY_ENV, "").strip().lower() in _TRUTHY


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_to_console: bool = True,
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "zip",
    log_to_console_only: bool = False,
):
    """
    Initialize loguru for EarthDaily Agriculture Manager.
    Should be called once at the start of workflow execution.

    Args:
        log_dir: Directory to store log files. Ignored when ``log_to_console_only``
            resolves to True.
        log_level: Logging level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to output logs to console (stderr).
        rotation: When to rotate logs (e.g., "1 day", "100 MB", "12:00")
        retention: How long to keep logs (e.g., "30 days", "1 week")
        compression: Compression format for old logs ("zip", "gz", "bz2", "xz")
        log_to_console_only: When True, **skip the file sink entirely** — useful
            for ephemeral container runtimes (Argo, ECS, Cloud Run) where local
            disk vanishes on pod exit and stdout/stderr is the only log surface
            captured by the orchestrator. The env var ``EDAGRO_LOG_CONSOLE_ONLY=1``
            overrides this kwarg so containers can flip it without code changes.
            Default ``False`` — existing notebook callers keep getting
            ``logs/earthdaily_<date>.log`` rotation.
    """
    # Env var wins over the kwarg so container deploys don't need code changes.
    console_only = _env_console_only() or log_to_console_only

    if not console_only:
        # Create logs directory only when we're going to write to it.
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler with colors (optional)
    if log_to_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            colorize=True,
        )

    # Add file handler with rotation — skipped in console-only mode.
    if not console_only:
        logger.add(
            f"{log_dir}/earthdaily_{{time:YYYY-MM-DD}}.log",
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            enqueue=True,  # Thread-safe for parallel processing
            backtrace=True,  # Better exception traces
            diagnose=True,  # Detailed error context
        )

    logger.info("✅ Logging system initialized")
    if console_only:
        logger.info("📺 Console-only mode (file sink disabled)")
    else:
        logger.info(f"📁 Log directory: {Path(log_dir).absolute()}")

    return logger
