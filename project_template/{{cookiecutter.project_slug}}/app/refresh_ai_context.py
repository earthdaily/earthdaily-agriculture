"""Refresh this project's auto-generated AI context.

Re-runs ``generate_ai_context.py --project-target .`` against the earthdaily-agriculture-internal
source code on disk, regenerating:

  - ``CLAUDE.md``                        — project-flavoured AI memory
  - ``.claude/skills/earthdaily-agriculture/SKILL.md``   — on-demand extractor skill
  - ``.claude/skills/earthdaily-agriculture/extractors.md`` — full per-extractor reference

The values stamped in at cookiecutter time (``project_name``, ``environment``,
``edagro_wheel_version``) are baked into this script so refreshes stay aligned
with the project's identity.

Designed to be called by the project's pre-commit hook (``.pre-commit-config.yaml``)
on wheel bumps — but also runnable manually:

    python app/refresh_ai_context.py

earthdaily-agriculture-internal discovery order (first hit wins):
    1. ``--template <path>`` flag (explicit local clone)
    2. ``EDAGRO_CLIENT_PATH`` env var
    3. Common sibling locations on disk
       (``~/Github/earthdaily-agriculture-internal``, ``~/Documents/Github/earthdaily-agriculture-internal``, …)

If no clone is found OR introspection fails, this script exits 0 with a clear
warning rather than blocking the commit. The project still works without
regenerated context.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Cookiecutter-baked identity. Changing these post-init requires re-running
# `cookiecutter` or hand-editing this file.
PROJECT_NAME = "{{ cookiecutter.project_name }}"
ENVIRONMENT = "{{ cookiecutter.environment }}"
WHEEL_VERSION = "{{ cookiecutter.edagro_wheel_version }}"

# Common locations a teammate might clone earthdaily-agriculture-internal. Extend freely —
# the discovery loop short-circuits on the first hit.
COMMON_LOCATIONS = [
    Path.home() / "Github" / "earthdaily-agriculture-internal",
    Path.home() / "Documents" / "Github" / "earthdaily-agriculture-internal",
    Path.home() / "Projects" / "earthdaily-agriculture-internal",
    Path.home() / "repos" / "earthdaily-agriculture-internal",
    Path.home() / "src" / "earthdaily-agriculture-internal",
    Path("C:/Users") / os.environ.get("USERNAME", "") / "Documents" / "Github" / "earthdaily-agriculture-internal",
]


def _looks_like_edagro_clone(candidate: Path) -> bool:
    return (candidate / "src" / "earthdaily.agriculture" / "scripts" / "generate_ai_context.py").is_file()


def _find_edagro_client(explicit: Path | None) -> Path | None:
    """Resolve a local earthdaily-agriculture-internal clone or return None."""
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if _looks_like_edagro_clone(candidate):
            return candidate
        print(
            f"[earthdaily-agriculture] WARN: --template {candidate} is not an earthdaily-agriculture-internal clone "
            "(missing src/earthdaily/agriculture/scripts/generate_ai_context.py).",
            file=sys.stderr,
        )
        return None

    env = os.environ.get("EDAGRO_CLIENT_PATH")
    if env:
        candidate = Path(env).expanduser().resolve()
        if _looks_like_edagro_clone(candidate):
            return candidate
        print(
            f"[earthdaily-agriculture] WARN: EDAGRO_CLIENT_PATH={env} is not an earthdaily-agriculture-internal clone.",
            file=sys.stderr,
        )

    for loc in COMMON_LOCATIONS:
        loc = loc.expanduser()
        if _looks_like_edagro_clone(loc):
            return loc

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--template", type=Path, default=None,
        help="Path to a local earthdaily-agriculture-internal clone (overrides auto-discovery).",
    )
    parser.add_argument(
        "--target", type=Path, default=Path.cwd(),
        help="Project directory to refresh (default: cwd).",
    )
    args = parser.parse_args()

    target = args.target.resolve()
    if not target.is_dir():
        print(f"[earthdaily-agriculture] target is not a directory: {target}", file=sys.stderr)
        return 1

    edagro_root = _find_edagro_client(args.template)
    if edagro_root is None:
        print(
            "[earthdaily-agriculture] earthdaily-agriculture-internal clone not found. Skipping AI-context refresh.\n"
            "         Set EDAGRO_CLIENT_PATH or pass --template <path> to enable it.",
            file=sys.stderr,
        )
        return 0  # non-blocking — don't fail the commit

    cmd = [
        sys.executable,
        "-m",
        "earthdaily.agriculture.ai_enablement.generate_ai_context",
        "--project-target",
        str(target),
        "--project-name",
        PROJECT_NAME,
        "--environment",
        ENVIRONMENT,
        "--wheel-version",
        WHEEL_VERSION,
    ]

    print(f"[earthdaily-agriculture] Refreshing AI context using {edagro_root}")
    try:
        subprocess.run(cmd, check=True, cwd=str(edagro_root / "src"))
    except subprocess.CalledProcessError as exc:
        # Almost always: the wheel was bumped but the dev hasn't `pip install
        # --force-reinstall` yet, so introspection imports the stale version.
        # Non-blocking; print actionable hint.
        print(
            f"[earthdaily-agriculture] WARN: AI-context refresh exited {exc.returncode}. If you "
            "just bumped the wheel, run `pip install --force-reinstall "
            f"dist/earthdaily_agriculture-{WHEEL_VERSION}-py3-none-any.whl` and try again.",
            file=sys.stderr,
        )
        return 0  # non-blocking

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
