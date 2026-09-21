"""
Notebook environment bootstrap.

Import this at the top of any notebook to ensure the project's ``src/`` directory
is on ``sys.path`` and the working directory is the project root.

Usage (first cell of every notebook)::

    import notebook_setup
    notebook_setup.init()

If the package is installed via ``pip install -e ".[jupyter]"``, this is optional
but still recommended for consistent .env loading.

Also home to the notebook's *gate* helpers — :func:`preflight` and :func:`stop`,
for the case where a run cannot continue until the operator changes a value. Both
halt the cell with an instruction instead of a traceback::

    from earthdaily.agriculture.notebook_setup import preflight

    preflight(
        f"Customer '{code}' does not exist",
        [
            ("creation enabled", CREATE_IF_MISSING, "CREATE_IF_MISSING = True"),
            ("creation confirmed", CONFIRM == code, f'CONFIRM = "{code}"'),
        ],
        notes=["A new tenant can only be removed from the back office."],
    )
"""

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml.

    Strategy 1: walk up from __file__ (works for editable installs / dev layout).
    Strategy 2: walk up from cwd (works when package is installed via wheel and
                the notebook has already chdir'd to the project root).
    """
    # Strategy 1: walk up from this file's location
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Strategy 2: walk up from cwd (handles installed-package scenarios)
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: use cwd
    return cwd


class RunBlocked(Exception):
    """A deliberate stop: the run is waiting on a value the operator must set.

    Not a failure — nothing went wrong and nothing was written. Raised by
    :func:`stop` and :func:`preflight`, and rendered in a notebook as its message
    alone: the frame that raised it is a guard, never the line you need to edit,
    so a traceback pointing at it is noise that hides the actual instruction.
    """


_RULE = "─" * 66

#: ``(label, ok, fix)`` — ``fix`` is the literal line to change, shown on failure.
Check = Tuple[str, bool, str]


def stop(summary: str, detail: str = "") -> None:
    """Print ``detail``, then halt the cell with no traceback.

    Args:
        summary: One line, used as the exception message — kept short because
            some notebook frontends show it when the traceback is empty.
        detail: The full operator-facing block, printed to stdout first.

    Raises:
        RunBlocked: always.
    """
    if detail:
        print(detail)
    raise RunBlocked(summary)


def preflight(
    title: str,
    checks: Sequence[Check],
    notes: Iterable[str] = (),
    rerun: str = "re-run this cell",
) -> None:
    """Report every prerequisite at once, and stop cleanly if any is unmet.

    Guards written in sequence charge an operator one round-trip per unmet
    condition — fix the first, re-run, discover the second. The caller evaluates
    all of ``checks`` up front, so a single run prints the complete list of what
    to change.

    Args:
        title: What is blocked, e.g. ``"Customer 'ACME' does not exist"``.
        checks: ``(label, ok, fix)`` triples, in the order to read them.
        notes: Context lines printed under the checklist (the plan, alternatives,
            anything irreversible), shown only when something is unmet.
        rerun: What to do once the values are set.

    Raises:
        RunBlocked: if any check is falsy.
    """
    rows = [(str(label), bool(ok), str(fix)) for label, ok, fix in checks]
    unmet = [row for row in rows if not row[1]]
    width = max((len(row[0]) for row in rows), default=0)

    out = []
    if unmet:
        out += [_RULE, f"⛔ {title}", "   Nothing has been written.", _RULE]
    out += [f"   {'✅' if ok else '❌'} {label.ljust(width)}   {'' if ok else fix}".rstrip() for label, ok, fix in rows]

    if not unmet:
        print("\n".join(out))
        return

    notes = list(notes)
    if notes:
        out += [""] + [f"   {note}".rstrip() for note in notes]
    out += [
        "",
        f"   {len(unmet)} of {len(rows)} unmet — set the value(s) above, then {rerun}.",
        _RULE,
    ]
    stop(f"{len(unmet)} of {len(rows)} prerequisite(s) unmet — see the checklist above", "\n".join(out))


def _install_blocked_renderer() -> bool:
    """Teach IPython to render :class:`RunBlocked` as its message alone.

    Scoped to that one exception type, so every other traceback is untouched. The
    cell still ends in an error, which is what stops "Run All" from carrying on
    past a gate.

    Returns:
        bool: True if the renderer was installed (i.e. we are under IPython).
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    if shell is None:  # plain python, pytest, a script
        return False

    def _render(shell, etype, value, tb, tb_offset=None):
        # stop() already printed the block; None becomes an empty traceback.
        return None

    shell.set_custom_exc((RunBlocked,), _render)
    return True


def init():
    """
    Set up the notebook environment:
    1. Add src/ to sys.path so ``from earthdaily.agriculture import ...`` works.
    2. Change working directory to project root (where inputs/, results/ etc. live).
    3. Load .env from src/ if present.
    4. Render RunBlocked without a traceback (see :func:`preflight`).
    """
    project_root = _find_project_root()
    src_dir = str(project_root / "src")

    # Ensure src/ is on the import path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Working directory = project root (workspace folders live here)
    os.chdir(str(project_root))

    # Load .env from src/ (where credentials live)
    env_file = project_root / "src" / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            pass

    _install_blocked_renderer()

    print(f"Project root: {project_root}")
    print(f"Working dir:  {os.getcwd()}")
