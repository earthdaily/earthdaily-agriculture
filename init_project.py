"""Bootstrap an empty repo into an EarthDaily Agriculture client project.

Drop this file into your empty repo and run it:

    cd path/to/my-empty-repo
    python init_project.py

The script:
  1. Locates the cookiecutter template. A local clone wins when one is
     available (pass ``--template``, set ``EDAGRO_CLIENT_PATH``, or rely on
     the common-location auto-discovery); otherwise it falls back to the
     public GitHub repo (see ``REMOTE_TEMPLATE_URL`` below).
  2. Forces the cookiecutter ``project_slug`` to the **current folder
     name** so the template populates the folder you're sitting in
     instead of creating a sibling, and so the git repo name, the
     folder name and the slug all stay aligned.
  3. Pre-fills every cookiecutter field (project_name, project_slug,
     environment, edagro_source_path, etc.) and only prompts you
     interactively for ``project_description``. Run with
     ``--non-interactive`` (or pass ``--project-description``) to
     suppress that prompt as well.

Template discovery order (first hit wins):
  1. ``--template <path>`` flag (explicit local path)
  2. ``EDAGRO_CLIENT_PATH`` env var (explicit local path)
  3. Script's own directory (if you ran it from inside an earthdaily-agriculture-internal clone)
  4. Common sibling locations on disk
     (~/Github/earthdaily-agriculture-internal, ~/Documents/Github/earthdaily-agriculture-internal, ...)
  5. ``REMOTE_TEMPLATE_URL`` (the public GitHub repo).
  6. Interactive prompt (skipped under ``--non-interactive``).

Flags:
    --target PATH                 Empty repo to populate (default: cwd)
    --template PATH               Local earthdaily-agriculture-internal clone path
    --project-name NAME           Override project_name (default: folder name)
    --project-description TEXT    Skip the prompt and pass this through
    --environment ENV             prod | preprod (default: prod)
    --non-interactive             No prompts at all; accept all defaults
    --no-internal-docs            Skip docs/internal/ (internal clones only)

Docs are copied into the new project by the template's post-generation hook,
from the repo you generate from — they are not vendored in the template. An
internal clone therefore yields ``docs/`` + ``docs/internal/``; the public repo
has no internal docs to copy and yields ``docs/`` alone. ``docs/study/`` is
always created, empty, for your own exploratory work.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Public GitHub URL for the earthdaily-agriculture-internal repo. Cookiecutter accepts a URL
# directly and clones it on the fly into a temp dir. Until the repo is
# public this URL is **not reachable** — the discovery loop below treats
# the URL as the last fallback and prefers a local clone whenever one is
# available. It points at the PUBLIC repo, which ships `project_template/` — the
# internal repo is not reachable for most people running this script, and this
# file itself ships in the public wheel.
REMOTE_TEMPLATE_URL = "https://github.com/earthdaily/earthdaily-agriculture"
REMOTE_TEMPLATE_DIRECTORY = "project_template"  # subdir inside the repo

# Common places a clone of either repo might live — the public
# `earthdaily-agriculture` or the internal `earthdaily-agriculture-internal`.
# Extend freely; the discovery loop short-circuits on the first hit.
_CLONE_DIRS = ["Github", "Documents/Github", "Projects", "repos", "src"]
#: Internal first — a contributor with both clones wants the internal template,
#: which carries docs/internal/ on top of the public set.
_CLONE_NAMES = ["earthdaily-agriculture-internal", "earthdaily-agriculture"]

COMMON_LOCATIONS = [Path.home().joinpath(*d.split("/")) / name for name in _CLONE_NAMES for d in _CLONE_DIRS] + [
    Path("C:/Users") / os.environ.get("USERNAME", "") / "Documents" / "Github" / name for name in _CLONE_NAMES
]


def _looks_like_edagro_clone(candidate: Path) -> bool:
    return (candidate / "project_template" / "cookiecutter.json").is_file()


def _find_template(
    explicit: Path | None,
    *,
    interactive: bool,
) -> tuple[str, Path | None]:
    """Resolve the cookiecutter template source.

    Returns ``(template_arg, local_root)`` where ``template_arg`` is what
    we'll hand to ``cookiecutter()`` (either a local ``project_template``
    path *or* the remote URL), and ``local_root`` is the earthdaily-agriculture-internal
    root on disk when we're using a local clone (else ``None``).
    """

    # 1. Explicit --template flag
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not _looks_like_edagro_clone(candidate):
            sys.exit(
                f"error: {candidate} does not look like an earthdaily-agriculture-internal clone (no project_template/cookiecutter.json)."
            )
        return str(candidate / "project_template"), candidate

    # 2. EDAGRO_CLIENT_PATH env var
    env = os.environ.get("EDAGRO_CLIENT_PATH")
    if env:
        candidate = Path(env).expanduser().resolve()
        if _looks_like_edagro_clone(candidate):
            return str(candidate / "project_template"), candidate
        print(
            f"warn: EDAGRO_CLIENT_PATH={env} does not contain project_template/cookiecutter.json — skipping.",
            file=sys.stderr,
        )

    # 3. If this script lives inside an earthdaily-agriculture-internal clone, use that.
    here = Path(__file__).resolve().parent
    if _looks_like_edagro_clone(here):
        return str(here / "project_template"), here

    # 4. Common sibling locations
    for loc in COMMON_LOCATIONS:
        loc = loc.expanduser()
        if _looks_like_edagro_clone(loc):
            print(f"Found earthdaily-agriculture-internal clone at: {loc}")
            return str(loc / "project_template"), loc

    # 5. Remote URL — works once the repo is public.
    print(
        f"\nNo local earthdaily-agriculture-internal clone found. Falling back to the public GitHub URL:\n    {REMOTE_TEMPLATE_URL}"
    )
    print(
        "Note: while the repo is private this URL will fail to clone. "
        "Either set EDAGRO_CLIENT_PATH to a local clone, pass "
        "--template <path>, or wait for the repo to go public.\n"
    )
    return REMOTE_TEMPLATE_URL, None


def _ensure_cookiecutter() -> None:
    try:
        import cookiecutter  # noqa: F401
    except ImportError:
        print("cookiecutter not found — installing into the current Python...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "cookiecutter"],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="Path to the empty repo (default: current working directory).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Path to a local earthdaily-agriculture-internal clone (overrides auto-discovery).",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Human-readable project name (default: target folder name).",
    )
    parser.add_argument(
        "--project-description",
        default=None,
        help="One-line project description (default: prompt interactively).",
    )
    parser.add_argument(
        "--environment",
        default="prod",
        choices=["prod", "preprod"],
    )
    parser.add_argument(
        "--project-type",
        default="extraction",
        choices=["extraction", "backoffice"],
        help=(
            "extraction (default): the analytics workflow scaffold. "
            "backoffice: an EarthDaily-INTERNAL account-provisioning scaffold — "
            "no workflow.yml, no app/, plus the provisioning notebooks. Requires "
            "an internal clone; a public checkout has nothing to copy."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="No prompts at all; accept all cookiecutter defaults.",
    )
    parser.add_argument(
        "--no-internal-docs",
        dest="include_internal_docs",
        action="store_false",
        default=None,
        help=(
            "Do not copy docs/internal/ into the project. Only has an effect when "
            "generating from an internal clone — a public checkout has no internal "
            "docs to copy. Use this for a project you intend to hand to a client."
        ),
    )
    args = parser.parse_args()

    target = args.target.resolve()
    if not target.is_dir():
        print(f"error: target is not a directory: {target}", file=sys.stderr)
        return 1

    template_arg, local_root = _find_template(args.template, interactive=not args.non_interactive)

    # When using a local clone, hand the post-gen hook the matching
    # ``src/`` path so the auto-generated CLAUDE.md / /edagro skill point
    # at the right source. For the remote-URL path we leave it as the
    # cookiecutter.json default (the hook will print a warning if it
    # can't reach the source; the project is still created cleanly).
    if local_root is not None:
        edagro_source = (local_root / "src").resolve()
        if not edagro_source.is_dir():
            print(
                f"error: EarthDaily Agriculture source not found at {edagro_source}",
                file=sys.stderr,
            )
            return 1
        # Forward slashes so the value is safe to embed inside JSON strings
        # (e.g. the rendered notebook). Windows pathlib handles "/" fine.
        edagro_source_str: str | None = edagro_source.as_posix()
    else:
        edagro_source_str = None

    slug = target.name
    project_name = args.project_name or slug
    if args.project_description is not None or args.non_interactive:
        project_description = args.project_description or "EarthDaily Agriculture analytics extraction project"
    else:
        project_description = (
            input("Project description [EarthDaily Agriculture analytics extraction project]: ").strip()
            or "EarthDaily Agriculture analytics extraction project"
        )

    print()
    print(f"  Template : {template_arg}")
    if edagro_source_str:
        print(f"  Source   : {edagro_source_str}")
    print(f"  Target   : {target}")
    print(f"  Slug     : {slug}  (locked to the target folder name)")
    print(f"  Name     : {project_name}")
    print(f"  Env      : {args.environment}")
    print()

    _ensure_cookiecutter()
    from cookiecutter.main import cookiecutter

    extra_context: dict[str, str] = {
        "project_name": project_name,
        "project_slug": slug,
        "project_description": project_description,
        "environment": args.environment,
        "project_type": args.project_type,
    }
    if edagro_source_str is not None:
        extra_context["edagro_source_path"] = edagro_source_str
    # Left as None unless --no-internal-docs was passed, so the template default wins.
    if args.include_internal_docs is False:
        extra_context["include_internal_docs"] = "no"

    cookiecutter_kwargs: dict[str, object] = {
        "no_input": True,
        "output_dir": str(target.parent),
        "overwrite_if_exists": True,
        "extra_context": extra_context,
    }
    # When using the remote URL, tell cookiecutter to pick the project_template
    # subdirectory inside the cloned repo.
    if template_arg.startswith(("http://", "https://", "git@", "git+")):
        cookiecutter_kwargs["directory"] = REMOTE_TEMPLATE_DIRECTORY

    cookiecutter(template_arg, **cookiecutter_kwargs)

    print()
    print(f"Initialized {target}")
    print("Next steps:")
    print(f"  cd {target}")
    print("  copy .env.template .env   (then fill in PROD_API_* / AWS_*)")
    print("  pip install -r requirements.txt")
    print('  git add -A && git commit -m "Initialize from EarthDaily Agriculture project template"')
    print()
    print("You can now `del init_project.py` if you don't want the bootstrap")
    print("tracked, or commit it as a recipe for the next bootstrap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
