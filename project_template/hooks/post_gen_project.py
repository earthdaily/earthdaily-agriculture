"""Cookiecutter post-generation hook.

Runs once, immediately after the project directory is rendered. Five jobs:

  0. Shape the project to its `project_type`. `extraction` is the default and
     keeps everything; `backoffice` prunes the extraction-only scaffolding and
     copies the provisioning notebooks + its own README from the source repo.
     Same reasoning as (1) below: nothing back-office-shaped is vendored into
     the template, because the template ships publicly. See `_copy_backoffice`.

  1. Copy the narrative docs into the project's `docs/` from the
     `earthdaily-agriculture` repo the template was generated from. The docs are
     **not** vendored into the template: whichever repo you generate from decides
     what the project gets, which is what keeps internal-only material out of
     public scaffolds without a single conditional. See `_copy_docs`.

  2. Generate the project's `CLAUDE.md` + `.claude/skills/earthdaily-agriculture/{SKILL.md,
     extractors.md}` from the live `earthdaily.agriculture` source — keeps the
     **project-local** `/earthdaily-agriculture` skill in sync with the actual extractor
     catalogue without committing static copies into the template.

  3. Install the **personal/global** `/earthdaily-agriculture` skill at
     `~/.claude/skills/earthdaily-agriculture/` (SKILL.md + extractors.md +
     runbooks/) if it doesn't already exist. Gives every developer the
     skill on first project init without a separate setup step. Existing
     installs are left untouched so we don't stomp local edits.

  4. Print clear next-step hints when any step is skipped.

Failure is non-fatal: if the EarthDaily Agriculture source path is missing or introspection
fails (e.g. Python running cookiecutter does not have `earthdaily.agriculture`'s deps
installed), we print a warning and let the project be created without the
generated context. The project still works; the user can re-run the generator
later.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

EDAGRO_SOURCE = Path(r"{{ cookiecutter.edagro_source_path }}")
# The generator lives inside the package since 2.6.0. It is invoked as a
# module so we don't need to know the filesystem layout of the source tree.
GENERATOR_MODULE = "earthdaily.agriculture.ai_enablement.generate_ai_context"
# Used only for the existence-check below — confirms the package is present
# at EDAGRO_SOURCE before we try to invoke it as a module. The path mirrors
# GENERATOR_MODULE: earthdaily/agriculture/ai_enablement/generate_ai_context.py.
SCRIPT = EDAGRO_SOURCE / "earthdaily" / "agriculture" / "ai_enablement" / "generate_ai_context.py"

PROJECT_NAME = "{{ cookiecutter.project_name }}"
PROJECT_DESCRIPTION = "{{ cookiecutter.project_description }}"
PROJECT_TYPE = "{{ cookiecutter.project_type }}".strip().lower()
ENVIRONMENT = "{{ cookiecutter.environment }}"
WHEEL_VERSION = "{{ cookiecutter.edagro_wheel_version }}"
IS_BACKOFFICE = PROJECT_TYPE == "backoffice"
INCLUDE_INTERNAL_DOCS = "{{ cookiecutter.include_internal_docs }}".strip().lower() in {
    "y",
    "yes",
    "true",
    "1",
}

# Docs are copied from the source repo at generation time rather than vendored
# into the template. The two repos lay them out differently, so probe for both:
#
#   internal (earthdaily-agriculture-internal)  docs/site/agriculture/  + docs/internal/
#   public   (earthdaily-agriculture)           docs/                   (the release
#                                                                        allowlist
#                                                                        remaps the
#                                                                        former to the
#                                                                        latter)
#
# A public clone therefore has no internal docs to find, so the internal/public
# boundary is enforced by what the source repo contains — not by a flag anyone
# has to remember to set.
PUBLIC_DOCS_CANDIDATES = (
    Path("docs") / "site" / "agriculture",  # internal layout
    Path("docs"),  # public layout
)
INTERNAL_DOCS_SUBDIR = Path("docs") / "internal"

# ── Back-office variant ──────────────────────────────────────────────────────
#
# The back-office files are NOT vendored into the template, for the same reason
# the docs aren't: the slug folder is a WHOLESALE include in the public
# allowlist, so anything placed under it ships to the public repo. (The exact
# pattern lives in release/public_allowlist.yml and is deliberately not quoted
# here: this file is itself Jinja-rendered, so the braces would be substituted.)
#
# Copying from the source repo at generation time means a public checkout has
# nothing to copy — a *physical* second condition alongside the build-time scrub
# of the `project_type` choice, rather than trusting one glob to hold the line.
#
# It also keeps the notebooks canonical in `backoffice/`: one copy, no drift.
BACKOFFICE_NOTEBOOKS_SUBDIR = Path("backoffice")
BACKOFFICE_SCAFFOLD_SUBDIR = Path("project_template_backoffice")

# Rendered by the shared template but meaningless for a provisioning project:
# the chain is not a WorkflowManager workflow, so the workflow config, the
# pipeline app, its CI job, the container files and the extraction notebook all
# go. `app/refresh_ai_context.py` stays — regenerating AI context is useful in
# both variants.
BACKOFFICE_PRUNE = (
    Path("configuration") / "workflow.yml",
    Path("app") / "run_pipeline.py",
    Path("app") / "transforms.py",
    Path("partials"),
    Path("cache"),
    Path("Dockerfile"),
    Path(".dockerignore"),
    Path(".github") / "workflows" / "automated_extraction.yml",
    Path("EDAgriculture_{{ cookiecutter.project_slug }}.ipynb"),
)

# Written by this hook, never copied over. `docs/study/` is the project's own
# exploratory space and ships as an empty scaffold.
DOCS_PRESERVE = {"study"}

# Personal skill install target. Overridable via the EDAGRO_PERSONAL_SKILL_DIR
# env var (useful for CI / sandboxes that don't want `~` writes).
PERSONAL_SKILL_DIR = Path(
    os.environ.get(
        "EDAGRO_PERSONAL_SKILL_DIR",
        str(Path.home() / ".claude" / "skills" / "earthdaily-agriculture"),
    )
).expanduser()


def _copy_tree_into(src: Path, dest: Path, *, skip: set[str] | None = None) -> int:
    """Copy the contents of `src` into `dest`, returning the number of files written.

    Entries named in `skip` are left alone, which is how `docs/study/` survives
    a copy that would otherwise merge over it.
    """
    skip = skip or set()
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in sorted(src.iterdir()):
        if entry.name in skip or entry.name.startswith("."):
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)
            written += sum(1 for p in (dest / entry.name).rglob("*") if p.is_file())
        else:
            shutil.copy2(entry, dest / entry.name)
            written += 1
    return written


def _copy_docs(target: Path) -> None:
    """Populate `<project>/docs/` from the repo this template was generated from.

    Non-fatal by design, like every other step in this hook: when the source
    repo cannot be located (the remote-URL flow never overrides
    `edagro_source_path`), the project is still created — just without the
    narrative docs, and the warning says how to get them.
    """
    docs_target = target / "docs"

    # An unset source path (the remote-URL flow never overrides it) would make
    # Path("") resolve to "." — which during this hook is the generated project
    # itself, so we would copy docs/ onto docs/. Bail out explicitly instead.
    if not str(EDAGRO_SOURCE).strip() or not EDAGRO_SOURCE.is_dir():
        print(
            "[earthdaily-agriculture] WARNING: no earthdaily-agriculture source path "
            f"({EDAGRO_SOURCE or '<unset>'}), so the narrative docs were not copied. "
            "Re-run init_project.py from a local clone, or pass --template <path>, "
            "to get them."
        )
        return

    repo_root = EDAGRO_SOURCE.resolve().parent

    public_src = next(
        (repo_root / c for c in PUBLIC_DOCS_CANDIDATES if (repo_root / c).is_dir()),
        None,
    )
    if public_src is None:
        print(
            f"[earthdaily-agriculture] WARNING: no docs directory found under {repo_root}. "
            "Skipping doc copy — the project is complete otherwise. Copy the "
            "guides in yourself from an earthdaily-agriculture checkout if you want them "
            "alongside the code."
        )
        return

    # Guard the flat public layout: never sweep `site/` or `internal/` into the
    # public set if a future layout puts them side by side.
    count = _copy_tree_into(public_src, docs_target, skip=DOCS_PRESERVE | {"site", "internal"})
    print(f"[earthdaily-agriculture] Copied {count} doc file(s) from {public_src} -> docs/")

    internal_src = repo_root / INTERNAL_DOCS_SUBDIR
    if not internal_src.is_dir():
        # The normal case for a public checkout — there is nothing to exclude,
        # so there is nothing to say.
        return
    if not INCLUDE_INTERNAL_DOCS:
        print(
            "[earthdaily-agriculture] Internal docs found but include_internal_docs=no "
            "— skipping docs/internal/."
        )
        return

    n_internal = _copy_tree_into(internal_src, docs_target / "internal", skip=DOCS_PRESERVE)
    print(
        f"[earthdaily-agriculture] Copied {n_internal} INTERNAL doc file(s) -> docs/internal/ "
        "(EarthDaily-only material — do not publish this project as-is)."
    )


def _remove(path: Path) -> bool:
    """Delete a file or directory. Returns True when something was removed."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def _prune_for_variant(target: Path) -> None:
    """Strip the extraction-only scaffolding from a back-office project.

    The two variants share ~70% of the surface (`.env.template`, `inputs/`,
    `results/`, `logs/`, README shape, pre-commit, the pinned wheel, the
    generated AI context), which is exactly why this is one template with a
    flag rather than two templates that would drift apart.
    """
    if not IS_BACKOFFICE:
        return

    removed = [p.as_posix() for p in BACKOFFICE_PRUNE if _remove(target / p)]

    # Directories that existed only to hold something we just pruned:
    # `configuration/` held workflow.yml, `.github/workflows/` held the
    # extraction cron. Deepest first, so a parent is considered after its child
    # has gone. Anything the user might still want (inputs/, results/, logs/)
    # ships with a .gitkeep and so is never empty.
    for rel in (Path(".github") / "workflows", Path(".github"), Path("configuration")):
        d = target / rel
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed.append(f"{rel.as_posix()}/")

    print(f"[earthdaily-agriculture] back-office variant: pruned {len(removed)} extraction-only path(s): {', '.join(removed)}")


def _copy_backoffice(target: Path) -> None:
    """Copy the provisioning notebooks + variant README into a back-office project.

    Non-fatal like every other step here, but *loud*: a back-office project with
    no notebooks is an empty shell, so the warning has to say what to do next
    rather than let someone discover it later.
    """
    if not IS_BACKOFFICE:
        return

    if not str(EDAGRO_SOURCE).strip() or not EDAGRO_SOURCE.is_dir():
        print(
            "[earthdaily-agriculture] WARNING: no earthdaily-agriculture source path, so the "
            "provisioning notebooks were NOT copied. This back-office project is "
            "an empty shell — re-run init_project.py from a local internal clone."
        )
        return

    repo_root = EDAGRO_SOURCE.resolve().parent

    nb_src = repo_root / BACKOFFICE_NOTEBOOKS_SUBDIR
    if nb_src.is_dir():
        nb_dest = target / "backoffice"
        nb_dest.mkdir(parents=True, exist_ok=True)
        notebooks = sorted(p for p in nb_src.glob("*.ipynb") if not p.name.startswith("."))
        for nb in notebooks:
            shutil.copy2(nb, nb_dest / nb.name)
        print(f"[earthdaily-agriculture] Copied {len(notebooks)} provisioning notebook(s) -> backoffice/")
    else:
        print(
            f"[earthdaily-agriculture] WARNING: no {BACKOFFICE_NOTEBOOKS_SUBDIR}/ under {repo_root} "
            "— this is a public checkout, which cannot scaffold a back-office "
            "project. Generate from the internal repo instead."
        )

    # The shared README is extraction-flavoured (Workflow mode, container
    # deploys); the variant carries its own, kept out of the template tree so
    # it never reaches the public payload.
    readme_src = repo_root / BACKOFFICE_SCAFFOLD_SUBDIR / "README.md"
    if readme_src.is_file():
        rendered = (
            readme_src.read_text(encoding="utf-8")
            .replace("@PROJECT_NAME@", PROJECT_NAME)
            .replace("@PROJECT_DESCRIPTION@", PROJECT_DESCRIPTION)
            .replace("@ENVIRONMENT@", ENVIRONMENT)
            .replace("@WHEEL_VERSION@", WHEEL_VERSION)
        )
        (target / "README.md").write_text(rendered, encoding="utf-8")
        print("[earthdaily-agriculture] Installed the back-office README.")


def _run_project_target(target: Path) -> bool:
    """Generate project-local CLAUDE.md + .claude/skills/earthdaily-agriculture/. Returns True on success."""
    cmd = [
        sys.executable,
        "-m",
        GENERATOR_MODULE,
        "--project-target",
        str(target),
        "--project-name",
        PROJECT_NAME,
        "--environment",
        ENVIRONMENT,
        "--wheel-version",
        WHEEL_VERSION,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(EDAGRO_SOURCE))
        return True
    except subprocess.CalledProcessError as exc:
        print(
            f"[earthdaily-agriculture] WARNING: project-local context generation failed "
            f"(exit {exc.returncode}). The project was still created; re-run "
            "the generator manually once earthdaily.agriculture' dependencies are importable."
        )
        return False


def _personal_skill_already_installed() -> bool:
    """Treat the personal skill as installed when SKILL.md exists."""
    return (PERSONAL_SKILL_DIR / "SKILL.md").is_file()


def _run_personal_skill_target() -> None:
    """Install ~/.claude/skills/earthdaily-agriculture/ when it doesn't already exist."""
    if _personal_skill_already_installed():
        print(
            f"[earthdaily-agriculture] Personal skill already installed at {PERSONAL_SKILL_DIR}; "
            "leaving it untouched. To refresh, run: "
            f"python -m {GENERATOR_MODULE} --personal-skill-target {PERSONAL_SKILL_DIR}"
        )
        return

    cmd = [
        sys.executable,
        "-m",
        GENERATOR_MODULE,
        "--personal-skill-target",
        str(PERSONAL_SKILL_DIR),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(EDAGRO_SOURCE))
        print(f"[earthdaily-agriculture] Personal skill installed at {PERSONAL_SKILL_DIR}")
    except subprocess.CalledProcessError as exc:
        print(
            f"[earthdaily-agriculture] WARNING: personal-skill install failed "
            f"(exit {exc.returncode}). The project was still created; install "
            "the personal skill later with: "
            f"python -m {GENERATOR_MODULE} --personal-skill-target {PERSONAL_SKILL_DIR}"
        )


def main() -> None:
    target = Path.cwd().resolve()

    # Independent of the generator below: a project should still get its guides
    # when `earthdaily.agriculture`'s deps aren't importable, and vice versa.
    try:
        _copy_docs(target)
    except OSError as exc:
        print(f"[earthdaily-agriculture] WARNING: doc copy failed ({exc}). Project still created.")

    # Variant shaping, before context generation so the generated CLAUDE.md
    # describes the project that actually exists on disk.
    try:
        _prune_for_variant(target)
        _copy_backoffice(target)
    except OSError as exc:
        print(f"[earthdaily-agriculture] WARNING: variant setup failed ({exc}). Project still created.")

    if not SCRIPT.exists():
        print(
            f"[earthdaily-agriculture] WARNING: generator not found at {SCRIPT}. "
            "Skipping CLAUDE.md / skill generation. "
            "Re-run later with: "
            f"python -m {GENERATOR_MODULE} --project-target {target} "
            f"--project-name \"{PROJECT_NAME}\" --environment {ENVIRONMENT} "
            f"--wheel-version {WHEEL_VERSION}"
        )
        return

    try:
        project_ok = _run_project_target(target)
        # Continue to the personal-skill step even if the project-local one
        # failed — the two are independent.
        _run_personal_skill_target()

        if project_ok:
            print(f"[earthdaily-agriculture] Project context generated in {target}")
    except FileNotFoundError:
        print(
            "[earthdaily-agriculture] WARNING: could not locate the Python interpreter; "
            "skipping context generation."
        )


if __name__ == "__main__":
    main()
