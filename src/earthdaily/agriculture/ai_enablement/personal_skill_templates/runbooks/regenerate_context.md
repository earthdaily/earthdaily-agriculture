---
title: Runbook — Regenerate AI context after repo changes
description: how / when to run `cd src && python -m earthdaily.agriculture.ai_enablement.generate_ai_context` (repo mode, project-target mode, personal-skill mode) and verify a clean diff.
visibility: public
---

# Runbook — Regenerate AI context after repo changes

`generate_ai_context.py` introspects every extractor class registered in `EXTRACTOR_REGISTRY` and rewrites:

- `CLAUDE.md` (project root) — small framing + auto-generated extractor reference.
- `agents.md` (project root) — per-extractor agent cards (params, output columns, methods).
- `src/scripts/gap_report.txt` — extractors with missing or incomplete docstring sections.

It also has two **target** modes for skill generation:

- `--project-target <path>` — writes the project-local `/earthdaily-agriculture` skill into a generated client project. Used by the cookiecutter post-gen hook.
- `--personal-skill-target <path>` — writes the personal/global `/earthdaily-agriculture` skill at `~/.claude/skills/earthdaily-agriculture/`.

---

## When to run

| Change | Re-run |
|---|---|
| Added a new extractor (`__init__`, `setup_*`, `get_*`, `format_*`, `process_*_bulk`) | Yes — repo mode. |
| Edited an extractor's docstring (the `"""..."""` block) | Yes — repo mode. The docstring drives CLAUDE.md / agents.md content. |
| Added or renamed an entry in `EXTRACTOR_REGISTRY` | Yes — repo mode. |
| Added a hard rule that affects every extractor (e.g. new column-mapping convention) | Yes — repo mode **+** personal-skill mode. |
| Bumped the wheel version | No — the version is read from `earthdaily.agriculture.__version__`, not stamped. |
| Edited a notebook or a doc under `docs/` | No — those aren't introspected. |

---

## Repo mode (most common)

```bash
cd src
python -m earthdaily.agriculture.ai_enablement.generate_ai_context
```

Outputs:

```
Generated:
  CLAUDE.md   -> ../CLAUDE.md
  agents.md   -> ../agents.md
  gap_report  -> scripts/gap_report.txt

Extractors: <N> (0 errors)
Gap summary: {...}
```

Verify and commit:

```bash
git diff CLAUDE.md agents.md
```

The diff should be limited to what you actually changed. If it sweeps unrelated extractors, somebody pushed a code change without re-running the generator first — investigate before committing.

**`gap_report.txt` is a build artifact** — checked into the repo but only as a snapshot for human review. A clean run has zero "Critical" or "Major" gaps. Fix the underlying docstrings before merging.

---

## Personal-skill mode

```bash
cd src
python -m earthdaily.agriculture.ai_enablement.generate_ai_context --personal-skill-target ~/.claude/skills/earthdaily-agriculture
```

Outputs into `~/.claude/skills/earthdaily-agriculture/`:

```
SKILL.md
extractors.md
runbooks/
  add_service_extractor.md
  add_processor_extractor.md
  setup_client_notebook.md
  debug_api_response.md
  regenerate_context.md
```

### Two flags that matter here

| Flag | What it does |
|---|---|
| `--include-internal` | Also ship runbooks marked `visibility: internal`. **Off by default.** |
| `--check` | Report whether the target's `runbooks/` is in sync, exit 1 if not, **write nothing**. For CI. |

**Runbooks are visibility-gated.** Each carries a `visibility:` key in its
frontmatter, and anything not *explicitly* `visibility: public` is treated as
internal — a missing or misspelled marker fails closed, so a new runbook stays
private until someone says otherwise. A default run prints what it left out:

```
  (skipped, internal) runbooks/<name>.md
```

The decision rides on the artifact rather than on a request-time argument: build
an internal skill for internal use, a public one otherwise. Same contract the MCP
repo's `build_usecases.py` uses for use-case pages.

**The runbook list inside `SKILL.md` is generated from what actually shipped**, so
there is no list to edit when you add one — and a public build can never point at
a runbook it does not carry.

Run this **once when you first install the skill**, and again whenever:

- A new extractor category is added (changes the SKILL.md decision table).
- A hard rule changes (e.g. a new "never do X on a Series" pattern).
- The lifecycle methods change (e.g. a sixth canonical method joins the five).
- A runbook is added, removed, or has its `visibility:` changed. Removals and
  retractions propagate — un-shipping one is deleting it (or marking it internal)
  and re-running.

You don't need to re-run it just because an individual extractor's docstring got tweaked — `extractors.md` is the only file affected, and the personal skill prefers pointing at the repo's `agents.md` for fine-grained detail anyway.

---

## Project-target mode (cookiecutter hook only)

```bash
cd src
python -m earthdaily.agriculture.ai_enablement.generate_ai_context \
    --project-target /path/to/new_client_project \
    --project-name "Acme Inc" \
    --environment prod \
    --wheel-version 2.4.0
```

The cookiecutter post-gen hook calls this automatically after `init_project.py` finishes — you should never need to invoke it manually unless the hook failed (e.g. `earthdaily.agriculture` wasn't importable when cookiecutter ran).

---

## Verifying a clean diff

```bash
# After running the generator
git status
git diff --stat CLAUDE.md agents.md src/scripts/gap_report.txt

# Three expected file changes; anything more is a sign you missed a step.
git add CLAUDE.md agents.md src/scripts/gap_report.txt
git commit -m "regenerate AI context after <change-summary>"
```

If `CLAUDE.md` keeps drifting on every regen even when nothing changed upstream, the introspection is non-deterministic somewhere — check for `datetime.now()` calls in the generator's output (timestamps should be in the gap report only, not in the user-facing context files).

---

## Failure modes

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: earthdaily.agriculture` | Run from `src/`, not from the repo root. The script `sys.path.insert`s `src/` but only if invoked from there. |
| Extractor missing from the output table | Not registered in `EXTRACTOR_REGISTRY` (top of `generate_ai_context.py`). |
| Gap report flags a "Missing: Output columns" you know is there | The docstring isn't in the expected format — check `EXPECTED_SECTIONS` in the script. |
| `--personal-skill-target` writes to the wrong place | `~` doesn't expand on Windows cmd — use `$env:USERPROFILE` (PowerShell) or the absolute path. |

---

## See also

- `add_service_extractor.md` / `add_processor_extractor.md` — when you've finished those, this is the final step.
- `docs/site/agriculture/12 - AI enablement.md` — design rationale for the three AI-context surfaces (CLAUDE.md, agents.md, the skill).
