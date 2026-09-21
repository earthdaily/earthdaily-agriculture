"""
Generate AI context files for EarthDaily Agriculture Extractors.

Introspects all extractor classes to produce:
  1. CLAUDE.md   — project instructions with auto-generated extractor reference
  2. agents.md   — per-extractor agent cards for AI coding assistants
  3. gap_report.txt — human-readable report highlighting missing or incomplete
     documentation per extractor (build artifact)

Usage:
    python -m earthdaily.agriculture.ai_enablement.generate_ai_context
    python -m earthdaily.agriculture.ai_enablement.generate_ai_context -v
    python -m earthdaily.agriculture.ai_enablement.generate_ai_context --project-target <path>
    python -m earthdaily.agriculture.ai_enablement.generate_ai_context --personal-skill-target ~/.claude/skills/earthdaily-agriculture
"""

import argparse
import importlib
import inspect
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Registry of all extractors to introspect
# ---------------------------------------------------------------------------
EXTRACTOR_REGISTRY = [
    # (module_path, class_name, analytic_category)
    ("earthdaily.agriculture.extractors.coverage_function", "CoverageExtractor", "Foundational"),
    ("earthdaily.agriculture.extractors.difference_functions", "DifferenceExtractor", "Benchmark & Warning"),
    ("earthdaily.agriculture.extractors.gdd_functions", "GDDExtractor", "Crop Development & Stressors"),
    ("earthdaily.agriculture.extractors.gdd_offset_functions", "GDDOffsetExtractor", "Crop Development & Stressors"),
    ("earthdaily.agriculture.extractors.zoning_functions", "ZoningExtractor", "Foundational"),
    (
        "earthdaily.agriculture.processors.processor_disease_risk_functions",
        "DiseaseExtractor",
        "Crop Development & Stressors",
    ),
    ("earthdaily.agriculture.extractors.FLM_functions", "FLMExtractor", "Foundational"),
    ("earthdaily.agriculture.extractors.VTS_functions", "VegationTsExtractor", "Foundational"),
    ("earthdaily.agriculture.extractors.VTS_functions", "MRTSExtractor", "Foundational"),
    ("earthdaily.agriculture.extractors.weather_functions", "WeatherExtractor", "Foundational"),
    (
        "earthdaily.agriculture.processors.processor_emergence_functions",
        "EmergenceExtractor",
        "Crop Development & Stressors",
    ),
    (
        "earthdaily.agriculture.processors.processor_harvest_functions",
        "HarvestExtractor",
        "Crop Development & Stressors",
    ),
    (
        "earthdaily.agriculture.processors.processor_greenness_functions",
        "GreennessExtractor",
        "Crop Development & Stressors",
    ),
    ("earthdaily.agriculture.processors.processor_baresoil_function", "BaresoilExtractor", "Sustainability"),
    ("earthdaily.agriculture.extractors.cropid_functions", "cropidExtractor", "Foundational"),
    (
        "earthdaily.agriculture.processors.processor_plantedarea_functions",
        "PlantedExtractor",
        "Crop Development & Stressors",
    ),
    (
        "earthdaily.agriculture.processors.processor_inseason_monitoring_functions",
        "InSeasonMonitoringExtractor",
        "Benchmark & Warning",
    ),
    ("earthdaily.agriculture.processors.processor_score_functions", "HistoricalScoreExtractor", "Risk Management"),
    ("earthdaily.agriculture.processors.processor_score_functions", "InseasonScoreExtractor", "Risk Management"),
    ("earthdaily.agriculture.extractors.regional_ts_extractor", "RegionalExtractor", "Regional"),
    ("earthdaily.agriculture.processors.processor_zarc_functions", "ZARCExtractor", "Risk Management"),
    (
        "earthdaily.agriculture.processors.processor_change_index_functions",
        "ChangeIndexExtractor",
        "Benchmark & Warning",
    ),
    (
        "earthdaily.agriculture.extractors.location_based_border_functions",
        "LocationBasedBorderExtractor",
        "Foundational",
    ),
    ("earthdaily.agriculture.services.entity_management", "EntityManager", "Entity Management"),
    ("earthdaily.agriculture.services.user_management", "UserManager", "Entity Management"),
    # Admin plane. All five subclass BaseExtractor, but only the two above were
    # ever registered, so the catalogue answered "is there a provisioning API?"
    # with silence — and a model reading it concluded, reasonably and wrongly,
    # that the client has no admin surface at all. Provisioning is precisely
    # where that hurts: the narrative docs are data-plane only, so the code is
    # the only place the answer lives.
    ("earthdaily.agriculture.services.history_manager", "HistoryManager", "Administration"),
]

# Expected docstring sections (used for gap detection)
EXPECTED_SECTIONS = [
    "Documentation:",
    "Notebook:",
    "Args",
    "Entity fields",
    "Output columns:",
]

# Management classes (different expectations)
MANAGEMENT_CLASSES = {
    "EntityManager",
    "UserManager",
    "HistoryManager",
}

#: A docstring "Entity fields" entry that is an actual field name, as opposed to
#: the prose that sits alongside it in several docstrings. Canonical entity keys
#: are snake_case identifiers, so anything else is a sentence and is kept as a
#: note rather than split on commas into fields that do not exist.
ENTITY_FIELD_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

# Common setup params to hide from agent cards (noise reduction)
COMMON_SETUP_PARAMS = {
    "column_mapping",
    "output_mapping",
    "exclude_columns",
    "output_columns",
    "partial_frequency",
    "publish_af",
    "skip_existing",
    "output_path",
}


# ---------------------------------------------------------------------------
# Docstring parser
# ---------------------------------------------------------------------------


def parse_docstring(docstring: str) -> dict:
    """Parse a structured class docstring into a metadata dict."""
    if not docstring:
        return {}

    lines = docstring.strip().splitlines()
    result: dict[str, Any] = {
        "description": "",
        "documentation_url": None,
        "notebook": None,
        "setup_args": [],
        "entity_fields": [],
        "entity_field_qualifiers": {},
        "entity_field_notes": [],
        "output_columns": [],
        "raw_sections_found": [],
    }

    # First non-empty lines until a known section = description
    desc_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if any(line.startswith(s) for s in EXPECTED_SECTIONS) or line.startswith("Supported operations"):
            break
        if line:
            desc_lines.append(line)
        i += 1
    result["description"] = " ".join(desc_lines)

    full_text = docstring

    # Documentation URL
    match = re.search(r"Documentation:\s*(https?://\S+)", full_text)
    if match:
        result["documentation_url"] = match.group(1)
        result["raw_sections_found"].append("Documentation:")

    # Notebook
    match = re.search(r"Notebook:\s*(\S+\.ipynb)", full_text)
    if match:
        result["notebook"] = match.group(1)
        result["raw_sections_found"].append("Notebook:")

    # Args section — capture the full indented block (every indented, non-blank
    # line) until the first blank line / dedented section header. Using
    # ``[ \t]+\S`` instead of ``\s+\w+`` so multi-line arg descriptions whose
    # continuation lines start with a non-word char (e.g. "'year'", "(required")
    # don't truncate the block and hide args declared after them.
    args_match = re.search(r"Args.*?:\s*\n((?:[ \t]+\S.*\n)+)", full_text)
    if args_match:
        result["raw_sections_found"].append("Args")
        for arg_line in args_match.group(1).strip().splitlines():
            arg_line = arg_line.strip()
            param_match = re.match(
                r"(\w+)\s*\(([^)]+)\):\s*(.+?)(?:\.\s*Default:\s*(.+))?$",
                arg_line,
            )
            if param_match:
                result["setup_args"].append(
                    {
                        "name": param_match.group(1),
                        "type": param_match.group(2).strip(),
                        "description": param_match.group(3).strip().rstrip("."),
                        "default": param_match.group(4).strip() if param_match.group(4) else None,
                    }
                )
            else:
                simple_match = re.match(r"(\w+)\s*\(([^)]+)\):\s*(.+)", arg_line)
                if simple_match:
                    result["setup_args"].append(
                        {
                            "name": simple_match.group(1),
                            "type": simple_match.group(2).strip(),
                            "description": simple_match.group(3).strip().rstrip("."),
                            "default": None,
                        }
                    )

    # Entity fields
    entity_match = re.search(r"Entity fields.*?:\s*\n\s*(.+?)(?:\n\s*\n|\n\s*Output|\Z)", full_text, re.DOTALL)
    if entity_match:
        result["raw_sections_found"].append("Entity fields")
        fields_text = entity_match.group(1).strip()
        # Qualifiers used to be discarded with `re.sub(r"\(.*?\)", "", line)` and
        # the card headed the survivors "Required Entity Fields". So an OPTIONAL
        # field was published as required — `historical_seasons` on Emergence
        # read as mandatory, contradicting the column-mapping reference, and the
        # only way to settle it was to open the source. Keep the qualifier.
        #
        # A qualifier binds to the group before its `;`, matching the docstring
        # convention `id, geometry (required); historical_seasons (optional)`.
        # Join continuation lines first. `PlantedExtractor` wraps a parenthetical
        # across two lines, and parsing line-by-line left the `(` unclosed, so the
        # qualifier stripper did not fire and `emergence_date` was lost entirely.
        logical_lines: list[str] = []
        for raw in fields_text.splitlines():
            raw = raw.strip().lstrip("- ")
            if not raw:
                continue
            if logical_lines and logical_lines[-1].count("(") > logical_lines[-1].count(")"):
                logical_lines[-1] += " " + raw
            else:
                logical_lines.append(raw)

        for line in logical_lines:
            for group in line.split(";"):
                group = group.strip()
                if not group:
                    continue
                qualifier_match = re.search(r"\(([^)]*)\)", group)
                qualifier = qualifier_match.group(1).strip() if qualifier_match else ""
                for token in re.split(r",", re.sub(r"\(.*?\)", "", group)):
                    token = token.strip().rstrip(".").strip()
                    if not token or token in ("via column_mapping", "required", "optional"):
                        continue
                    # `name - description` (StandingCrop, LocationBasedBorder) and
                    # bare `name` both reduce to the head token.
                    parts = re.split(r"\s+[-—–→:]\s+", token, maxsplit=1)
                    head, tail = parts[0].strip(), (parts[1] if len(parts) > 1 else "")
                    if not ENTITY_FIELD_RE.match(head):
                        # Prose, not a field — e.g. "sowing_date is NOT read: ...".
                        # It used to be split on commas and published as fields.
                        result["entity_field_notes"].append(token)
                        continue
                    result["entity_fields"].append(head)
                    note = " ".join(part for part in (qualifier, tail.strip()) if part)
                    if note:
                        result["entity_field_qualifiers"][head] = note

    # Output columns
    output_match = re.search(r"Output columns:\s*\n\s*(.+?)(?:\n\s*\n|\n\s*\"\"\"|\Z)", full_text, re.DOTALL)
    if output_match:
        result["raw_sections_found"].append("Output columns:")
        cols_text = output_match.group(1).strip()
        for line in cols_text.splitlines():
            line = line.strip().lstrip("- ")
            if ":" in line and not line.startswith("http"):
                line = line.split(":", 1)[1]
            for col in re.split(r"[,;]", line):
                col = col.strip().strip("+").strip()
                if col and not col.startswith("(") and not col.startswith("historical") and len(col) < 60:
                    result["output_columns"].append(col)

    # Supported operations (for management classes)
    ops_match = re.search(r"Supported operations:\s*\n((?:\s+-.+\n)*)", full_text)
    if ops_match:
        result["raw_sections_found"].append("Supported operations")
        result["supported_operations"] = [line.strip().lstrip("- ") for line in ops_match.group(1).strip().splitlines()]

    return result


# ---------------------------------------------------------------------------
# Class introspection
# ---------------------------------------------------------------------------


def get_setup_method(cls) -> dict | None:
    """Find the setup_*_parameters method and extract its signature."""
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("setup_") and name.endswith("_parameters"):
            sig = inspect.signature(method)
            params = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                params.append(
                    {
                        "name": pname,
                        "default": repr(param.default) if param.default is not inspect.Parameter.empty else None,
                        "kind": str(param.kind),
                    }
                )
            return {"method_name": name, "parameters": params}
    return None


def get_format_methods(cls) -> list:
    """Find all format_*_json methods."""
    return sorted(
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name.startswith("format_") and "json" in name
    )


def get_process_methods(cls) -> list:
    """Find all process_* methods (single and bulk)."""
    return sorted(
        name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction) if name.startswith("process_")
    )


def get_api_methods(cls) -> list:
    """Find all get_*_api methods."""
    return sorted(
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if "api" in name.lower() and name.startswith("get_")
    )


def introspect_class(module_path: str, class_name: str, category: str) -> dict:
    """Introspect a single extractor class."""
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as e:
        return {
            "class_name": class_name,
            "module": module_path,
            "category": category,
            "error": str(e),
        }

    try:
        source_file = inspect.getfile(cls)
        source_file = os.path.relpath(source_file)
    except Exception:
        source_file = module_path.replace(".", "/") + ".py"

    docstring = inspect.getdoc(cls) or ""
    parsed = parse_docstring(docstring)
    bases = [b.__name__ for b in cls.__bases__ if b.__name__ != "object"]

    entry = {
        "class_name": class_name,
        "module": module_path,
        "source_file": source_file,
        "category": category,
        "inherits": bases,
        "description": parsed.get("description", ""),
        "documentation_url": parsed.get("documentation_url"),
        "notebook": parsed.get("notebook"),
        "setup_method": get_setup_method(cls),
        "setup_args_from_docstring": parsed.get("setup_args", []),
        "entity_fields": parsed.get("entity_fields", []),
        "entity_field_qualifiers": parsed.get("entity_field_qualifiers", {}),
        "entity_field_notes": parsed.get("entity_field_notes", []),
        "output_columns": parsed.get("output_columns", []),
        "format_methods": get_format_methods(cls),
        "process_methods": get_process_methods(cls),
        "api_methods": get_api_methods(cls),
    }

    if "supported_operations" in parsed:
        entry["supported_operations"] = parsed["supported_operations"]

    entry["_sections_found"] = parsed.get("raw_sections_found", [])
    return entry


# ---------------------------------------------------------------------------
# Gap analysis (unchanged logic, produces gap_report.txt)
# ---------------------------------------------------------------------------


def analyze_gaps(entries: list) -> list:
    """Analyze documentation gaps for each extractor."""
    gaps = []

    for entry in entries:
        if "error" in entry:
            gaps.append(
                {
                    "class_name": entry["class_name"],
                    "severity": "ERROR",
                    "issues": [f"Failed to import: {entry['error']}"],
                }
            )
            continue

        issues = []
        is_management = entry["class_name"] in MANAGEMENT_CLASSES

        if not entry.get("description"):
            issues.append(("CRITICAL", "Missing class description"))
        elif len(entry["description"]) < 20:
            issues.append(("WARNING", f"Description too short ({len(entry['description'])} chars)"))

        if not entry.get("documentation_url"):
            issues.append(("CRITICAL", "Missing documentation URL"))
        elif entry["documentation_url"].endswith("/Api_reference/"):
            issues.append(("INFO", "Using generic Api_reference URL (no dedicated analytic page found)"))

        if not entry.get("notebook"):
            issues.append(("WARNING", "Missing notebook reference"))

        if not is_management and entry["class_name"] != "ChangeIndexExtractor":
            if not entry.get("setup_method"):
                issues.append(("CRITICAL", "No setup_*_parameters() method found"))

            docstring_args = {a["name"] for a in entry.get("setup_args_from_docstring", [])}
            if entry.get("setup_method"):
                sig_args = {
                    p["name"] for p in entry["setup_method"]["parameters"] if p["name"] not in COMMON_SETUP_PARAMS
                }
                undocumented = sig_args - docstring_args - COMMON_SETUP_PARAMS
                if undocumented:
                    issues.append(("INFO", f"Undocumented setup args: {sorted(undocumented)}"))

        if not entry.get("entity_fields") and not is_management:
            issues.append(("WARNING", "Missing entity fields documentation"))
        if not entry.get("output_columns") and not is_management:
            issues.append(("WARNING", "Missing output columns documentation"))
        if not entry.get("format_methods") and not is_management:
            issues.append(("INFO", "No format_*_json() methods found"))
        if not entry.get("api_methods") and not is_management:
            issues.append(("INFO", "No get_*_api() methods found"))
        if "BaseExtractor" not in entry.get("inherits", []) and entry["class_name"] != "ChangeIndexExtractor":
            issues.append(("WARNING", "Does not inherit from BaseExtractor"))

        gaps.append(
            {
                "class_name": entry["class_name"],
                "module": entry["module"],
                "category": entry.get("category", ""),
                "severity": max(
                    (i[0] for i in issues),
                    default="OK",
                    key=lambda x: {"CRITICAL": 3, "WARNING": 2, "INFO": 1, "OK": 0}.get(x, 0),
                ),
                "issues": [f"[{sev}] {msg}" for sev, msg in issues] if issues else ["OK - All sections documented"],
                "sections_found": entry.get("_sections_found", []),
            }
        )

    return gaps


def format_gap_report(gaps: list, entries: list) -> str:
    """Generate a human-readable gap report."""
    lines = []
    lines.append("=" * 80)
    lines.append("EarthDaily Agriculture Extractor Documentation Gap Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total extractors analyzed: {len(entries)}")
    lines.append("=" * 80)

    severity_counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "OK": 0, "ERROR": 0}
    for g in gaps:
        severity_counts[g["severity"]] = severity_counts.get(g["severity"], 0) + 1

    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    for sev in ("CRITICAL", "WARNING", "INFO", "OK", "ERROR"):
        lines.append(f"  {sev:10s}: {severity_counts.get(sev, 0)}")

    total_checks = len(gaps) * len(EXPECTED_SECTIONS)
    total_found = sum(len(g.get("sections_found", [])) for g in gaps)
    score = (total_found / total_checks * 100) if total_checks > 0 else 0
    lines.append(f"\n  Documentation completeness: {score:.0f}% ({total_found}/{total_checks} sections)")

    categories: dict[str, list] = {}
    for g in gaps:
        categories.setdefault(g.get("category", "Other"), []).append(g)

    for cat in sorted(categories.keys()):
        lines.append(f"\n\n{'=' * 80}")
        lines.append(f"Category: {cat}")
        lines.append("=" * 80)
        for g in sorted(categories[cat], key=lambda x: x["class_name"]):
            marker = {"CRITICAL": "!!", "WARNING": "! ", "INFO": "  ", "OK": "OK", "ERROR": "XX"}.get(
                g["severity"], "??"
            )
            lines.append(f"\n  [{marker}] {g['class_name']}")
            lines.append(f"      Module: {g.get('module', '?')}")
            sections = g.get("sections_found", [])
            missing = [
                s
                for s in EXPECTED_SECTIONS
                if s not in sections
                and not (g["class_name"] in MANAGEMENT_CLASSES and s in ("Args", "Output columns:"))
            ]
            if sections:
                lines.append(f"      Sections found: {', '.join(sections)}")
            if missing:
                lines.append(f"      Sections missing: {', '.join(missing)}")
            for issue in g["issues"]:
                lines.append(f"      {issue}")

    critical_items = [g for g in gaps if g["severity"] == "CRITICAL"]
    if critical_items:
        lines.append(f"\n\n{'=' * 80}")
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("\nCritical items to address first:")
        for g in critical_items:
            for issue in g["issues"]:
                if "[CRITICAL]" in issue:
                    lines.append(f"  - {g['class_name']}: {issue}")

    lines.append(f"\n{'=' * 80}")
    lines.append("End of report")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLAUDE.md generator
# ---------------------------------------------------------------------------

# Static header — hand-maintained project instructions.
# The script appends an auto-generated "Extractor Reference" section.
CLAUDE_MD_HEADER = r"""# CLAUDE.md — EarthDaily Agriculture Analytics Extraction Base

<!-- Auto-generated sections below "Extractor Reference" are produced by
     src/earthdaily/agriculture/ai_enablement/generate_ai_context.py — do not edit them manually. -->

## Project Overview

**Package:** `earthdaily-agriculture`
**Import name:** `earthdaily.agriculture`
**Purpose:** Python utilities for bulk analytics extraction from EarthDaily Agriculture APIs (vegetation indices, weather, crop detection, disease risk, etc.)
**Python:** >= 3.10, recommended 3.12.x (3.13 supported and tested in CI)

## Architecture

### Core Pattern: BaseExtractor Inheritance

All extractors inherit from `BaseExtractor` (`src/earthdaily/agriculture/core/base_extractor.py`) which provides:
- **Token management** — auto-refresh via `ensure_token_valid()`
- **Column mapping** — `DEFAULT_COLUMN_MAPPING` maps internal names (`id`, `geometry`, `crop`, `start_date`, `end_date`, `sowing_date`) to user DataFrame columns
- **Entity validation** — `validate_entity()`, `_is_entity_provided()` for safe Series/dict handling
- **Output formatting** — rename, exclude, select columns via `configure_output()`
- **Export format** — results write as CSV (default) or Parquet via `self.export_format`; see "Output Format" below
- **Report generation** — optional HTML report via `generate_report=True` on any bulk method
- **Logging** — loguru-based, shared with WorkflowManager or standalone

### Extractor Workflow

```
__init__(token, config, workflow_ref) -> setup_<type>_parameters() -> get_<type>_api(entity) -> format_<type>_json() -> process_single_entity() -> process_bulk_parallel()
```

### Decorators

- `@requires_token` — ensures valid token before API calls
- `@require_<type>_params` — ensures parameters are configured (e.g., `@require_coverage_params`, `@require_mrts_params`)

### Key Design Rules

- **Never use `if entity_data`** on pandas Series — always use `self._is_entity_provided(entity_data)` to avoid "truth value of a Series is ambiguous" errors
- **Column mapping fallback** — `get_entity_value(row, key)` looks up mapped column first, falls back to canonical key
- **API response None handling** — use `response.get('key') or []` instead of `response.get('key', [])` when API may return explicit `None` values
- **Minimal platform adherence** — entities can come from EarthDaily Agriculture platform or external datasets; avoid coupling to platform-specific field management
- **Always pass `include_groups=` to `groupby().apply()`** — pandas 3.x silently DROPS the grouping column from the result (2.2 deprecated it; 3.x just does it). `df.groupby("k").apply(fn)` returns a frame with no `k` column, nothing raises, and the loss surfaces far downstream. Prefer `groupby().sample()` / `.agg()` / `.transform()`, which keep the column; use `apply(..., include_groups=False)` where apply is genuinely needed. Enforced by the `check-groupby-apply` pre-commit hook (`SeriesGroupBy` — `groupby("k")["col"].apply(...)` — is exempt, it has no grouping column to drop)
- **`load_workflow()` returns the UNWRAPPED config** — for the new step format it hands back `raw["workflow"]`, so index `cfg["steps"]` / `cfg["settings"]` / `cfg["name"]`, never `cfg["workflow"]["steps"]` (bare `KeyError: 'workflow'`). The legacy `analytics:` format returns the wrapped dict instead, so the shape depends on the input. `manager.workflow_cfg` is the same object; `manager.workflow_cfg_raw` is the wrapped original

### Output Format (CSV / Parquet)

Results export as **CSV by default**; **Parquet** is an additive, opt-in alternative
(smaller files, preserved dtypes, faster reads — meaningful on large multi-year pulls).
The switch lives once on `BaseExtractor.export_format` and flows through the shared
`export_results()` writer, so it covers every extractor's final export with no
per-extractor `setup_*_parameters` change. Error files always stay CSV.

Resolution precedence: **workflow.yml `settings.export_format` > `EDAGRO_EXPORT_FORMAT` env var > default `csv`.**

```yaml
# workflow.yml — applies to every step (declarative, recommended)
workflow:
  settings:
    export_format: parquet
```

```bash
# headless / container runs (parity with EDAGRO_OUTPUT_PREFIX etc.) — a plain OS
# env var, NOT something for the .env credentials file
export EDAGRO_EXPORT_FORMAT=parquet
```

```python
# ad-hoc notebook / direct extractor use
extractor.export_format = "parquet"
```

Parquet needs `pyarrow` (already a dependency, used by the extraction cache). Per-step
formats are also possible via a step-level `export_format:` key in the workflow step.

### Manifest sidecar (opt-in)

Set `export_manifest` to also write a `<prefix>_manifest_<ts>.json` next to a successful
results export — a structured description of the dataset (structure, entity count, date
range, per-column completeness, plus any `manifest_metadata` you attach). Same design as
`export_format`: **one switch on `BaseExtractor`, flowing through the shared
`_finalize_extraction` path — every extractor gets it, no per-extractor change.** It only
fires when an export actually happened (`skip_export` skips it) and is non-fatal (a failure
warns, never breaks the run). Error files are unaffected.

Resolution precedence: **workflow.yml `settings.export_manifest` > `EDAGRO_EXPORT_MANIFEST`
env var > default `False`** (a step-level `export_manifest:` key overrides per step).

```python
# ad-hoc notebook / direct extractor use
extractor.export_manifest = True
extractor.manifest_metadata = {"region_dimension_file": "regions_b129.parquet", "join_key": "amu_id"}
```

### Durable raster output (`output_uri`)

The `postprocess="file"` writers — **FLM, Difference and Zoning** — save PNG / TIFF /
shapefile output through `BaseExtractor.save_map_file()`, which routes via `core/_fs`.
So `output_path` may itself be a remote URI (`s3://`, `gs://`, `az://`), same as every
other writer in the package.

Set `output_uri` to write a **second, durable copy** of every saved file to
`<output_uri>/<filename>` while `output_path` keeps the local working copy:

```yaml
workflow:
  settings:
    output_uri: s3://bucket/prefix/tifs
```

```python
extractor.output_uri = "s3://bucket/prefix/tifs"
```

Resolution precedence: **workflow.yml `settings.output_uri` > `EDAGRO_OUTPUT_URI` env var
> default `None`** (local-only); a step-level `output_uri:` key overrides per step.

Two destinations rather than one overloaded `output_path` because the analysis half
reads every raster back off local disk — a remote-only write turns a seconds-long pass
into thousands of network reads. **`saved_files` keeps recording the local path**, so
manifests stay openable by local readers. Unlike the manifest sidecar, a failed durable
write is **not** swallowed: it raises and the writer reports a normal per-entity error,
because a manifest claiming a raster that only ever existed on the runner is worse than
one that failed loudly.

## Package Structure — `earthdaily.agriculture`

### core/ — Foundation modules

| Module | Key exports |
|---|---|
| `base_extractor.py` | `BaseExtractor` — token, logging, column mapping, output formatting, report generation |
| `identity.py` | `EDAuthenticator` — OAuth2 token exchange, S3 client init |
| `api_utils.py` | `export_results()`, `retry_with_backoff_no_retry_on_400()`, `normalize_with_metadata()`, `filter_entities()`, `filter_timeseries_kpi()` |
| `geometry.py` | `validate_wkt()`, centroid extraction, geodataframe loading |
| `urls.py` | `agro_urls` dict with `preprod`/`prod` endpoints |
| `logging_setup.py` | loguru configuration with rotation/retention |
| `functions_enhanced.py` | `setup_environment()`, `get_seasonfield_list()`, `save_error_reports()`, `print_summary()` |

### reporting/ — Extraction reports

| Module | Key exports |
|---|---|
| `extraction_reporter.py` | `ExtractionReporter` — HTML report generator for bulk extraction runs |

### services/ — Orchestration & management

| Module | Key exports |
|---|---|
| `workflow_manager.py` | `WorkflowManager` — orchestrates auth, entity loading, extractor instantiation |
| `user_management.py` | `UserManager` — Grower/User account management; also a read-only workflow extractor (`setup_user_parameters()` → `process_entity_user_bulk_parallel()`) for bulk user lookups by id |
| `S3.py` | S3 interaction utilities |
| `layer_service_function.py` | Layer service queries |

## Project Structure

```
earthdaily-agriculture-internal/
+-- CLAUDE.md
+-- agents.md                   # Per-extractor agent cards (auto-generated)
+-- .gitignore
+-- pyproject.toml              # Package config, pytest settings
+-- requirements.txt            # Pinned dependencies
+-- README.md
+-- inputs/                     # Input shapefiles & data
+-- results/                    # Final extraction outputs
+-- partials/                   # Intermediate bulk results
+-- logs/                       # Application logs
+-- notebooks/                  # Extractor class showcases & dev notebooks
|   +-- EDAgriculture_*_Dev.ipynb
+-- customers/                  # Client-specific extraction notebooks
|   +-- EDAgriculture_<Client>_*.ipynb
+-- study/                      # Service delivery tools
|   +-- EDAgriculture_*.ipynb
+-- backoffice/                 # Backoffice tools (user management, etc.)
|   +-- *.ipynb
+-- tests/                      # pytest tests
|   +-- conftest.py
|   +-- test_greenness_*.py
+-- build/                      # Build artifacts (rebuild with `python -m build`)
+-- dist/                       # Wheel distributions
+-- src/
    +-- .env                    # Credentials (git-ignored, see template.env)
    +-- template.env            # Environment template
    +-- scripts/                # Standalone utility scripts
    +-- workflow/               # Workflow runner & YAML configs
    +-- earthdaily/agriculture/           # Core package
        +-- __init__.py         # Package version, key re-exports
        +-- notebook_setup.py   # Notebook environment bootstrap (project root, sys.path, .env)
        +-- core/               # Base classes, auth, API utils, geometry, logging, URLs
        +-- extractors/         # Service-level extractors
        +-- processors/         # Processor-based extractors
        +-- reporting/          # HTML report generation
        +-- services/           # Workflow manager, user management, S3
        +-- ai_enablement/      # CLAUDE.md/agents.md + skill generation (generate_ai_context.py)
```

### Workspace Directories

`inputs/`, `results/`, `partials/`, and `logs/` live at **project root** (not inside `src/`).
`setup_environment()` in `core/functions_enhanced.py` resolves the project root by finding `pyproject.toml`
and creates these directories there. All paths stored in `config` are absolute.

### Import Examples

```python
# Workflow (most common entry point)
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

# Individual extractors
from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
from earthdaily.agriculture.processors.processor_greenness_functions import GreennessExtractor

# Core utilities
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.api_utils import export_results

# Reporting
from earthdaily.agriculture.reporting import ExtractionReporter
```

### Notebook Setup

Notebooks live in `notebooks/`, `customers/`, `study/`, and `backoffice/` (all one level below project root).
Every notebook has a **bootstrap cell** (first code cell) that handles path setup:

```python
# Bootstrap: ensure src/ is on sys.path for earthdaily.agriculture imports
import sys
from pathlib import Path

_src = str(Path().resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from earthdaily.agriculture.notebook_setup import init
init()
```

If the package is installed via `pip install -e ".[jupyter]"`, the `sys.path` lines are
redundant but harmless. The `init()` call sets the working directory to project root and loads `.env`.

## Development Commands

```bash
# Install package in dev mode
pip install -e ".[test,jupyter]"

# Run tests
python -m pytest tests/ -v

# Build wheel
python -m build

# Regenerate CLAUDE.md and agents.md from source
cd src && python -m earthdaily.agriculture.ai_enablement.generate_ai_context
```

## Environment Setup

Credentials go in `src/.env` (copy from `src/template.env`):
- `PROD_API_CLIENT_ID`, `PROD_API_CLIENT_SECRET`, `PROD_API_USERNAME`, `PROD_API_PASSWORD`
- `PREPROD_API_*` variants for preprod environment
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` for S3

## Notebook Conventions

- **Naming:** `EDAgriculture_<Type>_<Purpose>.ipynb`
  - Function dev: `EDAgriculture_<Type>_Processor_Function_Dev.ipynb`
  - Service dev: `EDAgriculture_<Type>_Service_Function_Dev.ipynb`
  - Extraction: `EDAgriculture_<Type>_Extraction.ipynb`
- **Structure:** Step 0 (bootstrap cell) -> Step 1 (Init WorkflowManager) -> Step 2 (Load entities) -> Step 3 (Configure & extract)
- **Locations:**
  - `notebooks/` — extractor class dev & showcases
  - `customers/` — client-specific extractions (one folder per `<Client>`)
  - `study/` — service delivery (coverage check, zonal stats, custom map, ...)
  - `backoffice/` — user management & admin

## Common Pitfalls

- **LAI extraction requires `crop` field** — entities must have a crop value; the API returns empty otherwise
- **`clear_cover_min=100`** is very restrictive — use `95` for most extractions
- **`build/` gets stale** — rebuild after source changes
- **Column mapping with platform data** — platform returns `crop.id`, `sowingDate` etc.; use `column_mapping={"crop": "crop.id", "start_date": "sowingDate"}` to map them
- **Never commit `.env` files** — they contain API credentials

"""


def generate_claude_md(entries: list) -> str:
    """Generate CLAUDE.md with static header + auto-generated extractor reference."""
    lines = [CLAUDE_MD_HEADER.strip()]
    lines.append("")
    lines.append("## Extractor Reference (auto-generated)")
    lines.append("")
    lines.append(f"<!-- Generated by generate_ai_context.py on {datetime.now().strftime('%Y-%m-%d %H:%M')} -->")
    lines.append("")

    # Group by category
    categories: dict[str, list] = {}
    for entry in entries:
        if "error" in entry:
            continue
        categories.setdefault(entry["category"], []).append(entry)

    for cat in sorted(categories.keys()):
        lines.append(f"### {cat}")
        lines.append("")

        for e in sorted(categories[cat], key=lambda x: x["class_name"]):
            lines.append(f"#### `{e['class_name']}`")
            lines.append("")
            if e.get("description"):
                lines.append(e["description"])
                lines.append("")

            lines.append(f"- **Module:** `{e['module']}`")
            if e.get("inherits"):
                lines.append(f"- **Inherits:** {', '.join(e['inherits'])}")
            if e.get("documentation_url"):
                lines.append(f"- **Docs:** {e['documentation_url']}")
            if e.get("notebook"):
                lines.append(f"- **Notebook:** `{e['notebook']}`")

            # Setup method + key params
            setup = e.get("setup_method")
            if setup:
                key_params = [p for p in setup["parameters"] if p["name"] not in COMMON_SETUP_PARAMS]
                if key_params:
                    param_strs = []
                    for p in key_params:
                        s = f"`{p['name']}`"
                        if p["default"] is not None:
                            s += f" (default: {p['default']})"
                        param_strs.append(s)
                    lines.append(f"- **Setup:** `{setup['method_name']}()` — {', '.join(param_strs)}")
                else:
                    lines.append(f"- **Setup:** `{setup['method_name']}()`")

            if e.get("entity_fields"):
                qualifiers = e.get("entity_field_qualifiers") or {}
                rendered = [f"`{f}` ({qualifiers[f]})" if f in qualifiers else f"`{f}`" for f in e["entity_fields"]]
                lines.append(f"- **Entity fields:** {', '.join(rendered)}")

            if e.get("output_columns"):
                cols = e["output_columns"]
                if len(cols) <= 8:
                    lines.append(f"- **Output columns:** {', '.join(f'`{c}`' for c in cols)}")
                else:
                    lines.append(f"- **Output columns ({len(cols)}):** {', '.join(f'`{c}`' for c in cols[:8])}, ...")

            if e.get("supported_operations"):
                lines.append(f"- **Operations:** {', '.join(e['supported_operations'])}")

            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# agents.md generator
# ---------------------------------------------------------------------------


def generate_agents_md(entries: list) -> str:
    """Generate agents.md with per-extractor agent cards for AI assistants."""
    lines = []
    lines.append("# agents.md — EarthDaily Agriculture Extractor Agent Cards")
    lines.append("")
    lines.append("<!-- Auto-generated by src/earthdaily/agriculture/ai_enablement/generate_ai_context.py -->")
    lines.append(f"<!-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} -->")
    lines.append("")
    lines.append("Each card below describes one extractor class with enough context")
    lines.append("for an AI agent to use it correctly: import path, setup parameters,")
    lines.append("required entity fields, expected output columns, and key methods.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry in sorted(entries, key=lambda x: x["class_name"]):
        if "error" in entry:
            lines.append(f"## {entry['class_name']}")
            lines.append(f"**IMPORT ERROR:** {entry['error']}")
            lines.append("")
            continue

        lines.append(f"## {entry['class_name']}")
        lines.append("")
        if entry.get("description"):
            lines.append(f"> {entry['description']}")
            lines.append("")

        lines.append(f"**Category:** {entry['category']}  ")
        lines.append(f"**Import:** `from {entry['module']} import {entry['class_name']}`  ")
        if entry.get("inherits"):
            lines.append(f"**Inherits:** {', '.join(entry['inherits'])}  ")
        if entry.get("source_file"):
            lines.append(f"**Source:** `{entry['source_file']}`  ")
        if entry.get("documentation_url"):
            lines.append(f"**API Docs:** {entry['documentation_url']}  ")
        if entry.get("notebook"):
            lines.append(f"**Notebook:** `{entry['notebook']}`  ")
        lines.append("")

        # Setup method with full param table
        setup = entry.get("setup_method")
        if setup:
            lines.append(f"### Setup: `{setup['method_name']}()`")
            lines.append("")
            params = [p for p in setup["parameters"] if p["name"] not in COMMON_SETUP_PARAMS]
            if params:
                lines.append("| Parameter | Default |")
                lines.append("|---|---|")
                for p in params:
                    default = p["default"] if p["default"] is not None else "*required*"
                    lines.append(f"| `{p['name']}` | {default} |")
                lines.append("")

            # Docstring args with descriptions (richer than signature alone)
            doc_args = entry.get("setup_args_from_docstring", [])
            if doc_args:
                lines.append("**Parameter details:**")
                for a in doc_args:
                    desc = a["description"]
                    default_note = f" Default: {a['default']}" if a.get("default") else ""
                    lines.append(f"- `{a['name']}` ({a['type']}): {desc}{default_note}")
                lines.append("")

        # Entity fields
        #
        # NOT "Required Entity Fields": the qualifier used to be stripped during
        # parsing, so optional fields were published as required. That is how
        # `historical_seasons` came to contradict the column-mapping reference.
        if entry.get("entity_fields"):
            lines.append("### Entity Fields")
            lines.append("")
            qualifiers = entry.get("entity_field_qualifiers") or {}
            for f in entry["entity_fields"]:
                note = qualifiers.get(f)
                lines.append(f"- `{f}` — {note}" if note else f"- `{f}`")
            for note in entry.get("entity_field_notes") or []:
                lines.append(f"- _{note}_")
            lines.append("")

        # Output columns
        if entry.get("output_columns"):
            lines.append("### Output Columns")
            lines.append("")
            for c in entry["output_columns"]:
                lines.append(f"- `{c}`")
            lines.append("")

        # Methods
        methods_sections = [
            ("API Methods", entry.get("api_methods", [])),
            ("Process Methods", entry.get("process_methods", [])),
            ("Format Methods", entry.get("format_methods", [])),
        ]
        has_methods = any(m for _, m in methods_sections)
        if has_methods:
            lines.append("### Methods")
            lines.append("")
            for section_name, method_list in methods_sections:
                if method_list:
                    lines.append(f"**{section_name}:** {', '.join(f'`{m}()`' for m in method_list)}  ")
            lines.append("")

        # Supported operations (management classes)
        if entry.get("supported_operations"):
            lines.append("### Operations")
            lines.append("")
            for op in entry["supported_operations"]:
                lines.append(f"- {op}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Project-level context (for cookiecutter-generated projects)
# ---------------------------------------------------------------------------

PROJECT_CLAUDE_MD_TEMPLATE = """# CLAUDE.md — {project_name}

Generated EarthDaily Agriculture extraction project. Uses the local `earthdaily.agriculture` wheel from `dist/`.

## Quick context

- **Environment:** `{environment}` (`prod` | `preprod`)
- **earthdaily.agriculture version:** `{wheel_version}` — installed from `dist/earthdaily_agriculture-{wheel_version}-py3-none-any.whl`
- **Workflow:** `configuration/workflow.yml`
- **Custom transforms:** `app/transforms.py`
- **Inputs:** `inputs/` • **Outputs:** `results/` • **Partials:** `partials/` • **Logs:** `logs/`

## How to work in this project

1. Activate the venv (`.venv\\Scripts\\activate` on Windows, `source .venv/bin/activate` elsewhere).
2. Drive an extraction interactively from `EDAgriculture_<project_slug>.ipynb`, or call `manager.run_workflow()` against `configuration/workflow.yml`.
3. To refresh the wheel after upstream changes: rebuild it in the earthdaily-agriculture-internal repo with `python -m build`, copy the new `.whl` into `dist/`, then `pip install --force-reinstall dist/earthdaily_agriculture-<v>-py3-none-any.whl`.

## Use the `/earthdaily-agriculture` skill for extractor work

`.claude/skills/earthdaily-agriculture/SKILL.md` contains the extractor catalogue, paste-ready setup snippets, the YAML step recipe, and common pitfalls. The full per-extractor reference (params, output columns, methods) sits next to it in `extractors.md`.

## Bundled reference docs

The `docs/` folder ships with EarthDaily Agriculture's user-facing documentation. Open it when you need:

| Doc | When |
|---|---|
| `docs/01 - Quick_Start_Guide.md` | Bootstrapping a new client project from the cookiecutter template |
| `docs/03 - Extractor_parameters_reference.md` | All extractor setup kwargs in one place |
| `docs/04 - Extractor_column_mapping_reference.md` | Mapping your DataFrame columns onto canonical names |
| `docs/05 - Extractor_kpi_reference.md` | KPI filters per extractor |
| `docs/07 - Cache_design_context.md` | When to use the extraction cache |
| `docs/09 - Workflow_architecture.md` & `09b - Workflow_YAML_reference.md` | Multi-step workflow runner |
| `docs/13 - Cloud_storage_principles_and_usage.md` | **S3 / cloud-storage user guide** — the three recipes and their gotchas |
| `docs/14 - Deployment_patterns.md` | **Deployment** — Pattern A (GitHub Actions cron) vs Pattern B (Docker container) |
| `docs/extractors/` | Per-extractor deep-dives |

Showcase notebooks (also in `docs/`) — open and run: `EDAgriculture_S3_storage_Showcase.ipynb`, `EDAgriculture_KPI_Showcase.ipynb`, `EDAgriculture_viz_engine_Showcase.ipynb`, `EDAgriculture_extractor_cache_showcase.ipynb`, `EDAgriculture_Column_Mapping_Showcase.ipynb`, `EDAgriculture_Extreme_Event_Detection.ipynb`.

## Cloud storage in one line

Per `docs/13` Principle 1 — the path string is the mode. Switch any/all of the writer paths to `s3://...` at construction time:

```python
manager = WorkflowManager(
    "{environment}",
    output_result_dir="s3://my-bucket/runs/2026-01-01/results",
    partial_result_dir="s3://my-bucket/runs/2026-01-01/partials",
    cache_dir="s3://my-bucket/runs/2026-01-01/cache",   # auto-disables cache
)
```

The same works for **Azure Blob** (`az://container/...`) and GCS (`gs://bucket/...`) — the path scheme picks the backend.

Prerequisites by backend (install the extra in the upstream `earthdaily.agriculture` repo):

- **S3** (`s3://`): `pip install -e ".[s3]"` (pulls `s3fs` + `fsspec`); AWS credentials via the standard chain (env vars / IAM role / `AWS_ENDPOINT_URL` for MinIO).
- **Azure** (`az://`): `pip install -e ".[azure]"` (pulls `adlfs` + `fsspec`); credentials from env, first match wins — `AZURE_STORAGE_CONNECTION_STRING` → `AZURE_STORAGE_ACCOUNT_NAME` + `AZURE_STORAGE_SAS_TOKEN` → account name + `AZURE_STORAGE_ACCOUNT_KEY` → account name alone (adlfs default chain → Managed Identity). `.[cloud]` installs both backends.

## Container deploys — `EDAGRO_OUTPUT_PREFIX` + `EDAGRO_LOG_CONSOLE_ONLY`

For ephemeral runtimes (Argo / ECS / Cloud Run), set two env vars and `WorkflowManager` configures itself:

```bash
export EDAGRO_OUTPUT_PREFIX=s3://my-bucket/runs/2026-01-01/<workflow>   # or az://container/runs/...
export EDAGRO_LOG_CONSOLE_ONLY=1
```

- `EDAGRO_OUTPUT_PREFIX` — routes writers under `{{prefix}}/results`, `{{prefix}}/partials`, `{{prefix}}/cache` automatically. Precedence: explicit kwargs > env var > local defaults.
- `EDAGRO_LOG_CONSOLE_ONLY=1` — skip the loguru file sink, emit logs only to stdout (so the orchestrator captures them the standard way).

Equivalent code kwargs: `WorkflowManager(..., output_result_dir=..., partial_result_dir=..., cache_dir=..., log_to_console_only=True)`. The env vars win if both are set.

See `docs/14 - Deployment_patterns.md` for the full Pattern A (GH Actions cron) vs Pattern B (Docker container) decision guide.

## Don'ts

- Don't commit `.env` (credentials).
- Don't commit `dist/*.whl` to a public repo.
- Don't run extractions without a `column_mapping` if your DataFrame uses non-canonical column names — see the `/earthdaily-agriculture` skill for examples.
"""


SKILL_MD_HEADER = """---
name: earthdaily-agriculture
description: Use when picking an EarthDaily Agriculture extractor, configuring its parameters, wiring it into a WorkflowManager pipeline, or rebuilding/refreshing the local earthdaily.agriculture wheel in dist/. Covers the extractor catalogue, paste-ready setup snippets, YAML workflow recipes, and common pitfalls.
---

# /earthdaily-agriculture — EarthDaily Agriculture extractor recipes

This project uses `earthdaily.agriculture` (installed from the local wheel in `dist/`) to bulk-extract analytics from the EarthDaily Agriculture APIs. Use this skill to pick the right extractor, configure it correctly, and wire it into the workflow.

For full per-extractor parameters, output columns, and methods, open `extractors.md` in this directory.
"""


SKILL_MD_RECIPES = """
## Recipe — direct extractor mode (notebook / ad-hoc)

```python
from earthdaily.agriculture.services.workflow_manager import WorkflowManager
from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor

manager = WorkflowManager("prod", log_level="INFO")
manager.load_seasonfields()                     # or manager.sfd_list = load_geodataframe("inputs/...")

extractor = CoverageExtractor(
    manager.bearer_token, manager.token_expiration,
    config=manager.config, workflow_ref=manager,  # workflow_ref → auto token refresh
)
extractor.setup_coverage_parameters(
    vegetation_index="NDVI", start_date="2025-01-01", clear_cover_min=80,
    column_mapping={"crop": "crop.id"},          # only if your DF uses non-canonical names
)
results = extractor.process_entity_coverage_bulk_parallel(
    entity_list=manager.sfd_list, max_workers=10,
    output_path=manager.output_result_dir, prefix="coverage",
)
df = results["results_df"]
```

## Recipe — YAML workflow step

```yaml
# configuration/workflow.yml
workflow:
  settings:
    max_workers: 10
    column_mapping: { crop: crop.id, start_date: sowingDate }
  steps:
    - name: coverage
      extractor: CoverageExtractor
      module: earthdaily.agriculture.extractors.coverage_function
      setup:
        method: setup_coverage_parameters
        params: { vegetation_index: NDVI, start_date: "2025-01-01", clear_cover_min: 90 }
      run:
        method: process_entity_coverage_bulk_parallel
        params: { prefix: coverage, skip_export: true }
```

Driver: `manager.load_workflow("configuration/workflow.yml"); manager.run_workflow(entity_list=manager.sfd_list)`.

## Wheel runbook

- **Refresh after upstream changes:** in the earthdaily-agriculture-internal repo run `python -m build`, copy the new `.whl` into this project's `dist/`, then `pip install --force-reinstall dist/earthdaily_agriculture-<v>-py3-none-any.whl`.
- **Version sync:** `requirements.txt` pins `dist/earthdaily_agriculture-<version>-py3-none-any.whl` — bump that version string when refreshing.
- **Install fresh:** `python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt`.

## Common pitfalls

- **LAI requires `crop`** — entities without a crop value return empty.
- **`clear_cover_min=100`** is very restrictive — most extractions want 80–95.
- **Platform DataFrames** return `crop.id`, `sowingDate` etc. — pass `column_mapping={"crop": "crop.id", "start_date": "sowingDate"}`.
- **Never `if entity_data:` on a Series** — use `self._is_entity_provided(entity_data)`.
- **Empty results, no error?** flip `WorkflowManager(..., log_level="DEBUG")` to see the actual API URL and response.
- **Token expires mid-run** if you build an extractor without `workflow_ref=manager`.
"""


PERSONAL_SKILL_HEADER = """---
name: earthdaily-agriculture
description: Use when the user mentions earthdaily-agriculture, earthdaily.agriculture, earthdaily-agriculture, BaseExtractor, WorkflowManager, adding or debugging an EarthDaily Agriculture extractor (CoverageExtractor, GreennessExtractor, MRTSExtractor, VegationTsExtractor, FLMExtractor, WeatherExtractor, RegionalExtractor, ZARCExtractor, cropid, disease extractor, harvest extractor, emergence extractor, in-season score, historical score, tillage, covercrop, baresoil, field level maps, planted area, change index, standing crop, location-based border, entity management), regenerating agents.md, EarthDaily Agriculture API 400 errors, setup_*_parameters, process_*_bulk_parallel, or bootstrapping a client extraction notebook. Stands alone — does NOT require the earthdaily-agriculture-internal repo's CLAUDE.md to be loaded.
---

# /earthdaily-agriculture — Personal skill for the EarthDaily Agriculture Python package

You are helping the user work with `earthdaily.agriculture`, a Python package that bulk-extracts agricultural analytics from EarthDaily Agriculture / Geosys APIs. Source of truth is the **earthdaily-agriculture-internal repo** (typically cloned at `~/Documents/Github/earthdaily-agriculture-internal` on Windows or `~/projects/earthdaily-agriculture-internal` on macOS/Linux). This skill works *anywhere* — client notebooks, scratch dirs, even folders with no `CLAUDE.md`.

For the canonical, always-current reference, point the user at:
- `<earthdaily-agriculture-internal>/CLAUDE.md` — architecture overview + extractor reference.
- `<earthdaily-agriculture-internal>/agents.md` — per-extractor agent cards (params, output columns, methods).
- `<earthdaily-agriculture-internal>/docs/site/agriculture/` — public-facing numbered docs (`01 - Quick_Start_Guide.md`, `03 - Extractor_parameters_reference.md`, `09 - Workflow_architecture.md`, `13 - Cloud_storage_principles_and_usage.md`, `14 - Deployment_patterns.md`, etc.). Internal docs (release pipeline design, repo changelog) live under `<earthdaily-agriculture-internal>/docs/internal/` and are not part of the public package.

For the per-extractor catalogue current at the time this skill was generated, see `extractors.md` next to this file.

## Mental model — the 5-method extractor lifecycle

Every extractor subclasses `BaseExtractor` and implements:

```
__init__(token, config, workflow_ref)
    → setup_<type>_parameters(...)
        → get_<type>_api(entity_data)
            → format_<type>_json(response, entity_data)
                → process_single_entity_<type>(row)
                    → process_<type>_bulk_parallel(entity_list, ...)
```

Decorators: `@requires_token` on `get_*_api`; `@require_<type>_params` on every public method that needs config. Always pass `workflow_ref=manager` so token refresh works mid-run.

## extractors/ vs processors/ — what goes where

| `src/earthdaily/agriculture/extractors/` | `src/earthdaily/agriculture/processors/` |
|---|---|
| Synchronous service endpoints | Processor endpoints (often heavier compute) |
| One HTTP call → response | May be sync OR async (submit → poll → retrieve) |
| Examples: Coverage, FLM, VTS, MRTS, Weather, Regional, cropid | Examples: Emergence, Harvest, Disease, Greenness, Baresoil, Score (Historical/In-season), Planted, ZARC, ChangeIndex, InSeasonMonitoring (async) |

## `BaseExtractor` contract — what to trust

- **Token management** — `@requires_token` + `workflow_ref` mean you never write auth yourself.
- **Column mapping** — `DEFAULT_COLUMN_MAPPING` + `get_entity_value(row, key)` + `_is_entity_provided(entity_data)` handle platform-vs-canonical column names.
- **API response validation** — `validate_api_response(response_json, entity_data)` returns False on empty/None responses. Call it first inside `format_*_json`.
- **Output formatting** — `configure_output(rename, exclude, select)` applies rename → exclude → select in that order.
- **Export format** — `self.export_format` selects `"csv"` (default) or `"parquet"` for results. Set via workflow.yml `settings.export_format`, the `EDAGRO_EXPORT_FORMAT` env var, or `extractor.export_format = "parquet"` (precedence in that order). One switch on `BaseExtractor` → covers every extractor; **don't** add it to `setup_*_parameters`. Errors stay CSV; parquet needs `pyarrow` (already a dep).
- **Reports** — pass `generate_report=True` to any `process_*_bulk_parallel` to write an HTML report.
- **Logging** — `loguru`-based, shared with `WorkflowManager` or standalone.

## Hard rules (don't trip on these)

- Never `if entity_data:` on a `pd.Series` — use `self._is_entity_provided(entity_data)`.
- Use `response.get('key') or []` for APIs that return explicit `None` (not `response.get('key', [])` — the default arg only fires when the key is missing).
- LAI extraction requires `crop` on every entity; the API returns empty otherwise.
- `clear_cover_min=100` is very restrictive — default `95` (`80` for cloudy regions).
- Rebuild `dist/earthdaily_agriculture-*.whl` (via `python -m build`) after source changes — stale wheels masquerade as code bugs.
- Never commit `.env` or `dist/*.whl` to public repos.

## Standard import surface

```python
from earthdaily.agriculture.services.workflow_manager import WorkflowManager
from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
from earthdaily.agriculture.processors.processor_greenness_functions import GreennessExtractor
from earthdaily.agriculture.core.geometry import validate_wkt, load_geodataframe
from earthdaily.agriculture.core.api_utils import export_results
from earthdaily.agriculture.reporting import ExtractionReporter
```

## Credentials — create the `.env` yourself, before anything else

Nothing authenticates until a `.env` exists, and the package will not create or
prompt for one. `WorkflowManager("prod")` calls `setup_environment()` in its
constructor, which raises `ValueError("Missing required environment variables")`
the moment it cannot find all four values. A user who has not scaffolded a project
from the cookiecutter template has no `.env` at all — this is the first thing to
check when someone new reports that nothing works.

Write the file by hand. Four values per environment:

```env
# prod
PROD_API_CLIENT_ID=<client_id>
PROD_API_CLIENT_SECRET=<client_secret>
PROD_API_USERNAME=<username>
PROD_API_PASSWORD=<password>

# preprod — same four with a PREPROD_ prefix; only needed for WorkflowManager("preprod")

# only when writing results to s3://
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

Credentials come from the user's EarthDaily contact — there is no self-service
signup, so never invent placeholder values and tell them to "try it".

### Where the file may live

`setup_environment()` loads the **first** of these that exists, then stops. Any of
them is a valid choice; the folder is the user's to pick:

| # | Path | When this is the one that wins |
|---|---|---|
| 1 | `<project_root>/.env` | `project_root=` was passed to `WorkflowManager` |
| 2 | `<project_root>/src/.env` | same, for a package-shaped layout |
| 3 | `<code_root>/src/.env` | working inside a repo checkout — `code_root` is the nearest ancestor holding `pyproject.toml` |
| 4 | `<cwd>/.env` | no `pyproject.toml` above you — the normal case for a notebook in an ordinary folder |
| 5 | first `.env` found walking up from cwd | nothing above matched |

So there are two answers to "where do I put it":

- **Next to your work.** Drop `.env` in the folder you run from. Rule 4 finds it,
  no arguments needed. This is the right default for a plain folder.
- **Somewhere deliberate** — a shared credentials directory, outside any repo:

  ```python
  manager = WorkflowManager("prod", project_root="/abs/path/to/that/folder")
  ```

  Note the side effect: `project_root` is also the workspace root, so `results/`,
  `partials/`, `cache/` and `logs/` are created there too. Pass
  `output_result_dir=` if the outputs belong somewhere else.

### Confirm which file was read

The loader logs the path it used — treat this as the check, not the extraction
result:

```
🔑 Loaded credentials from: /abs/path/.env
```

Distinguish the two failures by their log line; they have different fixes.

- `No .env file found (searched N known locations and upward from <cwd>)` — the
  file is not in any of the five places. Wrong **location**; the contents are
  irrelevant until this is fixed.
- `❌ Missing environment variables for prod: PROD_API_USERNAME (or API_USERNAME), …`
  — a `.env` *was* read, but a key is absent or misspelled. Wrong **contents**.

### Two traps

- `notebook_setup.init()` **chdirs to the project root** it resolves, which is not
  necessarily the folder the notebook sits in. That changes what `cwd` means for
  rule 4, so a `.env` beside the notebook can stop being found after `init()` runs.
  `init()` prints the directory it chose — read it.
- Pointing anyone at a clone of this repo as their working folder does not give
  them credentials. It is the package source; it ships no `.env`, only
  `project_template/{{cookiecutter.project_slug}}/.env.template` to copy.

The unprefixed `API_CLIENT_ID` / `API_USERNAME` / … names still work as a fallback
for any environment, but prefer the `PROD_`/`PREPROD_` forms so one file serves
both. A `.env` is plaintext credentials: keep it gitignored and out of notebook
output cells.

### Check this before writing extraction code, not after it fails

Any request to actually run an extraction — "pull NDVI for this geometry", "get
coverage for these fields" — needs two things a code environment may not have.
Establish both up front, and say plainly if either is missing:

1. **Credentials**, per the above. If there is no `.env`, ask the user which
   folder they want to work from and write the template there for them to fill
   in. Do not guess the location, and do not proceed as though it will work.
2. **Network egress to the API.** Extractions call `identity.geosys-na.net` and
   `api.geosys-na.net`. A sandbox, container or remote VM without those hosts
   allowlisted cannot run one however correct the code is — and the package
   being absent from that environment is a second, separate blocker. In that
   situation hand the user a runnable script for the machine that *does* have
   both, rather than a partial attempt.

## Notebook bootstrap (paste verbatim as Step 0)

```python
import sys
from pathlib import Path

_src = str(Path().resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from earthdaily.agriculture.notebook_setup import init
init()
```

## When to use each runbook

<!--RUNBOOK-POINTERS-->

## Tagging conventions for this skill

When you reason beyond what the runbooks literally state, tag the inference: `[inference]`. The user's CLAUDE.md and runbooks are the source of truth; speculate sparingly.
"""


def generate_personal_skill_md(entries: list, runbook_pointers: str = "") -> str:
    """Personal-skill SKILL.md with extractor decision table and runbook pointers.

    ``runbook_pointers`` is the rendered "When to use each runbook" list, built
    from the runbooks that actually shipped (see :func:`render_runbook_pointers`).
    Passed in rather than computed here so this stays a pure formatter.

    Differs from the project-local skill in three ways:
      1. Description triggers fire across ALL projects (not just inside one).
      2. Points at runbooks/*.md instead of inlining the recipes.
      3. References the earthdaily-agriculture-internal repo's CLAUDE.md / agents.md as the
         canonical reference, since this skill is loaded outside that repo.
    """
    lines = [PERSONAL_SKILL_HEADER.replace("<!--RUNBOOK-POINTERS-->", runbook_pointers).rstrip()]
    lines.append("")
    lines.append(f"<!-- Generated by generate_ai_context.py on {datetime.now().strftime('%Y-%m-%d %H:%M')} -->")
    lines.append("")
    lines.append("## Extractor catalogue (snapshot — see extractors.md for full detail)")
    lines.append("")
    lines.append("| Category | Extractor | What it does |")
    lines.append("|---|---|---|")

    categories: dict[str, list] = {}
    for entry in entries:
        if "error" in entry:
            continue
        categories.setdefault(entry.get("category", "Other"), []).append(entry)

    for cat in sorted(categories.keys()):
        for e in sorted(categories[cat], key=lambda x: x["class_name"]):
            desc = (e.get("description") or "").split(". ")[0]
            if len(desc) > 90:
                desc = desc[:87] + "..."
            lines.append(f"| {cat} | `{e['class_name']}` | {desc} |")

    lines.append("")
    lines.append(
        "For full per-extractor parameters, output columns, and methods see "
        "`extractors.md` next to this file (snapshot of agents.md at generation time)."
    )
    lines.append("")
    return "\n".join(lines)


def _runbook_frontmatter_value(text: str, key: str) -> str:
    """One scalar out of a runbook's leading ``---`` block. Empty when absent."""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    if end == -1:
        return ""
    for line in text[3:end].splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip() == key:
            return value.strip().strip("\"'").strip()
    return ""


def render_runbook_pointers(script_root: Path, shipped: Sequence[str]) -> str:
    """The SKILL.md "When to use each runbook" list, built from what actually shipped.

    Generated rather than hardcoded because the two can disagree, and both ways of
    disagreeing are harmful: a pointer to a runbook that is not in the bundle is a
    dead link, and — since runbooks are now visibility-gated — hardcoding an
    internal one would advertise a VPN-only admin walkthrough in a public
    artifact. Driving the list from the shipped set makes both impossible.

    Descriptions come from each runbook's own ``description:`` frontmatter, so
    the text lives next to what it describes instead of in a template far away.
    """
    source_dir = script_root / "personal_skill_templates" / "runbooks"
    lines = []
    for name in shipped:
        desc = _runbook_frontmatter_value((source_dir / name).read_text(encoding="utf-8"), "description")
        lines.append(f"- **`runbooks/{name}`** — {desc}" if desc else f"- **`runbooks/{name}`**")
    return "\n".join(lines)


def _runbook_is_internal(text: str) -> bool:
    """True when a runbook must not enter a public bundle.

    Anything not explicitly ``visibility: public`` counts as internal: a runbook
    with a missing, misspelled or unparsed marker **fails closed**. That is the
    whole point — a new runbook is private until someone says otherwise, so this
    class of leak cannot happen by omission.

    Deliberately a three-line reader rather than a YAML dependency: the contract
    is one key in a leading ``---`` block. Mirrors ``is_internal()`` in the MCP
    server's ``usecases.py``, which gates the same bundle the same way.
    """
    if not text.startswith("---"):
        return True
    end = text.find("\n---", 3)
    if end == -1:
        return True
    for line in text[3:end].splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "visibility":
            return value.strip().strip("\"'").strip() != "public"
    return True


def _copy_runbooks(
    script_root: Path,
    target_runbooks_dir: Path,
    include_internal: bool = False,
    check: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Sync runbook templates into a target directory.

    Returns ``(shipped, skipped, drift)``. Held to the same contract as the MCP
    server's ``scripts/build_usecases.py``, because both feed the same bundle:

    - **internal runbooks are excluded by default** (`include_internal` opts in),
      so the decision rides on the artifact rather than on a request-time flag —
      build a public bundle for a public deployment, an internal one otherwise;
    - **removals propagate.** A runbook deleted from the source is deleted from
      the target, so un-shipping one is deleting the source and re-running. It
      previously lingered in the target forever, because this only ever copied;
    - **`check` writes nothing** and reports drift instead, so CI can assert a
      target is current without a network call or a side effect.
    """
    source_dir = script_root / "personal_skill_templates" / "runbooks"
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Runbook templates not found at {source_dir}. "
            "This script expects sibling templates under personal_skill_templates/runbooks/."
        )

    wanted: dict[str, str] = {}
    skipped: list[str] = []
    for src_path in sorted(source_dir.glob("*.md")):
        text = src_path.read_text(encoding="utf-8")
        if _runbook_is_internal(text) and not include_internal:
            skipped.append(src_path.name)
            continue
        wanted[src_path.name] = text

    existing: dict[str, str] = {}
    if target_runbooks_dir.is_dir():
        existing = {p.name: p.read_text(encoding="utf-8") for p in target_runbooks_dir.glob("*.md")}

    added = sorted(set(wanted) - set(existing))
    removed = sorted(set(existing) - set(wanted))
    changed = sorted(n for n in set(wanted) & set(existing) if wanted[n] != existing[n])

    if check:
        drift = (
            [f"missing from target: {n}" for n in added]
            + [f"in target but not in source: {n}" for n in removed]
            + [f"stale in target: {n}" for n in changed]
        )
        return sorted(wanted), skipped, drift

    target_runbooks_dir.mkdir(parents=True, exist_ok=True)
    for name in removed:
        (target_runbooks_dir / name).unlink()
    for name in added + changed:
        shutil.copy2(source_dir / name, target_runbooks_dir / name)

    return sorted(wanted), skipped, []


def generate_project_claude_md(project_name: str, environment: str, wheel_version: str) -> str:
    return PROJECT_CLAUDE_MD_TEMPLATE.format(
        project_name=project_name,
        environment=environment,
        wheel_version=wheel_version,
    )


def generate_skill_md(entries: list) -> str:
    """Recipes-style skill body with an auto-generated extractor decision table."""
    lines = [SKILL_MD_HEADER.rstrip()]
    lines.append("")
    lines.append(f"<!-- Generated by generate_ai_context.py on {datetime.now().strftime('%Y-%m-%d %H:%M')} -->")
    lines.append("")
    lines.append("## Pick-an-extractor")
    lines.append("")
    lines.append("| Category | Extractor | What it does |")
    lines.append("|---|---|---|")

    categories: dict[str, list] = {}
    for entry in entries:
        if "error" in entry:
            continue
        categories.setdefault(entry.get("category", "Other"), []).append(entry)

    for cat in sorted(categories.keys()):
        for e in sorted(categories[cat], key=lambda x: x["class_name"]):
            desc = (e.get("description") or "").split(". ")[0]
            if len(desc) > 90:
                desc = desc[:87] + "..."
            lines.append(f"| {cat} | `{e['class_name']}` | {desc} |")
    lines.append("")

    lines.append(SKILL_MD_RECIPES.rstrip())
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLAUDE.md, agents.md, and gap report from EarthDaily Agriculture extractor introspection."
    )
    parser.add_argument(
        "--claude-md",
        default="../CLAUDE.md",
        help="Path for CLAUDE.md (default: ../CLAUDE.md — project root)",
    )
    parser.add_argument(
        "--agents-md",
        default="../agents.md",
        help="Path for agents.md (default: ../agents.md — project root)",
    )
    parser.add_argument(
        "--report",
        "-r",
        default="scripts/gap_report.txt",
        help="Path for the gap report (default: scripts/gap_report.txt)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress to stdout",
    )
    parser.add_argument(
        "--project-target",
        default=None,
        help="When set, write project-flavored CLAUDE.md + .claude/skills/earthdaily-agriculture/{SKILL.md,extractors.md} into this directory and skip the repo-level outputs. Used by the cookiecutter post-gen hook.",
    )
    parser.add_argument(
        "--project-name",
        default="EarthDaily Agriculture Project",
        help="Project name for the generated CLAUDE.md (only used with --project-target)",
    )
    parser.add_argument(
        "--environment",
        default="prod",
        help="Target environment: prod or preprod (only used with --project-target)",
    )
    parser.add_argument(
        "--wheel-version",
        default="2.4.0",
        help="earthdaily.agriculture wheel version pinned by the project (only used with --project-target)",
    )
    parser.add_argument(
        "--personal-skill-target",
        default=None,
        help=(
            "When set, write a personal/global Claude Code skill into this directory "
            "(typically ~/.claude/skills/earthdaily-agriculture). Produces SKILL.md, extractors.md, "
            "and runbooks/*.md. Skips the repo-level outputs. Distinct from --project-target "
            "which writes a project-local skill inside a generated client project."
        ),
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help=(
            "Also ship runbooks marked `visibility: internal`. Off by default: the target may be "
            "the MCP bundle, which is built into a wheel and can be hosted, so the decision rides "
            "on the artifact — build an internal skill for internal use, a public one otherwise. "
            "Mirrors `build_usecases.py --include-internal` in the MCP server repo."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report whether the target's runbooks/ is in sync and exit 1 if not; write nothing. "
            "For CI. Only meaningful with --personal-skill-target."
        ),
    )
    args = parser.parse_args()

    if args.check and not args.personal_skill_target:
        parser.error("--check only applies to --personal-skill-target")
    if args.include_internal and not args.personal_skill_target:
        parser.error("--include-internal only applies to --personal-skill-target")

    # Ensure we can import from src/ when invoked as a direct script (not via
    # `python -m earthdaily.agriculture.ai_enablement.generate_ai_context`, which already handles
    # imports correctly via the installed package). After the move into the package,
    # the script lives at src/earthdaily/agriculture/ai_enablement/generate_ai_context.py, so we
    # need three .parent hops to reach src/.
    src_dir = Path(__file__).resolve().parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if args.verbose:
        print(f"Source directory: {src_dir}")
        print(f"Introspecting {len(EXTRACTOR_REGISTRY)} extractor classes...\n")

    # Introspect all classes
    entries = []
    for module_path, class_name, category in EXTRACTOR_REGISTRY:
        if args.verbose:
            print(f"  {class_name:40s} from {module_path}...", end=" ")
        entry = introspect_class(module_path, class_name, category)
        entries.append(entry)
        if args.verbose:
            print("ERROR" if "error" in entry else "OK")

    # Analyze gaps
    gaps = analyze_gaps(entries)

    # Generate outputs
    claude_md = generate_claude_md(entries)
    agents_md = generate_agents_md(entries)
    gap_report = format_gap_report(gaps, entries)

    # ── Personal-skill mode: write a global skill into ~/.claude/skills/earthdaily-agriculture ──
    if args.personal_skill_target:
        target = Path(args.personal_skill_target).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)

        # Runbooks first: SKILL.md's pointer list is built from what actually
        # shipped, and --check must not write anything before it reports.
        script_root = Path(__file__).resolve().parent
        runbooks_copied, runbooks_skipped, runbook_drift = _copy_runbooks(
            script_root=script_root,
            target_runbooks_dir=target / "runbooks",
            include_internal=args.include_internal,
            check=args.check,
        )

        if not args.check:
            personal_skill_md = generate_personal_skill_md(
                entries,
                runbook_pointers=render_runbook_pointers(script_root, runbooks_copied),
            )
            (target / "SKILL.md").write_text(personal_skill_md, encoding="utf-8")
            (target / "extractors.md").write_text(agents_md, encoding="utf-8")

        if args.check:
            if runbook_drift:
                print("\nrunbooks out of date:", file=sys.stderr)
                for problem in runbook_drift:
                    print(f"  - {problem}", file=sys.stderr)
                print("\nre-run without --check to sync", file=sys.stderr)
                sys.exit(1)
            print(f"ok: runbooks/ matches source ({len(runbooks_copied)} shipped, {len(runbooks_skipped)} internal)")
            return

        errors = sum(1 for e in entries if "error" in e)
        print(f"\nPersonal skill written to {target}:")
        print("  SKILL.md")
        print("  extractors.md")
        for name in runbooks_copied:
            print(f"  runbooks/{name}")
        for name in runbooks_skipped:
            print(f"  (skipped, internal) runbooks/{name}")
        print(f"\nExtractors catalogued: {len(entries)} ({errors} errors)")
        print(
            "\nInstall hint: Claude Code auto-loads personal skills from "
            "~/.claude/skills/. If you generated into a different path, copy or "
            "symlink the folder under ~/.claude/skills/earthdaily-agriculture to activate it."
        )
        return

    # ── Project-target mode: write into a generated project, not the repo ──
    if args.project_target:
        target = Path(args.project_target).resolve()
        skill_dir = target / ".claude" / "skills" / "earthdaily-agriculture"
        skill_dir.mkdir(parents=True, exist_ok=True)

        project_claude = generate_project_claude_md(args.project_name, args.environment, args.wheel_version)
        skill_md = generate_skill_md(entries)

        (target / "CLAUDE.md").write_text(project_claude, encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (skill_dir / "extractors.md").write_text(agents_md, encoding="utf-8")

        errors = sum(1 for e in entries if "error" in e)
        print(f"\nProject context generated in {target}:")
        print("  CLAUDE.md")
        print("  .claude/skills/earthdaily-agriculture/SKILL.md")
        print("  .claude/skills/earthdaily-agriculture/extractors.md")
        print(f"\nExtractors documented: {len(entries)} ({errors} errors)")
        return

    # ── Repo mode: write the repo-level CLAUDE.md, agents.md, gap report ──
    for path, content in [
        (args.claude_md, claude_md),
        (args.agents_md, agents_md),
        (args.report, gap_report),
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # Summary
    errors = sum(1 for e in entries if "error" in e)
    severity_counts = {}
    for g in gaps:
        severity_counts[g["severity"]] = severity_counts.get(g["severity"], 0) + 1

    print("\nGenerated:")
    print(f"  CLAUDE.md    -> {args.claude_md}")
    print(f"  agents.md    -> {args.agents_md}")
    print(f"  gap_report   -> {args.report}")
    print(f"\nExtractors: {len(entries)} ({errors} errors)")
    print(f"Gap summary: {severity_counts}")


if __name__ == "__main__":
    main()
