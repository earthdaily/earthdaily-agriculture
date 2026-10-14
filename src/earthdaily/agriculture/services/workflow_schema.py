"""The ``workflow.yml`` schema (step format), declared once.

``WorkflowManager`` reads ``workflow.yml`` through ``.get("…")`` calls spread over its module;
until now the only list of those keys was the handwritten Workflow YAML reference (doc 09b),
and a misspelled key (``export_fromat:``) was silently ignored. This module is the one list:

- the ``TypedDict``\\s document the shape for readers, IDEs and coding agents;
- ``KEY_DOCS`` is rendered by ``generate_ai_context`` into the WorkflowManager card and the key
  tables of doc 09b;
- ``unknown_keys()`` is what ``load_workflow()`` uses to warn about keys nothing reads;
- ``tests/test_workflow_schema.py`` checks that every key ``WorkflowManager`` reads is declared
  here, and every key declared here is read.

The legacy ``analytics:`` format is not covered.
"""

from __future__ import annotations

import difflib
from typing import Any, Literal, TypedDict

ConditionOperator = Literal["eq", "ne", "in", "notnull"]
ExportFormat = Literal["csv", "parquet"]


class StepCall(TypedDict, total=False):
    """``setup:`` / ``run:`` — which method to call on the extractor, with which kwargs."""

    method: str
    params: dict[str, Any]


class TransformBlock(TypedDict, total=False):
    module: str
    function: str
    params: dict[str, Any]


class ConditionBlock(TypedDict, total=False):
    depends_on: str
    column: str
    operator: ConditionOperator
    value: Any


class EntitySource(TypedDict, total=False):
    """For the caller's own entity loading; the package only rewrites date sentinels in ``params``."""

    params: dict[str, Any]


class WorkflowSettings(TypedDict, total=False):
    column_mapping: dict[str, str]
    export_format: ExportFormat
    export_manifest: bool
    fail_safe: bool
    max_parallel_steps: int
    max_workers: int
    output_prefix: str
    output_uri: str
    retry_failed_only: bool | str
    timezone: str


class WorkflowStep(TypedDict, total=False):
    name: str
    enabled: bool
    depends_on: str | list[str]
    input_from: str
    transform: TransformBlock
    condition: ConditionBlock
    extractor: str
    module: str
    setup: StepCall
    run: StepCall
    export_format: ExportFormat
    export_manifest: bool
    output_uri: str
    retry_failed_only: bool | str


class Workflow(TypedDict, total=False):
    """The block under the top-level ``workflow:`` key (``load_workflow()`` returns this, unwrapped)."""

    name: str
    description: str
    settings: WorkflowSettings
    steps: list[WorkflowStep]
    entity_source: EntitySource


#: ``run.params`` keys WorkflowManager takes for itself (with their defaults); every other key
#: in ``run.params`` is forwarded to the bulk method as a keyword argument.
RUN_PARAMS_RESERVED: dict[str, tuple[str, str]] = {
    "max_workers": ("settings.max_workers, else 10", "threads for this step"),
    "prefix": ("the step name", "output file prefix (composed with settings.output_prefix)"),
    "skip_export": ("true", "results stay in memory unless set to false: a step does NOT write files by default"),
    "fail_safe": ("settings.fail_safe, else false", "record per-entity failures instead of raising"),
    "generate_report": ("false", "write the HTML extraction report"),
}

#: One line per key, per block. Rendered into the cards and doc 09b; the gate keeps it in sync.
KEY_DOCS: dict[str, dict[str, str]] = {
    "workflow": {
        "name": "workflow name, shown in logs and reports",
        "description": "free text",
        "settings": "defaults for every step (see settings)",
        "steps": "the steps, required (see step)",
        "entity_source": "free-form block for your own entity loading; date sentinels in its params are rewritten",
    },
    "settings": {
        "max_workers": "default threads per step (default 10)",
        "max_parallel_steps": "independent steps run at once within a DAG level (default 4)",
        "fail_safe": "record per-entity failures in failed_ids instead of raising (default false)",
        "retry_failed_only": "process only the ids in the newest (true) or the named `failed_ids_<prefix>_*.csv`",
        "column_mapping": "canonical field -> your column (id, geometry, crop, ...); GLOBAL: no per-step override",
        "output_prefix": "prefix for every output file name of the run",
        "export_format": "csv (default) or parquet, for every step; a step-level key overrides it",
        "export_manifest": "also write a `<prefix>_manifest_<ts>.json` next to each export",
        "output_uri": "second, durable copy of FLM / Difference / Zoning map files (s3:// gs:// az://)",
        "timezone": "IANA zone for the 'today' date sentinels (default UTC)",
    },
    "step": {
        "name": "unique step id, required; referenced by depends_on / input_from",
        "enabled": "false loads the step but skips it; dependents cascade-skip (default true)",
        "depends_on": "step name(s) that must finish first; drives the DAG",
        "input_from": "'original' or a step name: whose results_df becomes this step's entity_list",
        "transform": "a function run first on the entities (before condition and extractor; see transform)",
        "condition": "after the transform, keep only the entity ids passing a test on an upstream step's results",
        "extractor": "extractor class name, e.g. CoverageExtractor; needs module",
        "module": "import path of the extractor class",
        "setup": "setup call: method + params (see call)",
        "run": "bulk call: method + params, required with extractor (see call, run.params)",
        "export_format": "per-step override of settings.export_format",
        "export_manifest": "per-step override of settings.export_manifest",
        "output_uri": "per-step override of settings.output_uri",
        "retry_failed_only": "per-step override of settings.retry_failed_only",
    },
    "call": {
        "method": "method name on the extractor (setup_*_parameters / process_*_bulk_*)",
        "params": "keyword arguments; strings in setup.params may use date sentinels (today, today-30d, ...)",
    },
    "transform": {
        "module": "import path of the module holding the function, required",
        "function": "function name, required",
        "params": "passed as the function's params argument (WorkflowManager adds _workflow_manager)",
    },
    "condition": {
        "depends_on": "the upstream step whose results_df is filtered, required",
        "column": "column of that results_df, required",
        "operator": "eq (default), ne, in, notnull",
        "value": "compared value (a list for in); required except for notnull",
    },
    "entity_source": {
        "params": "parameters for your entity loader; date sentinels are rewritten",
    },
}

#: Block -> its TypedDict (``call`` covers both setup and run).
BLOCK_TYPES: dict[str, type] = {
    "workflow": Workflow,
    "settings": WorkflowSettings,
    "step": WorkflowStep,
    "call": StepCall,
    "transform": TransformBlock,
    "condition": ConditionBlock,
    "entity_source": EntitySource,
}

#: Traps worth stating on the WorkflowManager card (each backed by a test).
TRAPS: list[str] = [
    "`load_workflow()` returns the UNWRAPPED config for the step format: index `cfg['steps']`, "
    "never `cfg['workflow']['steps']` (a bare `KeyError: 'workflow'`). The legacy `analytics:` "
    "format returns the wrapped dict; `manager.workflow_cfg_raw` always keeps the wrapped original.",
    "A step does not write files unless `run.params.skip_export: false`; results stay in memory.",
    "`settings.column_mapping` is global: there is no per-step column mapping.",
    "Any other `settings` key is passed to a step's setup method only if its signature accepts it.",
]


def unknown_keys(workflow: dict, setup_params: dict[str, set[str]] | None = None) -> list[str]:
    """Paths of keys the step format does not read, each with a suggestion when one is close.

    ``setup_params`` maps step name -> parameter names of that step's setup method: an extra
    ``settings`` key is fine when some step's setup accepts it (documented pass-through).
    """
    problems: list[str] = []

    def check(block: str, mapping: Any, path: str, extra_ok: frozenset[str] | set[str] = frozenset()) -> None:
        if not isinstance(mapping, dict):
            return
        known = set(BLOCK_TYPES[block].__annotations__)
        for key in mapping:
            if key in known or key in extra_ok:
                continue
            close = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=0.75)
            problems.append(f"{path}.{key}" + (f" (did you mean '{close[0]}'?)" if close else ""))

    check("workflow", workflow, "workflow")
    accepted = set().union(*(setup_params or {}).values()) if setup_params else set()
    check("settings", workflow.get("settings"), "workflow.settings", accepted)
    check("entity_source", workflow.get("entity_source"), "workflow.entity_source")
    for i, step in enumerate(workflow.get("steps") or []):
        if not isinstance(step, dict):
            continue
        where = f"workflow.steps[{i}:{step.get('name', '?')}]"
        check("step", step, where)
        for call in ("setup", "run"):
            check("call", step.get(call), f"{where}.{call}")
        check("transform", step.get("transform"), f"{where}.transform")
        check("condition", step.get("condition"), f"{where}.condition")
    return problems
