"""W7a gate: the workflow.yml schema (services/workflow_schema.py) matches what WorkflowManager reads.

The schema is the one list of workflow.yml keys, rendered into the WorkflowManager card and the
key tables of doc 09b. It is only worth trusting if it cannot drift from the code, so:

- every key WorkflowManager reads from a YAML block is declared for that block, and
- every declared key is read somewhere (no documented key that does nothing).

Reads are found by AST: ``<var>.get("k")`` / ``<var>.pop("k")`` / ``<var>["k"]``, with the
variable names below mapped to the YAML block they hold.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from earthdaily.agriculture.services import workflow_manager, workflow_schema
from earthdaily.agriculture.services.workflow_schema import (
    BLOCK_TYPES,
    KEY_DOCS,
    RUN_PARAMS_RESERVED,
    unknown_keys,
)

pytestmark = pytest.mark.public

# Variable (as written in workflow_manager.py) -> the YAML block it holds.
VAR_BLOCK = {
    "self.workflow_cfg": "workflow",
    "workflow_cfg": "workflow",
    "self.workflow_cfg or {}": "workflow",
    "settings": "settings",
    "step_cfg": "step",
    "step": "step",
    "cond": "condition",
    "c": "condition",
    "transform_conf": "transform",
    "t": "transform",
    "setup_cfg": "call",
    "run_cfg": "call",
    "setup": "call",
    "run": "call",
    "step.get('setup') or {}": "call",
    "workflow_cfg.get('entity_source') or {}": "entity_source",
}
# `raw` holds a step while the legacy/step formats are told apart; its "workflow" key is the
# top-level wrapper, not a step key.
RAW_STEP_KEYS_EXCEPT = {"workflow"}


def _reads() -> dict[str, set[str]]:
    tree = ast.parse(inspect.getsource(workflow_manager))
    reads: dict[str, set[str]] = {}

    def add(var: str, key: str) -> None:
        if var == "raw":
            if key in RAW_STEP_KEYS_EXCEPT:
                return
            var = "step"
        block = VAR_BLOCK.get(var)
        if block:
            reads.setdefault(block, set()).add(key)

    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("get", "pop", "setdefault")
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        ):
            add(ast.unparse(n.func.value), n.args[0].value)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            add(ast.unparse(n.value), n.slice.value)
    return reads


READS = _reads()


@pytest.mark.parametrize("block", sorted(BLOCK_TYPES))
def test_every_key_read_is_declared(block):
    declared = set(BLOCK_TYPES[block].__annotations__)
    undeclared = READS.get(block, set()) - declared
    assert not undeclared, (
        f"WorkflowManager reads {sorted(undeclared)} from the '{block}' block, but workflow_schema does not "
        "declare them: add them (with a KEY_DOCS line) so they are documented and not flagged as unknown."
    )


@pytest.mark.parametrize("block", sorted(BLOCK_TYPES))
def test_every_declared_key_is_read(block):
    dead = set(BLOCK_TYPES[block].__annotations__) - READS.get(block, set())
    assert not dead, f"workflow_schema declares {sorted(dead)} for '{block}', but WorkflowManager never reads them."


@pytest.mark.parametrize("block", sorted(BLOCK_TYPES))
def test_key_docs_match_the_types(block):
    assert set(KEY_DOCS[block]) == set(BLOCK_TYPES[block].__annotations__)


def test_reserved_run_params_match_the_code():
    tree = ast.parse(inspect.getsource(workflow_manager))
    popped = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "pop"
        and ast.unparse(n.func.value) == "run_params"
    }
    assert popped == set(RUN_PARAMS_RESERVED)


class TestUnknownKeys:
    WORKFLOW = {
        "name": "w",
        "settings": {"max_workers": 4, "export_fromat": "parquet", "partial_frequency": 10},
        "steps": [
            {
                "name": "coverage",
                "extractor": "CoverageExtractor",
                "module": "m",
                "setup": {"method": "setup_coverage_parameters", "params": {}},
                "run": {"method": "process_entity_coverage_bulk_parallel", "parms": {}},
                "condition": {"depends_on": "x", "column": "c", "opertor": "eq"},
            }
        ],
    }

    def test_misspellings_are_reported_with_a_suggestion(self):
        problems = unknown_keys(self.WORKFLOW, {"coverage": {"partial_frequency"}})
        assert "workflow.settings.export_fromat (did you mean 'export_format'?)" in problems
        assert "workflow.steps[0:coverage].run.parms (did you mean 'params'?)" in problems
        assert "workflow.steps[0:coverage].condition.opertor (did you mean 'operator'?)" in problems

    def test_settings_key_accepted_by_a_setup_method_is_fine(self):
        problems = unknown_keys(self.WORKFLOW, {"coverage": {"partial_frequency"}})
        assert not any("partial_frequency" in p for p in problems)

    def test_settings_key_no_setup_accepts_is_reported(self):
        assert any("partial_frequency" in p for p in unknown_keys(self.WORKFLOW, {}))

    def test_a_clean_workflow_has_no_problems(self):
        clean = {"steps": [{"name": "a", "run": {"method": "m", "params": {}}}], "settings": {"max_workers": 2}}
        assert unknown_keys(clean) == []


def test_schema_module_documents_the_legacy_gap():
    assert "legacy" in (workflow_schema.__doc__ or "").lower()


def test_load_workflow_reports_a_typo(tmp_path):
    from earthdaily.agriculture.services.workflow_manager import WorkflowManager

    path = tmp_path / "workflow.yml"
    path.write_text(
        "workflow:\n"
        "  name: typo\n"
        "  settings:\n"
        "    export_fromat: parquet\n"
        "  steps:\n"
        "    - name: a\n"
        "      run: {method: m, params: {}}\n",
        encoding="utf-8",
    )
    manager = WorkflowManager.__new__(WorkflowManager)  # load_workflow needs no auth
    manager.load_workflow(str(path))
    assert manager.workflow_unknown_keys == ["workflow.settings.export_fromat (did you mean 'export_format'?)"]


def test_doc_09b_key_tables_are_current():
    """Doc 09b's Key reference is generated; a schema change must regenerate it."""
    from pathlib import Path

    from earthdaily.agriculture.ai_enablement import generate_ai_context as gen

    doc = Path(__file__).resolve().parent.parent / gen.WORKFLOW_DOC_PATH
    if not doc.is_file():
        pytest.skip("doc 09b not in this checkout")
    text = doc.read_text(encoding="utf-8")
    block = text.split(gen.WORKFLOW_DOC_BEGIN, 1)[1].split(gen.WORKFLOW_DOC_END, 1)[0]
    assert block.strip() == "\n".join(gen.render_workflow_tables(heading_level=3)).strip(), (
        "doc 09b's generated key tables are stale: run generate_ai_context (cd src && python -m "
        "earthdaily.agriculture.ai_enablement.generate_ai_context)"
    )
