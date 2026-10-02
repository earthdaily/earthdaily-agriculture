"""
Every extractor's bulk method must accept what WorkflowManager passes it.

WorkflowManager calls a step's run method with a fixed set of keyword arguments,
unconditionally (`_execute_single_step`). A bulk method missing one of them raises
``TypeError`` the moment it runs as a workflow step — locally, in GitHub Actions or
in a container — while direct notebook calls keep working, so nothing surfaces it.
ZARCExtractor shipped that way (no ``params``) in 2.6.0.

The required set is read from WorkflowManager's own source, so the two cannot drift.
Introspects whatever modules are installed: in the public payload that is exactly
the public extractors.
"""

import ast
import importlib
import inspect
import pkgutil
import textwrap

import pytest

import earthdaily.agriculture as pkg
from earthdaily.agriculture.core.base_extractor import BaseExtractor
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public

_SKIP_MODULES = ("ai_enablement", "notebook_setup")


def _required_kwargs() -> set[str]:
    """The `explicitly_passed` set literal in WorkflowManager._execute_single_step."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(WorkflowManager._execute_single_step)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "explicitly_passed" for t in node.targets
        ):
            return {elt.value for elt in node.value.elts}
    raise AssertionError("`explicitly_passed` not found in WorkflowManager._execute_single_step")


def _bulk_methods():
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if any(s in info.name for s in _SKIP_MODULES):
            continue
        importlib.import_module(info.name)

    seen, stack = set(), list(BaseExtractor.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())

    out = []
    for cls in seen:
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            # Own methods only, so an inherited one is checked once, on its owner.
            if name.startswith("process_") and "bulk" in name and fn.__qualname__.split(".")[0] == cls.__name__:
                out.append(pytest.param(cls, name, id=f"{cls.__name__}.{name}"))
    return sorted(out, key=lambda p: p.id)


def test_required_set_is_what_we_expect():
    """Pin it: a change to the call site should be a conscious edit here too."""
    assert _required_kwargs() == {
        "entity_list",
        "params",
        "max_workers",
        "output_path",
        "fail_safe",
        "prefix",
        "generate_report",
        "skip_export",
    }


def test_there_are_bulk_methods_to_check():
    assert len(_bulk_methods()) >= 10


@pytest.mark.parametrize(("cls", "method"), _bulk_methods())
def test_bulk_method_accepts_every_workflow_kwarg(cls, method):
    params = inspect.signature(getattr(cls, method)).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = sorted(_required_kwargs() - set(params))
    assert not missing, (
        f"{cls.__name__}.{method} does not accept {missing}; WorkflowManager passes them "
        "unconditionally, so this extractor raises TypeError as a workflow step."
    )
