"""W5 gate: docstrings must not drift from the allowed values declared in the signature.

A parameter typed with a ``Literal`` (see ``core.choices``) has ONE source for its allowed
values: the type. The class docstring (read by ``generate_ai_context`` into CLAUDE.md /
agents.md) and the setup method's docstring (what ``help()`` and IDEs show) may explain those
values, but two things are checked for every such parameter, in both docstrings:

1. the entry's type names the alias (``filter (CoverageFilter): ...``), not ``(str)``, so a
   reader is pointed at the declared list instead of a copy;
2. a LIST of quoted values in the entry (``('NDVI', 'EVI', 'LAI')``, ``'a', 'b' or 'c'``)
   contains only allowed values. That is the drift that already happened: VegetationTs
   documented six indices while accepting two. A single quoted word elsewhere in a sentence
   (an output column, another parameter's value) is not a list and is not checked.

A list that deliberately names a value that is NOT allowed goes in
``NOT_ALLOWED_BUT_DOCUMENTED`` below, with the reason.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

from earthdaily.agriculture.ai_enablement import generate_ai_context as gen
from earthdaily.agriculture.core.choices import literal_alias

pytestmark = pytest.mark.public

# (class, parameter) -> values the docstring may mention although they are not allowed, and why.
NOT_ALLOWED_BUT_DOCUMENTED: dict[tuple[str, str], set[str]] = {}

_Q = r"(?:``[^`]+``|'[^']+'|\"[^\"]+\")"
# Two or more quoted values joined by commas / "or" / "and" / "/": a value list.
VALUE_LIST = re.compile(rf"{_Q}(?:\s*(?:,\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+|\s*/\s*){_Q})+")
QUOTED = re.compile(r"``([^`]+)``|'([^']+)'|\"([^\"]+)\"")


def _listed_values(text: str) -> list[str]:
    """Quoted values that appear in a list (two or more joined by , / or / and / slash)."""
    return [
        next(g for g in groups if g) for run in VALUE_LIST.finditer(text) for groups in QUOTED.findall(run.group(0))
    ]


def _registry_classes():
    for module_path, class_name, _category in gen.EXTRACTOR_REGISTRY:
        try:
            cls = getattr(importlib.import_module(module_path), class_name)
        except Exception:  # private module absent from the public payload
            continue
        yield cls


def _arg_entry(doc: str, name: str) -> tuple[str, str] | None:
    """``(type, text)`` of a Google-style Args entry, continuation lines joined."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(name)} \(([^)]*)\):(.*)$", line)
        if not m:
            continue
        indent, text = len(m.group(1)), [m.group(3)]
        for nxt in lines[i + 1 :]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) > indent:
                text.append(nxt.strip())
            else:
                break
        return m.group(2), " ".join(text)
    return None


def _cases():
    for cls in _registry_classes():
        setup = gen.get_setup_method(cls)
        if not setup:
            continue
        method = getattr(cls, setup["method_name"])
        hints = gen._type_hints(method)
        for p in setup["parameters"]:
            if not p.get("choices"):
                continue
            alias = literal_alias(hints[p["name"]])
            alias_name = next(
                (n for n, v in vars(inspect.getmodule(cls)).items() if v is alias and not n.startswith("_")), None
            ) or next(
                (n for n in ("MergeMode", "DataSource", "DetailLevel", "ExtractionMode") if n in str(hints[p["name"]])),
                None,
            )
            for where, doc in (("class", inspect.getdoc(cls) or ""), ("setup", inspect.getdoc(method) or "")):
                yield pytest.param(
                    cls.__name__,
                    p["name"],
                    p["choices"],
                    alias_name,
                    where,
                    doc,
                    id=f"{cls.__name__}.{p['name']}[{where}]",
                )


@pytest.mark.parametrize("cls_name, param, choices, alias_name, where, doc", list(_cases()))
def test_docstring_entry_matches_declared_values(cls_name, param, choices, alias_name, where, doc):
    entry = _arg_entry(doc, param)
    if entry is None:
        pytest.skip(f"{param} not documented in the {where} docstring")
    typ, text = entry
    assert alias_name and alias_name in typ, (
        f"{cls_name} {where} docstring: `{param} ({typ})` should name its type `{alias_name}` — the allowed "
        "values live in that Literal; don't copy them into the docstring."
    )
    allowed = {str(c).lower() for c in choices if c is not None}
    exempt = {v.lower() for v in NOT_ALLOWED_BUT_DOCUMENTED.get((cls_name, param), set())}
    listed = _listed_values(text)
    unknown = sorted({m for m in listed if m.lower() not in allowed | exempt})
    assert not unknown, (
        f"{cls_name} {where} docstring for `{param}` mentions {unknown}, which the type does not allow "
        f"(allowed: {list(choices)}). Fix the docstring, or the Literal if the API accepts them."
    )


class TestGateSelfCheck:
    """The detector itself: it must catch the drift it exists for, and nothing else."""

    def test_catches_the_vegetation_ts_drift(self):
        # The pre-2026-10-06 docstring: six indices listed, NDVI and EVI accepted.
        listed = _listed_values("Index type ('NDVI', 'EVI', 'CVI', 'GNDVI', 'NDWI', 'LAI'). Default: 'NDVI'")
        assert {"CVI", "GNDVI", "NDWI", "LAI"} <= set(listed)

    @pytest.mark.parametrize(
        "text",
        [
            "'a', 'b' or 'c'",
            'Supported values: "ndvi", "evi", or "ndvi|evi"',
            "one of ``x`` / ``y``",
        ],
    )
    def test_list_shapes_are_recognized(self, text):
        assert len(_listed_values(text)) >= 2

    def test_single_mentions_are_not_a_list(self):
        text = "the 'mask' output column tells them apart; used with postprocess='file'. Default: 'Auto'"
        assert _listed_values(text) == []


# ---------------------------------------------------------------------------
# (a) every allowed value reaches the generated cards
# ---------------------------------------------------------------------------


def _entries():
    for module_path, class_name, category in gen.EXTRACTOR_REGISTRY:
        entry = gen.introspect_class(module_path, class_name, category)
        if "error" not in entry:  # private module absent from the public payload
            yield pytest.param(entry, id=class_name)


@pytest.mark.parametrize("entry", list(_entries()))
def test_allowed_values_render_in_both_cards(entry):
    setup = entry.get("setup_method") or {}
    params = [p for p in setup.get("parameters", []) if p.get("choices") and p["name"] not in gen.COMMON_SETUP_PARAMS]
    if not params:
        pytest.skip("no Literal-typed setup parameter")
    claude, agents = gen.generate_claude_md([entry]), gen.generate_agents_md([entry])
    for p in params:
        for value in (v for v in p["choices"] if v is not None):
            assert str(value) in claude, f"{entry['class_name']}.{p['name']}={value!r} missing from the CLAUDE.md card"
            assert (
                f"`{value}`" in agents
            ), f"{entry['class_name']}.{p['name']}={value!r} missing from the agents.md card"


# ---------------------------------------------------------------------------
# (b) documented output columns exist
# ---------------------------------------------------------------------------
#
# Two class docstrings listed columns no formatter produces (2026-09-17 audit), and on
# 2026-10-06 eight more were found the same way (Historical score's `score, percentile, rank`,
# ZARC's `risk_level`, ...). A documented column is backed when its name appears as a string
# literal in the extractor's module (or a package module it imports, e.g. Soil's sda_client), or
# quoted in the sample responses of that extractor's tests: formatters that flatten the API JSON
# have no literal for pass-through fields. Prose ("+ change metrics") and patterns
# (`zone_{n}_area`, `offset_<N>`, `emergence_year_1..5`) are not column names and are skipped.

COLUMN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TESTS_DIR = Path(__file__).resolve().parent


def _string_literals(module) -> set[str]:
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return set()
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@pytest.mark.parametrize("entry", list(_entries()))
def test_documented_output_columns_exist(entry):
    columns = [c.strip() for c in entry.get("output_columns") or [] if COLUMN.match(c.strip())]
    if not columns:
        pytest.skip("no column names documented")
    module = importlib.import_module(entry["module"])
    pool = _string_literals(module)
    for obj in vars(module).values():  # package helper modules it imports (not core)
        helper = inspect.getmodule(obj)
        if (
            helper
            and helper is not module
            and helper.__name__.startswith("earthdaily.agriculture.")
            and ".core." not in helper.__name__
        ):
            pool |= _string_literals(helper)
    stem = Path(module.__file__).stem
    key = re.sub(r"processor_|_functions?$|_ts_extractor$", "", stem).split("_")[0].lower()
    tests_text = "".join(p.read_text(encoding="utf-8") for p in TESTS_DIR.glob(f"test_*{key}*.py"))

    def backed(col: str) -> bool:
        word = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(col)}(?![A-Za-z0-9_])")
        return (
            any(len(lit) < 300 and word.search(lit) for lit in pool)
            or re.search(rf"[\"']{re.escape(col)}[\"']", tests_text) is not None
        )

    unbacked = [c for c in columns if not backed(c)]
    assert not unbacked, (
        f"{entry['class_name']} documents output columns {unbacked} that nothing in its code or tests produces. "
        "Fix the 'Output columns:' section of the class docstring."
    )
