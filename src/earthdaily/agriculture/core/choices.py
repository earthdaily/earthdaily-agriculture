"""Allowed parameter values, declared once as ``Literal`` types and validated from them.

A parameter's allowed values live in its signature (``filter: CoverageFilter = "none"``), so a
coding agent, an IDE and ``generate_ai_context`` all read the list the validator enforces.
Before this, each extractor kept a ``valid_* = {...}`` set inside the method body and the
docstring repeated it by hand. The two drifted: 8 cases were found on 2026-09-17, and the
generated CLAUDE.md / agents.md copied the wrong lists.

Usage::

    CoverageFilter = Literal["none", "duplicate", "crop_coverage"]

    def setup_coverage_parameters(self, filter: CoverageFilter = "none", ...):
        filter = check_choice("filter", filter, CoverageFilter)

Values the API accepts in any case (``mask="auto"`` → ``"Auto"``) are registered once with
:func:`register_case_insensitive`; :func:`check_choice` then returns the canonical casing.
"""

from __future__ import annotations

import types
from typing import Any, Literal, Union, get_args, get_origin

#: ``merge_existing`` on every bulk method: how results already on disk are combined with new ones.
MergeMode = Literal["auto", "preserve", "mark"]

# Value sets shared by several extractors (identical in the Phase 0 inventory, 2026-10-06).
# An extractor-specific set stays in its own module: sharing is only for lists that are the
# same by meaning, not just by coincidence.

#: ``data_source`` of the season processors: low-resolution or medium-resolution imagery.
DataSource = Literal["LR", "MR"]
#: ``detail_level`` of the historical / in-season score processors.
DetailLevel = Literal["full", "summary"]
#: ``extraction_mode`` of the vegetation time-series extractors.
ExtractionMode = Literal["period", "specific_dates", "windows"]

_CASE_INSENSITIVE: list[Any] = []


def register_case_insensitive(choices: Any) -> None:
    """Mark a ``Literal`` alias as matched case-insensitively by :func:`check_choice`.

    A registry rather than a flag on each call, so ``generate_ai_context`` can tell readers
    which parameters ignore case without reading method bodies.
    """
    if not is_case_insensitive(choices):
        _CASE_INSENSITIVE.append(choices)


def is_case_insensitive(choices: Any) -> bool:
    return any(choices == registered for registered in _CASE_INSENSITIVE)


def _literals_in(annotation: Any) -> list[Any]:
    """Every ``Literal`` an annotation is built from: itself, union members, ``list[...]`` items."""
    origin = get_origin(annotation)
    if origin is Literal:
        return [annotation]
    if origin in (Union, types.UnionType):
        return [lit for arg in get_args(annotation) for lit in _literals_in(arg)]
    if origin is list:
        return [lit for arg in get_args(annotation) for lit in _literals_in(arg)]
    return []


def choice_values(annotation: Any) -> tuple | None:
    """The allowed values of an annotation built from ``Literal`` types, or ``None``.

    Handles ``Literal[...]``, ``Literal[...] | None`` (``None`` means "not set", not a choice
    to list), ``list[Literal[...]]`` and unions of those, e.g. a parameter that takes one value,
    a list of them, or a ``"none"`` sentinel. Values keep their declared order, deduplicated.
    """
    values: list[Any] = []
    for lit in _literals_in(annotation):
        values += [v for v in get_args(lit) if v not in values]
    return tuple(values) or None


def literal_alias(annotation: Any) -> Any:
    """The main ``Literal`` inside an annotation (the first one, unwrapping ``| None`` / ``list``)."""
    literals = _literals_in(annotation)
    return literals[0] if literals else None


def check_choice(name: str, value: Any, choices: Any) -> Any:
    """Return ``value`` in its canonical form, or raise ``ValueError`` naming ``name``.

    The message starts ``Invalid <name>`` so callers and tests can match on the parameter.
    """
    allowed = get_args(choices)
    case_insensitive = is_case_insensitive(choices)
    if case_insensitive and isinstance(value, str):
        canonical = {str(a).lower(): a for a in allowed}.get(value.lower())
        if canonical is not None:
            return canonical
    elif value in allowed:
        return value
    note = " (case-insensitive)" if case_insensitive else ""
    raise ValueError(f"Invalid {name} {value!r}. Must be one of {list(allowed)}{note}.")


def check_choices(name: str, values: Any, choices: Any) -> list:
    """:func:`check_choice` for a list parameter: every item must be allowed.

    Returns the items in canonical form, in input order; the error names every bad item at once.
    """
    good, bad = [], []
    for v in values:
        try:
            good.append(check_choice(name, v, choices))
        except ValueError:
            bad.append(v)
    if bad:
        note = " (case-insensitive)" if is_case_insensitive(choices) else ""
        raise ValueError(f"Invalid {name} {bad!r}. Each must be one of {list(get_args(choices))}{note}.")
    return good
