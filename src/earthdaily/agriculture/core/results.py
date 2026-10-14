"""The dict every ``process_*_bulk_*`` method returns.

All 31 bulk methods return the same seven core keys (Phase 0 inventory, 2026-10-06); this
names them so a caller, a coding agent and the generated docs read one contract instead of
guessing ``result["results_df"]`` vs ``result["df"]``. It documents what the methods return.
It does not change it.

Extractors that return more keys annotate a subclass, so each generated card lists only the
keys that extractor can actually return.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class BulkResult(TypedDict, total=False):
    """Result of a bulk extraction. Core keys are always present; cache keys only with caching on."""

    results_df: pd.DataFrame
    global_errors: list[Any]
    failed_ids: list[Any]
    total_entities: int
    total_calculations: int
    successful_calculations: int
    failed_calculations: int
    cache_hit: int
    cache_miss: int


class SpatialBulkResult(BulkResult, total=False):
    """Weather / GDD / GDD-offset: adds the spatial-grouping counts when grouping is on."""

    representative_calls: int
    spatial_groups: int
    grouped_from: int


class RegionalBulkResult(BulkResult, total=False):
    """Regional: the observed and daily-average series, also returned as separate tables."""

    observed_results_df: pd.DataFrame
    daily_avg_results_df: pd.DataFrame


class SoilBulkResult(BulkResult, total=False):
    """Soil: per-map-unit detail rows when ``return_mapunit_detail=True``."""

    mapunit_detail_df: pd.DataFrame


#: Every result type a bulk method may be annotated with (the generator recognizes these).
RESULT_TYPES: tuple[type, ...] = (BulkResult, SpatialBulkResult, RegionalBulkResult, SoilBulkResult)

#: Keys every bulk method returns. The TypedDicts are ``total=False`` because of the optional
#: keys; these are the ones a caller can rely on.
CORE_KEYS: tuple[str, ...] = (
    "results_df",
    "failed_ids",
    "global_errors",
    "total_entities",
    "total_calculations",
    "successful_calculations",
    "failed_calculations",
)

#: One line per key, rendered into the generated docs. A test keeps it in sync with the
#: union of the result types' keys.
KEY_DOCS: dict[str, str] = {
    "results_df": "the extracted rows (after output formatting)",
    "failed_ids": "ids of entities that errored OR returned no data; what retry_failed_only re-runs",
    "global_errors": "one record per entity error: entity_id plus the error's fields",
    "total_entities": "entities in the input",
    "total_calculations": "entities submitted (after filter_column / filter_value)",
    "successful_calculations": "entities that returned at least one row",
    "failed_calculations": "total_calculations - successful_calculations (errors and empty results)",
    "cache_hit": "entities served from the extraction cache (when caching is on)",
    "cache_miss": "entities fetched from the API (when caching is on)",
    "representative_calls": "API calls made under spatial grouping",
    "spatial_groups": "spatial groups formed (spatial grouping only)",
    "grouped_from": "entities folded into those groups (spatial grouping only)",
    "observed_results_df": "the observed-value rows, as a separate table",
    "daily_avg_results_df": "the daily-average rows, as a separate table",
    "mapunit_detail_df": "per-map-unit rows behind each field's area-weighted values",
}


def result_keys(result_type: type) -> list[str]:
    """The keys of a result type, core keys first, in a stable order."""
    keys = list(result_type.__annotations__)
    return [k for k in CORE_KEYS if k in keys] + [k for k in keys if k not in CORE_KEYS]
