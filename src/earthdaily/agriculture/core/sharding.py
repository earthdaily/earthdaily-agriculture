"""Sharded stateless runs: split -> run one shard per container -> retry failed -> merge.

The pieces a scheduler needs around an extractor that runs in a stateless container.
Everything goes through :mod:`earthdaily.agriculture.core._fs`, so every path may be
local or a remote URI (``s3://``, ``az://``).

Layout under one run's base prefix::

    <base>/shards.json                    the manifest written by split_entities()
    <base>/input/shard-<i>.parquet        each shard's entities
    <base>/shard-<i>/results/...          each shard's outputs (its EDAGRO_OUTPUT_PREFIX)
    <base>/merged/results/...             merge_shards() output

A shard is retried in two ways: a container that crashes is re-run whole by the
scheduler, and entities that failed inside a finished shard are re-run with
``retry_failed_only``. Both leave several ``*_final`` files in a shard's results, so
:func:`merge_shards` keeps, for each entity, the rows from the newest file that has it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd

from earthdaily.agriculture.core._fs import (
    ensure_dir,
    glob_files,
    is_remote_path,
    join_path,
    storage_options_for,
    write_bytes,
)

MANIFEST_NAME = "shards.json"


# --------------------------------------------------------------------------------------
# Tables (entity lists and results), local or remote
# --------------------------------------------------------------------------------------


def read_table(uri: str) -> pd.DataFrame:
    """Read a ``.parquet`` or ``.csv`` table from a local path or remote URI."""
    so = storage_options_for(uri)
    if str(uri).lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(uri, storage_options=so)
    return pd.read_csv(uri, storage_options=so)


def write_table(df: pd.DataFrame, uri: str) -> str:
    """Write *df* as parquet (``.parquet``) or csv to a local path or remote URI."""
    if not is_remote_path(uri):
        ensure_dir(os.path.dirname(str(uri)) or ".")
    so = storage_options_for(uri)
    if str(uri).lower().endswith((".parquet", ".pq")):
        df.to_parquet(uri, index=False, storage_options=so)
    else:
        df.to_csv(uri, index=False, storage_options=so)
    return uri


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


def select_shard(entities: pd.DataFrame, shard: int, num_shards: int) -> pd.DataFrame:
    """Every ``num_shards``-th row starting at ``shard`` — by position, never by label.

    A label-based ``index % n`` raised on a string index (a CSV read with ``index_col``)
    and silently skewed shards on a non-unique one.
    """
    if num_shards is None or num_shards <= 1:
        return entities.reset_index(drop=True)
    if shard is None or not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard!r}")
    return entities.iloc[shard::num_shards].reset_index(drop=True)


def plan_num_shards(
    n_entities: int,
    entities_per_minute: float,
    target_minutes: float = 15.0,
    overhead_seconds: float = 5.0,
    headroom: float = 0.8,
) -> int:
    """Number of shards so each container finishes inside ``target_minutes``.

    ``entities_per_minute`` is one container's measured throughput for the extractor
    (at its ``max_workers``). ``headroom`` keeps slack for slow API periods; the
    container's fixed start-up cost (imports, token) comes off the budget first.
    """
    if n_entities <= 0:
        return 1
    if entities_per_minute <= 0 or target_minutes <= 0:
        raise ValueError("entities_per_minute and target_minutes must be positive")
    budget_minutes = max(target_minutes - overhead_seconds / 60.0, 0.0) * headroom
    per_shard = max(int(entities_per_minute * budget_minutes), 1)
    return max(math.ceil(n_entities / per_shard), 1)


def shard_prefix(base: str, shard: int) -> str:
    """The ``EDAGRO_OUTPUT_PREFIX`` a shard's container should use."""
    return join_path(base, f"shard-{shard}")


def split_entities(entities: pd.DataFrame | str, base: str, num_shards: int) -> dict[str, Any]:
    """Write one entity file per shard plus ``<base>/shards.json``; return the manifest.

    Each shard gets ``<base>/input/shard-<i>.parquet`` — a container reads only its own
    slice instead of the full list. Empty shards (more shards than entities) are not
    written and not listed.
    """
    df = read_table(entities) if isinstance(entities, str) else entities
    shards = []
    for i in range(num_shards):
        part = select_shard(df, i, num_shards)
        if part.empty:
            continue
        uri = join_path(base, "input", f"shard-{i}.parquet")
        write_table(part, uri)
        shards.append({"shard": i, "input": uri, "n_entities": int(len(part)), "output_prefix": shard_prefix(base, i)})
    manifest = {"base": base, "num_shards": num_shards, "n_entities": int(len(df)), "shards": shards}
    write_bytes(join_path(base, MANIFEST_NAME), json.dumps(manifest, indent=2).encode("utf-8"))
    return manifest


def read_manifest(base_or_uri: str) -> dict[str, Any]:
    """Load ``shards.json`` from a base prefix or a direct URI."""
    uri = base_or_uri if str(base_or_uri).endswith(".json") else join_path(base_or_uri, MANIFEST_NAME)
    if not is_remote_path(uri):
        with open(uri, encoding="utf-8") as fh:
            return json.load(fh)
    import fsspec  # lazy: only remote reads need it (it ships with the [s3] / [azure] extras)

    with fsspec.open(uri, "r", encoding="utf-8", **(storage_options_for(uri) or {})) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


def find_final_results(results_dir: str, prefix: str | None = None) -> list[str]:
    """Every ``*_final.{csv,parquet}`` in *results_dir*, newest first.

    Prefers files whose name contains *prefix*; falls back to any ``*_final`` file.
    Filenames carry a sortable timestamp (``..._YYYYmmdd_HHMMSS_ffffff_final.ext``).
    """
    patterns = ([f"*{prefix}*_final"] if prefix else []) + ["*_final"]
    for pattern in patterns:
        matches = [m for ext in ("csv", "parquet") for m in glob_files(join_path(results_dir, f"{pattern}.{ext}"))]
        if matches:
            return sorted(set(matches), reverse=True)
    return []


def find_report(results_dir: str, prefix: str | None = None) -> str | None:
    """The newest ``*_final`` result in *results_dir* — what ``REPORT_PATH=`` points at."""
    finals = find_final_results(results_dir, prefix)
    return finals[0] if finals else None


def _entity_key(df: pd.DataFrame) -> str | None:
    for col in ("entity_id", "id"):
        if col in df.columns:
            return col
    return None


def combine_finals(paths: list[str]) -> pd.DataFrame:
    """Combine one shard's ``*_final`` files (newest first) without duplicating entities.

    For each entity, the rows come from the newest file that contains it: a retry pass
    (only the failed entities) adds what the first pass lacked, and a re-run shard
    replaces the older run's rows instead of doubling them. Tables without an entity
    column are simply concatenated.
    """
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for path in paths:
        df = read_table(path)
        key = _entity_key(df)
        if key is None:
            frames.append(df)
            continue
        fresh = df[~df[key].astype(str).isin(seen)]
        seen.update(df[key].astype(str).unique())
        frames.append(fresh)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_shards(
    base: str,
    num_shards: int | None = None,
    prefix: str | None = None,
    require_all: bool = False,
    export_format: str | None = None,
) -> dict[str, Any]:
    """Merge every shard's results under *base* into ``<base>/merged/results``.

    ``num_shards`` defaults to the manifest's. Shards with no ``*_final`` are reported
    (``missing``); ``require_all=True`` raises instead of publishing a partial merge.

    Returns ``{"report_path", "rows", "shards_merged", "missing"}``.
    """
    from earthdaily.agriculture.core.api_utils import export_results

    # The manifest lists only the shards that were written (empty ones are skipped).
    expected = [s["shard"] for s in read_manifest(base)["shards"]] if num_shards is None else list(range(num_shards))

    frames, missing = [], []
    for i in expected:
        finals = find_final_results(join_path(shard_prefix(base, i), "results"), prefix)
        if not finals:
            missing.append(i)
            continue
        frames.append(combine_finals(finals))

    if missing and require_all:
        raise RuntimeError(f"{len(missing)}/{len(expected)} shard(s) produced no results: {missing}")
    if not frames:
        raise RuntimeError("no shard produced results — nothing to merge")

    merged = pd.concat(frames, ignore_index=True)
    out_prefix = f"{prefix}_merged" if prefix else "merged"
    report, _ = export_results(
        merged, [], join_path(base, "merged", "results"), prefix=out_prefix, verbose=False, export_format=export_format
    )
    return {"report_path": report, "rows": int(len(merged)), "shards_merged": len(frames), "missing": missing}
