"""
Data manifest generation and loading.

Generates a structured JSON manifest describing dataset contents:
metadata, structure, completeness, and file references.

Works with DataFrames (CSV/parquet extractions) and xarray Datasets (zarr).

Usage:
    from earthdaily.agriculture.export import generate_manifest, load_manifest, print_manifest
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_manifest(
    source,
    output_path: str | None = None,
    prefix: str = "data",
    entity_col: str = "amu_id",
    date_col: str = "date",
    metadata: dict | None = None,
    file_entries: list[dict] | None = None,
) -> dict:
    """
    Generate a data manifest from a DataFrame or xarray Dataset.

    The manifest captures structure, completeness, and provenance so consumers
    can understand the dataset without loading it.

    Args:
        source:       pd.DataFrame or xr.Dataset to describe.
        output_path:  Directory to save the manifest JSON. None = don't save.
        prefix:       Filename prefix (e.g. "regional_multi_index").
        entity_col:   Column/coordinate used as entity identifier.
        date_col:     Column/coordinate used as date axis.
        metadata:     Optional extra metadata (extraction params, package version, etc.).
        file_entries: Optional list of file descriptors from export_dataframe().

    Returns:
        dict: The manifest content.
    """
    try:
        import xarray as xr

        has_xarray = True
    except ImportError:
        has_xarray = False

    if isinstance(source, pd.DataFrame):
        manifest = _manifest_from_dataframe(source, entity_col, date_col)
    elif has_xarray and isinstance(source, xr.Dataset):
        manifest = _manifest_from_xarray(source, entity_col, date_col)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}. Expected DataFrame or xr.Dataset.")

    # Add header
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
        **manifest,
    }

    if metadata:
        manifest["metadata"] = metadata

    if file_entries:
        manifest["files"] = file_entries
        manifest["file_count"] = len(file_entries)

    # Save
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = os.path.join(output_path, f"{prefix}_manifest_{ts}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)
        manifest["_manifest_path"] = manifest_path
        logger.info(f"Manifest saved: {manifest_path}")

    return manifest


def _manifest_from_dataframe(df: pd.DataFrame, entity_col: str, date_col: str) -> dict:
    """Build manifest sections from a DataFrame."""
    manifest = {
        "source_type": "dataframe",
        "structure": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        },
    }

    # Entities
    if entity_col in df.columns:
        manifest["entities"] = {
            "column": entity_col,
            "count": int(df[entity_col].nunique()),
        }

    # Date range
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            manifest["date_range"] = {
                "column": date_col,
                "min": str(dates.min().date()),
                "max": str(dates.max().date()),
                "unique_dates": int(dates.nunique()),
            }

    # Completeness per column
    completeness = {}
    skip = {entity_col, date_col}
    for col in df.columns:
        if col in skip:
            continue
        total = len(df)
        non_null = int(df[col].notna().sum())
        completeness[col] = {
            "dtype": str(df[col].dtype),
            "non_null": non_null,
            "total": total,
            "coverage_pct": round(non_null / total * 100, 1) if total > 0 else 0,
        }
    manifest["completeness"] = completeness

    return manifest


def _manifest_from_xarray(ds, entity_col: str, date_col: str) -> dict:
    """Build manifest sections from an xarray Dataset."""
    manifest = {
        "source_type": "xarray_dataset",
        "structure": {
            "dimensions": {dim: int(size) for dim, size in ds.dims.items()},
            "coordinates": list(ds.coords),
            "variables": list(ds.data_vars),
        },
    }

    # Entities
    if entity_col in ds.coords:
        manifest["entities"] = {
            "coordinate": entity_col,
            "count": int(len(ds[entity_col])),
        }

    # Date range
    if date_col in ds.coords:
        dates = pd.to_datetime(ds[date_col].values)
        manifest["date_range"] = {
            "coordinate": date_col,
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
            "unique_dates": int(len(dates)),
        }

    # Crop dimension (if present)
    if "crop" in ds.coords:
        manifest["crops"] = list(ds.crop.values)

    # Completeness per variable
    completeness = {}
    for var_name in ds.data_vars:
        arr = ds[var_name]
        total = int(np.prod(arr.shape))
        non_nan = int(arr.count().values)
        completeness[var_name] = {
            "shape": list(arr.shape),
            "dims": list(arr.dims),
            "non_null": non_nan,
            "total": total,
            "coverage_pct": round(non_nan / total * 100, 1) if total > 0 else 0,
        }

        # Per-crop breakdown if crop dimension exists
        if "crop" in arr.dims:
            per_crop = {}
            for crop in ds.crop.values:
                crop_arr = arr.sel(crop=crop)
                crop_total = int(np.prod(crop_arr.shape))
                crop_nn = int(crop_arr.count().values)
                per_crop[str(crop)] = {
                    "non_null": crop_nn,
                    "total": crop_total,
                    "coverage_pct": round(crop_nn / crop_total * 100, 1) if crop_total > 0 else 0,
                }
            completeness[var_name]["per_crop"] = per_crop

    manifest["completeness"] = completeness

    return manifest


def load_manifest(path: str | Path) -> dict:
    """
    Load a manifest from a JSON file.

    Args:
        path: Path to the manifest JSON.

    Returns:
        dict: Manifest content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_manifest(manifest: dict) -> None:
    """
    Print a formatted summary of a manifest.

    Args:
        manifest: Manifest dict (from generate_manifest or load_manifest).
    """
    sep = "=" * 60

    print(sep)
    print(f"Data Manifest — {manifest.get('prefix', '?')}")
    print(f"Generated: {manifest.get('generated_at', '?')}")
    print(f"Source type: {manifest.get('source_type', '?')}")
    print(sep)

    # Structure
    structure = manifest.get("structure", {})
    if "dimensions" in structure:
        print(f"Dimensions: {structure['dimensions']}")
        print(f"Variables: {structure.get('variables', [])}")
    else:
        print(f"Shape: {structure.get('rows', '?')} rows x {structure.get('columns', '?')} columns")
        print(f"Columns: {structure.get('column_names', [])}")

    # Entities
    entities = manifest.get("entities", {})
    if entities:
        key = entities.get("column") or entities.get("coordinate", "?")
        print(f"\nEntities ({key}): {entities.get('count', '?')}")

    # Date range
    dr = manifest.get("date_range", {})
    if dr:
        print(f"Date range: {dr.get('min', '?')} to {dr.get('max', '?')} ({dr.get('unique_dates', '?')} dates)")

    # Crops
    crops = manifest.get("crops")
    if crops:
        print(f"Crops: {crops}")

    # Completeness
    completeness = manifest.get("completeness", {})
    if completeness:
        print(f"\n{sep}")
        print("Completeness:")
        for var, info in completeness.items():
            nn = info.get("non_null", 0)
            total = info.get("total", 0)
            pct = info.get("coverage_pct", 0)
            print(f"  {var:25s}  {nn:>10,} / {total:>10,}  ({pct:.1f}%)")

            per_crop = info.get("per_crop", {})
            if per_crop:
                for crop, ci in per_crop.items():
                    c_nn = ci.get("non_null", 0)
                    c_total = ci.get("total", 0)
                    c_pct = ci.get("coverage_pct", 0)
                    print(f"    {crop:23s}  {c_nn:>10,} / {c_total:>10,}  ({c_pct:.1f}%)")

    # Files
    files = manifest.get("files", [])
    if files:
        print(f"\n{sep}")
        print(f"Files ({len(files)}):")
        for f in files:
            size_kb = f.get("size_bytes", 0) / 1024
            line = f"  {f.get('filename', '?'):40s}  {f.get('rows', '?'):>8} rows  {size_kb:>8.1f} KB"
            cloud_upload = f.get("cloud_upload")
            if cloud_upload:
                line += f"  → {cloud_upload.get('uri', '?')}"
            print(line)

    # Cloud publish summary (added by earthdaily.agriculture.export.cloud_publish.publish_local_run_to_cloud)
    cloud_publish = manifest.get("cloud_publish")
    if cloud_publish:
        print(f"\n{sep}")
        print("Cloud publish:")
        print(f"  bucket          : {cloud_publish.get('bucket', '?')}")
        print(f"  publish_prefix  : {cloud_publish.get('publish_prefix', '?')}")
        print(
            f"  uploaded / skipped / total : "
            f"{cloud_publish.get('uploaded_files', '?')} / "
            f"{cloud_publish.get('skipped_files', '?')} / "
            f"{cloud_publish.get('total_files', '?')}"
        )
        print(f"  published_at    : {cloud_publish.get('published_at', '?')}")

    # Metadata
    meta = manifest.get("metadata", {})
    if meta:
        print(f"\n{sep}")
        print("Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    print(sep)
