"""
File export utilities.

Export DataFrames to parquet/csv with file descriptors for manifest generation.
Ported from Fertilizer_optimized_placement/app/file_export.py and adapted for earthdaily.agriculture.

Usage:
    from earthdaily.agriculture.export import export_dataframe
"""

import os
from datetime import datetime

import pandas as pd
from loguru import logger


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns with numeric-like data so parquet doesn't choke on string 'NaN'."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            original_non_null = df[col].notna().sum()
            converted_non_null = converted.notna().sum()
            if original_non_null > 0 and converted_non_null / original_non_null > 0.5:
                df[col] = converted
    return df


def export_dataframe(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str,
    formats: list[str] | None = None,
    timestamp: str | None = None,
    exclude_columns: list[str] | None = None,
) -> list[dict]:
    """
    Export a DataFrame to one or more file formats.

    Args:
        df:              DataFrame to export.
        output_dir:      Directory to write files into.
        prefix:          Filename prefix (e.g. "coverage_results").
        formats:         List of formats. Supported: "parquet", "csv". Default: ["parquet"].
        timestamp:       Optional timestamp string for filenames. Auto-generated if None.
        exclude_columns: Columns to drop before export (e.g. ["geometry"]).

    Returns:
        List of file descriptors:
            [{"path": str, "filename": str, "format": str, "rows": int,
              "columns": list, "size_bytes": int}]
    """
    if formats is None:
        formats = ["parquet"]

    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    export_df = df
    if exclude_columns:
        drop_cols = [c for c in exclude_columns if c in df.columns]
        if drop_cols:
            export_df = df.drop(columns=drop_cols)

    exported = []
    for fmt in formats:
        if fmt == "parquet":
            filename = f"{prefix}_{ts}.parquet"
            filepath = os.path.join(output_dir, filename)
            _sanitize_for_parquet(export_df).to_parquet(filepath, index=False)
        elif fmt == "csv":
            filename = f"{prefix}_{ts}.csv"
            filepath = os.path.join(output_dir, filename)
            export_df.to_csv(filepath, index=False)
        else:
            logger.warning(f"Unsupported export format '{fmt}' — skipping")
            continue

        file_size = os.path.getsize(filepath)
        exported.append(
            {
                "path": filepath,
                "filename": filename,
                "format": fmt,
                "rows": len(export_df),
                "columns": list(export_df.columns),
                "size_bytes": file_size,
            }
        )
        logger.info(f"Exported {prefix}.{fmt}: {len(export_df)} rows, {file_size / 1024:.1f} KB")

    return exported
