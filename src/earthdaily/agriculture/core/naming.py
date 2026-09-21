"""
Filename generator for EarthDaily Agriculture exports (charts, CSVs, reports, etc.).

Builds filenames from a consistent pattern of tokens:

    {prefix}_{input_file}_{analytic}_{chart_type}_{datetime}.{ext}

All tokens are optional — omitted tokens are simply skipped.

Usage:

    from earthdaily.agriculture.core.naming import build_filename

    # Full pattern
    build_filename(input_file="iowa_amus.csv", analytic="NDVI Time Series",
                   chart_type="season_overlay", ext="png")
    # -> "iowa_amus_ndvi_time_series_season_overlay_20260320_143012.png"

    # Minimal — just analytic + datetime
    build_filename(analytic="Emergence", ext="csv")
    # -> "emergence_20260320_143012.csv"

    # With prefix — for multi-report workflows
    build_filename(prefix="agro_terrint", analytic="comparison_report", ext="html")
    # -> "agro_terrint_comparison_report_20260320_143012.html"

    # With custom extra tokens
    build_filename(input_file="regions.shp", analytic="VVI",
                   chart_type="boxplot", extra="prod", ext="html")
    # -> "regions_vvi_boxplot_prod_20260320_143012.html"

    # With a target directory
    build_filename(analytic="Harvest", chart_type="histogram", ext="png",
                   output_dir="results/charts")
    # -> "results/charts/harvest_histogram_20260320_143012.png"
"""

import re
from datetime import datetime
from pathlib import Path


def _slugify(text: str) -> str:
    """Convert a human-readable string to a filename-safe slug."""
    text = str(text).strip()
    # Take stem if it looks like a file path
    if "/" in text or "\\" in text or "." in text:
        text = Path(text).stem
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)  # drop non-alphanumeric
    text = re.sub(r"[\s\-]+", "_", text)  # whitespace/hyphens -> underscore
    text = text.strip("_")
    return text


def build_filename(
    prefix: str = None,
    input_file: str = None,
    analytic: str = None,
    chart_type: str = None,
    extra: str = None,
    ext: str = "png",
    output_dir: str = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    Build a filename from a pattern of tokens.

    Tokens are joined with underscores. Each token is slugified for
    filesystem safety. A timestamp is always appended.

    Args:
        prefix:        Project/run prefix (e.g. "agro_terrint", "north_america").
                       Placed first in the filename for easy sorting/filtering.
        input_file:    Source data filename (e.g. "iowa_amus.csv").
                       The extension is stripped automatically.
        analytic:      Analytic name (e.g. "NDVI Time Series").
        chart_type:    Chart/export type (e.g. "season_overlay", "histogram",
                       "boxplot", "timeseries", "summary").
        extra:         Any additional qualifier (e.g. "prod", entity ID).
        ext:           File extension without dot (default "png").
        output_dir:    Target directory. If provided the returned string
                       is a full path; otherwise just the filename.
        timestamp_fmt: strftime format for the timestamp suffix.

    Returns:
        Filename string (or full path if output_dir is set).
    """
    parts = []
    for token in (prefix, input_file, analytic, chart_type, extra):
        if token is not None:
            slug = _slugify(token)
            if slug:
                parts.append(slug)

    # Timestamp
    parts.append(datetime.now().strftime(timestamp_fmt))

    filename = "_".join(parts)

    # Extension
    ext = ext.lstrip(".")
    filename = f"{filename}.{ext}"

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return str(path / filename)

    return filename
