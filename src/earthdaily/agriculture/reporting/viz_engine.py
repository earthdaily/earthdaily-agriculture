"""
Generic YAML-driven visualization engine for EarthDaily Agriculture analytics.

Usage:
    from earthdaily.agriculture.reporting.viz_engine import (kpi_summary, kpi_groupby, distribution_chart,
        per_entity_chart, entity_chart, timeseries_chart, build_cross_summary,
        choropleth_map, column_stats, print_column_stats, grouped_stats,
        crosstab_chart, scatter_comparison, comparison_chart,
        generate_viz_config, load_viz_config)

    # Season overlay for a single entity — config from YAML
    timeseries_chart(df, col_specs, mode="season", entity_id="abc123",
                     season_config=viz_cfg["analytics"]["NDVI Time Series"]["season"])

    # Override YAML season config at call time
    timeseries_chart(df, col_specs, mode="season", entity_id="abc123",
                     season_config=season_cfg, season_start="15/05", season_duration=150)

Functions read column specs from a visualization YAML config and render
Plotly charts + KPI text summaries for any analytic DataFrame.

Column spec fields (from YAML):
    name         – column name in the DataFrame
    label        – display label (optional, derived from name if omitted)
    type         – numeric | date | categorical | timeseries
    agg          – aggregation for multi-row data (mean, max, min, last)
    color_scale  – threshold list [{above: <val>, color: "<hex>"}, ...] for per-entity bars
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def prepare_df(df, col_specs, entity_col, date_col):
    """Coerce column types and aggregate multi-row-per-entity data if needed."""
    df = df.copy()
    needs_agg = any(c.get("agg") for c in col_specs)

    for spec in col_specs:
        col = spec["name"]
        if col not in df.columns:
            continue
        if spec["type"] in ("numeric", "timeseries"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif spec["type"] == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if needs_agg and entity_col in df.columns:
        agg_map = {spec["name"]: spec["agg"] for spec in col_specs if spec.get("agg") and spec["name"] in df.columns}
        if agg_map:
            return df.groupby(entity_col).agg(agg_map).reset_index(), True

    return df, False


def resolve_color(value, color_scale):
    """Resolve color from a threshold list (evaluated top-down)."""
    if not color_scale:
        return "#2196F3"
    for rule in color_scale:
        if value >= rule["above"]:
            return rule["color"]
    return "#2196F3"


# ══════════════════════════════════════════════════════════════════════════════
#  KPI SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def kpi_summary(df, col_specs, entity_col="id", date_col="date"):
    """Print a text KPI summary for each column spec."""
    prepared, _ = prepare_df(df, col_specs, entity_col, date_col)
    n_entities = df[entity_col].nunique() if entity_col in df.columns else len(df)

    print(f"KPI Summary — {n_entities} entities")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            print(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print("=" * 70)

    for spec in col_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())

        if col not in prepared.columns:
            print(f"\n  {label}: column not found")
            continue

        col_type = spec["type"]

        if col_type in ("numeric", "timeseries"):
            v = pd.to_numeric(prepared[col], errors="coerce").dropna()
            if len(v) == 0:
                print(f"\n  {label}: no data")
                continue
            print(f"\n  {label}:")
            print(f"    Mean: {v.mean():.4f}  |  Median: {v.median():.4f}  |  Std: {v.std():.4f}")
            print(f"    Min:  {v.min():.4f}  |  Max:    {v.max():.4f}  |  Count: {len(v)}")

        elif col_type == "date":
            d = pd.to_datetime(prepared[col], errors="coerce")
            valid = d.dropna()
            total = len(prepared)
            print(f"\n  {label}:")
            print(f"    Detected: {len(valid)}/{total} ({len(valid) / total * 100:.1f}%)")
            if len(valid) > 0:
                print(f"    Earliest: {valid.min().strftime('%Y-%m-%d')}")
                print(f"    Latest:   {valid.max().strftime('%Y-%m-%d')}")
                print(f"    Median:   {valid.median().strftime('%Y-%m-%d')}")

        elif col_type == "categorical":
            print(f"\n  {label}:")
            print(prepared[col].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTION CHART — numeric, date, categorical columns
# ══════════════════════════════════════════════════════════════════════════════


def distribution_chart(df, col_specs, entity_col="id", date_col="date", env=""):
    """
    Render distribution charts for non-timeseries columns.

    Handles:
      - numeric:     histogram across entities
      - date:        histogram by day-of-year
      - categorical: value-counts bar chart

    Timeseries columns are skipped — use timeseries_chart() for those.
    For per-entity bars, use per_entity_chart().
    """
    # Filter out timeseries specs
    specs = [s for s in col_specs if s["type"] != "timeseries"]
    if not specs:
        print("No distribution columns configured (all columns are timeseries).")
        return

    # Aggregate multi-row data if needed
    prepared, _ = prepare_df(df, specs, entity_col, date_col)

    # ── Numeric: histogram ─────────────────────────────────────────────────────
    numeric_specs = [s for s in specs if s["type"] == "numeric"]
    for spec in numeric_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        v = pd.to_numeric(prepared[col], errors="coerce").dropna()
        if len(v) == 0:
            continue

        stats_text = f"Mean: {v.mean():.4f} | Median: {v.median():.4f} | Std: {v.std():.4f} | n={len(v)}"

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=v,
                nbinsx=spec.get("nbins", max(5, len(v) // 3)),
                marker_color="#2196F3",
                marker_line_color="black",
                marker_line_width=1,
                opacity=0.75,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"Distribution of {label}", font_size=14),
            xaxis_title=label,
            yaxis_title="Count",
            template="plotly_white",
            height=400,
            annotations=[
                dict(
                    text=stats_text,
                    xref="paper",
                    yref="paper",
                    x=0.98,
                    y=0.95,
                    showarrow=False,
                    font=dict(size=11, color="#475569"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0",
                    borderwidth=1,
                    borderpad=4,
                )
            ],
        )
        fig.show()

    # ── Date: histogram ────────────────────────────────────────────────────────
    date_specs = [s for s in specs if s["type"] == "date"]
    for spec in date_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        d = pd.to_datetime(prepared[col], errors="coerce")
        valid = prepared[d.notna()].copy()
        valid[col] = d[d.notna()]
        if len(valid) == 0:
            print(f"No {label} dates to plot.")
            continue

        fig = px.histogram(
            valid,
            x=valid[col].dt.dayofyear,
            nbins=15,
            title=f"{label} Distribution (Day of Year)",
            labels={"x": "Day of Year"},
            color_discrete_sequence=["#4CAF50"],
        )
        fig.update_layout(height=400)
        fig.show()

    # ── Categorical: value counts bar chart ───────────────────────────────────
    cat_specs = [s for s in specs if s["type"] == "categorical"]
    for spec in cat_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        counts = prepared[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(
            counts, x=col, y="count", color=col, title=f"{label} Breakdown", labels={col: label, "count": "Count"}
        )
        fig.update_layout(height=400)
        fig.show()


def per_entity_chart(df, col_specs, entity_col="id", date_col="date", env=""):
    """
    Render per-entity horizontal bar charts for non-timeseries columns.

    Handles:
      - numeric: per-entity horizontal bar sorted by value
      - date:    per-entity bar by day-of-year

    Timeseries and categorical columns are skipped.
    For distribution histograms, use distribution_chart().
    """
    specs = [s for s in col_specs if s["type"] != "timeseries"]
    if not specs:
        return

    prepared, _ = prepare_df(df, specs, entity_col, date_col)

    # ── Numeric: per-entity bar ────────────────────────────────────────────────
    numeric_specs = [s for s in specs if s["type"] == "numeric"]
    for spec in numeric_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        color_scale = spec.get("color_scale")
        sorted_df = prepared.dropna(subset=[col]).sort_values(col)
        if len(sorted_df) == 0:
            continue
        colors = [resolve_color(v, color_scale) for v in sorted_df[col]]
        fig = go.Figure(
            go.Bar(
                x=sorted_df[col],
                y=sorted_df[entity_col],
                orientation="h",
                marker_color=colors,
                text=sorted_df[col].round(2),
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"{label} per Entity",
            xaxis_title=label,
            yaxis=dict(tickfont=dict(size=8)),
            height=max(400, len(sorted_df) * 25),
        )
        fig.show()

    # ── Date: per-entity bar ───────────────────────────────────────────────────
    date_specs = [s for s in specs if s["type"] == "date"]
    for spec in date_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        d = pd.to_datetime(prepared[col], errors="coerce")
        valid = prepared[d.notna()].copy()
        valid[col] = d[d.notna()]
        if len(valid) == 0:
            continue

        sorted_df = valid.sort_values(col)
        fig = go.Figure(
            go.Bar(
                x=sorted_df[col].dt.dayofyear,
                y=sorted_df[entity_col],
                orientation="h",
                marker_color="#4CAF50",
                text=sorted_df[col].dt.strftime("%Y-%m-%d"),
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"{label} per Entity",
            xaxis_title="Day of Year",
            yaxis=dict(tickfont=dict(size=8)),
            height=max(400, len(sorted_df) * 25),
        )
        fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTITY BAR CHART — per-entity comparison of selected properties
# ══════════════════════════════════════════════════════════════════════════════


def entity_chart(df, col_specs, entity_col="id", name_col=None, date_col="date"):
    """
    Render a horizontal grouped bar chart showing selected properties per entity.

    Each numeric column in col_specs becomes a bar group. Entities are shown on
    the y-axis (using name_col for labels if provided, otherwise entity_col).

    Args:
        df:         DataFrame with one or more rows per entity.
        col_specs:  List of column spec dicts (only numeric/timeseries are plotted).
        entity_col: Column used to identify entities.
        name_col:   Optional column for display labels (falls back to entity_col).
        date_col:   Date column (used for aggregation if multi-row per entity).
    """
    numeric_specs = [s for s in col_specs if s["type"] in ("numeric", "timeseries")]
    if not numeric_specs:
        print("No numeric columns configured for entity chart.")
        return

    prepared, _ = prepare_df(df, col_specs, entity_col, date_col)

    # Resolve display labels
    label_col = name_col if name_col and name_col in prepared.columns else entity_col
    if label_col not in prepared.columns:
        print(f"Column '{label_col}' not found in DataFrame.")
        return

    labels = prepared[label_col].astype(str).tolist()

    # Single property → simple horizontal bar
    if len(numeric_specs) == 1:
        spec = numeric_specs[0]
        col = spec["name"]
        col_label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            print(f"Column '{col}' not found in DataFrame.")
            return

        sorted_df = prepared.dropna(subset=[col]).sort_values(col)
        sorted_labels = sorted_df[label_col].astype(str).tolist()
        values = sorted_df[col].tolist()
        color_scale = spec.get("color_scale")
        colors = [resolve_color(v, color_scale) for v in values]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=sorted_labels,
                orientation="h",
                marker_color=colors,
                marker_line_color="black",
                marker_line_width=0.5,
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"{col_label} per Entity", font_size=14),
            xaxis_title=col_label,
            yaxis=dict(tickfont=dict(size=8)),
            template="plotly_white",
            height=max(400, len(sorted_df) * 25),
        )
        fig.show()
        return

    # Multiple properties → grouped horizontal bar
    fig = go.Figure()
    for spec in numeric_specs:
        col = spec["name"]
        col_label = spec.get("label", col.replace("_", " ").title())
        if col not in prepared.columns:
            continue
        values = pd.to_numeric(prepared[col], errors="coerce").tolist()
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                name=col_label,
                text=[f"{v:.2f}" if pd.notna(v) else "" for v in values],
                textposition="auto",
                hovertemplate="%{y}: %{x:.4f}<extra>" + col_label + "</extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        title=dict(text="Entity Comparison", font_size=14),
        yaxis=dict(tickfont=dict(size=8)),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=max(400, len(prepared) * 25 * len(numeric_specs)),
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  TIME SERIES CHART — requires a date column and a value column
# ══════════════════════════════════════════════════════════════════════════════


def timeseries_chart(
    df,
    col_specs,
    entity_col="id",
    date_col="date",
    env="",
    mode="all",
    aggregation="mean",
    entity_id=None,
    season_config=None,
    season_start=None,
    season_end=None,
    season_duration=None,
):
    """
    Render time series charts for columns with type=timeseries.

    Args:
        mode: "all" to show individual entity lines + box plot,
              "aggregation" to show central tendency +/- std band,
              "season" to slice one entity's time series into growing seasons
                       and overlay them on a common day/month x-axis.
        aggregation: Central tendency for aggregation mode — "mean" or "median".
        entity_id: (season mode only) Entity ID to display.
        season_config: (season mode only) Dict with season slicing parameters,
                       typically read from the visualization YAML:
                         season_start (str, required): "DD/MM" e.g. "01/04" for April 1st
                         season_end   (str):           "DD/MM" e.g. "01/10" for October 1st
                         season_duration (int):         alternative to season_end — length in days
        season_start:    Override season_config.season_start at call time ("DD/MM").
        season_end:      Override season_config.season_end at call time ("DD/MM").
        season_duration: Override season_config.season_duration at call time (days).

    Non-timeseries columns are skipped — use distribution_chart() for those.
    """
    ts_specs = [s for s in col_specs if s["type"] == "timeseries"]
    if not ts_specs:
        print("No timeseries columns configured for this analytic.")
        return

    for spec in ts_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            continue

        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts[col] = pd.to_numeric(ts[col], errors="coerce")

        if mode == "all":
            _timeseries_all(ts, col, label, entity_col, date_col)
        elif mode == "aggregation":
            _timeseries_aggregation(ts, col, label, entity_col, date_col, aggregation)
        elif mode == "season":
            if entity_id is None:
                print("Season mode requires entity_id. Pass entity_id=<id>.")
                return
            # Merge: YAML config as base, call-time kwargs as overrides
            cfg = dict(season_config or {})
            if season_start is not None:
                cfg["season_start"] = season_start
            if season_end is not None:
                cfg["season_end"] = season_end
            if season_duration is not None:
                cfg["season_duration"] = season_duration
            resolved_start = cfg.get("season_start")
            resolved_end = cfg.get("season_end")
            resolved_duration = cfg.get("season_duration")
            if resolved_start is None:
                print('Season mode requires season_start ("DD/MM") in season_config or as argument.')
                return
            if resolved_end is None and resolved_duration is None:
                print('Season mode requires season_end ("DD/MM") or season_duration (days).')
                return
            _timeseries_season(
                ts,
                col,
                label,
                entity_col,
                date_col,
                entity_id,
                resolved_start,
                resolved_end,
                resolved_duration,
                aggregation,
            )
        else:
            print(f"Unknown mode '{mode}'. Use 'all', 'aggregation', or 'season'.")


def _timeseries_all(ts, col, label, entity_col, date_col):
    """Mode 'all': one line per entity."""
    fig = px.line(
        ts,
        x=date_col,
        y=col,
        color=entity_col,
        title=f"{label} Time Series — All Entities",
        labels={col: label, date_col: "Date"},
    )
    fig.update_layout(height=500, showlegend=True, legend_title_text="Entity")
    fig.show()


def _timeseries_aggregation(ts, col, label, entity_col, date_col, aggregation="mean"):
    """Mode 'aggregation': central tendency +/- std over time."""
    agg_label = aggregation.capitalize()

    # Group by date across all entities
    grouped = ts.groupby(date_col)[col]
    if aggregation == "median":
        center = grouped.median().reset_index(name="center")
    else:
        center = grouped.mean().reset_index(name="center")
    std = grouped.std().reset_index(name="std")
    agg_df = center.merge(std, on=date_col).sort_values(date_col)
    agg_df["std"] = agg_df["std"].fillna(0)

    n_entities = ts[entity_col].nunique()

    fig = go.Figure()

    # +/- std band
    fig.add_trace(
        go.Scatter(
            x=list(agg_df[date_col]) + list(agg_df[date_col][::-1]),
            y=list(agg_df["center"] + agg_df["std"]) + list((agg_df["center"] - agg_df["std"])[::-1]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line=dict(width=0),
            name="+/- Std",
            hoverinfo="skip",
        )
    )

    # Central tendency line
    fig.add_trace(
        go.Scatter(
            x=agg_df[date_col],
            y=agg_df["center"],
            mode="lines+markers",
            name=agg_label,
            line=dict(color="#2196F3", width=2),
            marker=dict(size=4),
            hovertemplate=f"Date: %{{x}}<br>{agg_label}: %{{y:.4f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{label} — {agg_label} +/- Std ({n_entities} entities)",
        xaxis_title="Date",
        yaxis_title=label,
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  SEASON MODE — slice one entity into growing seasons and overlay
# ══════════════════════════════════════════════════════════════════════════════


def _parse_ddmm(ddmm):
    """Parse a 'DD/MM' string into (day, month) integers."""
    parts = ddmm.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Expected 'DD/MM', got '{ddmm}'")
    return int(parts[0]), int(parts[1])


# Reference year for the common x-axis (leap year so Feb 29 works)
_REF_YEAR = 2000


def _timeseries_season(
    ts, col, label, entity_col, date_col, entity_id, season_start, season_end, season_duration, aggregation="mean"
):
    """
    Mode 'season': slice one entity's time series into yearly windows and
    overlay them on a common day/month x-axis.

    Each year becomes a separate trace in the Plotly chart (clickable legend).
    An additional 'Average' trace shows the mean/median across all years.

    The x-axis displays calendar day/month so the chart is immediately readable.

    Args:
        season_start:    Season start as "DD/MM" (e.g. "01/04" for April 1st).
        season_end:      Season end as "DD/MM" (e.g. "01/10" for October 1st).
                         Mutually exclusive with season_duration (end takes priority).
        season_duration: Alternative to season_end — season length in days.
    """
    from datetime import date as dt_date

    start_day, start_month = _parse_ddmm(season_start)

    # Resolve season end date (day, month) from season_end or season_duration
    if season_end is not None:
        end_day, end_month = _parse_ddmm(season_end)
    else:
        # Compute end day/month from duration using the reference year
        ref_start = pd.Timestamp(dt_date(_REF_YEAR, start_month, start_day))
        ref_end = ref_start + pd.Timedelta(days=int(season_duration))
        end_day, end_month = ref_end.day, ref_end.month

    # Filter to the selected entity
    entity_ts = ts[ts[entity_col] == entity_id].copy()
    if entity_ts.empty:
        print(f"No data found for entity '{entity_id}'.")
        return

    entity_ts = entity_ts.dropna(subset=[date_col, col]).sort_values(date_col)

    # Determine year range
    min_year = entity_ts[date_col].dt.year.min()
    max_year = entity_ts[date_col].dt.year.max()

    seasons = {}
    for year in range(min_year, max_year + 1):
        try:
            s_start = pd.Timestamp(dt_date(year, start_month, start_day))
            s_end = pd.Timestamp(dt_date(year, end_month, end_day))
        except ValueError:
            continue

        if s_end <= s_start:
            # Cross-year season (e.g. Oct → Mar): extend end to next year
            try:
                s_end = pd.Timestamp(dt_date(year + 1, end_month, end_day))
            except ValueError:
                continue

        mask = (entity_ts[date_col] >= s_start) & (entity_ts[date_col] < s_end)
        season_data = entity_ts[mask].copy()

        if season_data.empty:
            continue

        # Map real dates to the reference year for alignment on day/month axis
        season_data["_ref_date"] = season_data[date_col].apply(
            lambda d: (
                d.replace(year=_REF_YEAR) if not (d.month == 2 and d.day == 29) else d.replace(year=_REF_YEAR, day=28)
            )
        )

        season_label = f"{year}/{year + 1}" if start_month >= 7 else str(year)
        seasons[season_label] = season_data

    if not seasons:
        end_str = season_end or f"{season_duration}d from start"
        print(f"No seasons found for entity '{entity_id}' with start={season_start}, end={end_str}.")
        return

    # Build the Plotly figure
    fig = go.Figure()

    # Collect all values aligned by reference date for averaging
    all_ref_values = {}

    for season_label, sdf in sorted(seasons.items()):
        sdf_sorted = sdf.sort_values("_ref_date")
        fig.add_trace(
            go.Scatter(
                x=sdf_sorted["_ref_date"],
                y=sdf_sorted[col],
                mode="lines+markers",
                name=season_label,
                marker=dict(size=4),
                line=dict(width=1.5),
                hovertemplate=(
                    f"Season: {season_label}<br>Date: %{{customdata}}<br>{label}: %{{y:.4f}}<extra></extra>"
                ),
                customdata=sdf_sorted[date_col].dt.strftime("%Y-%m-%d"),
            )
        )

        for _, row in sdf_sorted.iterrows():
            ref_d = row["_ref_date"]
            all_ref_values.setdefault(ref_d, []).append(row[col])

    # Compute average series
    if all_ref_values:
        avg_dates = sorted(all_ref_values.keys())
        if aggregation == "median":
            avg_values = [np.median(all_ref_values[d]) for d in avg_dates]
            avg_label = "Median"
        else:
            avg_values = [np.mean(all_ref_values[d]) for d in avg_dates]
            avg_label = "Mean"

        fig.add_trace(
            go.Scatter(
                x=avg_dates,
                y=avg_values,
                mode="lines",
                name=f"{avg_label} ({len(seasons)} seasons)",
                line=dict(color="black", width=3, dash="dash"),
                hovertemplate=(f"{avg_label}<br>{label}: %{{y:.4f}}<extra></extra>"),
            )
        )

    end_display = season_end or f"+{season_duration}d"
    fig.update_layout(
        title=(
            f"{label} — Season Overlay — Entity: {entity_id}<br>"
            f"<sup>Window: {season_start} to {end_display}, "
            f"{len(seasons)} season(s)</sup>"
        ),
        xaxis_title="Day / Month",
        xaxis=dict(tickformat="%d %b"),
        yaxis_title=label,
        height=550,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        hovermode="x unified",
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-ANALYTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def build_cross_summary(analytics_dict, entities_df, viz_analytics_cfg):
    """Build an entity-level summary table by aggregating one KPI per analytic column."""
    s = entities_df[["id", "crop", "area_ha", "size_class"]].copy()

    for name, df in analytics_dict.items():
        if df is None or name not in viz_analytics_cfg:
            continue
        cfg = viz_analytics_cfg[name]
        entity_col = cfg.get("entity_col", "id")

        for spec in cfg.get("columns", []):
            col = spec["name"]
            if col not in df.columns:
                continue

            col_label = f"{name}_{col}"

            if spec["type"] in ("numeric", "timeseries"):
                agg_fn = spec.get("agg", "mean")
                agg_df = df.groupby(entity_col)[col].agg(agg_fn).reset_index()
                agg_df.columns = [entity_col, col_label]

            elif spec["type"] == "date":
                tmp = df.copy()
                tmp[col] = pd.to_datetime(tmp[col], errors="coerce")
                agg_df = tmp.groupby(entity_col)[col].first().reset_index()
                agg_df.columns = [entity_col, col_label]

            elif spec["type"] == "categorical":
                agg_df = df.groupby(entity_col)[col].first().reset_index()
                agg_df.columns = [entity_col, col_label]
            else:
                continue

            s = s.merge(agg_df, left_on="id", right_on=entity_col, how="left")
            if entity_col != "id" and entity_col in s.columns:
                s.drop(columns=[entity_col], inplace=True)

    return s


# ══════════════════════════════════════════════════════════════════════════════
#  CHOROPLETH MAP — geographic visualization from a GeoDataFrame
# ══════════════════════════════════════════════════════════════════════════════


def choropleth_map(gdf, value_col, entity_col="id", name_col=None, label=None, map_config=None):
    """
    Render a choropleth map from a GeoDataFrame colored by *value_col*.

    Args:
        gdf:         GeoDataFrame with geometry and a numeric value column.
        value_col:   Column name for the choropleth color values.
        entity_col:  Column identifying entities.
        name_col:    Optional column for hover display names.
        label:       Display label for the color axis (overrides map_config).
        map_config:  Dict with map config from visualization YAML:
                       color_ramp: list of hex colors for the continuous scale
                       label:      display label for the value axis
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        print("choropleth_map requires a GeoDataFrame.")
        return

    if value_col not in gdf.columns:
        print(f"Column '{value_col}' not found in GeoDataFrame.")
        return

    cfg = map_config or {}
    color_ramp = cfg.get("color_ramp", ["#FFEDA0", "#FD8D3C", "#BD0026"])
    label = label or cfg.get("label", value_col.replace("_", " ").title())

    plot_gdf = gdf.to_crs(epsg=4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()

    hover_cols = [entity_col] if entity_col in plot_gdf.columns else []
    if name_col and name_col in plot_gdf.columns and name_col != entity_col:
        hover_cols.append(name_col)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*centroid.*")
        center_lat = plot_gdf.geometry.centroid.y.mean()
        center_lon = plot_gdf.geometry.centroid.x.mean()

    fig = px.choropleth_mapbox(
        plot_gdf,
        geojson=plot_gdf.geometry,
        locations=plot_gdf.index,
        color=value_col,
        color_continuous_scale=color_ramp,
        mapbox_style="open-street-map",
        center={"lat": center_lat, "lon": center_lon},
        zoom=3,
        opacity=0.7,
        hover_data=hover_cols or None,
        labels={value_col: label},
    )

    fig.update_layout(
        title=f"{label} — Choropleth Map",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN STATS — detailed descriptive statistics (ported from data_analysis)
# ══════════════════════════════════════════════════════════════════════════════


def column_stats(df, column, top_n=10):
    """
    Return comprehensive descriptive statistics for a DataFrame column.

    Includes null/unique counts, and for numeric columns: mean, median, std,
    min, max, quartiles.  Always includes top-N value counts.

    Args:
        df:      DataFrame to analyse.
        column:  Column name.
        top_n:   Number of top values to show.

    Returns:
        dict with stats, or None if column not found.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return None

    col = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(col)

    stats = {
        "column": column,
        "dtype": str(col.dtype),
        "total": len(col),
        "non_null": int(col.notna().sum()),
        "null": int(col.isna().sum()),
        "null_pct": round(col.isna().mean() * 100, 2),
        "unique": int(col.nunique()),
        "unique_pct": round(col.nunique() / max(len(col), 1) * 100, 2),
    }

    if is_numeric:
        v = col.dropna()
        if len(v) > 0:
            stats.update(
                {
                    "mean": round(float(v.mean()), 4),
                    "median": round(float(v.median()), 4),
                    "std": round(float(v.std()), 4),
                    "min": round(float(v.min()), 4),
                    "max": round(float(v.max()), 4),
                    "q25": round(float(v.quantile(0.25)), 4),
                    "q75": round(float(v.quantile(0.75)), 4),
                }
            )

    stats["top_values"] = (
        col.value_counts(dropna=False)
        .head(top_n)
        .to_frame("count")
        .assign(pct=lambda d: (d["count"] / len(col) * 100).round(2))
    )

    return stats


def print_column_stats(df, column, top_n=10):
    """Print formatted descriptive statistics for a DataFrame column."""
    s = column_stats(df, column, top_n)
    if s is None:
        return

    print(f"{'=' * 70}")
    print(f"  Column: {s['column']}  ({s['dtype']})")
    print(f"{'=' * 70}")
    print(f"  Total: {s['total']:,}  |  Non-null: {s['non_null']:,}  |  Null: {s['null']:,} ({s['null_pct']}%)")
    print(f"  Unique: {s['unique']:,} ({s['unique_pct']}%)")

    if "mean" in s:
        print(f"\n  Mean: {s['mean']}  |  Median: {s['median']}  |  Std: {s['std']}")
        print(f"  Min:  {s['min']}  |  Max:    {s['max']}")
        print(f"  Q25:  {s['q25']}  |  Q75:    {s['q75']}")

    print(f"\n  Top {top_n} values:")
    print(s["top_values"].to_string())
    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
#  GROUPED STATS — break down a column by a groupby variable
# ══════════════════════════════════════════════════════════════════════════════


def grouped_stats(df, column, groupby, decimals=4):
    """
    Compute statistics for *column* broken down by *groupby*.

    For numeric columns returns: count, unique, mean, median, std, min, max.
    For categorical columns returns a crosstab (value counts pivoted).

    Args:
        df:       DataFrame.
        column:   Column to analyse.
        groupby:  Column to group by (becomes the table columns).
        decimals: Rounding precision for numeric stats.

    Returns:
        pd.DataFrame with groups as columns.
    """
    if column not in df.columns or groupby not in df.columns:
        print(f"Column '{column}' or '{groupby}' not found.")
        return None

    if pd.api.types.is_numeric_dtype(df[column]):
        return (
            df.groupby(groupby)[column]
            .agg(
                count="count",
                unique="nunique",
                mean=lambda x: round(x.mean(), decimals),
                median=lambda x: round(x.median(), decimals),
                std=lambda x: round(x.std(), decimals),
                min="min",
                max="max",
            )
            .T
        )
    else:
        return df.groupby([groupby, column]).size().unstack(fill_value=0)


def kpi_groupby(df, col_specs, groupby, entity_col="id", date_col="date", decimals=4, stats=None):
    """
    Cross-table of KPI statistics broken down by a grouping column.

    Combines the multi-KPI awareness of kpi_summary with the group-by
    capability of grouped_stats, producing a single DataFrame with
    KPI labels as rows and (group, stat) as a MultiIndex on columns.

    Args:
        df:          DataFrame (one row per entity, e.g. a summary table).
        col_specs:   List of column specs (dicts with 'name', 'type', and
                     optional 'label').  Only numeric/timeseries specs are
                     included.
        groupby:     Column name to group by (e.g. 'area_class').
        entity_col:  Entity identifier column (used by prepare_df).
        date_col:    Date column (used by prepare_df).
        decimals:    Rounding precision for numeric stats.
        stats:       List of stat names to compute.  Defaults to
                     ['count', 'mean', 'median', 'std', 'min', 'max'].

    Returns:
        pd.DataFrame with KPI labels as index and a MultiIndex
        (group, stat) as columns, or None if no data.
    """
    if groupby not in df.columns:
        print(f"Groupby column '{groupby}' not found.")
        return None

    if stats is None:
        stats = ["count", "mean", "median", "std", "min", "max"]

    numeric_specs = [s for s in col_specs if s["type"] in ("numeric", "timeseries")]
    if not numeric_specs:
        print("No numeric columns configured.")
        return None

    stat_fns = {
        "count": "count",
        "unique": "nunique",
        "mean": lambda x: round(x.mean(), decimals),
        "median": lambda x: round(x.median(), decimals),
        "std": lambda x: round(x.std(), decimals),
        "min": "min",
        "max": "max",
    }
    agg_dict = {k: stat_fns[k] for k in stats if k in stat_fns}

    rows = {}
    for spec in numeric_specs:
        col = spec["name"]
        label = spec.get("label", col.replace("_", " ").title())
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        grp = v.groupby(df[groupby]).agg(**agg_dict)
        # grp: rows = group values, cols = stat names → transpose so
        # each stat becomes accessible per group
        for group_val in grp.index:
            for stat_name in grp.columns:
                rows.setdefault(label, {})[(group_val, stat_name)] = grp.loc[group_val, stat_name]

    if not rows:
        print("No data to display.")
        return None

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=[groupby, "stat"])
    result = result.sort_index(axis=1)
    result.index.name = "KPI"
    return result.round(decimals)


# ══════════════════════════════════════════════════════════════════════════════
#  CROSSTAB HEATMAP — categorical × categorical as Plotly heatmap
# ══════════════════════════════════════════════════════════════════════════════


def crosstab_chart(df, row_col, col_col, normalize=False):
    """
    Render a Plotly heatmap of the crosstab between two categorical columns.

    Args:
        df:        DataFrame.
        row_col:   Column for the heatmap rows.
        col_col:   Column for the heatmap columns.
        normalize: If True show percentages (column-normalised) instead of counts.
    """
    if row_col not in df.columns or col_col not in df.columns:
        print(f"Column '{row_col}' or '{col_col}' not found.")
        return

    ct = pd.crosstab(df[row_col], df[col_col])
    if normalize:
        ct = (ct / ct.sum(axis=0) * 100).round(2)

    label = "%" if normalize else "Count"

    fig = go.Figure(
        go.Heatmap(
            z=ct.values,
            x=[str(c) for c in ct.columns],
            y=[str(r) for r in ct.index],
            colorscale="Blues",
            text=ct.values,
            texttemplate="%{text}",
            hovertemplate=f"{row_col}: %{{y}}<br>{col_col}: %{{x}}<br>{label}: %{{z}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Crosstab: {row_col} × {col_col}" + (" (%)" if normalize else ""),
        xaxis_title=col_col,
        yaxis_title=row_col,
        height=max(350, len(ct) * 35 + 120),
        template="plotly_white",
    )
    fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SCATTER COMPARISON — source1 vs source2 with 1:1 line & stats
# ══════════════════════════════════════════════════════════════════════════════


def _compute_comparison_stats(v1, v2):
    """Compute correlation, R², RMSE, MAE, bias, std_diff between two arrays."""
    diff = v1 - v2
    corr = float(np.corrcoef(v1, v2)[0, 1])
    return {
        "correlation": round(corr, 4),
        "r_squared": round(corr**2, 4),
        "rmse": round(float(np.sqrt(np.mean(diff**2))), 4),
        "mae": round(float(np.mean(np.abs(diff))), 4),
        "bias": round(float(np.mean(diff)), 4),
        "std_diff": round(float(np.std(diff, ddof=1)), 4),
        "n": len(v1),
    }


def scatter_comparison(
    df,
    col_1,
    col_2,
    label_1=None,
    label_2=None,
    date_col=None,
    value_threshold=0.0,
    title="Scatter Comparison",
):
    """
    Render an interactive scatter plot of *col_1* vs *col_2* with a 1:1 line
    and R²/RMSE/MAE/bias annotation.

    Args:
        df:              DataFrame containing both columns.
        col_1:           Column name for the x-axis (source 1 / prod).
        col_2:           Column name for the y-axis (source 2 / preprod).
        label_1:         Display label for col_1 (defaults to col_1).
        label_2:         Display label for col_2 (defaults to col_2).
        date_col:        Optional date column shown on hover.
        value_threshold: If > 0, points where both values are below threshold
                         are shown in light gray and excluded from stats.
        title:           Chart title.
    """
    if col_1 not in df.columns or col_2 not in df.columns:
        print(f"Column '{col_1}' or '{col_2}' not found.")
        return

    label_1 = label_1 or col_1
    label_2 = label_2 or col_2

    common = df.dropna(subset=[col_1, col_2]).copy()
    if len(common) < 2:
        print("Not enough common data points for scatter comparison.")
        return

    v1 = common[col_1].values.astype(float)
    v2 = common[col_2].values.astype(float)

    fig = go.Figure()

    # Split by threshold if requested
    if value_threshold > 0.0:
        above = (v1 >= value_threshold) & (v2 >= value_threshold)

        hover_tpl = f"{label_1}: %{{x:.4f}}<br>{label_2}: %{{y:.4f}}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=v1[~above],
                y=v2[~above],
                mode="markers",
                name=f"below {value_threshold}",
                marker=dict(size=4, color="lightgray", opacity=0.4),
                hovertemplate=hover_tpl,
            )
        )

        hover_above = hover_tpl
        if date_col and date_col in common.columns:
            dates = common[date_col].astype(str).values
            hover_above = f"{label_1}: %{{x:.4f}}<br>{label_2}: %{{y:.4f}}<br>date: %{{text}}<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=v1[above],
                    y=v2[above],
                    mode="markers",
                    name=f"above {value_threshold}",
                    marker=dict(size=5, opacity=0.5),
                    text=dates[above],
                    hovertemplate=hover_above,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=v1[above],
                    y=v2[above],
                    mode="markers",
                    name=f"above {value_threshold}",
                    marker=dict(size=5, opacity=0.5),
                    hovertemplate=hover_above,
                )
            )

        stats = _compute_comparison_stats(v1[above], v2[above]) if above.sum() >= 2 else {}
    else:
        hover_tpl = f"{label_1}: %{{x:.4f}}<br>{label_2}: %{{y:.4f}}<extra></extra>"
        if date_col and date_col in common.columns:
            dates = common[date_col].astype(str).values
            hover_tpl = f"{label_1}: %{{x:.4f}}<br>{label_2}: %{{y:.4f}}<br>date: %{{text}}<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=v1,
                    y=v2,
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                    text=dates,
                    hovertemplate=hover_tpl,
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=v1,
                    y=v2,
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                    hovertemplate=hover_tpl,
                    showlegend=False,
                )
            )

        stats = _compute_comparison_stats(v1, v2)

    # 1:1 reference line
    all_vals = np.concatenate([v1, v2])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = (vmax - vmin) * 0.05
    fig.add_trace(
        go.Scatter(
            x=[vmin - margin, vmax + margin],
            y=[vmin - margin, vmax + margin],
            mode="lines",
            name="1:1 line",
            line=dict(color="red", dash="dash", width=1),
            opacity=0.6,
        )
    )

    # Stats annotation
    if stats:
        ann = f"R²={stats['r_squared']}  RMSE={stats['rmse']}  MAE={stats['mae']}  bias={stats['bias']}  n={stats['n']}"
        if value_threshold > 0.0:
            ann += f"  threshold={value_threshold}"
        fig.add_annotation(
            text=ann,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,248,220,0.85)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(text=title, font_size=14),
        xaxis_title=label_1,
        yaxis_title=label_2,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.12),
    )
    fig.show()
    return fig


def comparison_chart(
    df,
    col_1,
    col_2,
    date_col="date",
    label_1=None,
    label_2=None,
    value_threshold=0.0,
    title="Time Series Comparison",
):
    """
    Render a side-by-side time-series overlay + scatter plot for two columns.

    Combines a line overlay (left) with a scatter/1:1 plot (right) — the
    Plotly equivalent of data_analysis.plot_timeseries_comparison().

    Args:
        df:              DataFrame with both value columns and a date column.
        col_1:           First value column (e.g. prod).
        col_2:           Second value column (e.g. preprod).
        date_col:        Date column name.
        label_1:         Display label for col_1.
        label_2:         Display label for col_2.
        value_threshold: Minimum value for stats computation (see scatter_comparison).
        title:           Overall chart title.
    """
    if col_1 not in df.columns or col_2 not in df.columns:
        print(f"Column '{col_1}' or '{col_2}' not found.")
        return

    label_1 = label_1 or col_1
    label_2 = label_2 or col_2

    ts = df.copy()
    if date_col in ts.columns:
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts[col_1] = pd.to_numeric(ts[col_1], errors="coerce")
    ts[col_2] = pd.to_numeric(ts[col_2], errors="coerce")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{title} — Time Series", f"{title} — Scatter"],
        horizontal_spacing=0.08,
    )

    # ── Left panel: overlay line plot ─────────────────────────────────────────
    if date_col in ts.columns:
        fig.add_trace(
            go.Scatter(
                x=ts[date_col],
                y=ts[col_1],
                mode="lines",
                name=label_1,
                opacity=0.85,
                line=dict(width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ts[date_col],
                y=ts[col_2],
                mode="lines",
                name=label_2,
                opacity=0.85,
                line=dict(width=1.5),
            ),
            row=1,
            col=1,
        )
        if value_threshold > 0.0:
            fig.add_hline(
                y=value_threshold,
                line_dash="dot",
                line_color="gray",
                opacity=0.7,
                annotation_text=f"threshold={value_threshold}",
                annotation_position="top left",
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)

    # ── Right panel: scatter with 1:1 line ────────────────────────────────────
    common = ts.dropna(subset=[col_1, col_2])
    if len(common) >= 2:
        v1 = common[col_1].values.astype(float)
        v2 = common[col_2].values.astype(float)

        if value_threshold > 0.0:
            above = (v1 >= value_threshold) & (v2 >= value_threshold)
            fig.add_trace(
                go.Scatter(
                    x=v1[~above],
                    y=v2[~above],
                    mode="markers",
                    name=f"below {value_threshold}",
                    marker=dict(size=4, color="lightgray", opacity=0.4),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=v1[above],
                    y=v2[above],
                    mode="markers",
                    name="data",
                    marker=dict(size=5, opacity=0.5),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            stat_v1, stat_v2 = v1[above], v2[above]
        else:
            fig.add_trace(
                go.Scatter(
                    x=v1,
                    y=v2,
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            stat_v1, stat_v2 = v1, v2

        # 1:1 line
        all_vals = np.concatenate([v1, v2])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        margin = (vmax - vmin) * 0.05
        fig.add_trace(
            go.Scatter(
                x=[vmin - margin, vmax + margin],
                y=[vmin - margin, vmax + margin],
                mode="lines",
                name="1:1",
                line=dict(color="red", dash="dash", width=1),
                opacity=0.6,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Stats annotation
        if len(stat_v1) >= 2:
            stats = _compute_comparison_stats(stat_v1, stat_v2)
            ann = (
                f"R²={stats['r_squared']}  RMSE={stats['rmse']}  "
                f"MAE={stats['mae']}  bias={stats['bias']}  n={stats['n']}"
            )
            fig.add_annotation(
                text=ann,
                xref="x2 domain",
                yref="y2 domain",
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,248,220,0.85)",
                bordercolor="gray",
                borderwidth=1,
            )

        fig.update_xaxes(title_text=label_1, row=1, col=2)
        fig.update_yaxes(title_text=label_2, scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig.update_layout(
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.25),
        margin=dict(t=80, b=40),
    )
    fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  VIZ CONFIG GENERATOR
# ══════════════════════════════════════════════════════════════════════════════


def generate_viz_config(
    df,
    entity_col="id",
    date_col="date",
    save_to=None,
    timeseries_threshold=10,
):
    """
    Auto-generate a visualization config (col_specs) from a DataFrame.

    Inspects column types and value distributions to classify each column as
    timeseries, numeric, categorical, or date. Returns a dict ready for
    viz_engine functions, and optionally saves it as YAML.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        entity_col (str): Column used as entity identifier. Default: "id".
        date_col (str): Column used as date axis. Default: "date".
        save_to (str | Path, optional): Path to save the config as YAML.
            Example: "results/coverage_viz.yaml"
        timeseries_threshold (int): Minimum unique values per entity to
            classify a numeric column as timeseries (vs numeric). Default: 10.

    Returns:
        dict: Config with keys "entity_col", "date_col", "col_specs".

    Example:
        >>> config = generate_viz_config(df, entity_col="id", date_col="date")
        >>> timeseries_chart(df, config["col_specs"],
        ...     entity_col=config["entity_col"], date_col=config["date_col"])
    """
    skip_cols = {entity_col, date_col, "geometry", "wkt", "_cached_at"}
    has_date = date_col in df.columns
    has_entity = entity_col in df.columns

    # Compute per-entity unique counts for timeseries detection
    if has_date and has_entity:
        entity_counts = df.groupby(entity_col).size()
        median_rows_per_entity = entity_counts.median() if len(entity_counts) > 0 else 0
    else:
        median_rows_per_entity = 0

    col_specs = []
    for col in df.columns:
        if col in skip_cols:
            continue

        dtype = df[col].dtype
        non_null = df[col].dropna()
        n_unique = non_null.nunique()
        label = col.replace("_", " ").replace(".", " ").title()

        # Date columns
        if pd.api.types.is_datetime64_any_dtype(dtype):
            col_specs.append({"name": col, "label": label, "type": "date"})
            continue

        # Try parsing as date if string column looks like dates
        if dtype is object and n_unique > 0:
            sample = non_null.head(5)
            try:
                parsed = pd.to_datetime(sample, format="%Y-%m-%d")
                if parsed.notna().all():
                    col_specs.append({"name": col, "label": label, "type": "date"})
                    continue
            except (ValueError, TypeError):
                pass

        # Numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            if has_date and median_rows_per_entity >= timeseries_threshold:
                col_specs.append({"name": col, "label": label, "type": "timeseries"})
            else:
                col_specs.append({"name": col, "label": label, "type": "numeric"})
            continue

        # Categorical / string columns
        col_specs.append({"name": col, "label": label, "type": "categorical"})

    config = {
        "entity_col": entity_col,
        "date_col": date_col,
        "col_specs": col_specs,
    }

    # Save as YAML if requested
    if save_to is not None:
        _save_viz_config_yaml(config, save_to)

    # Print preview
    _print_viz_config(config, df)

    return config


def _save_viz_config_yaml(config, path):
    """Write a viz config dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f'entity_col: "{config["entity_col"]}"')
    lines.append(f'date_col: "{config["date_col"]}"')
    lines.append("")
    lines.append("col_specs:")
    for spec in config["col_specs"]:
        lines.append(f'  - name: "{spec["name"]}"')
        lines.append(f'    label: "{spec["label"]}"')
        lines.append(f'    type: "{spec["type"]}"')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved viz config to {path}")


def _print_viz_config(config, df):
    """Print a formatted summary of the generated viz config."""
    specs = config["col_specs"]
    ts_cols = [s for s in specs if s["type"] == "timeseries"]
    num_cols = [s for s in specs if s["type"] == "numeric"]
    cat_cols = [s for s in specs if s["type"] == "categorical"]
    date_cols = [s for s in specs if s["type"] == "date"]

    print(f"Viz config generated: {len(specs)} columns")
    print(f"  entity_col: {config['entity_col']}")
    print(f"  date_col:   {config['date_col']}")

    if ts_cols:
        print(f"  timeseries ({len(ts_cols)}): {', '.join(s['name'] for s in ts_cols)}")
    if num_cols:
        print(f"  numeric    ({len(num_cols)}): {', '.join(s['name'] for s in num_cols)}")
    if cat_cols:
        print(f"  categorical({len(cat_cols)}): {', '.join(s['name'] for s in cat_cols)}")
    if date_cols:
        print(f"  date       ({len(date_cols)}): {', '.join(s['name'] for s in date_cols)}")


def load_viz_config(path):
    """
    Load a viz config from a YAML file.

    Args:
        path (str | Path): Path to the YAML config file.

    Returns:
        dict: Config with keys "entity_col", "date_col", "col_specs".

    Example:
        >>> config = load_viz_config("results/coverage_viz.yaml")
        >>> timeseries_chart(df, config["col_specs"],
        ...     entity_col=config["entity_col"], date_col=config["date_col"])
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load viz configs. Install with: pip install pyyaml")

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "col_specs" not in config:
        raise ValueError(f"Invalid viz config: missing 'col_specs' key in {path}")

    return config
