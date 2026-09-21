"""
HTML report generator for extraction runs.

Usage (automatic via BaseExtractor):
    results_df, status = extractor._finalize_extraction(
        ..., generate_report=True
    )

Usage (standalone):
    from earthdaily.agriculture.reporting import ExtractionReporter

    reporter = ExtractionReporter()
    reporter.set_extraction_context(
        extractor_name="GreennessExtractor",
        prefix="greenness",
        env="production",
        parameters={"start_date": "2025-01-01", ...},
        column_mapping={"id": "entity_id", ...},
    )
    reporter.set_results(
        results_df=results_df,
        total_entities=100,
        total_calculations=95,
        successful=90,
        failed=5,
        failed_ids=["id1", "id2", ...],
        global_errors=[{"entity_id": "id1", "message": "..."}],
        elapsed_seconds=123.4,
    )
    html = reporter.render_html(output_path="results/greenness_report.html")
"""

from __future__ import annotations

import base64
import html as html_lib
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd

from earthdaily.agriculture.reporting.templates import REPORT_CSS

logger = logging.getLogger(__name__)

#: Corporate site the mark links to — see docs/internal/19, §26.
BRAND_URL = "https://earthdaily.com"

#: Footer links, in the order §26 sets: the useful one first, the legal pair after.
#: A surface attributed to EarthDaily has to say where its terms live.
DOCS_URL = "https://docs.earthdaily.com/agro/"
TERMS_URL = "https://earthdaily.com/terms-of-use"
PRIVACY_URL = "https://earthdaily.com/privacy-policy"
FOOTER_LINKS = (("Documentation", DOCS_URL), ("Terms of use", TERMS_URL), ("Privacy policy", PRIVACY_URL))

#: Packaged mark. The light-ground, full-colour asset, because the report renders
#: on a light hero (§26: pick the asset for the ground it sits on). When the report
#: grows a dark theme, the white-gradient asset joins it here.
_LOGO_ASSET = "eda_logo_light.png"

#: Rendered height of the mark. The packaged asset is twice this, so it stays crisp
#: on a HiDPI screen — keep the two in step when changing either.
_LOGO_RENDER_HEIGHT = 56


@lru_cache(maxsize=1)
def _logo_data_uri() -> str | None:
    """The mark as a ``data:`` URI, or None when the asset cannot be read.

    Embedded rather than hotlinked because these reports are self-contained by
    design (§26): they are written to S3, mailed around and opened offline, and a
    strict CSP blocks external hosts — a remote ``<img>`` would fail exactly where
    the report is being relied on. Read through importlib.resources so it works
    from a wheel, and cached because the bytes never change within a process.
    """
    try:
        from importlib.resources import files

        raw = (files("earthdaily.agriculture.reporting") / "assets" / _LOGO_ASSET).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception as exc:  # pragma: no cover - depends on install layout
        # Never fail a report over branding: an install that stripped package data
        # still gets its numbers, with a text attribution instead of the mark.
        logger.warning("Corporate mark unavailable (%s); falling back to a text link", exc)
        return None


def _render_brand(height: int = _LOGO_RENDER_HEIGHT) -> str:
    """The corporate mark, linked to the corporate site (§26)."""
    data_uri = _logo_data_uri()
    if not data_uri:
        return f"<a class='er-brand' href='{BRAND_URL}' target='_blank' rel='noopener noreferrer'>EarthDaily</a>"
    return (
        f"<a class='er-brand' href='{BRAND_URL}' target='_blank' rel='noopener noreferrer'>"
        f"<img src='{data_uri}' alt='EarthDaily' height='{height}'>"
        "</a>"
    )


class ExtractionReporter:
    """Generate self-contained HTML reports for bulk extraction runs."""

    def __init__(
        self,
        *,
        include_map: bool = False,
        include_data_preview: bool = True,
        preview_rows: int = 20,
        map_geometry_column: str = "geometry",
        map_max_features: int = 500,
        max_errors_shown: int = 50,
    ) -> None:
        self.include_map = include_map
        self.include_data_preview = include_data_preview
        self.preview_rows = preview_rows
        self.map_geometry_column = map_geometry_column
        self.map_max_features = map_max_features
        self.max_errors_shown = max_errors_shown

        # Populated via set_extraction_context / set_results
        self._context: dict[str, Any] = {}
        self._results_df: pd.DataFrame | None = None
        self._stats: dict[str, Any] = {}
        self._errors: list[dict[str, Any]] = []
        self._failed_ids: list[str] = []
        self._entity_df: pd.DataFrame | None = None  # original entities for map

    # ------------------------------------------------------------------
    # Public API: populate report data
    # ------------------------------------------------------------------

    def set_extraction_context(
        self,
        *,
        extractor_name: str,
        prefix: str,
        env: str | None = None,
        parameters: dict[str, Any] | None = None,
        column_mapping: dict[str, str] | None = None,
        output_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Set metadata about the extraction run."""
        self._context = {
            "extractor_name": extractor_name,
            "prefix": prefix,
            "env": env,
            "parameters": parameters or {},
            "column_mapping": column_mapping or {},
            "output_path": output_path,
            # `datetime.utcnow()` deprecated in 3.12; use timezone-aware UTC.
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        }
        if extra:
            self._context.update(extra)

    def set_results(
        self,
        *,
        results_df: pd.DataFrame | None = None,
        total_entities: int = 0,
        total_calculations: int = 0,
        successful: int = 0,
        failed: int = 0,
        failed_ids: list[str] | None = None,
        global_errors: list[dict[str, Any]] | None = None,
        elapsed_seconds: float | None = None,
        finalization_status: dict[str, Any] | None = None,
    ) -> None:
        """Set extraction results and statistics."""
        self._results_df = results_df
        self._stats = {
            "total_entities": total_entities,
            "total_calculations": total_calculations,
            "successful": successful,
            "failed": failed,
            "elapsed_seconds": elapsed_seconds,
            "output_rows": len(results_df) if results_df is not None else 0,
            "finalization": finalization_status or {},
        }
        self._failed_ids = list(failed_ids or [])
        self._errors = list(global_errors or [])

    def set_entity_data(self, entity_df: pd.DataFrame) -> None:
        """Provide original entity DataFrame for map rendering."""
        self._entity_df = entity_df

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_html(self, output_path: str | None = None) -> str:
        """Build self-contained HTML report and optionally write to file."""
        parts: list[str] = [REPORT_CSS]
        parts.append("<div class='extraction-report'><div class='er-container'>")

        parts.append(self._render_hero())
        parts.append(self._render_stats())

        if self.include_map:
            map_html = self._render_map()
            if map_html:
                parts.append(map_html)

        if self._errors:
            parts.append(self._render_errors())

        parts.append(self._render_parameters())

        if self.include_data_preview and self._results_df is not None and not self._results_df.empty:
            parts.append(self._render_data_preview())

        parts.append(self._render_footer())
        parts.append("</div></div>")

        html_out = "\n".join(parts)

        if output_path:
            try:
                # Routes through fsspec for remote URIs (s3://, gs://, ...) and
                # falls back to Path.write_text + parent mkdir for local paths.
                from earthdaily.agriculture.core._fs import write_text

                write_text(output_path, html_out, encoding="utf-8")
                logger.info("Extraction report written to %s", output_path)
            except Exception as exc:
                logger.warning("Failed to write extraction report: %s", exc)

        return html_out

    # ------------------------------------------------------------------
    # Private: HTML section builders
    # ------------------------------------------------------------------

    def _render_hero(self) -> str:
        ctx = self._context
        extractor = ctx.get("extractor_name", "Extraction")
        prefix = ctx.get("prefix", "")
        env = ctx.get("env", "")
        ts = ctx.get("generated_at", "")

        title = f"{extractor} Report"
        subtitle_parts = []
        if prefix:
            subtitle_parts.append(f"Prefix: {_esc(prefix)}")
        if env:
            subtitle_parts.append(f"Environment: {_esc(env)}")
        if ts:
            subtitle_parts.append(f"Generated: {_esc(ts)}")

        # The mark sits once per surface, in a fixed place (§26) — here, opposite the
        # title in the hero.
        return (
            "<div class='er-hero'>"
            "<div class='er-hero-head'>"
            "<div class='er-hero-titles'>"
            f"<h2>{_esc(title)}</h2>"
            f"<div class='er-subtitle'>{' &middot; '.join(subtitle_parts)}</div>"
            "</div>"
            f"{_render_brand()}"
            "</div>"
            "</div>"
        )

    def _render_stats(self) -> str:
        s = self._stats
        total_ent = s.get("total_entities", 0)
        total_calc = s.get("total_calculations", 0)
        successful = s.get("successful", 0)
        failed = s.get("failed", 0)
        elapsed = s.get("elapsed_seconds")
        output_rows = s.get("output_rows", 0)
        rate = (successful / total_calc * 100) if total_calc > 0 else 0

        finalization = s.get("finalization", {})
        exported = finalization.get("exported", False)
        export_path = finalization.get("export_path", "")

        cards = [
            ("Total Entities", str(total_ent), ""),
            ("Calculated", str(total_calc), ""),
            ("Successful", str(successful), "success"),
            ("Failed", str(failed), "danger" if failed > 0 else ""),
            ("Output Rows", str(output_rows), ""),
        ]
        if elapsed is not None:
            minutes, secs = divmod(elapsed, 60)
            if minutes >= 1:
                cards.append(("Duration", f"{int(minutes)}m {secs:.1f}s", ""))
            else:
                cards.append(("Duration", f"{elapsed:.1f}s", ""))

        if exported and export_path:
            cards.append(("Exported To", _esc(str(export_path)), ""))

        parts = ["<div class='er-section'>", "<div class='er-stats'>"]
        for label, value, css_class in cards:
            cls = f" {css_class}" if css_class else ""
            parts.append(
                f"<div class='er-stat-card'><div class='label'>{label}</div><div class='value{cls}'>{value}</div></div>"
            )
        parts.append("</div>")

        # Success rate bar
        parts.append(
            f"<div style='font-size:12px;color:#475569;margin-top:8px;'>"
            f"Success rate: {rate:.1f}%</div>"
            f"<div class='er-rate-bar'>"
            f"<div class='er-rate-fill' style='width:{rate:.1f}%'></div>"
            f"</div>"
        )
        parts.append("</div>")
        return "\n".join(parts)

    def _render_errors(self) -> str:
        errors = self._errors[: self.max_errors_shown]
        truncated = len(self._errors) - len(errors)

        parts = [
            "<div class='er-section'>",
            f"<h3>Errors ({len(self._errors)} total)</h3>",
            "<table class='er-table'>",
            "<thead><tr><th>Entity ID</th><th>Error</th><th>Code</th></tr></thead>",
            "<tbody>",
        ]
        for err in errors:
            eid = _esc(str(err.get("entity_id", "")))
            msg = _esc(str(err.get("message", err.get("error_message", ""))))
            code = _esc(str(err.get("error_code", "")))
            parts.append(f"<tr class='er-error-row'><td>{eid}</td><td>{msg}</td><td>{code}</td></tr>")
        parts.append("</tbody></table>")
        if truncated > 0:
            parts.append(
                f"<div style='font-size:12px;color:#64748b;margin-top:6px;'>"
                f"... and {truncated} more errors not shown</div>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _render_parameters(self) -> str:
        ctx = self._context
        params = ctx.get("parameters", {})
        mapping = ctx.get("column_mapping", {})

        if not params and not mapping:
            return ""

        parts = ["<div class='er-section er-params'>"]
        if params:
            params_json = json.dumps(params, indent=2, default=str)
            parts.append(f"<details><summary>Extraction Parameters</summary><pre>{_esc(params_json)}</pre></details>")
        if mapping:
            mapping_json = json.dumps(mapping, indent=2, default=str)
            parts.append(f"<details><summary>Column Mapping</summary><pre>{_esc(mapping_json)}</pre></details>")
        parts.append("</div>")
        return "\n".join(parts)

    def _render_data_preview(self) -> str:
        df = self._results_df
        if df is None or df.empty:
            return ""

        preview = df.head(self.preview_rows)
        cols = list(preview.columns)

        parts = [
            "<div class='er-section'>",
            f"<h3>Data Preview (first {len(preview)} of {len(df)} rows)</h3>",
            "<table class='er-table'>",
            "<thead><tr>",
        ]
        for col in cols:
            parts.append(f"<th>{_esc(str(col))}</th>")
        parts.append("</tr></thead><tbody>")

        for _, row in preview.iterrows():
            parts.append("<tr>")
            for col in cols:
                val = row[col]
                cell = "" if pd.isna(val) else _esc(str(val))
                parts.append(f"<td>{cell}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></div>")
        return "\n".join(parts)

    def _render_map(self) -> str | None:
        """Render a folium map of entity geometries."""
        df = self._entity_df
        if df is None or self.map_geometry_column not in (df.columns if df is not None else []):
            return None
        try:
            import folium
            from shapely import wkb, wkt
        except ImportError:
            logger.info("Map skipped: folium or shapely not available")
            return None

        try:
            series = df[self.map_geometry_column].dropna().head(self.map_max_features)
            if series.empty:
                return None

            geometries = []
            for val in series:
                try:
                    if isinstance(val, (bytes, bytearray)):
                        geom = wkb.loads(val)
                    else:
                        geom = wkt.loads(str(val))
                    geometries.append(geom)
                except Exception:
                    continue

            if not geometries:
                return None

            xs = [g.bounds[0] for g in geometries] + [g.bounds[2] for g in geometries]
            ys = [g.bounds[1] for g in geometries] + [g.bounds[3] for g in geometries]
            center = [(min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2]

            fmap = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")
            fmap.fit_bounds([[min(ys), min(xs)], [max(ys), max(xs)]])

            for geom in geometries:
                try:
                    folium.GeoJson(geom.__geo_interface__).add_to(fmap)
                except Exception:
                    continue

            map_html = fmap._repr_html_()
            return f"<div class='er-map-card'><h3>Entity Locations</h3>{map_html}</div>"
        except Exception as exc:
            logger.warning("Map generation failed: %s", exc)
            return None

    def _render_footer(self) -> str:
        links = " &middot; ".join(
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{_esc(text)}</a>" for text, url in FOOTER_LINKS
        )
        return f"<div class='er-footer'>Generated by earthdaily-agriculture ExtractionReporter &middot; {links}</div>"


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(text, quote=True)
