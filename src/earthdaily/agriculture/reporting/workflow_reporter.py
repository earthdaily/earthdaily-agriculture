"""
Workflow-level end-of-run HTML + JSON reporter.

Companion to :class:`~earthdaily.agriculture.reporting.extraction_reporter.ExtractionReporter`,
which reports on a *single extractor's* bulk run. ``WorkflowRunReporter``
operates one level up — over the **dict-of-step-results** that
``WorkflowManager.run_workflow`` returns — and produces a per-run HTML
report plus a machine-readable JSON receipt (`schema_version: 1`).

Usage — standalone (two-call pattern, recommended):

    from earthdaily.agriculture.reporting import WorkflowRunReporter

    reporter = WorkflowRunReporter(
        include_step_table=True,
        include_data_preview=False,
        max_errors_shown=50,
    )
    reporter.set_run_context(
        # `workflow_cfg` is the UNWRAPPED inner dict for the new step format, so
        # index `name` directly — `["workflow"]["name"]` raises KeyError.
        workflow_name=manager.workflow_cfg["name"],
        prefix=manager.workflow_cfg["settings"].get("output_prefix"),
        env=manager.env,
        entity_count=len(entity_df),
        parameters={"limit": args.limit},
    )
    results = manager.run_workflow(entity_list=entity_df)
    reporter.set_step_results(results, elapsed_seconds=elapsed)

    # ``report_path`` is normally not known until after the consumer builds
    # the final per-field CSV downstream of run_workflow. Amend it before
    # rendering.
    reporter.amend_run_context(report_path=str(final_csv_path))

    html = reporter.render_html(output_path="results/run_<prefix>.html")
    receipt = reporter.render_json(output_path="results/run_<prefix>.json")

JSON schema (v1)::

    {
      "schema_version": 1,
      "generated_at": "2026-05-13T18:42:11.123Z",
      "workflow": {"name": "...", "prefix": "...", "env": "prod"},
      "entities": {"loaded": 578, "limited_to": null},
      "duration_seconds": 612.3,
      "report_path": "results/<Client>_Individual_Field_Report_run_20260513.csv",
      "steps": [
        {"name": "mrts", "kind": "extractor", "rows": 6843, "errors": 0, "failed_ids": []},
        {"name": "last_clear_image", "kind": "transform", "rows": 578, "errors": 0}
      ],
      "errors": [{"step": "planted_area", "entity_id": "abc123", "message": "..."}]
    }
"""

from __future__ import annotations

import base64
import csv
import html as html_lib
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from earthdaily.agriculture.reporting.templates import REPORT_CSS

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class WorkflowRunReporter:
    """Generate self-contained HTML + machine-readable JSON for a workflow run.

    Mirrors :class:`ExtractionReporter`'s public shape on purpose — the two
    are intended to feel familiar to a reader who's used the extractor-level
    reporter before.
    """

    def __init__(
        self,
        *,
        include_step_table: bool = True,
        include_data_preview: bool = False,
        preview_rows: int = 10,
        max_errors_shown: int = 50,
    ) -> None:
        self.include_step_table = include_step_table
        self.include_data_preview = include_data_preview
        self.preview_rows = preview_rows
        self.max_errors_shown = max_errors_shown

        # Populated by set_run_context / set_step_results / amend_run_context.
        self._context: dict[str, Any] = {}
        self._step_summaries: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._elapsed_seconds: float | None = None
        # Raw step result dicts kept around for the optional data preview path
        # so set_step_results doesn't strip information the renderer might want.
        self._raw_results: dict[str, dict[str, Any]] = {}
        # Optional Plotly DAG figure to embed in the HTML report.
        # Populated via set_workflow_dag(); when set, the report grows an
        # interactive DAG section right after the hero (modebar PNG/SVG
        # download lives on the chart itself).
        self._workflow_dag_fig: Any = None

    # ------------------------------------------------------------------
    # Public API: populate report data
    # ------------------------------------------------------------------

    def set_run_context(
        self,
        *,
        workflow_name: str,
        prefix: str | None = None,
        env: str | None = None,
        entity_count: int = 0,
        limited_to: int | None = None,
        report_path: str | None = None,
        parameters: dict[str, Any] | None = None,
        column_mapping: dict[str, str] | None = None,
        resolved_yaml: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Set workflow-level metadata. Mirrors set_extraction_context."""
        self._context = {
            "workflow_name": workflow_name,
            "prefix": prefix,
            "env": env,
            "entity_count": entity_count,
            "limited_to": limited_to,
            "report_path": report_path,
            "parameters": parameters or {},
            "column_mapping": column_mapping or {},
            "resolved_yaml": resolved_yaml,
            "generated_at": _iso_utc_now(),
        }
        if extra:
            self._context.update(extra)

    def amend_run_context(self, **updates: Any) -> None:
        """Update individual fields on a previously-set run context.

        The canonical use case: the consumer doesn't know ``report_path``
        until *after* ``run_workflow`` finishes and downstream code has
        produced the final per-field CSV. Pass it here before
        ``render_html`` / ``render_json``:

            reporter.amend_run_context(report_path=str(final_csv))

        Unknown keys are accepted (added to ``self._context``) so consumers
        can stash their own metadata without subclassing.
        """
        if not self._context:
            raise RuntimeError("amend_run_context() called before set_run_context(); call set_run_context first.")
        # Refresh generated_at so consumers can tell rendered HTML/JSON apart
        # from the original set_run_context() timestamp if they care.
        self._context.update(updates)
        self._context["generated_at"] = _iso_utc_now()

    def set_step_results(
        self,
        results: dict[str, dict[str, Any]],
        *,
        elapsed_seconds: float | None = None,
        step_kinds: dict[str, str] | None = None,
    ) -> None:
        """Walk the dict returned by ``WorkflowManager.run_workflow``.

        Tolerates non-dict entries (transforms occasionally return bare
        DataFrames in test setups) and missing keys (``results_df``,
        ``global_errors``, ``failed_ids`` are all optional per step).

        Args:
            results: ``{step_name: step_result_dict}`` as produced by
                ``run_workflow``. Each value is either a dict with
                ``results_df``/``global_errors``/``failed_ids`` keys, or a
                bare DataFrame (older transform contract — handled
                gracefully).
            elapsed_seconds: Wall-clock run duration. Optional.
            step_kinds: Optional ``{step_name: "extractor"|"transform"|"transform_only"}``
                mapping. Filled by ``WorkflowManager`` when calling this
                reporter; standalone callers can pass an empty dict and
                each step lands as ``"step"``.
        """
        self._elapsed_seconds = elapsed_seconds
        self._raw_results = dict(results)
        self._step_summaries = []
        self._errors = []

        kinds = step_kinds or {}

        for step_name, step_res in results.items():
            df, errors, failed_ids, skipped, reason, cache = _unpack_step_result(step_res)
            kind = kinds.get(step_name, "step")
            summary: dict[str, Any] = {
                "name": step_name,
                "kind": kind,
                "rows": int(len(df)) if df is not None else 0,
                "errors": len(errors),
                "failed_ids": list(failed_ids),
            }
            if skipped:
                summary["skipped"] = True
                if reason is not None:
                    summary["reason"] = reason
            if cache is not None:
                summary["cache"] = cache
            self._step_summaries.append(summary)

            # Stamp each error with the originating step. Accept both shapes:
            #   {"entity_id": ..., "message": ...}  (extractor-style)
            #   {"step": ..., "error": ...}         (workflow-level)
            for err in errors:
                normalized = dict(err)
                normalized.setdefault("step", step_name)
                if "message" not in normalized and "error" in normalized:
                    normalized["message"] = normalized["error"]
                self._errors.append(normalized)

    def set_workflow_dag(self, fig: Any) -> None:
        """Attach a Plotly DAG figure to embed in the HTML report.

        Typically the run-results DAG from
        :meth:`WorkflowManager.visualize_workflow_results`. The HTML report
        renders it interactively (Plotly modebar exposes PNG/SVG download).

        Args:
            fig: A ``plotly.graph_objects.Figure``. Set to ``None`` to clear.
        """
        self._workflow_dag_fig = fig

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_html(self, output_path: str | None = None) -> str:
        """Build the HTML report and optionally write it via fsspec."""
        if not self._context:
            raise RuntimeError("render_html() called before set_run_context().")

        parts: list[str] = [REPORT_CSS, _TOOLBAR_CSS]
        parts.append("<div class='extraction-report'><div class='er-container'>")
        parts.append(self._render_hero())
        parts.append(self._render_toolbar())
        dag_section = self._render_workflow_dag()
        if dag_section:
            parts.append(dag_section)
        parts.append(self._render_run_summary())

        if self.include_step_table and self._step_summaries:
            parts.append(self._render_step_table())

        if self._errors:
            parts.append(self._render_errors())

        parts.append(self._render_parameters())

        if self.include_data_preview:
            preview = self._render_data_preview()
            if preview:
                parts.append(preview)

        parts.append(self._render_footer())
        parts.append("</div></div>")

        html_out = "\n".join(parts)

        if output_path:
            try:
                from earthdaily.agriculture.core._fs import write_text

                write_text(output_path, html_out, encoding="utf-8")
                logger.info("Workflow report written to %s", output_path)
            except Exception as exc:
                logger.warning("Failed to write workflow report: %s", exc)

        return html_out

    def render_json(self, output_path: str | None = None) -> dict[str, Any]:
        """Build the JSON receipt and optionally write it via fsspec."""
        if not self._context:
            raise RuntimeError("render_json() called before set_run_context().")

        ctx = self._context
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": ctx.get("generated_at", _iso_utc_now()),
            "workflow": {
                "name": ctx.get("workflow_name"),
                "prefix": ctx.get("prefix"),
                "env": ctx.get("env"),
            },
            "entities": {
                "loaded": ctx.get("entity_count", 0),
                "limited_to": ctx.get("limited_to"),
            },
            "duration_seconds": self._elapsed_seconds,
            "report_path": ctx.get("report_path"),
            "steps": list(self._step_summaries),
            "errors": list(self._errors),
        }
        # Optional / off-by-default payload fields.
        if ctx.get("parameters"):
            payload["parameters"] = ctx["parameters"]
        if ctx.get("column_mapping"):
            payload["column_mapping"] = ctx["column_mapping"]
        if ctx.get("resolved_yaml") is not None:
            payload["resolved_yaml"] = ctx["resolved_yaml"]

        if output_path:
            try:
                from earthdaily.agriculture.core._fs import write_text

                write_text(
                    output_path,
                    json.dumps(payload, indent=2, default=str),
                    encoding="utf-8",
                )
                logger.info("Workflow JSON receipt written to %s", output_path)
            except Exception as exc:
                logger.warning("Failed to write workflow JSON receipt: %s", exc)

        return payload

    # ------------------------------------------------------------------
    # Private: HTML section builders
    # ------------------------------------------------------------------

    def _render_hero(self) -> str:
        ctx = self._context
        title = f"{ctx.get('workflow_name', 'Workflow')} Run Report"
        subtitle_parts: list[str] = []
        if ctx.get("prefix"):
            subtitle_parts.append(f"Prefix: {_esc(str(ctx['prefix']))}")
        if ctx.get("env"):
            subtitle_parts.append(f"Environment: {_esc(str(ctx['env']))}")
        if ctx.get("generated_at"):
            subtitle_parts.append(f"Generated: {_esc(str(ctx['generated_at']))}")
        return (
            "<div class='er-hero'>"
            f"<h2>{_esc(title)}</h2>"
            f"<div class='er-subtitle'>{' &middot; '.join(subtitle_parts)}</div>"
            "</div>"
        )

    def _render_toolbar(self) -> str:
        """Hero toolbar with download buttons for JSON receipt + CSV summaries.

        All downloads are baked in as base64 data URIs at render time so the
        HTML file remains self-contained — opens from disk or email and the
        downloads still work offline. Chart PNG/SVG export lives on the
        embedded Plotly modebar (rendered separately by
        :meth:`_render_workflow_dag`).
        """
        wf_name = self._context.get("workflow_name") or "workflow"
        stem = _slug(wf_name)
        buttons: list[str] = []

        # JSON receipt — always available, mirrors render_json() payload.
        try:
            payload = self.render_json(output_path=None)
            json_uri = _data_uri(json.dumps(payload, indent=2, default=str), "application/json")
            buttons.append(_dl_button(json_uri, f"{stem}_receipt.json", "JSON receipt"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Toolbar: failed to build JSON receipt link: %s", exc)

        # Step summary CSV — flattened from _step_summaries.
        if self._step_summaries:
            csv_text = _step_summaries_to_csv(self._step_summaries)
            buttons.append(_dl_button(_data_uri(csv_text, "text/csv"), f"{stem}_steps.csv", "Step summary CSV"))

        # Errors CSV — only when non-empty (no point in an empty file).
        if self._errors:
            errs_csv = _errors_to_csv(self._errors)
            buttons.append(_dl_button(_data_uri(errs_csv, "text/csv"), f"{stem}_errors.csv", "Errors CSV"))

        if not buttons:
            return ""
        return "<div class='er-toolbar'>" + "".join(buttons) + "</div>"

    def _render_workflow_dag(self) -> str:
        """Embed the workflow DAG (when ``set_workflow_dag(fig)`` was called).

        Uses Plotly's HTML snippet (``include_plotlyjs="cdn"`` so the report
        stays small — first open requires network for plotly-latest.min.js
        but subsequent opens hit the browser cache). The Plotly modebar
        exposes PNG (custom filename, 2× scale) and the standard zoom /
        pan controls.
        """
        fig = self._workflow_dag_fig
        if fig is None:
            return ""

        try:
            wf_name = self._context.get("workflow_name") or "workflow"
            stem = f"{_slug(wf_name)}_dag_results"
            config = {
                "displaylogo": False,
                "displayModeBar": True,
                "toImageButtonOptions": {
                    "filename": stem,
                    "format": "png",
                    "scale": 2,
                },
            }
            snippet = fig.to_html(include_plotlyjs="cdn", full_html=False, config=config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to render workflow DAG into report: %s", exc)
            return ""

        return f"<div class='er-section'><h3>Workflow DAG (run results)</h3>{snippet}</div>"

    def _render_run_summary(self) -> str:
        ctx = self._context
        total_steps = len(self._step_summaries)
        failed_steps = sum(1 for s in self._step_summaries if s.get("errors", 0))
        skipped_steps = sum(1 for s in self._step_summaries if s.get("skipped"))
        successful_steps = total_steps - failed_steps - skipped_steps
        total_rows = sum(s.get("rows", 0) for s in self._step_summaries)
        total_errors = len(self._errors)

        cards: list[tuple[str, str, str]] = [
            ("Entities Loaded", str(ctx.get("entity_count", 0)), ""),
            ("Steps", str(total_steps), ""),
            ("Successful", str(successful_steps), "success"),
            ("Skipped", str(skipped_steps), ""),
            ("Failed", str(failed_steps), "danger" if failed_steps else ""),
            ("Total Output Rows", str(total_rows), ""),
            ("Total Errors", str(total_errors), "danger" if total_errors else ""),
        ]
        if self._elapsed_seconds is not None:
            cards.append(("Duration", _format_duration(self._elapsed_seconds), ""))
        if ctx.get("report_path"):
            cards.append(("Report Path", _esc(str(ctx["report_path"])), ""))

        parts = ["<div class='er-section'>", "<div class='er-stats'>"]
        for label, value, css_class in cards:
            cls = f" {css_class}" if css_class else ""
            parts.append(
                f"<div class='er-stat-card'><div class='label'>{label}</div><div class='value{cls}'>{value}</div></div>"
            )
        parts.append("</div></div>")
        return "\n".join(parts)

    def _render_step_table(self) -> str:
        parts = [
            "<div class='er-section'>",
            f"<h3>Steps ({len(self._step_summaries)})</h3>",
            "<table class='er-table'>",
            "<thead><tr><th>Step</th><th>Kind</th><th>Rows</th><th>Errors</th><th>Status</th></tr></thead>",
            "<tbody>",
        ]
        any_cache = any("cache" in s for s in self._step_summaries)
        if any_cache:
            parts[3] = (
                "<thead><tr><th>Step</th><th>Kind</th><th>Rows</th><th>Errors</th>"
                "<th>Cache</th><th>Status</th></tr></thead>"
            )

        for summary in self._step_summaries:
            name = _esc(str(summary.get("name", "")))
            kind = _esc(str(summary.get("kind", "step")))
            rows = summary.get("rows", 0)
            errors = summary.get("errors", 0)
            row_class = " class='er-error-row'" if errors else ""
            if summary.get("skipped"):
                status_html = _esc(f"skipped ({summary.get('reason', 'cascade')})")
            elif errors:
                status_html = "<span style='color:#b91c1c;font-weight:600'>failed</span>"
            else:
                status_html = "<span style='color:#15803d;font-weight:600'>ok</span>"

            cache_cell = ""
            if any_cache:
                cache_info = summary.get("cache") or {}
                hits = cache_info.get("hits")
                misses = cache_info.get("misses")
                if hits is not None or misses is not None:
                    cache_cell = f"<td>{_esc(f'{hits or 0}/{misses or 0}')}</td>"
                else:
                    cache_cell = "<td>-</td>"

            parts.append(
                f"<tr{row_class}><td>{name}</td><td>{kind}</td>"
                f"<td>{rows}</td><td>{errors}</td>"
                f"{cache_cell}<td>{status_html}</td></tr>"
            )

        parts.append("</tbody></table></div>")
        return "\n".join(parts)

    def _render_errors(self) -> str:
        errors = self._errors[: self.max_errors_shown]
        truncated = len(self._errors) - len(errors)

        parts = [
            "<div class='er-section'>",
            f"<h3>Errors ({len(self._errors)} total)</h3>",
            "<table class='er-table'>",
            "<thead><tr><th>Step</th><th>Entity ID</th><th>Message</th></tr></thead>",
            "<tbody>",
        ]
        for err in errors:
            step = _esc(str(err.get("step", "")))
            eid = _esc(str(err.get("entity_id", "")))
            msg = _esc(str(err.get("message", err.get("error", ""))))
            parts.append(f"<tr class='er-error-row'><td>{step}</td><td>{eid}</td><td>{msg}</td></tr>")
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
        params = ctx.get("parameters") or {}
        mapping = ctx.get("column_mapping") or {}
        resolved = ctx.get("resolved_yaml")

        if not params and not mapping and resolved is None:
            return ""

        parts = ["<div class='er-section er-params'>"]
        if params:
            params_json = json.dumps(params, indent=2, default=str)
            parts.append(f"<details><summary>Run Parameters</summary><pre>{_esc(params_json)}</pre></details>")
        if mapping:
            mapping_json = json.dumps(mapping, indent=2, default=str)
            parts.append(f"<details><summary>Column Mapping</summary><pre>{_esc(mapping_json)}</pre></details>")
        if resolved is not None:
            resolved_json = json.dumps(resolved, indent=2, default=str)
            parts.append(
                f"<details><summary>Resolved Workflow YAML</summary><pre>{_esc(resolved_json)}</pre></details>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _render_data_preview(self) -> str:
        """Render the head() of the largest step's results_df.

        Off by default — workflows often have many wide DataFrames and
        previewing all of them blows up the report. When ``include_data_preview``
        is True we pick the step with the most rows as the representative.
        """
        if not self._raw_results:
            return ""
        candidates = []
        for name, step in self._raw_results.items():
            df, *_ = _unpack_step_result(step)
            if df is not None and not df.empty:
                candidates.append((name, df))
        if not candidates:
            return ""
        name, df = max(candidates, key=lambda t: len(t[1]))
        preview = df.head(self.preview_rows)
        cols = list(preview.columns)
        parts = [
            "<div class='er-section'>",
            f"<h3>Data Preview — {_esc(name)} (first {len(preview)} of {len(df)} rows)</h3>",
            "<table class='er-table'><thead><tr>",
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

    def _render_footer(self) -> str:
        return "<div class='er-footer'>Generated by earthdaily-agriculture WorkflowRunReporter</div>"


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def _unpack_step_result(
    step_res: Any,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]], list[str], bool, str | None, dict[str, Any] | None]:
    """Normalize one step's result into (df, errors, failed_ids, skipped, reason, cache).

    Tolerates the three shapes a step can land in:
      1. The standard dict from ``run_workflow``:
         ``{"results_df": df, "global_errors": [...], "failed_ids": [...], ...}``
      2. A bare DataFrame (older transform contract).
      3. ``None`` or any other unexpected value (treated as an empty step).
    """
    if isinstance(step_res, pd.DataFrame):
        return step_res, [], [], False, None, None
    if not isinstance(step_res, dict):
        return None, [], [], False, None, None

    df = step_res.get("results_df")
    if df is None:
        # Some transforms stash the DataFrame at the top level under "data"
        # — be tolerant.
        df = step_res.get("data")
    if df is not None and not isinstance(df, pd.DataFrame):
        df = None

    errors = list(step_res.get("global_errors") or [])
    failed_ids = list(step_res.get("failed_ids") or [])
    skipped = bool(step_res.get("skipped"))
    reason = step_res.get("reason")
    cache = step_res.get("cache")
    if cache is not None and not isinstance(cache, dict):
        cache = None
    return df, errors, failed_ids, skipped, reason, cache


def _iso_utc_now() -> str:
    """Timezone-aware UTC ISO-8601 timestamp matching ExtractionReporter."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{seconds:.1f}s"


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(text, quote=True)


# ----------------------------------------------------------------------
# Toolbar helpers — self-contained download buttons baked as data URIs.
# Kept module-level so they can be unit-tested without instantiating the
# reporter, and so the HTML stays valid when opened from disk.
# ----------------------------------------------------------------------

_TOOLBAR_CSS = (
    "<style>"
    ".er-toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px;}"
    ".er-toolbar a{display:inline-flex;align-items:center;gap:6px;"
    "padding:6px 12px;border-radius:6px;background:#0f172a;color:#f8fafc;"
    "text-decoration:none;font-size:13px;font-weight:500;border:1px solid #1e293b;}"
    ".er-toolbar a:hover{background:#1e293b;}"
    ".er-toolbar a::before{content:'⬇';margin-right:2px;opacity:0.8;}"
    "</style>"
)


def _slug(text: str) -> str:
    """Lowercased ASCII slug; safe for use as a download filename stem."""
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)).strip("_")
    return out.lower() or "workflow"


def _data_uri(content: str, mime: str) -> str:
    """Encode ``content`` as a ``data:`` URI for use in an ``<a download>`` link."""
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _dl_button(href: str, filename: str, label: str) -> str:
    return f"<a href='{href}' download='{_esc(filename)}'>{_esc(label)}</a>"


def _step_summaries_to_csv(summaries: list[dict[str, Any]]) -> str:
    """Flatten ``_step_summaries`` into a CSV (one row per step).

    Fields: name, kind, rows, errors, skipped, reason, cache_hits, cache_misses.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "kind", "rows", "errors", "skipped", "reason", "cache_hits", "cache_misses"])
    for s in summaries:
        cache = s.get("cache") or {}
        writer.writerow(
            [
                s.get("name", ""),
                s.get("kind", ""),
                s.get("rows", 0),
                s.get("errors", 0),
                bool(s.get("skipped", False)),
                s.get("reason", "") or "",
                cache.get("hits", "") if isinstance(cache, dict) else "",
                cache.get("misses", "") if isinstance(cache, dict) else "",
            ]
        )
    return buf.getvalue()


def _errors_to_csv(errors: list[dict[str, Any]]) -> str:
    """Flatten the error list into a CSV (one row per error)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["step", "entity_id", "message"])
    for err in errors:
        writer.writerow(
            [
                err.get("step", ""),
                err.get("entity_id", ""),
                err.get("message", err.get("error", "")),
            ]
        )
    return buf.getvalue()
