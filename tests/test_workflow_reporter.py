"""Tests for earthdaily.agriculture.reporting.workflow_reporter.WorkflowRunReporter.

Covers the six cases called out in the todo plus a handful of guard-rail
checks (state-order errors, render-to-file via fsspec helpers, kind dispatch,
non-dict step values, skipped steps).

WorkflowRunReporter operates over plain dicts and DataFrames — no network
or filesystem dependencies — so these tests run as pure unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.reporting import WorkflowRunReporter
from earthdaily.agriculture.reporting.workflow_reporter import SCHEMA_VERSION

pytestmark = pytest.mark.public

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter() -> WorkflowRunReporter:
    return WorkflowRunReporter()


@pytest.fixture
def populated(reporter: WorkflowRunReporter) -> WorkflowRunReporter:
    """A reporter with a run context set, ready for set_step_results."""
    reporter.set_run_context(
        workflow_name="emergence_greenness_disease",
        prefix="run_20260513",
        env="prod",
        entity_count=578,
        parameters={"limit": 10},
    )
    return reporter


# ---------------------------------------------------------------------------
# Case 1 — empty step dict
# ---------------------------------------------------------------------------


def test_empty_step_dict_renders_without_error(populated: WorkflowRunReporter) -> None:
    populated.set_step_results({}, elapsed_seconds=0.1)

    html = populated.render_html()
    assert "Run Report" in html
    # No step table when there are no steps to summarise.
    assert "<h3>Steps" not in html
    # No errors section either.
    assert "<h3>Errors" not in html

    payload = populated.render_json()
    assert payload["steps"] == []
    assert payload["errors"] == []
    assert payload["entities"]["loaded"] == 578


# ---------------------------------------------------------------------------
# Case 2 — mixed extractor + transform results
# ---------------------------------------------------------------------------


def test_mixed_extractor_and_transform_steps(populated: WorkflowRunReporter) -> None:
    extractor_df = pd.DataFrame({"entity_id": ["a", "b", "c"], "ndvi": [0.5, 0.6, 0.7]})
    transform_df = pd.DataFrame({"entity_id": ["a", "b"], "joined": [1, 2]})

    results = {
        "mrts": {"results_df": extractor_df, "global_errors": [], "failed_ids": []},
        "last_clear_image": {"results_df": transform_df, "global_errors": [], "failed_ids": []},
    }
    populated.set_step_results(
        results,
        elapsed_seconds=42.0,
        step_kinds={"mrts": "extractor", "last_clear_image": "transform"},
    )

    payload = populated.render_json()
    assert {s["name"] for s in payload["steps"]} == {"mrts", "last_clear_image"}
    by_name = {s["name"]: s for s in payload["steps"]}
    assert by_name["mrts"]["kind"] == "extractor"
    assert by_name["mrts"]["rows"] == 3
    assert by_name["last_clear_image"]["kind"] == "transform"
    assert by_name["last_clear_image"]["rows"] == 2

    # Step kinds surface in the HTML table too.
    html = populated.render_html()
    assert "<td>mrts</td>" in html and "<td>extractor</td>" in html
    assert "<td>last_clear_image</td>" in html and "<td>transform</td>" in html


# ---------------------------------------------------------------------------
# Case 3 — transform-returned bare DataFrame
# ---------------------------------------------------------------------------


def test_transform_returned_bare_dataframe(populated: WorkflowRunReporter) -> None:
    bare_df = pd.DataFrame({"entity_id": ["a", "b", "c", "d"], "x": [1, 2, 3, 4]})
    results = {
        "ok_step": {"results_df": pd.DataFrame({"entity_id": ["a"]})},
        "weird_transform": bare_df,  # Older transform contract: bare DF, no dict wrapper
        "none_step": None,  # Truly empty / not run
        "string_step": "skipped-elsewhere",  # Defensive: anything non-DataFrame, non-dict
    }
    populated.set_step_results(results, elapsed_seconds=1.0)

    by_name = {s["name"]: s for s in populated.render_json()["steps"]}

    assert by_name["weird_transform"]["rows"] == 4
    assert by_name["weird_transform"]["errors"] == 0
    assert by_name["none_step"]["rows"] == 0
    assert by_name["string_step"]["rows"] == 0


# ---------------------------------------------------------------------------
# Case 4 — errors with and without entity_id
# ---------------------------------------------------------------------------


def test_errors_with_and_without_entity_id(populated: WorkflowRunReporter) -> None:
    results = {
        "planted_area": {
            "results_df": pd.DataFrame({"entity_id": ["a"]}),
            "global_errors": [
                {"entity_id": "abc123", "message": "geometry rejected"},
                {"entity_id": "xyz789", "message": "timeout after 30s"},
            ],
            "failed_ids": ["abc123", "xyz789"],
        },
        "join_step": {
            "results_df": pd.DataFrame(),
            # Workflow-level error from a transform — no entity_id.
            "global_errors": [{"step": "join_step", "error": "KeyError: 'crop'"}],
            "failed_ids": [],
        },
    }
    populated.set_step_results(results, elapsed_seconds=5.0)
    payload = populated.render_json()

    # Three errors total across both shapes.
    assert len(payload["errors"]) == 3

    # Every error is tagged with its originating step (defaults to step name
    # for entity-style entries, preserves workflow-level "step" field).
    assert all("step" in err for err in payload["errors"])

    by_step = {err["step"]: err for err in payload["errors"] if "entity_id" not in err}
    assert "join_step" in by_step
    # Workflow-level error gets "message" populated from "error" for HTML rendering.
    assert by_step["join_step"]["message"] == "KeyError: 'crop'"

    # HTML surfaces both shapes; the entity-id column is empty for the
    # workflow-level error but the message column is populated.
    html = populated.render_html()
    assert "abc123" in html
    assert "KeyError: &#x27;crop&#x27;" in html  # html-escaped


# ---------------------------------------------------------------------------
# Case 5 — cache-hit metadata surfaced in the step table
# ---------------------------------------------------------------------------


def test_cache_metadata_in_step_table(populated: WorkflowRunReporter) -> None:
    results = {
        "cached_step": {
            "results_df": pd.DataFrame({"entity_id": ["a", "b"]}),
            "global_errors": [],
            "failed_ids": [],
            "cache": {"hits": 7, "misses": 3},
        },
        "uncached_step": {
            "results_df": pd.DataFrame({"entity_id": ["c"]}),
            "global_errors": [],
            "failed_ids": [],
            # No cache key → step should not surface a cache cell.
        },
    }
    populated.set_step_results(results, elapsed_seconds=1.0)
    html = populated.render_html()

    # Header gains a Cache column because at least one step has cache info.
    assert "<th>Cache</th>" in html
    # Cached step shows hits/misses.
    assert "7/3" in html

    # JSON receipt carries the cache dict through.
    cached_summary = next(s for s in populated.render_json()["steps"] if s["name"] == "cached_step")
    assert cached_summary["cache"] == {"hits": 7, "misses": 3}


def test_step_table_omits_cache_column_when_no_step_has_cache(populated: WorkflowRunReporter) -> None:
    populated.set_step_results(
        {"a": {"results_df": pd.DataFrame({"x": [1]})}},
        elapsed_seconds=0.1,
    )
    html = populated.render_html()
    assert "<th>Cache</th>" not in html


# ---------------------------------------------------------------------------
# Case 6 — JSON schema_version round-trip
# ---------------------------------------------------------------------------


def test_json_schema_version_and_shape(populated: WorkflowRunReporter, tmp_path: Path) -> None:
    populated.set_step_results(
        {"step1": {"results_df": pd.DataFrame({"x": [1, 2]})}},
        elapsed_seconds=12.5,
    )
    populated.amend_run_context(report_path="results/final.csv")

    out_path = tmp_path / "receipt.json"
    payload = populated.render_json(output_path=str(out_path))

    # In-memory payload sanity.
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["workflow"] == {
        "name": "emergence_greenness_disease",
        "prefix": "run_20260513",
        "env": "prod",
    }
    assert payload["entities"] == {"loaded": 578, "limited_to": None}
    assert payload["duration_seconds"] == 12.5
    assert payload["report_path"] == "results/final.csv"
    assert payload["generated_at"].endswith("Z")

    # File round-trip: written content == in-memory payload (modulo JSON).
    on_disk = json.loads(out_path.read_text())
    assert on_disk == payload


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_render_html_without_context_raises(reporter: WorkflowRunReporter) -> None:
    with pytest.raises(RuntimeError, match="set_run_context"):
        reporter.render_html()


def test_render_json_without_context_raises(reporter: WorkflowRunReporter) -> None:
    with pytest.raises(RuntimeError, match="set_run_context"):
        reporter.render_json()


def test_amend_run_context_without_set_raises(reporter: WorkflowRunReporter) -> None:
    with pytest.raises(RuntimeError, match="set_run_context"):
        reporter.amend_run_context(report_path="x")


def test_amend_run_context_updates_fields_and_refreshes_timestamp(
    populated: WorkflowRunReporter,
) -> None:
    # Control the clock so "amend refreshes the timestamp" is verified
    # deterministically. Real microsecond timestamps can collide when two
    # `_iso_utc_now()` calls land in the same microsecond on a fast machine,
    # which made the bare `!=` assertion flaky.
    clock = iter(f"2026-01-01T00:00:{i:02d}.000000Z" for i in range(60))
    with patch(
        "earthdaily.agriculture.reporting.workflow_reporter._iso_utc_now",
        side_effect=lambda: next(clock),
    ):
        populated.set_step_results({}, elapsed_seconds=0.0)
        first_ts = populated.render_json()["generated_at"]
        populated.amend_run_context(report_path="results/post-hoc.csv", custom_tag="x")
        payload = populated.render_json()

    assert payload["report_path"] == "results/post-hoc.csv"
    # Refreshed timestamp on amend — verifies the contract documented on
    # the method.
    assert payload["generated_at"] != first_ts


def test_render_html_writes_file_via_fsspec_helper(populated: WorkflowRunReporter, tmp_path: Path) -> None:
    populated.set_step_results({}, elapsed_seconds=0.0)
    out_path = tmp_path / "subdir" / "report.html"  # parent doesn't exist yet
    html = populated.render_html(output_path=str(out_path))
    # write_text helper creates parent dirs for local paths.
    assert out_path.read_text(encoding="utf-8") == html


def test_skipped_step_classified_as_skipped(populated: WorkflowRunReporter) -> None:
    results = {
        "ok": {"results_df": pd.DataFrame({"x": [1]})},
        "disabled": {
            "results_df": pd.DataFrame(),
            "global_errors": [],
            "failed_ids": [],
            "skipped": True,
            "reason": "disabled",
        },
    }
    populated.set_step_results(results, elapsed_seconds=1.0)
    by_name = {s["name"]: s for s in populated.render_json()["steps"]}
    assert by_name["disabled"]["skipped"] is True
    assert by_name["disabled"]["reason"] == "disabled"

    html = populated.render_html()
    assert "skipped (disabled)" in html


def test_step_kind_defaults_to_generic_when_not_provided(populated: WorkflowRunReporter) -> None:
    """Standalone callers that don't pass step_kinds get kind='step' for every entry."""
    populated.set_step_results(
        {"unnamed": {"results_df": pd.DataFrame({"x": [1]})}},
        elapsed_seconds=0.1,
    )
    payload = populated.render_json()
    assert payload["steps"][0]["kind"] == "step"


def test_parameters_block_renders_only_when_populated(
    populated: WorkflowRunReporter,
) -> None:
    populated.set_step_results({}, elapsed_seconds=0.0)
    html = populated.render_html()
    # populated fixture passed parameters={"limit": 10} → parameters details block present.
    assert "Run Parameters" in html


def test_parameters_block_omitted_when_no_params(reporter: WorkflowRunReporter) -> None:
    reporter.set_run_context(
        workflow_name="bare",
        prefix=None,
        env="prod",
        entity_count=0,
    )
    reporter.set_step_results({}, elapsed_seconds=0.0)
    html = reporter.render_html()
    assert "Run Parameters" not in html
    assert "Column Mapping" not in html
    assert "Resolved Workflow YAML" not in html


def test_resolved_yaml_included_only_when_provided(reporter: WorkflowRunReporter) -> None:
    """Per todo open question #1, resolved YAML is opt-in via the context."""
    reporter.set_run_context(
        workflow_name="bare",
        env="prod",
        entity_count=1,
        resolved_yaml={"workflow": {"name": "bare", "steps": []}},
    )
    reporter.set_step_results({}, elapsed_seconds=0.0)

    html = reporter.render_html()
    assert "Resolved Workflow YAML" in html

    payload = reporter.render_json()
    assert payload["resolved_yaml"] == {"workflow": {"name": "bare", "steps": []}}


def test_errors_truncated_to_max_errors_shown_in_html() -> None:
    reporter = WorkflowRunReporter(max_errors_shown=2)
    reporter.set_run_context(workflow_name="w", env="prod", entity_count=100)
    results = {
        "step": {
            "results_df": pd.DataFrame(),
            "global_errors": [{"entity_id": f"e{i}", "message": f"boom {i}"} for i in range(5)],
            "failed_ids": [f"e{i}" for i in range(5)],
        }
    }
    reporter.set_step_results(results, elapsed_seconds=0.0)

    html = reporter.render_html()
    # Only the first 2 errors render in the table.
    assert "boom 0" in html and "boom 1" in html
    assert "boom 2" not in html
    # Truncation hint is present.
    assert "... and 3 more errors not shown" in html

    # JSON receipt carries ALL errors regardless of the HTML truncation knob.
    assert len(reporter.render_json()["errors"]) == 5
