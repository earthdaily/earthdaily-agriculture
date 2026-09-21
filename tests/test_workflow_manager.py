"""
Tests for WorkflowManager — focused on the `input_from` step field.

Strategy: bypass WorkflowManager.__init__ (which does auth/S3/env setup) and
manually set the attributes that load_workflow / run_workflow rely on. Use
transform-only steps (no extractor) so we can verify the entity_list each
step receives without invoking real API calls.
"""

import inspect
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public

# ---------------------------------------------------------------------------
# Module-level transforms — must be importable by name from the YAML configs
# (importlib.import_module("tests.test_workflow_manager")).
# ---------------------------------------------------------------------------

_RECORDED_INPUTS: dict = {}


def record_and_passthrough(entity_list, upstream_results, params):
    """Record entity_list under params['record_as'] and return it unchanged."""
    _RECORDED_INPUTS[params["record_as"]] = entity_list.copy()
    return entity_list


def make_marker_df(entity_list, upstream_results, params):
    """Return a synthetic DataFrame so downstream steps can detect they
    received the upstream's results_df, not the original entity_list."""
    return pd.DataFrame({"id": [f"marker_{params['marker']}"], "source": [params["marker"]]})


def reshape_and_record(entity_list, upstream_results, params):
    """Record the input then reshape — used to prove transform runs on the
    resolved entity_list (not the original)."""
    _RECORDED_INPUTS[params["record_as"]] = entity_list.copy()
    out = entity_list.copy()
    out["reshaped"] = True
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_recorded():
    _RECORDED_INPUTS.clear()
    yield
    _RECORDED_INPUTS.clear()


@pytest.fixture
def manager():
    """A WorkflowManager with __init__ bypassed."""
    with patch.object(WorkflowManager, "__init__", lambda self, *a, **kw: None):
        wm = WorkflowManager()
    wm.workflow_cfg = None
    wm.workflow_steps = None
    wm.step_map = None
    wm.workflow_results = {}
    wm.config = {}
    wm.bearer_token = "fake-token"
    wm.token_expiration = None
    wm.output_result_dir = "/tmp/output"
    wm.partial_result_dir = "/tmp/partials"
    wm.cache_dir = "/tmp/cache"
    wm.sfd_list = None
    return wm


@pytest.fixture
def original_entities():
    return pd.DataFrame({"id": ["orig_a", "orig_b"], "source": ["original", "original"]})


def _write_yaml(tmp_path, workflow_dict):
    p = tmp_path / "workflow.yml"
    p.write_text(yaml.dump({"workflow": workflow_dict}))
    return str(p)


def _record_transform(name, record_as):
    return {
        "transform": {
            "module": "tests.test_workflow_manager",
            "function": "record_and_passthrough",
            "params": {"record_as": record_as},
        }
    }


def _marker_transform(marker):
    return {
        "transform": {
            "module": "tests.test_workflow_manager",
            "function": "make_marker_df",
            "params": {"marker": marker},
        }
    }


# ---------------------------------------------------------------------------
# Resolution behavior tests
# ---------------------------------------------------------------------------


def test_input_from_original_overrides_depends_on(manager, original_entities, tmp_path):
    """input_from: 'original' wins over depends_on default."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "upstream", **_marker_transform("UP")},
            {
                "name": "downstream",
                "depends_on": "upstream",
                "input_from": "original",
                **_record_transform("downstream", "downstream"),
            },
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    received = _RECORDED_INPUTS["downstream"]
    pd.testing.assert_frame_equal(
        received.reset_index(drop=True),
        original_entities.reset_index(drop=True),
    )


def test_input_from_named_step_uses_results_df(manager, original_entities, tmp_path):
    """input_from: <step_name> reads that step's results_df."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "upstream", **_marker_transform("UP")},
            {
                "name": "downstream",
                "depends_on": "upstream",
                "input_from": "upstream",
                **_record_transform("downstream", "downstream"),
            },
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    received = _RECORDED_INPUTS["downstream"]
    assert list(received["id"]) == ["marker_UP"]
    assert list(received["source"]) == ["UP"]


def test_input_from_unknown_step_raises_at_load(manager, tmp_path):
    """Reference to a step that does not exist → ValueError at load time."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "a", **_marker_transform("A")},
            {"name": "b", "input_from": "does_not_exist", **_marker_transform("B")},
        ],
    }
    with pytest.raises(ValueError, match="unknown step"):
        manager.load_workflow(_write_yaml(tmp_path, cfg))


def test_input_from_later_step_raises_at_load(manager, tmp_path):
    """Reference to a step in a later execution level → ValueError at load time."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "a", "input_from": "b", **_marker_transform("A")},
            {"name": "b", "depends_on": "a", **_marker_transform("B")},
        ],
    }
    with pytest.raises(ValueError, match="earlier execution level"):
        manager.load_workflow(_write_yaml(tmp_path, cfg))


def test_input_from_parallel_step_raises_at_load(manager, tmp_path):
    """Reference to a step in the same execution level → ValueError at load time."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "a", **_marker_transform("A")},
            {"name": "b", "input_from": "a", **_marker_transform("B")},
            # c is parallel to b (both depend only on root); referencing it is illegal
            {"name": "c", "input_from": "b", **_marker_transform("C")},
        ],
    }
    # b and c are both at level 1 — c referencing b is same-level, must fail
    with pytest.raises(ValueError, match="earlier execution level"):
        manager.load_workflow(_write_yaml(tmp_path, cfg))


def test_input_from_omitted_with_depends_on_auto_chains(manager, original_entities, tmp_path):
    """Omitted input_from + single depends_on → uses upstream's results_df."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "upstream", **_marker_transform("UP")},
            {
                "name": "downstream",
                "depends_on": "upstream",
                **_record_transform("downstream", "downstream"),
            },
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    received = _RECORDED_INPUTS["downstream"]
    assert list(received["id"]) == ["marker_UP"]


def test_input_from_omitted_no_depends_on_uses_original(manager, original_entities, tmp_path):
    """Omitted input_from + no depends_on → uses original entity_list."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "solo", **_record_transform("solo", "solo")},
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    received = _RECORDED_INPUTS["solo"]
    pd.testing.assert_frame_equal(
        received.reset_index(drop=True),
        original_entities.reset_index(drop=True),
    )


def test_input_from_with_transform_runs_on_resolved_df(manager, original_entities, tmp_path):
    """input_from + transform → transform runs against the resolved DataFrame,
    not the original entity_list."""
    cfg = {
        "name": "test",
        "steps": [
            {"name": "upstream", **_marker_transform("UP")},
            {
                "name": "downstream",
                # Without depends_on, both steps land in execution level 0 and the
                # input_from validator rejects same-level references. Adding
                # depends_on is correct semantically — the transform consumes
                # upstream's results_df, so it must run after upstream.
                "depends_on": "upstream",
                "input_from": "upstream",
                "transform": {
                    "module": "tests.test_workflow_manager",
                    "function": "reshape_and_record",
                    "params": {"record_as": "downstream"},
                },
            },
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    received = _RECORDED_INPUTS["downstream"]
    # Transform was handed the upstream's results_df (marker rows), not the original
    assert list(received["id"]) == ["marker_UP"]
    assert "reshaped" not in received.columns

    # Transform output (with reshaped col) is what's stored as the step's result
    assert manager.workflow_results["downstream"]["results_df"]["reshaped"].all()


def test_input_from_empty_upstream_skips_step(manager, original_entities, tmp_path):
    """If the resolved upstream's results_df is empty, the step is skipped."""

    def _empty_df_transform(entity_list, upstream_results, params):
        return entity_list.iloc[0:0]

    # Inject the empty transform into this module's namespace so the YAML
    # importlib lookup can find it.
    import sys

    sys.modules[__name__]._empty_df_transform = _empty_df_transform

    cfg = {
        "name": "test",
        "steps": [
            {
                "name": "upstream",
                "transform": {
                    "module": "tests.test_workflow_manager",
                    "function": "_empty_df_transform",
                    "params": {},
                },
            },
            {
                "name": "downstream",
                "depends_on": "upstream",
                **_record_transform("downstream", "downstream"),
            },
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    manager.run_workflow(entity_list=original_entities)

    # Upstream produced no rows → downstream skipped, transform never ran
    assert "downstream" not in _RECORDED_INPUTS
    assert manager.workflow_results["downstream"]["skipped"] is True


# ---------------------------------------------------------------------------
# Visualization / inspection
# ---------------------------------------------------------------------------


def test_inspect_workflow_surfaces_input_from(manager, tmp_path):
    cfg = {
        "name": "test",
        "steps": [
            {"name": "a", **_marker_transform("A")},
            {"name": "b", "depends_on": "a", **_marker_transform("B")},
            {"name": "c", "depends_on": "a", "input_from": "original", **_marker_transform("C")},
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    info = manager.inspect_workflow()

    by_name = {s["name"]: s for s in info["steps"]}
    assert by_name["a"]["input_from"] == "original"
    assert by_name["a"]["input_from_explicit"] is False
    assert by_name["b"]["input_from"] == "a"  # default-to-depends_on
    assert by_name["b"]["input_from_explicit"] is False
    assert by_name["c"]["input_from"] == "original"  # explicit override
    assert by_name["c"]["input_from_explicit"] is True


def test_visualize_workflow_includes_input_from(manager, tmp_path):
    cfg = {
        "name": "test",
        "steps": [
            {"name": "a", **_marker_transform("A")},
            {"name": "b", "depends_on": "a", **_marker_transform("B")},
        ],
    }
    manager.load_workflow(_write_yaml(tmp_path, cfg))
    fig = manager.visualize_workflow()

    # Plotly Figure — first trace is the node trace; its hovertext list holds
    # one HTML blob per step containing the resolved input_from.
    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
    hover_blobs = list(fig.data[0].hovertext)
    combined = "\n".join(hover_blobs)
    assert "input_from:</b> original" in combined
    assert "input_from:</b> a" in combined


# ---------------------------------------------------------------------------
# run.params forwarding
#
# `_execute_single_step` used to call the run method with a HARDCODED kwarg list,
# so every option it did not name was dropped on the floor without a word. A step
# declaring `spatial_grouping: true` ran, produced correct data, and never grouped
# — the 2.5.7 headline feature was unreachable from the workflow runner. Nothing
# raised, so no test asserting "the output is right" could have caught it.
# ---------------------------------------------------------------------------

_RUN_CALLS: list = []


class RecordingExtractor:
    """Stand-in accepting the same options the real point-based extractors do."""

    def __init__(self, bearer_token, token_expiration, config=None, workflow_ref=None):
        self.column_mapping = {}

    def setup_recording_parameters(self, **kwargs):
        return None

    def process_recording_bulk_parallel(
        self,
        entity_list=None,
        params=None,
        max_workers=5,
        output_path=None,
        fail_safe=False,
        prefix=None,
        generate_report=False,
        skip_export=True,
        use_cache=None,
        spatial_grouping=False,
        spatial_precision=5,
        spatial_max_window_days=400,
        partial_frequency=50,
        page_limit=None,
    ):
        _RUN_CALLS.append(
            {
                "max_workers": max_workers,
                "skip_export": skip_export,
                "use_cache": use_cache,
                "spatial_grouping": spatial_grouping,
                "spatial_precision": spatial_precision,
                "spatial_max_window_days": spatial_max_window_days,
                "partial_frequency": partial_frequency,
                "page_limit": page_limit,
            }
        )
        return {"results_df": pd.DataFrame({"id": ["a"]}), "global_errors": [], "failed_ids": []}


def _recording_step(run_params):
    return {
        "name": "recording_step",
        "module": "tests.test_workflow_manager",
        "extractor": "RecordingExtractor",
        "setup": {"method": "setup_recording_parameters", "params": {}},
        "run": {"method": "process_recording_bulk_parallel", "params": run_params},
    }


@pytest.fixture
def run_manager(manager):
    manager.workflow_cfg = {"settings": {}}
    manager._run_prefix = None
    _RUN_CALLS.clear()
    yield manager
    _RUN_CALLS.clear()


class TestRunParamsForwarding:
    def test_spatial_and_cache_options_reach_the_extractor(self, run_manager, original_entities):
        """Criterion 1: a YAML step declaring spatial_grouping actually groups."""
        step = _recording_step(
            {
                "use_cache": True,
                "spatial_grouping": True,
                "spatial_precision": 5,
                "spatial_max_window_days": 1100,
            }
        )
        run_manager._execute_single_step(step, original_entities)

        assert len(_RUN_CALLS) == 1
        call = _RUN_CALLS[0]
        assert call["spatial_grouping"] is True, "spatial_grouping was dropped before reaching the extractor"
        assert call["use_cache"] is True
        assert call["spatial_precision"] == 5
        assert call["spatial_max_window_days"] == 1100

    def test_other_silently_dropped_options_are_forwarded_too(self, run_manager, original_entities):
        step = _recording_step({"partial_frequency": 25, "page_limit": 9000})
        run_manager._execute_single_step(step, original_entities)

        assert _RUN_CALLS[0]["partial_frequency"] == 25
        assert _RUN_CALLS[0]["page_limit"] == 9000

    def test_unknown_key_is_warned_by_name(self, run_manager, original_entities):
        """Criterion 2: a typo must be loud, not silent.

        Captured through a loguru sink — the package logs via loguru, which bypasses
        stdlib logging entirely, so caplog sees nothing.
        """
        from loguru import logger as _logger

        messages: list = []
        sink_id = _logger.add(messages.append, level="WARNING")
        try:
            step = _recording_step({"spatial_precison": 5})  # deliberate typo
            run_manager._execute_single_step(step, original_entities)
        finally:
            _logger.remove(sink_id)

        text = "".join(str(m) for m in messages)
        assert "spatial_precison" in text, f"typo not surfaced; warnings were: {text!r}"
        # ...and it must not have been forwarded as a real kwarg.
        assert _RUN_CALLS[0]["spatial_precision"] == 5  # untouched default

    def test_explicit_kwargs_still_win(self, run_manager, original_entities):
        """max_workers/skip_export keep their existing dedicated handling."""
        step = _recording_step({"max_workers": 3, "skip_export": False, "spatial_grouping": True})
        run_manager._execute_single_step(step, original_entities)

        assert _RUN_CALLS[0]["max_workers"] == 3
        assert _RUN_CALLS[0]["skip_export"] is False
        assert _RUN_CALLS[0]["spatial_grouping"] is True

    def test_forwarding_is_signature_driven_not_a_hardcoded_list(self):
        """Criterion 6: fail if the passthrough regresses to a fixed kwarg list.

        Asserted on the source because the defect was an *absence* — a hardcoded
        call site silently ignoring everything it did not name.
        """
        source = inspect.getsource(WorkflowManager._execute_single_step)
        assert "**passthrough" in source, (
            "the run call must forward remaining run.params; a hardcoded kwarg list "
            "silently drops every option it does not name"
        )
        assert "inspect.signature(run_fn)" in source, "forwarding must be filtered against the target signature"
