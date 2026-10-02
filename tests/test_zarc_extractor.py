"""
ZARCExtractor — run as a workflow step, run-level params, and setup defaults.

The bulk method used to have no ``params`` argument, so every WorkflowManager step
using ZARC raised ``TypeError`` before a single request (direct notebook calls were
fine, which is why it went unnoticed). Separately, a row without a crop sent
``crop=None`` to the API instead of the setup default.

No network: ``requests.post`` is patched and the URL / payload it receives is what
these tests assert on.
"""

import inspect
import time
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from earthdaily.agriculture.processors.processor_zarc_functions import ZARCExtractor
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public

POST = "earthdaily.agriculture.processors.processor_zarc_functions.requests.post"
GEOM = "POLYGON ((-50 -15, -50 -14.99, -49.99 -14.99, -50 -15))"


def _ok_response(entity_id="f1"):
    resp = Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "id": entity_id,
        "data": {"status": "OK", "emergence_date": "2025-10-01", "sowing_date": "2025-09-11"},
    }
    return resp


def _query(post_mock, call=0):
    """The ZARC query string of the n-th POST, as {key: value}."""
    url = post_mock.call_args_list[call].args[0]
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


@pytest.fixture
def extractor(tmp_path):
    ex = ZARCExtractor(
        "fake-token",
        time.time() + 3600,
        {"env": "prod", "output_result_dir": str(tmp_path), "project_root": str(tmp_path)},
    )
    return ex


@pytest.fixture
def manager(tmp_path):
    """A WorkflowManager with __init__ bypassed (same approach as test_workflow_manager)."""
    with patch.object(WorkflowManager, "__init__", lambda self, *a, **kw: None):
        wm = WorkflowManager()
    wm.workflow_cfg = {"settings": {}}
    wm.workflow_steps = None
    wm.step_map = None
    wm.workflow_results = {}
    wm.config = {"env": "prod", "output_result_dir": str(tmp_path), "project_root": str(tmp_path)}
    wm.bearer_token = "fake-token"
    wm.token_expiration = time.time() + 3600
    wm.output_result_dir = str(tmp_path)
    wm.partial_result_dir = str(tmp_path)
    wm.cache_dir = str(tmp_path / "cache")
    wm.sfd_list = None
    wm._run_prefix = None
    return wm


def _zarc_step(setup_params=None, run_params=None):
    return {
        "name": "zarc",
        "module": "earthdaily.agriculture.processors.processor_zarc_functions",
        "extractor": "ZARCExtractor",
        "setup": {"method": "setup_zarc_parameters", "params": {"use_cache": False, **(setup_params or {})}},
        "run": {"method": "process_zarc_bulk_extraction_parallel", "params": run_params or {}},
    }


class TestWorkflowStep:
    def test_zarc_runs_as_a_workflow_step(self, manager):
        """The regression: WorkflowManager always passes params=, which used to TypeError."""
        entities = pd.DataFrame([{"id": "f1", "geometry": GEOM, "emergence_date": "2025-10-01"}])
        with patch(POST, return_value=_ok_response()) as post:
            result = manager._execute_single_step(_zarc_step({"crop": "SOYBEANS"}), entities)

        assert post.call_count == 1
        assert len(result["results_df"]) == 1
        assert result["failed_ids"] == []
        assert _query(post)["crop"] == "SOYBEANS"

    def test_run_level_params_reach_the_request(self, manager):
        """`run.params.params` is how a workflow overrides setup for one step."""
        entities = pd.DataFrame([{"id": "f1", "geometry": GEOM}])  # no emergence_date, no crop
        step = _zarc_step({"crop": "SOYBEANS"}, {"params": {"emergence_date": "2025-11-01", "crop": "CORN"}})
        with patch(POST, return_value=_ok_response()) as post:
            result = manager._execute_single_step(step, entities)

        assert result["failed_ids"] == []
        q = _query(post)
        assert q["date_emergence"] == "2025-11-01"
        assert q["crop"] == "CORN"


class TestDefaults:
    def test_setup_crop_applies_when_the_row_has_none(self, extractor):
        """A row without a crop used to send `crop=None`."""
        extractor.setup_zarc_parameters(crop="SOYBEANS", use_cache=False)
        with patch(POST, return_value=_ok_response()) as post:
            extractor.process_single_entity_zarc({"id": "f1", "geometry": GEOM, "emergence_date": "2025-10-01"})
        assert _query(post)["crop"] == "SOYBEANS"

    def test_row_crop_still_wins(self, extractor):
        extractor.setup_zarc_parameters(crop="SOYBEANS", use_cache=False)
        row = {"id": "f1", "geometry": GEOM, "emergence_date": "2025-10-01", "crop": "CORN"}
        with patch(POST, return_value=_ok_response()) as post:
            extractor.process_single_entity_zarc(row)
        assert _query(post)["crop"] == "CORN"

    def test_setup_emergence_date_fills_rows_without_one(self, extractor):
        extractor.setup_zarc_parameters(emergence_date="2025-10-15T00:00:00", use_cache=False)
        assert extractor.zarc_params["emergence_date"] == "2025-10-15"
        df = pd.DataFrame(
            [{"id": "f1", "geometry": GEOM}, {"id": "f2", "geometry": GEOM, "emergence_date": "2025-12-01"}]
        )
        with patch(POST, side_effect=[_ok_response("f1"), _ok_response("f2")]) as post:
            result = extractor.process_zarc_bulk_extraction_parallel(df, max_workers=1, skip_export=True)

        assert result["failed_ids"] == []
        sent = sorted(_query(post, i)["date_emergence"] for i in range(2))
        assert sent == ["2025-10-15", "2025-12-01"]  # the row's own date wins

    def test_invalid_setup_emergence_date_is_rejected(self, extractor):
        with pytest.raises(ValueError, match="emergence_date"):
            extractor.setup_zarc_parameters(emergence_date="not-a-date")

    def test_unset_emergence_date_leaves_the_cache_key_unchanged(self, extractor):
        """zarc_params is the cache key — an unused option must not add a key."""
        extractor.setup_zarc_parameters()
        assert "emergence_date" not in extractor.zarc_params

    def test_missing_emergence_date_everywhere_is_a_per_entity_error(self, extractor):
        extractor.setup_zarc_parameters(use_cache=False)
        df = pd.DataFrame([{"id": "f1", "geometry": GEOM}])
        with patch(POST) as post:
            result = extractor.process_zarc_bulk_extraction_parallel(df, max_workers=1, skip_export=True)
        post.assert_not_called()
        assert result["failed_ids"] == ["f1"]


def test_params_is_appended_so_positional_callers_are_unaffected():
    """Notebooks call this method with keywords, but anything positional must not shift."""
    names = list(inspect.signature(ZARCExtractor.process_zarc_bulk_extraction_parallel).parameters)
    assert names[:3] == ["self", "entity_list", "max_workers"]
    assert names[-1] == "params"


class TestYamlDates:
    """An unquoted YAML date (`emergence_date: 2025-11-01`) loads as a datetime.date.

    normalize_date passed date objects through unchanged, so they reached strptime and
    every row failed after five retries with an unhelpful message.
    """

    def test_unquoted_yaml_dates_work_in_a_workflow_step(self, manager, tmp_path):
        import yaml

        step_yaml = """
name: zarc
module: earthdaily.agriculture.processors.processor_zarc_functions
extractor: ZARCExtractor
setup:
  method: setup_zarc_parameters
  params: {crop: SOYBEANS, use_cache: false, emergence_date: 2025-10-01}
run:
  method: process_zarc_bulk_extraction_parallel
  params:
    params: {emergence_date: 2025-11-01}
"""
        step = yaml.safe_load(step_yaml)
        assert type(step["run"]["params"]["params"]["emergence_date"]).__name__ == "date"  # the trap
        entities = pd.DataFrame([{"id": "f1", "geometry": GEOM}])
        with patch(POST, return_value=_ok_response()) as post:
            result = manager._execute_single_step(step, entities)

        assert result["failed_ids"] == []
        assert _query(post)["date_emergence"] == "2025-11-01"  # run-level wins over setup

    def test_a_date_object_in_setup_is_stored_as_a_string(self, extractor):
        from datetime import date

        extractor.setup_zarc_parameters(emergence_date=date(2025, 10, 1), use_cache=False)
        assert extractor.zarc_params["emergence_date"] == "2025-10-01"


def test_blank_nb_days_cell_falls_back_to_the_run_override(extractor):
    """A blank CSV cell ("") used to bypass the run-level override and get the setup value."""
    extractor.setup_zarc_parameters(nb_days_sowing_emergence=20, use_cache=False)
    df = pd.DataFrame([{"id": "f1", "geometry": GEOM, "emergence_date": "2025-10-01", "nb_days_sowing_emergence": ""}])
    with patch(POST, return_value=_ok_response()) as post:
        extractor.process_zarc_bulk_extraction_parallel(
            df, max_workers=1, skip_export=True, params={"nb_days_sowing_emergence": 30}
        )
    assert _query(post)["nb_days_sowing_emergence"] == "30"
