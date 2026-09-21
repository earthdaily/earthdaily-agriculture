"""
Tests for WorkflowManager covering the runner surface exercised in
``EDAgro_Workflow_Runner_Dev.ipynb``:

    - ``__init__`` and module wiring (auth/S3 mocked out)
    - ``load_workflow`` validation: file presence, missing/wrong top-level keys,
      duplicate / missing step names, cycle detection, legacy 'analytics' format
    - ``inspect_workflow`` output shape (name / settings / steps / execution_levels,
      transform vs extractor classification)
    - ``visualize_workflow`` happy path + load-required guard
    - ``run_workflow`` precondition errors and ``self.sfd_list`` fallback
    - ``run_workflow`` end-to-end with extractor configs (notebook cells 28/30)

The existing ``test_workflow_manager.py`` already covers ``input_from`` resolution
exhaustively — these tests deliberately do NOT duplicate that surface.

Strategy: bypass ``WorkflowManager.__init__`` for runner tests (same pattern as
the existing test file). For ``__init__`` itself, mock the auth/S3/env dependencies.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public

# ---------------------------------------------------------------------------
# Module-level transforms / extractor stubs used in the YAML configs
# (importlib resolves them via "tests.test_workflow_runner.<name>")
# ---------------------------------------------------------------------------

_RECORDED: dict = {}


def _passthrough(entity_list, upstream_results, params):
    """Return entity_list unchanged."""
    return entity_list


def _raise_in_transform(entity_list, upstream_results, params):
    """Always raises — used to exercise run_workflow's try/finally restore path."""
    raise RuntimeError("boom from _raise_in_transform")


def _marker_df(entity_list, upstream_results, params):
    return pd.DataFrame({"id": [f"marker_{params['marker']}"], "source": [params["marker"]]})


class _StubExtractor:
    """Tiny extractor stand-in for run_workflow tests.

    The runner calls ``__init__(bearer_token, token_expiration, config, workflow_ref=...)``,
    then ``setup_xxx(...)`` and ``run_xxx(entity_list=..., **bulk_kwargs)``.

    The ``column_mapping`` attribute is required because the runner does
    ``extractor.column_mapping.update(col_mapping)`` when settings define one.
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        self.bearer_token = bearer_token
        self.token_expiration = token_expiration
        self.config = config
        self.workflow_ref = workflow_ref
        self.setup_calls = []
        self.run_calls = []
        self.column_mapping = {}

    def setup_stub(self, **kwargs):
        self.setup_calls.append(kwargs)

    def run_stub(self, entity_list, **kwargs):
        # Record the call so tests can assert on the kwargs the runner forwarded.
        self.run_calls.append(kwargs)
        out = entity_list.copy()
        out["stub_ran"] = True
        # Honor skip_export by not writing anything; if a test wants a side
        # effect on output_path it can patch this method directly.
        if not kwargs.get("skip_export", False) and kwargs.get("output_path"):
            try:
                import os

                os.makedirs(kwargs["output_path"], exist_ok=True)
                with open(os.path.join(kwargs["output_path"], "stub_export.csv"), "w") as f:
                    f.write("id\n" + "\n".join(out["id"].astype(str).tolist()))
            except Exception:
                pass
        return {
            "results_df": out,
            "global_errors": [],
            "failed_ids": [],
            "total_calculations": len(entity_list),
            "successful_calculations": len(entity_list),
            "failed_calculations": 0,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_recorded():
    _RECORDED.clear()
    yield
    _RECORDED.clear()


@pytest.fixture
def manager():
    """A WorkflowManager with __init__ bypassed (matches the pattern in
    test_workflow_manager.py)."""
    with patch.object(WorkflowManager, "__init__", lambda self, *a, **kw: None):
        wm = WorkflowManager()
    wm.workflow_cfg = None
    wm.workflow_steps = None
    wm.step_map = None
    wm.workflow_results = {}
    wm.config = {"env": "preprod"}
    wm.bearer_token = "fake-token"
    wm.token_expiration = None
    wm.output_result_dir = "/tmp/output"
    wm.partial_result_dir = "/tmp/partials"
    wm.cache_dir = "/tmp/cache"
    wm.sfd_list = None
    wm._run_prefix = None
    return wm


@pytest.fixture
def entities():
    return pd.DataFrame({"id": ["a", "b", "c"], "geometry": ["P((0))", "P((1))", "P((2))"]})


def _write_yaml(tmp_path, workflow_dict, key="workflow"):
    p = tmp_path / "workflow.yml"
    p.write_text(yaml.dump({key: workflow_dict}))
    return str(p)


def _passthrough_step(name, depends_on=None, input_from=None):
    step = {
        "name": name,
        "transform": {
            "module": "tests.test_workflow_runner",
            "function": "_passthrough",
            "params": {},
        },
    }
    if depends_on is not None:
        step["depends_on"] = depends_on
    if input_from is not None:
        step["input_from"] = input_from
    return step


# ===================================================================
# WorkflowManager.__init__ — basic wiring (auth/S3 mocked)
# ===================================================================


class TestWorkflowManagerInit:
    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_wires_config_and_authenticator(self, mock_setup_env, mock_authenticator, mock_setup_logging):
        """Constructor reads env from setup_environment, builds an EDAuthenticator,
        and copies its token onto the manager."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        wm = WorkflowManager(env="preprod")

        assert wm.env == "preprod"
        assert wm.project_root == "/proj"
        assert wm.bearer_token == "tok"
        assert wm.output_result_dir == "/proj/results"
        assert wm.partial_result_dir == "/proj/partials"
        assert wm.cache_dir == "/proj/cache"
        # Workflow runner state initialized to empty
        assert wm.workflow_cfg is None
        assert wm.workflow_steps is None
        assert wm.step_map is None
        assert wm.workflow_results == {}
        assert wm.sfd_list is None
        # S3 init was attempted on the authenticator instance
        mock_auth_instance.initialize_s3_client.assert_called_once()

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_propagates_log_to_console_only_to_setup_logging(
        self, mock_setup_env, mock_authenticator, mock_setup_logging
    ):
        """log_to_console_only kwarg must reach setup_logging so containerized
        deploys can opt out of the file sink."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        WorkflowManager(env="preprod", log_to_console_only=True)

        assert mock_setup_logging.call_count == 1
        kwargs = mock_setup_logging.call_args.kwargs
        assert kwargs["log_to_console_only"] is True

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_path_kwargs_override_config(self, mock_setup_env, mock_authenticator, mock_setup_logging):
        """output_result_dir / partial_result_dir / cache_dir kwargs override the
        defaults setup_environment returned, both in manager.config and on the
        mirrored attributes — so they can't drift after construction."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        wm = WorkflowManager(
            env="preprod",
            output_result_dir="s3://my-bucket/runs/2026-05-12/results",
            partial_result_dir="s3://my-bucket/runs/2026-05-12/partials",
            cache_dir="s3://my-bucket/runs/2026-05-12/cache",
        )

        # Both the config dict and the mirrored attribute reflect the kwarg.
        for key, attr, expected in [
            ("output_result_dir", "output_result_dir", "s3://my-bucket/runs/2026-05-12/results"),
            ("partial_result_dir", "partial_result_dir", "s3://my-bucket/runs/2026-05-12/partials"),
            ("cache_dir", "cache_dir", "s3://my-bucket/runs/2026-05-12/cache"),
        ]:
            assert wm.config[key] == expected, f"config[{key!r}] not overridden"
            assert getattr(wm, attr) == expected, f"wm.{attr} not overridden"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_no_storage_kwargs_keeps_setup_environment_defaults(
        self, mock_setup_env, mock_authenticator, mock_setup_logging
    ):
        """No storage kwargs → manager.config and the mirrored attributes keep
        whatever setup_environment returned (the local-default contract from
        doc 13)."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        wm = WorkflowManager(env="preprod")

        assert wm.config["output_result_dir"] == "/proj/results"
        assert wm.output_result_dir == "/proj/results"
        assert wm.config["partial_result_dir"] == "/proj/partials"
        assert wm.partial_result_dir == "/proj/partials"
        assert wm.config["cache_dir"] == "/proj/cache"
        assert wm.cache_dir == "/proj/cache"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_partial_storage_kwargs_only_override_those(
        self, mock_setup_env, mock_authenticator, mock_setup_logging
    ):
        """Passing only one of the three storage kwargs overrides just that
        one — the other two keep their setup_environment defaults."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        wm = WorkflowManager(env="preprod", output_result_dir="s3://only-this/results")

        assert wm.output_result_dir == "s3://only-this/results"
        assert wm.partial_result_dir == "/proj/partials"
        assert wm.cache_dir == "/proj/cache"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_edagro_output_prefix_env_var_routes_all_three_paths(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """EDAGRO_OUTPUT_PREFIX env var is the canonical Pattern B (docs/site/agriculture/14)
        invocation: orchestrators pass it once, we derive /results, /partials,
        /cache. Trailing slash on the prefix is tolerated."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", "s3://my-bucket/runs/2026-01-01/")

        wm = WorkflowManager(env="preprod")

        assert wm.output_result_dir == "s3://my-bucket/runs/2026-01-01/results"
        assert wm.partial_result_dir == "s3://my-bucket/runs/2026-01-01/partials"
        assert wm.cache_dir == "s3://my-bucket/runs/2026-01-01/cache"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_explicit_kwargs_win_over_edagro_output_prefix_env(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """Kwargs > env var > setup_environment defaults. An explicit kwarg
        overrides the env-var-derived path for that one path only."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", "s3://env-bucket/run")

        wm = WorkflowManager(env="preprod", output_result_dir="s3://explicit/results")

        # output_result_dir: kwarg wins
        assert wm.output_result_dir == "s3://explicit/results"
        # partials and cache: env var still wins (no kwarg)
        assert wm.partial_result_dir == "s3://env-bucket/run/partials"
        assert wm.cache_dir == "s3://env-bucket/run/cache"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_unset_edagro_output_prefix_keeps_local_defaults(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """No env var, no kwargs → local defaults from setup_environment."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.delenv("EDAGRO_OUTPUT_PREFIX", raising=False)

        wm = WorkflowManager(env="preprod")

        assert wm.output_result_dir == "/proj/results"
        assert wm.partial_result_dir == "/proj/partials"
        assert wm.cache_dir == "/proj/cache"

    # ── `storage` flag — notebook-friendly override for the env var ──────────

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_local_ignores_edagro_output_prefix_env(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """storage='local' is the notebook escape hatch: a stray
        EDAGRO_OUTPUT_PREFIX in the dev .env shouldn't silently route writes
        to S3 when the user explicitly asked for local."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", "s3://stray-bucket/run")

        wm = WorkflowManager(env="preprod", storage="local")

        assert wm.output_result_dir == "/proj/results"
        assert wm.partial_result_dir == "/proj/partials"
        assert wm.cache_dir == "/proj/cache"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_s3_raises_when_no_prefix_and_no_kwargs(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """storage='s3' fails fast if neither the env var nor explicit remote
        kwargs are provided — catches misconfigured prod runs early instead
        of silently writing to local disk."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.delenv("EDAGRO_OUTPUT_PREFIX", raising=False)

        with pytest.raises(ValueError, match="storage='s3' requires"):
            WorkflowManager(env="preprod", storage="s3")

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_s3_accepts_explicit_remote_kwarg_without_env(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """An explicit s3:// kwarg satisfies storage='s3' even without the env var."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.delenv("EDAGRO_OUTPUT_PREFIX", raising=False)

        wm = WorkflowManager(
            env="preprod",
            storage="s3",
            output_result_dir="s3://explicit-bucket/results",
        )
        # Other two paths fall through to the setup_environment local defaults
        # because no env var and no other kwargs.
        assert wm.output_result_dir == "s3://explicit-bucket/results"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_auto_is_the_default_and_honours_env(
        self, mock_setup_env, mock_authenticator, mock_setup_logging, monkeypatch
    ):
        """storage='auto' (the default) keeps the env-var-respecting behaviour."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        monkeypatch.setenv("EDAGRO_OUTPUT_PREFIX", "s3://auto-bucket/run")

        # No explicit storage arg → defaults to "auto" → env var wins.
        wm = WorkflowManager(env="preprod")
        assert wm.output_result_dir == "s3://auto-bucket/run/results"

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_storage_invalid_value_raises(self, mock_setup_env, mock_authenticator, mock_setup_logging):
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_authenticator.return_value = mock_auth_instance

        with pytest.raises(ValueError, match="Invalid storage"):
            WorkflowManager(env="preprod", storage="cloud")  # not in the allowed set

    @patch("earthdaily.agriculture.services.workflow_manager.setup_logging")
    @patch("earthdaily.agriculture.services.workflow_manager.EDAuthenticator")
    @patch("earthdaily.agriculture.services.workflow_manager.setup_environment")
    def test_init_tolerates_s3_init_failure(self, mock_setup_env, mock_authenticator, mock_setup_logging):
        """If S3 client init raises, init still succeeds (S3 is optional)."""
        mock_setup_env.return_value = {
            "env": "preprod",
            "project_root": "/proj",
            "output_result_dir": "/proj/results",
            "partial_result_dir": "/proj/partials",
            "cache_dir": "/proj/cache",
        }
        mock_auth_instance = MagicMock()
        mock_auth_instance.bearer_token = "tok"
        mock_auth_instance.expiration_date = None
        mock_auth_instance.initialize_s3_client.side_effect = RuntimeError("no aws creds")
        mock_authenticator.return_value = mock_auth_instance

        # Should NOT raise
        wm = WorkflowManager(env="preprod")
        assert wm.env == "preprod"


# ===================================================================
# load_workflow — validation paths
# ===================================================================


class TestLoadWorkflow:
    def test_file_not_found_raises(self, manager):
        with pytest.raises(FileNotFoundError, match="not found"):
            manager.load_workflow("/does/not/exist.yml")

    def test_missing_top_level_keys_raises(self, manager, tmp_path):
        """A YAML without 'workflow' or 'analytics' top-level keys is invalid."""
        p = tmp_path / "bad.yml"
        p.write_text(yaml.dump({"foo": "bar"}))
        with pytest.raises(ValueError, match="must contain 'workflow' or 'analytics'"):
            manager.load_workflow(str(p))

    def test_legacy_analytics_format_returns_raw(self, manager, tmp_path):
        """The legacy {analytics: ...} format is accepted with a warning,
        stored as workflow_cfg, and skips step parsing."""
        p = tmp_path / "legacy.yml"
        p.write_text(yaml.dump({"analytics": {"emergence": {"module": "x", "class": "Y"}}}))
        result = manager.load_workflow(str(p))
        # Stored as raw config
        assert "analytics" in result
        # No steps parsed
        assert manager.workflow_steps is None
        assert manager.step_map is None

    def test_step_missing_name_raises(self, manager, tmp_path):
        cfg = {
            "name": "test",
            "steps": [{"transform": {"module": "x", "function": "y", "params": {}}}],
        }
        with pytest.raises(ValueError, match="missing required 'name'"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_duplicate_step_name_raises(self, manager, tmp_path):
        cfg = {
            "name": "test",
            "steps": [_passthrough_step("a"), _passthrough_step("a")],
        }
        with pytest.raises(ValueError, match="Duplicate step name"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_unknown_depends_on_raises(self, manager, tmp_path):
        cfg = {
            "name": "test",
            "steps": [_passthrough_step("a", depends_on="ghost")],
        }
        with pytest.raises(ValueError, match="depends on unknown step"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_circular_dependency_raises(self, manager, tmp_path):
        """A→B and B→A produces a cycle that breaks topological sort."""
        cfg = {
            "name": "test",
            "steps": [
                _passthrough_step("a", depends_on="b"),
                _passthrough_step("b", depends_on="a"),
            ],
        }
        with pytest.raises(ValueError, match="Circular dependency"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_input_from_non_string_raises(self, manager, tmp_path):
        """input_from must be a string ('original' or a step name)."""
        cfg = {
            "name": "test",
            "steps": [
                _passthrough_step("a"),
                {
                    "name": "b",
                    "depends_on": "a",
                    "input_from": 123,  # invalid — int
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_passthrough",
                        "params": {},
                    },
                },
            ],
        }
        with pytest.raises(ValueError, match="input_from must be a string"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_load_returns_parsed_config_and_populates_state(self, manager, tmp_path):
        cfg = {
            "name": "Test Workflow",
            "settings": {"max_workers": 5},
            "steps": [_passthrough_step("a"), _passthrough_step("b", depends_on="a")],
        }
        result = manager.load_workflow(_write_yaml(tmp_path, cfg))

        assert result["name"] == "Test Workflow"
        assert manager.workflow_steps is not None
        assert len(manager.workflow_steps) == 2
        assert set(manager.step_map.keys()) == {"a", "b"}


# ===================================================================
# inspect_workflow
# ===================================================================


class TestInspectWorkflow:
    def test_inspect_without_load_raises(self, manager):
        with pytest.raises(RuntimeError, match="No workflow loaded"):
            manager.inspect_workflow()

    def test_inspect_returns_top_level_metadata(self, manager, tmp_path):
        cfg = {
            "name": "My Workflow",
            "description": "demo",
            "settings": {"max_workers": 7, "partial_frequency": 25},
            "steps": [_passthrough_step("a")],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        info = manager.inspect_workflow()

        assert info["name"] == "My Workflow"
        assert info["description"] == "demo"
        assert info["settings"]["max_workers"] == 7
        assert isinstance(info["steps"], list)
        assert isinstance(info["execution_levels"], list)

    def test_inspect_execution_levels_respects_dependencies(self, manager, tmp_path):
        """Two parallel roots feed into one downstream step → 2 levels."""
        cfg = {
            "name": "test",
            "steps": [
                _passthrough_step("a"),
                _passthrough_step("b"),
                _passthrough_step("c", depends_on=["a", "b"]),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        info = manager.inspect_workflow()
        levels = info["execution_levels"]
        assert len(levels) == 2
        assert set(levels[0]) == {"a", "b"}
        assert levels[1] == ["c"]

    def test_inspect_classifies_transform_vs_extractor(self, manager, tmp_path):
        """Transform-only step exposes has_transform=True/has_extractor=False."""
        cfg = {
            "name": "test",
            "steps": [
                _passthrough_step("transform_step"),
                {
                    "name": "extractor_step",
                    "extractor": "EmergenceExtractor",
                    "module": "earthdaily.agriculture.processors.processor_emergence_functions",
                    "setup": {"method": "setup_emergence_parameters", "params": {}},
                    "run": {"method": "process_emergence_bulk_extraction_parallel", "params": {}},
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        info = manager.inspect_workflow()

        by_name = {s["name"]: s for s in info["steps"]}
        assert by_name["transform_step"]["has_transform"] is True
        assert by_name["transform_step"]["has_extractor"] is False
        assert by_name["transform_step"]["transform"] == "tests.test_workflow_runner._passthrough"

        assert by_name["extractor_step"]["has_extractor"] is True
        assert by_name["extractor_step"]["has_transform"] is False
        assert by_name["extractor_step"]["extractor"] == "EmergenceExtractor"


# ===================================================================
# visualize_workflow — load guard
# ===================================================================


class TestVisualizeWorkflow:
    def test_visualize_without_load_raises(self, manager):
        """visualize_workflow requires a loaded workflow (notebook cell 10)."""
        with pytest.raises((RuntimeError, AttributeError, TypeError)):
            manager.visualize_workflow()


# ===================================================================
# run_workflow — preconditions and entity_list defaulting
# ===================================================================


class TestRunWorkflow:
    def test_run_without_workflow_raises(self, manager, entities):
        """run_workflow requires load_workflow first."""
        with pytest.raises(RuntimeError, match="No workflow loaded"):
            manager.run_workflow(entity_list=entities)

    def test_run_without_entities_raises(self, manager, tmp_path):
        """If entity_list is None and self.sfd_list is None, raise."""
        cfg = {"name": "test", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        with pytest.raises(ValueError, match="No entities"):
            manager.run_workflow()

    def test_run_with_empty_entities_raises(self, manager, tmp_path):
        cfg = {"name": "test", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        with pytest.raises(ValueError, match="No entities"):
            manager.run_workflow(entity_list=pd.DataFrame())

    def test_run_falls_back_to_sfd_list(self, manager, tmp_path, entities):
        """Notebook cell 12: run_workflow() with no arg uses self.sfd_list."""
        cfg = {"name": "test", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.sfd_list = entities  # populated like manager.load_seasonfields would

        results = manager.run_workflow()

        assert "solo" in results
        # Transform-only step stores its DataFrame return value as results_df
        assert len(results["solo"]["results_df"]) == 3

    def test_run_uses_entity_list_argument(self, manager, tmp_path, entities):
        """Notebook cell 25: passing entity_list explicitly overrides sfd_list."""
        cfg = {"name": "test", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.sfd_list = pd.DataFrame({"id": ["IGNORED"]})  # would be wrong if used

        subset = entities.head(2)
        results = manager.run_workflow(entity_list=subset)

        assert len(results["solo"]["results_df"]) == 2
        assert list(results["solo"]["results_df"]["id"]) == ["a", "b"]

    def test_run_returns_dict_keyed_by_step_name(self, manager, tmp_path, entities):
        cfg = {
            "name": "test",
            "steps": [
                _passthrough_step("a"),
                _passthrough_step("b", depends_on="a"),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities)

        assert set(results.keys()) == {"a", "b"}
        # Each step's result has the standard keys
        for step in ("a", "b"):
            assert "results_df" in results[step]
            assert "global_errors" in results[step]


# ===================================================================
# run_workflow — extractor step (programmatic YAML, mirrors notebook cell 28)
# ===================================================================


class TestRunWorkflowExtractorStep:
    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_extractor_step_invokes_setup_and_run(self, mock_import, manager, tmp_path, entities):
        """Programmatically construct a YAML with an extractor step (notebook cell 28
        pattern). Verify the extractor is instantiated, setup is called, and the run
        method is invoked with the resolved entity_list."""
        # Mock the module import to return our stub's containing module
        mock_module = MagicMock()
        mock_module._StubExtractor = _StubExtractor
        mock_import.return_value = mock_module

        cfg = {
            "name": "Stub workflow",
            "settings": {"max_workers": 4, "partial_frequency": 25},
            "steps": [
                {
                    "name": "stub",
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {
                        "method": "setup_stub",
                        "params": {"foo": "bar"},
                    },
                    "run": {
                        "method": "run_stub",
                        "params": {},
                    },
                }
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities)

        # Extractor's run output flowed through to results_df
        assert "stub" in results
        df = results["stub"]["results_df"]
        # Stub's run_stub adds a 'stub_ran' column to every row
        assert df["stub_ran"].all()
        assert len(df) == 3


# ===================================================================
# run_workflow — WorkflowRunReporter integration (generate_report kwarg)
# ===================================================================


class TestRunWorkflowGenerateReport:
    """Phase 2 of the WorkflowRunReporter integration.

    The manager owns: building the reporter, populating context from the
    loaded YAML, timing the run, and calling ``set_step_results`` after the
    DAG completes (even on exception). The reporter instance is stashed at
    ``manager.last_run_reporter`` so callers can amend it (e.g. set
    ``report_path`` after a downstream export step) and render HTML / JSON.
    """

    def test_generate_report_false_does_not_build_reporter(self, manager, tmp_path, entities):
        """Default behaviour: byte-identical to pre-Phase-2 except for the
        new ``last_run_reporter`` attribute being set to None."""
        cfg = {"name": "test", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        results = manager.run_workflow(entity_list=entities)

        assert manager.last_run_reporter is None
        # Results shape unchanged.
        assert "__run_receipt__" not in results
        assert set(results.keys()) == {"solo"}

    def test_generate_report_true_populates_run_context(self, manager, tmp_path, entities):
        """generate_report=True builds the reporter, populates context from
        the loaded YAML, and stashes it on the manager."""
        manager.env = "preprod"
        cfg = {
            "name": "Greenness Pipeline",
            "settings": {"output_prefix": "client_a_run", "max_workers": 4},
            "steps": [_passthrough_step("solo")],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        manager.run_workflow(entity_list=entities, generate_report=True)

        reporter = manager.last_run_reporter
        assert reporter is not None
        receipt = reporter.render_json()
        assert receipt["schema_version"] == 1
        assert receipt["workflow"]["name"] == "Greenness Pipeline"
        assert receipt["workflow"]["prefix"] == "client_a_run"
        assert receipt["workflow"]["env"] == "preprod"
        assert receipt["entities"]["loaded"] == 3
        # duration recorded (monotonic clock — non-negative real)
        assert isinstance(receipt["duration_seconds"], float)
        assert receipt["duration_seconds"] >= 0

    def test_step_kinds_classified_from_step_map(self, manager, tmp_path, entities):
        """A mix of transform-only / extractor / transform+extractor steps
        should each land with their canonical kind label in the report."""
        manager.env = "preprod"
        cfg = {
            "name": "mixed",
            "steps": [
                # Transform-only
                {
                    "name": "shape",
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_passthrough",
                        "params": {},
                    },
                },
                # Extractor-only (uses our _StubExtractor)
                {
                    "name": "stub",
                    "depends_on": "shape",
                    "input_from": "shape",
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {"method": "setup_stub", "params": {}},
                    "run": {"method": "run_stub", "params": {}},
                },
                # Transform + extractor
                {
                    "name": "combo",
                    "depends_on": "stub",
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_passthrough",
                        "params": {},
                    },
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {"method": "setup_stub", "params": {}},
                    "run": {"method": "run_stub", "params": {}},
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        # No importlib patch — both _passthrough and _StubExtractor are real
        # symbols in this test module, so the runner's importlib.import_module
        # finds them on the natural path.
        manager.run_workflow(entity_list=entities, generate_report=True)

        receipt = manager.last_run_reporter.render_json()
        by_name = {s["name"]: s for s in receipt["steps"]}
        assert by_name["shape"]["kind"] == "transform"
        assert by_name["stub"]["kind"] == "extractor"
        assert by_name["combo"]["kind"] == "transform+extractor"

    def test_report_options_split_between_constructor_and_context(self, manager, tmp_path, entities):
        """report_options accepts both reporter-constructor knobs and
        run-context fields. Constructor kwargs change the reporter's
        behaviour; context kwargs end up in the rendered JSON."""
        manager.env = "preprod"
        cfg = {"name": "t", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        manager.run_workflow(
            entity_list=entities,
            generate_report=True,
            report_options={
                # Constructor knob — tightens the in-HTML error truncation.
                "max_errors_shown": 7,
                # Context fields — surface in the JSON receipt.
                "parameters": {"limit": 10, "scenario": "smoke"},
                "column_mapping": {"id": "entity_id"},
            },
        )

        reporter = manager.last_run_reporter
        assert reporter.max_errors_shown == 7
        receipt = reporter.render_json()
        assert receipt["parameters"] == {"limit": 10, "scenario": "smoke"}
        assert receipt["column_mapping"] == {"id": "entity_id"}

    def test_unknown_report_options_key_raises(self, manager, tmp_path, entities):
        """Typos in report_options must not silently be ignored — Phase 2
        contract is that unknown keys raise ValueError listing the
        accepted names so the user can fix the typo."""
        manager.env = "preprod"
        cfg = {"name": "t", "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        with pytest.raises(ValueError, match="Unknown report_options keys"):
            manager.run_workflow(
                entity_list=entities,
                generate_report=True,
                report_options={"max_errrors_shown": 5},  # typo
            )

    def test_reporter_populated_even_when_step_raises(self, manager, tmp_path, entities):
        """If a step raises mid-run, the reporter still captures whatever
        step results were collected before the exception. The manager
        re-raises after the finally block, so callers see the original
        exception but can still inspect manager.last_run_reporter."""
        manager.env = "preprod"
        cfg = {
            "name": "fails",
            "steps": [
                _passthrough_step("a"),
                {
                    "name": "b",
                    "depends_on": "a",
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_raise_in_transform",
                        "params": {},
                    },
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        with pytest.raises(RuntimeError, match="boom from _raise_in_transform"):
            manager.run_workflow(entity_list=entities, generate_report=True)

        # Reporter still populated — first step succeeded, second raised.
        reporter = manager.last_run_reporter
        assert reporter is not None
        receipt = reporter.render_json()
        names = {s["name"] for s in receipt["steps"]}
        assert "a" in names  # the step that completed before the failure
        # Workflow recorded the second step in workflow_results, so reporter
        # sees it too — captured pre-raise inside the run_workflow loop.

    def test_amend_run_context_and_render_round_trip(self, manager, tmp_path, entities):
        """Two-call API: after run_workflow returns the consumer typically
        builds a final per-field CSV downstream, then amends the reporter
        with the resulting report_path and renders both HTML and JSON."""
        manager.env = "preprod"
        cfg = {"name": "t", "settings": {"output_prefix": "pref"}, "steps": [_passthrough_step("solo")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))

        manager.run_workflow(entity_list=entities, generate_report=True)
        reporter = manager.last_run_reporter

        final_csv = tmp_path / "final.csv"
        final_csv.write_text("id\na\nb\nc\n")

        reporter.amend_run_context(report_path=str(final_csv))
        html_out = tmp_path / "run.html"
        json_out = tmp_path / "run.json"
        reporter.render_html(output_path=str(html_out))
        receipt = reporter.render_json(output_path=str(json_out))

        assert html_out.exists()
        assert json_out.exists()
        assert receipt["report_path"] == str(final_csv)
        assert receipt["workflow"]["prefix"] == "pref"


# ===================================================================
# BUG 1 — _apply_condition must not require a `value` key for value-less
# operators like `notnull`.
# ===================================================================


class TestApplyConditionNotNullOperator:
    def _make_step(self, operator, column="historical_season", **extra):
        cond = {"depends_on": "upstream", "column": column, "operator": operator}
        cond.update(extra)
        return {"name": "downstream", "condition": cond}

    def test_notnull_without_value_key_does_not_raise(self, manager):
        """`operator: notnull` must work without a `value` key in the YAML.

        Repro for the original bug: the YAML had only `depends_on` / `column` /
        `operator: notnull` and `_apply_condition` raised KeyError('value').
        """
        manager.workflow_cfg = {"settings": {}}

        upstream_results = {
            "upstream": {
                "results_df": pd.DataFrame(
                    {
                        "id": ["a", "b", "c"],
                        "entity_id": ["a", "b", "c"],
                        "historical_season": [2024, None, 2025],
                    }
                )
            }
        }
        entities = pd.DataFrame({"id": ["a", "b", "c"], "geometry": ["P", "P", "P"]})

        step = self._make_step("notnull")  # NOTE: no `value` key
        filtered = manager._apply_condition(step, entities, upstream_results)

        # Only 'a' and 'c' have non-null historical_season
        assert sorted(filtered["id"].tolist()) == ["a", "c"]

    def test_eq_still_requires_value(self, manager):
        """Value-bearing operators must still raise a clear error if `value` is missing."""
        manager.workflow_cfg = {"settings": {}}
        upstream_results = {"upstream": {"results_df": pd.DataFrame({"id": ["a"], "entity_id": ["a"], "col": ["x"]})}}
        entities = pd.DataFrame({"id": ["a"]})

        step = self._make_step("eq")  # no `value`
        with pytest.raises(ValueError, match="'eq' requires a 'value'"):
            manager._apply_condition(step, entities, upstream_results)

    def test_in_still_requires_value(self, manager):
        manager.workflow_cfg = {"settings": {}}
        upstream_results = {"upstream": {"results_df": pd.DataFrame({"id": ["a"], "entity_id": ["a"], "col": ["x"]})}}
        entities = pd.DataFrame({"id": ["a"]})

        step = self._make_step("in")  # no `value`
        with pytest.raises(ValueError, match="'in' requires a 'value'"):
            manager._apply_condition(step, entities, upstream_results)

    def test_eq_with_value_works(self, manager):
        """Sanity check: existing eq behavior still works."""
        manager.workflow_cfg = {"settings": {}}
        upstream_results = {
            "upstream": {
                "results_df": pd.DataFrame({"id": ["a", "b"], "entity_id": ["a", "b"], "status": ["ok", "fail"]})
            }
        }
        entities = pd.DataFrame({"id": ["a", "b"]})

        step = self._make_step("eq", column="status", value="ok")
        filtered = manager._apply_condition(step, entities, upstream_results)
        assert filtered["id"].tolist() == ["a"]


# ===================================================================
# BUG 2 — `skip_export` from a step's run.params must be forwarded to the
# bulk method, not just used to gate output_path locally.
# ===================================================================


class TestSkipExportForwarding:
    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_skip_export_forwarded_to_run_fn(self, mock_import, manager, tmp_path, entities):
        """`skip_export: true` in run.params must reach the bulk method as a
        kwarg. Bulk methods default to skip_export=False and would otherwise
        export via _finalize_extraction's fallback to self.output_path."""
        mock_module = MagicMock()
        mock_module._StubExtractor = _StubExtractor
        mock_import.return_value = mock_module

        manager.output_result_dir = str(tmp_path / "output")

        cfg = {
            "name": "Skip-export workflow",
            "settings": {"max_workers": 2},
            "steps": [
                {
                    "name": "stub",
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {"method": "setup_stub", "params": {}},
                    "run": {
                        "method": "run_stub",
                        "params": {"skip_export": True},
                    },
                }
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.run_workflow(entity_list=entities)

        # The runner instantiates a fresh extractor — recover it from the import mock
        # and assert the kwargs forwarded to run_stub.
        # Easier path: assert no file was written to output_result_dir.
        import os

        if os.path.isdir(manager.output_result_dir):
            written = os.listdir(manager.output_result_dir)
        else:
            written = []
        assert written == [], f"skip_export=true but files were written: {written}"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_skip_export_default_does_export(self, mock_import, manager, tmp_path, entities):
        """When skip_export is not set, the runner exports normally — a file
        should land in output_result_dir. This is the foil to the bug-fix test."""
        mock_module = MagicMock()
        mock_module._StubExtractor = _StubExtractor
        mock_import.return_value = mock_module

        manager.output_result_dir = str(tmp_path / "output")

        cfg = {
            "name": "Default-export workflow",
            "settings": {"max_workers": 2},
            "steps": [
                {
                    "name": "stub",
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {"method": "setup_stub", "params": {}},
                    "run": {
                        "method": "run_stub",
                        "params": {"skip_export": False},
                    },
                }
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.run_workflow(entity_list=entities)

        import os

        written = os.listdir(manager.output_result_dir)
        assert "stub_export.csv" in written

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_skip_export_kwarg_passed_to_run_fn(self, mock_import, manager, tmp_path, entities):
        """Direct assertion: the kwargs the runner forwarded include
        skip_export=True. Captured via the stub's run_calls list."""
        captured = {}

        class _CaptureStub(_StubExtractor):
            def run_stub(self, entity_list, **kwargs):
                captured.update(kwargs)
                return super().run_stub(entity_list, **kwargs)

        mock_module = MagicMock()
        mock_module._StubExtractor = _CaptureStub
        mock_import.return_value = mock_module

        manager.output_result_dir = str(tmp_path / "output")

        cfg = {
            "name": "Capture skip_export",
            "settings": {"max_workers": 2},
            "steps": [
                {
                    "name": "stub",
                    "extractor": "_StubExtractor",
                    "module": "tests.test_workflow_runner",
                    "setup": {"method": "setup_stub", "params": {}},
                    "run": {
                        "method": "run_stub",
                        "params": {"skip_export": True},
                    },
                }
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.run_workflow(entity_list=entities)

        assert captured.get("skip_export") is True
        # And output_path is None when skip_export=True (existing gating logic)
        assert captured.get("output_path") is None


# ===================================================================
# run_workflow(run_prefix=...) — three-layer prefix composition
# ===================================================================


def _stub_step(step_prefix=None):
    """Build a stub-extractor step. If step_prefix is given, set run.params.prefix."""
    step = {
        "name": "stub",
        "extractor": "_StubExtractor",
        "module": "tests.test_workflow_runner",
        "setup": {"method": "setup_stub", "params": {}},
        "run": {"method": "run_stub", "params": {"skip_export": True}},
    }
    if step_prefix is not None:
        step["run"]["params"]["prefix"] = step_prefix
    return step


def _captured_prefix(
    mock_import, manager, tmp_path, entities, *, output_prefix=None, run_prefix=None, step_prefix=None
):
    """Helper: build a one-step workflow with the given prefix layers, run it,
    and return the prefix actually forwarded to the bulk method."""
    # Capture the kwargs the runner forwards to run_stub.
    captured = {}

    def _run_stub(self, entity_list, **kwargs):
        captured.update(kwargs)
        return {"results_df": entity_list, "global_errors": [], "failed_ids": []}

    with patch.object(_StubExtractor, "run_stub", _run_stub):
        mock_module = MagicMock()
        mock_module._StubExtractor = _StubExtractor
        mock_import.return_value = mock_module

        cfg = {"name": "prefix test", "settings": {}, "steps": [_stub_step(step_prefix)]}
        if output_prefix is not None:
            cfg["settings"]["output_prefix"] = output_prefix

        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.run_workflow(entity_list=entities, run_prefix=run_prefix)

    return captured.get("prefix")


class TestRunPrefixComposition:
    """All 8 combinations of (output_prefix, run_prefix, step_prefix) presence.
    Strings are kept slug-clean (lowercase + underscores) so the slugified value
    equals the input — keeps the assertions decoupled from _slugify internals."""

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_no_prefixes_uses_step_name(self, mock_import, manager, tmp_path, entities):
        # 0/0/0 — no output, no run, no step prefix → falls back to step name
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities)
        assert prefix == "stub"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_step_prefix_only(self, mock_import, manager, tmp_path, entities):
        # 0/0/1
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities, step_prefix="coverage")
        assert prefix == "coverage"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_run_prefix_only(self, mock_import, manager, tmp_path, entities):
        # 0/1/0 — falls back to step name for the trailing slot
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities, run_prefix="run_a")
        assert prefix == "run_a_stub"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_run_and_step_prefix(self, mock_import, manager, tmp_path, entities):
        # 0/1/1
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities, run_prefix="run_a", step_prefix="coverage")
        assert prefix == "run_a_coverage"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_output_prefix_only(self, mock_import, manager, tmp_path, entities):
        # 1/0/0
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities, output_prefix="proj")
        assert prefix == "proj_stub"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_output_and_step_prefix(self, mock_import, manager, tmp_path, entities):
        # 1/0/1
        prefix = _captured_prefix(
            mock_import, manager, tmp_path, entities, output_prefix="proj", step_prefix="coverage"
        )
        assert prefix == "proj_coverage"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_output_and_run_prefix(self, mock_import, manager, tmp_path, entities):
        # 1/1/0
        prefix = _captured_prefix(mock_import, manager, tmp_path, entities, output_prefix="proj", run_prefix="run_a")
        assert prefix == "proj_run_a_stub"

    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_all_three_layers(self, mock_import, manager, tmp_path, entities):
        # 1/1/1
        prefix = _captured_prefix(
            mock_import,
            manager,
            tmp_path,
            entities,
            output_prefix="proj",
            run_prefix="run_a",
            step_prefix="coverage",
        )
        assert prefix == "proj_run_a_coverage"


class TestRunPrefixLifecycle:
    @patch("earthdaily.agriculture.services.workflow_manager.importlib.import_module")
    def test_run_prefix_resets_between_calls(self, mock_import, manager, tmp_path, entities):
        """A second run_workflow() without run_prefix must NOT inherit the previous tag."""
        first = _captured_prefix(mock_import, manager, tmp_path, entities, run_prefix="run_a")
        second = _captured_prefix(mock_import, manager, tmp_path, entities)  # no run_prefix
        assert first == "run_a_stub"
        assert second == "stub"
        assert manager._run_prefix is None


# ===================================================================
# Step-level `enabled` flag
# ===================================================================


class TestStepEnabledFlag:
    def test_load_rejects_non_bool_enabled(self, manager, tmp_path):
        cfg = {
            "name": "bad",
            "steps": [
                {
                    "name": "a",
                    "enabled": "yes",  # string, not bool — must be rejected
                    "transform": {"module": "tests.test_workflow_runner", "function": "_passthrough", "params": {}},
                }
            ],
        }
        with pytest.raises(ValueError, match="'enabled' must be a boolean"):
            manager.load_workflow(_write_yaml(tmp_path, cfg))

    def test_inspect_workflow_surfaces_enabled(self, manager, tmp_path):
        cfg = {
            "name": "introspect",
            "steps": [
                _passthrough_step("a"),  # enabled defaults to True
                {**_passthrough_step("b"), "enabled": False},
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        info = manager.inspect_workflow()
        by_name = {s["name"]: s for s in info["steps"]}
        assert by_name["a"]["enabled"] is True
        assert by_name["b"]["enabled"] is False

    def test_disabled_step_is_skipped(self, manager, tmp_path, entities):
        cfg = {
            "name": "skip-me",
            "steps": [
                {**_passthrough_step("a"), "enabled": False},
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities)
        assert results["a"]["skipped"] is True
        assert results["a"]["reason"] == "disabled"
        assert results["a"]["results_df"].empty

    def test_disabled_upstream_cascades_skip_to_downstream(self, manager, tmp_path, entities):
        """B depends_on A; A is disabled. B should auto-skip via the existing
        empty-upstream branch — no special downstream wiring needed."""
        cfg = {
            "name": "cascade",
            "steps": [
                {**_passthrough_step("a"), "enabled": False},
                _passthrough_step("b", depends_on="a"),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities)
        assert results["a"]["skipped"] is True
        assert results["a"]["reason"] == "disabled"
        # B falls through the empty-upstream skip branch (no `reason` key — it's
        # the existing input_from-empty path, not the disabled path).
        assert results["b"]["skipped"] is True
        assert results["b"]["results_df"].empty

    def test_enabled_default_true_runs_normally(self, manager, tmp_path, entities):
        """A step without an `enabled` field behaves exactly as today (runs)."""
        cfg = {"name": "default", "steps": [_passthrough_step("a")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities)
        assert results["a"].get("skipped") is not True
        assert len(results["a"]["results_df"]) == 3


# ===================================================================
# run_workflow(disabled_steps=[...]) — headless equivalent of the widgets
# ===================================================================


class TestRunWorkflowDisabledSteps:
    def test_disabled_steps_kwarg_skips_only_listed(self, manager, tmp_path, entities):
        cfg = {
            "name": "headless-disable",
            "steps": [
                _passthrough_step("a"),
                _passthrough_step("b"),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities, disabled_steps=["b"])
        assert results["a"].get("skipped") is not True
        assert results["b"]["skipped"] is True
        assert results["b"]["reason"] == "disabled"

    def test_disabled_steps_unknown_name_raises(self, manager, tmp_path, entities):
        cfg = {"name": "x", "steps": [_passthrough_step("a")]}
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        with pytest.raises(ValueError, match="unknown step name 'bogus'"):
            manager.run_workflow(entity_list=entities, disabled_steps=["bogus"])

    def test_disabled_steps_overrides_are_restored_after_run(self, manager, tmp_path, entities):
        """After run_workflow returns, the loaded YAML's enabled state is unchanged.
        A step with no `enabled` field should still have no `enabled` field."""
        cfg = {
            "name": "restore",
            "steps": [
                _passthrough_step("a"),
                {**_passthrough_step("b"), "enabled": True},  # explicit True
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        manager.run_workflow(entity_list=entities, disabled_steps=["a", "b"])

        # Step "a" had no `enabled` field — must not have one after restore.
        assert "enabled" not in manager.step_map["a"]
        # Step "b" had explicit True — must still be True.
        assert manager.step_map["b"]["enabled"] is True

    def test_disabled_steps_restored_when_a_step_raises_mid_run(self, manager, tmp_path, entities):
        """If a step raises inside the run loop, the try/finally must still
        roll back any runtime disabled_steps overrides."""
        cfg = {
            "name": "raise-mid-run",
            "steps": [
                {**_passthrough_step("a"), "enabled": True},
                {
                    "name": "b",
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_raise_in_transform",
                        "params": {},
                    },
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        with pytest.raises(RuntimeError, match="boom"):
            # Disable "a" at runtime, then "b" raises during execution.
            manager.run_workflow(entity_list=entities, disabled_steps=["a"])
        # The runtime override on "a" must be rolled back to its pre-call state.
        assert manager.step_map["a"]["enabled"] is True

    def test_parallel_step_raising_propagates_and_rolls_back_overrides(self, manager, tmp_path, entities):
        """A step that raises inside the *parallel* execution path must propagate
        the exception out of run_workflow (fail-fast) AND the try/finally must
        still roll back any runtime disabled_steps overrides. Mirrors
        test_disabled_steps_restored_when_a_step_raises_mid_run but adds an
        extra parallel sibling step so we go through the multi-step branch."""
        cfg = {
            "name": "raise-parallel",
            "steps": [
                # Three steps at the same level (no depends_on). At runtime we
                # disable "a"; "b" raises; "c" runs alongside "b" so len(level) > 1
                # and the parallel branch is exercised.
                {**_passthrough_step("a"), "enabled": True},
                {
                    "name": "b",
                    "transform": {
                        "module": "tests.test_workflow_runner",
                        "function": "_raise_in_transform",
                        "params": {},
                    },
                },
                _passthrough_step("c"),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        with pytest.raises(RuntimeError, match="boom"):
            manager.run_workflow(entity_list=entities, disabled_steps=["a"])
        # Runtime override on "a" must be rolled back even though "b" raised
        # in the parallel branch.
        assert manager.step_map["a"]["enabled"] is True

    def test_disabled_steps_cascade_to_downstream(self, manager, tmp_path, entities):
        """Same cascade-skip behaviour as YAML-level enabled=false."""
        cfg = {
            "name": "cascade-headless",
            "steps": [
                _passthrough_step("a"),
                _passthrough_step("b", depends_on="a"),
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        results = manager.run_workflow(entity_list=entities, disabled_steps=["a"])
        assert results["a"]["skipped"] is True
        assert results["a"]["reason"] == "disabled"
        assert results["b"]["skipped"] is True
        assert results["b"]["results_df"].empty


# ===================================================================
# Dynamic-date sentinels — load_workflow rewrites today / today-Nd / etc.
# ===================================================================


class TestDateSentinelResolver:
    """Direct unit tests for the private resolver helper."""

    def test_today_resolves_to_today_iso(self):
        from datetime import date

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        assert _resolve_date_sentinel("today", None) == date.today().isoformat()

    def test_yesterday_resolves_one_day_back(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        assert _resolve_date_sentinel("yesterday", None) == (date.today() - timedelta(days=1)).isoformat()

    def test_today_minus_n_days(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        assert _resolve_date_sentinel("today-7d", None) == (date.today() - timedelta(days=7)).isoformat()

    def test_today_minus_n_weeks(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        assert _resolve_date_sentinel("today-2w", None) == (date.today() - timedelta(weeks=2)).isoformat()

    def test_today_minus_n_months_uses_30_day_approx(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        # 30-day approximation, documented in the spec.
        assert _resolve_date_sentinel("today-1m", None) == (date.today() - timedelta(days=30)).isoformat()

    def test_today_plus_n_days_for_forecast_window(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        # Forecast use case: weather extractor wants "today + 10 days".
        assert _resolve_date_sentinel("today+10d", None) == (date.today() + timedelta(days=10)).isoformat()

    def test_today_plus_n_weeks_and_months(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        assert _resolve_date_sentinel("today+2w", None) == (date.today() + timedelta(weeks=2)).isoformat()
        assert _resolve_date_sentinel("today+1m", None) == (date.today() + timedelta(days=30)).isoformat()

    def test_offset_without_unit_defaults_to_days(self):
        from datetime import date, timedelta

        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        # "today+10" should resolve as if "today+10d" was passed.
        assert _resolve_date_sentinel("today+10", None) == (date.today() + timedelta(days=10)).isoformat()
        # "today-5" should resolve as "today-5d".
        assert _resolve_date_sentinel("today-5", None) == (date.today() - timedelta(days=5)).isoformat()

    def test_non_sentinel_string_returns_none(self):
        from earthdaily.agriculture.services.workflow_manager import _resolve_date_sentinel

        # Already-resolved ISO date — returned unchanged (i.e. resolver says "not a sentinel").
        assert _resolve_date_sentinel("2025-03-01", None) is None
        # Random string — same.
        assert _resolve_date_sentinel("hello", None) is None
        # Out-of-grammar patterns — same.
        assert _resolve_date_sentinel("today-Xd", None) is None
        assert _resolve_date_sentinel("today*1d", None) is None  # invalid sign
        assert _resolve_date_sentinel("tomorrow", None) is None  # not in grammar


class TestLoadWorkflowResolvesDateSentinels:
    """End-to-end: load_workflow rewrites sentinels in entity_source.params and steps[].setup.params."""

    def test_setup_params_sentinels_rewritten_in_place(self, manager, tmp_path):
        from datetime import date

        cfg = {
            "name": "dates",
            "steps": [
                {
                    "name": "mrts",
                    "extractor": "MRTSExtractor",
                    "module": "earthdaily.agriculture.extractors.VTS_functions",
                    "setup": {
                        "method": "setup_mrts_parameters",
                        "params": {
                            "vegetation_index": "NDVI",
                            "start_date": "today-30d",
                            "end_date": "today",
                            "clear_cover_min": 90,  # int — must pass through
                        },
                    },
                    "run": {"method": "process_mrts_bulk", "params": {"prefix": "mrts"}},
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        params = manager.workflow_cfg["steps"][0]["setup"]["params"]
        # Both sentinels resolved
        assert params["start_date"] != "today-30d"
        assert params["end_date"] != "today"
        # Resolved to real ISO dates
        from datetime import datetime as _dt

        _dt.fromisoformat(params["start_date"])
        _dt.fromisoformat(params["end_date"])
        # end_date is today
        assert params["end_date"] == date.today().isoformat()
        # Non-string param untouched
        assert params["clear_cover_min"] == 90
        # Non-sentinel string untouched
        assert params["vegetation_index"] == "NDVI"

    def test_literal_dates_pass_through_unchanged(self, manager, tmp_path):
        cfg = {
            "name": "literal",
            "steps": [
                {
                    **_passthrough_step("a"),
                    "setup": {"method": "setup_x", "params": {"start_date": "2025-03-01"}},
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        assert manager.workflow_cfg["steps"][0]["setup"]["params"]["start_date"] == "2025-03-01"

    def test_entity_source_params_also_rewritten(self, manager, tmp_path):
        from datetime import date, timedelta

        cfg = {
            "name": "entity-dates",
            "entity_source": {
                "method": "load_seasonfields",
                "params": {"sowing_date_gte": "today-180d"},
            },
            "steps": [_passthrough_step("a")],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        resolved = manager.workflow_cfg["entity_source"]["params"]["sowing_date_gte"]
        assert resolved == (date.today() - timedelta(days=180)).isoformat()

    def test_idempotent_on_already_resolved_config(self, manager, tmp_path):
        """Resolved ISO dates aren't sentinels, so a second load is a no-op
        (within the same calendar day)."""
        cfg = {
            "name": "idempotent",
            "steps": [
                {
                    **_passthrough_step("a"),
                    "setup": {"method": "setup_x", "params": {"end_date": "today"}},
                },
            ],
        }
        path = _write_yaml(tmp_path, cfg)
        manager.load_workflow(path)
        first = manager.workflow_cfg["steps"][0]["setup"]["params"]["end_date"]
        manager.load_workflow(path)
        second = manager.workflow_cfg["steps"][0]["setup"]["params"]["end_date"]
        # File on disk still says "today"; resolver gives the same answer.
        assert first == second

    def test_settings_timezone_changes_resolution(self, manager, tmp_path, monkeypatch):
        """settings.timezone routes "today" through that zone instead of UTC."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            pytest.skip("zoneinfo unavailable")

        # Freeze datetime.now used inside _today_in_tz to a moment where the
        # UTC date and the Chicago date differ.
        import datetime as _dt

        from earthdaily.agriculture.services import workflow_manager as wm_mod

        class _FixedDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                # 2026-05-12 04:30 UTC = 2026-05-11 23:30 America/Chicago.
                base = _dt.datetime(2026, 5, 12, 4, 30, tzinfo=_dt.timezone.utc)
                return base if tz is None else base.astimezone(tz)

        monkeypatch.setattr(wm_mod, "datetime", _FixedDatetime)

        cfg = {
            "name": "tz",
            "settings": {"timezone": "America/Chicago"},
            "steps": [
                {
                    **_passthrough_step("a"),
                    "setup": {"method": "setup_x", "params": {"end_date": "today"}},
                },
            ],
        }
        manager.load_workflow(_write_yaml(tmp_path, cfg))
        assert manager.workflow_cfg["steps"][0]["setup"]["params"]["end_date"] == "2026-05-11"
