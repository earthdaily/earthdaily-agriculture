"""
Workflow Manager - Orchestrator for Prefect-based workflows.
Handles environment setup, authentication, entity loading, and workflow execution.
"""

import importlib
import inspect
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from typing import Any, Literal

try:
    from zoneinfo import ZoneInfo  # stdlib, py3.9+
except ImportError:  # pragma: no cover — py<3.9
    ZoneInfo = None  # type: ignore[assignment,misc]

import pandas as pd
import yaml
from loguru import logger

from earthdaily.agriculture.core.functions_enhanced import setup_environment
from earthdaily.agriculture.core.geometry import load_geodataframe

# Internal Project Utilities
from earthdaily.agriculture.core.identity import EDAuthenticator
from earthdaily.agriculture.core.logging_setup import setup_logging
from earthdaily.agriculture.core.naming import _slugify
from earthdaily.agriculture.services.entity_management import EntityManager

# Sentinel for "field absent on the step dict" — used by run_workflow() when
# applying and restoring runtime `disabled_steps` overrides. Distinguishes
# `step["enabled"] = True` (explicit) from "field not in the YAML" so we can
# pop the key on restore instead of leaving an artifact behind.
_ENABLED_UNSET = object()

# Accepted keys for `run_workflow(report_options=...)`. Kept tight so typos
# raise a clear error rather than silently being ignored.
_REPORTER_CTOR_KEYS = frozenset(
    {
        "include_step_table",
        "include_data_preview",
        "preview_rows",
        "max_errors_shown",
    }
)
_REPORTER_CTX_KEYS = frozenset(
    {
        "parameters",
        "column_mapping",
        "resolved_yaml",
    }
)


def _classify_workflow_step(step: dict) -> str:
    """Classify a workflow step entry as extractor / transform / both / generic.

    Mirrors the inspect_workflow() / visualize_workflow() taxonomy so the
    WorkflowRunReporter's step table shows the same kind label as the DAG
    visualisation.
    """
    has_extractor = bool(step.get("extractor"))
    has_transform = bool(step.get("transform"))
    if has_extractor and has_transform:
        return "transform+extractor"
    if has_extractor:
        return "extractor"
    if has_transform:
        return "transform"
    return "step"


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic-date sentinels — resolved at load_workflow() time inside any
# `setup.params` block. See docs/site/agriculture/09b for the grammar. Only the exact patterns
# below are rewritten; everything else passes through. Default TZ is UTC; a
# workflow can opt into a local TZ via `settings.timezone` (IANA name).
# ─────────────────────────────────────────────────────────────────────────────
# Pattern: today, today±N, today±Nd, today±Nw, today±Nm.
# The unit is optional and defaults to days when omitted (so "today+10" ≡ "today+10d").
_DATE_SENTINEL_RE = re.compile(r"^today(?:([+-])(\d+)([dwm])?)?$")


def _today_in_tz(tz: str | None) -> date:
    """date.today() in the named IANA tz (UTC if tz is None or zoneinfo missing)."""
    if tz is None or ZoneInfo is None:
        return date.today()
    return datetime.now(ZoneInfo(tz)).date()


def _resolve_date_sentinel(value: str, tz: str | None) -> str | None:
    """Return resolved ISO date string, or None if ``value`` is not a sentinel.

    Supports past (``today-N``) and future (``today+N``) offsets. The unit
    suffix is optional and defaults to days — ``today+10`` is equivalent
    to ``today+10d``. Future-dated sentinels are typically needed for
    forecast windows, e.g. a weather extractor with
    ``end_date: "today+10"`` for a 10-day forecast.
    """
    if value == "yesterday":
        return (_today_in_tz(tz) - timedelta(days=1)).isoformat()
    m = _DATE_SENTINEL_RE.match(value)
    if not m:
        return None
    sign, n_str, unit = m.groups()
    if n_str is None:
        return _today_in_tz(tz).isoformat()
    n = int(n_str)
    if unit is None or unit == "d":
        delta = timedelta(days=n)
    elif unit == "w":
        delta = timedelta(weeks=n)
    else:  # "m" — 30-day approximation; discouraged for precise windows.
        delta = timedelta(days=30 * n)
    today = _today_in_tz(tz)
    resolved = today + delta if sign == "+" else today - delta
    return resolved.isoformat()


class WorkflowManager:
    """
    Orchestrates workflow execution by:
    1. Setting up environment and credentials
    2. Managing authentication tokens
    3. Loading entities to process
    4. Instantiating extractors
    5. Executing Prefect flows with prepared inputs
    """

    def __init__(
        self,
        env: str = None,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_console_only: bool = False,
        project_root: str = None,
        output_result_dir: str | None = None,
        partial_result_dir: str | None = None,
        cache_dir: str | None = None,
        storage: Literal["auto", "local", "s3"] = "auto",
        eds_env: str | None = None,
    ):
        """
        Initialize workflow manager with configuration.

        Args:
            env: Environment to use ('prod' or 'preprod'). If None, reads from ENVIRONMENT env var.
            log_level: Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)
            log_to_console: Whether to output logs to console (stderr).
            log_to_console_only: When True, skip the file-rotation log sink and
                emit only to the console. The env var
                ``EDAGRO_LOG_CONSOLE_ONLY=1`` overrides this kwarg so
                containerized deploys (Argo / ECS / Cloud Run) can flip it
                without code changes. Default ``False`` — existing notebook
                callers keep writing to ``logs/earthdaily_<date>.log``.
            project_root: Override for workspace root (where results/, partials/, etc. are created).
                If None, resolves automatically via pyproject.toml lookup.
            output_result_dir: Override the final-results write path. Defaults
                to ``<project_root>/results`` when None. Accepts any path that
                fsspec understands — local (``/abs/path``), ``s3://bucket/...``,
                ``gs://bucket/...``, etc. Per doc 13 §1 ("paths are the mode"),
                pointing this at ``s3://...`` is the single switch that makes
                the whole writer surface land on cloud storage.
            partial_result_dir: Override the partial-results write path. Same
                semantics as ``output_result_dir``. Defaults to
                ``<project_root>/partials`` when None.
            cache_dir: Override the cache directory. Defaults to
                ``<project_root>/cache`` when None. **Remote paths** (``s3://`` etc.)
                disable caching with a one-time warning — atomic-rename has no
                object-store equivalent.
            eds_env: Override the environment for the **EarthDaily Data
                Studio** provider, which LRTS uses. Defaults to ``env``, and
                that default is the intended behaviour:

                    ``prod``    -> EDS prod     (api.earthdaily.com)
                    ``preprod`` -> EDS BLEEDING (bleeding-api.test-internal…)

                **This is an escape hatch, not part of the normal flow.** It
                exists only because an account may be entitled to LRTS on one
                environment and not the other — a back-end entitlement matter,
                nothing to do with the client. Leave it unset unless you are
                deliberately crossing environments.

                It selects both the EDS token and the LRTS data host, which must
                agree: they are separate identity deployments and a token from
                one is refused by the other.
            storage: High-level storage mode. Notebook-friendly toggle that
                sits between the env var and the explicit kwargs. Values:

                - ``"auto"`` (default): honour ``EDAGRO_OUTPUT_PREFIX`` if set,
                  otherwise local defaults. Matches the orchestrator-driven
                  Pattern B flow from doc 14.
                - ``"local"``: ignore ``EDAGRO_OUTPUT_PREFIX``; force local
                  defaults. Use in dev notebooks when a stray env var would
                  silently route writes to S3.
                - ``"s3"``: require ``EDAGRO_OUTPUT_PREFIX`` to be set or
                  explicit ``output_result_dir`` etc. kwargs to be passed.
                  Raises ``ValueError`` otherwise — catches misconfigured
                  S3 runs early instead of silently falling back to local.

                Explicit ``output_result_dir`` / ``partial_result_dir`` /
                ``cache_dir`` kwargs still win over this flag.
        """
        # Initialize base configuration first to get project_root
        self.config = setup_environment(env=env, project_root=project_root)
        self.env = self.config["env"]
        self.project_root = self.config["project_root"]

        # Writer-path precedence (highest first):
        #   1. Explicit constructor kwargs (output_result_dir / partial_result_dir / cache_dir)
        #   2. `storage` flag — notebook-friendly override for the env-var behaviour
        #   3. EDAGRO_OUTPUT_PREFIX env var (when storage=="auto") — the canonical
        #      Pattern B invocation from docs/site/agriculture/14. Orchestrators pass
        #      EDAGRO_OUTPUT_PREFIX=s3://my-bucket/runs/2026-01-01 and we
        #      derive /results, /partials, /cache from it. Cache then
        #      auto-disables because the path is remote (see doc 13 §5).
        #   4. setup_environment defaults — local <project_root>/{results,partials,cache}.
        if storage not in ("auto", "local", "s3"):
            raise ValueError(f"Invalid storage={storage!r}. Choose from 'auto', 'local', 's3'.")

        output_prefix_env = os.environ.get("EDAGRO_OUTPUT_PREFIX")

        if storage == "s3":
            # Require S3 routing — either via env var or via explicit kwargs.
            has_explicit_remote = any(
                p is not None and "://" in p for p in (output_result_dir, partial_result_dir, cache_dir)
            )
            if not output_prefix_env and not has_explicit_remote:
                raise ValueError(
                    "storage='s3' requires EDAGRO_OUTPUT_PREFIX to be set, or at "
                    "least one of output_result_dir / partial_result_dir / cache_dir "
                    "to be a remote URI (s3://, gs://, ...). Got neither."
                )

        # Apply the env-var prefix unless storage='local' explicitly opts out.
        if output_prefix_env and storage != "local":
            prefix = output_prefix_env.rstrip("/")
            self.config["output_result_dir"] = f"{prefix}/results"
            self.config["partial_result_dir"] = f"{prefix}/partials"
            self.config["cache_dir"] = f"{prefix}/cache"

        if output_result_dir is not None:
            self.config["output_result_dir"] = output_result_dir
        if partial_result_dir is not None:
            self.config["partial_result_dir"] = partial_result_dir
        if cache_dir is not None:
            self.config["cache_dir"] = cache_dir

        # Setup logging using project-root logs directory (file sink skipped
        # automatically when log_to_console_only / EDAGRO_LOG_CONSOLE_ONLY).
        log_dir = os.path.join(self.project_root, "logs")
        setup_logging(
            log_dir=log_dir,
            log_level=log_level,
            log_to_console=log_to_console,
            log_to_console_only=log_to_console_only,
        )

        logger.info("🚀 Initializing WorkflowManager")
        logger.info(f"   Project root: {self.project_root}")

        # Create EDAuthenticator instance
        # EDS (LRTS) may run against a different environment — see `eds_env`.
        self.eds_env = eds_env or self.env

        logger.info("🔐 Initializing EDAuthenticator...")
        self.authenticator = EDAuthenticator(env=self.env)

        # Keep backward compatibility with existing code
        self.bearer_token = self.authenticator.bearer_token
        self.token_expiration = self.authenticator.expiration_date

        # Auto-initialize S3 client (skip gracefully if no credentials)
        try:
            logger.info("☁️  Initializing S3 client...")
            self.authenticator.initialize_s3_client()
        except Exception as e:
            logger.warning(f"⚠️  S3 client initialization skipped: {e}")
            logger.warning("   S3-dependent features will not be available.")

        # Workflow state
        self.sfd_list = None
        self.partials_path = None
        self.data_source = None  # 'api' or 'file'

        # Workflow runner state (populated by load_workflow)
        self.workflow_cfg: dict[str, Any] | None = None
        #: The workflow YAML exactly as parsed, wrapper included — a read-only
        #: companion to :attr:`workflow_cfg`, which for the new step format holds
        #: the UNWRAPPED inner dict. Provided so callers that genuinely need the
        #: wrapped form (round-tripping the YAML) do not have to re-wrap by hand
        #: or guess which shape they were handed. Never consumed internally.
        self.workflow_cfg_raw: dict[str, Any] | None = None
        self.workflow_steps: list[dict[str, Any]] | None = None
        self.step_map: dict[str, dict[str, Any]] | None = None
        self.workflow_results: dict[str, Any] = {}
        self._run_prefix: str | None = None  # Optional per-run prefix set by run_workflow(run_prefix=...)

        # Set convenient attributes (now absolute paths at project root)
        self.output_result_dir = self.config["output_result_dir"]
        self.partial_result_dir = self.config["partial_result_dir"]
        self.cache_dir = self.config["cache_dir"]

        logger.success("✅ WorkflowManager initialized")
        logger.info(f"   Environment: {self.env}")
        logger.info(f"   Results dir: {self.output_result_dir}")
        logger.info(f"   Partials dir: {self.partial_result_dir}")
        logger.info(f"   Cache dir: {self.cache_dir}")

    def initialize_system(self):
        """Initialize system configuration."""
        self.config = setup_environment()
        logger.success(f"✅ Environment initialized: {self.config['env']}")

    def authenticate(self):
        """Obtain and store authentication token."""
        logger.info("🔐 Requesting authentication token...")
        self.bearer_token, self.token_expiration = EDAuthenticator.get_new_token(self.env)
        logger.success(f"✅ Token acquired (valid until {self.token_expiration})")
        return self.bearer_token, self.token_expiration

    def refresh_token_if_needed(self):
        """Refresh token if expired or close to expiration."""
        logger.debug("Checking token validity...")
        self.bearer_token, self.token_expiration = EDAuthenticator.check_token(
            expiration_date=self.token_expiration, bearer_token=self.bearer_token, env=self.env
        )
        return self.bearer_token, self.token_expiration

    # ── Second identity provider (EarthDaily Data Studio) ───────────────────
    #
    # Geosys auth is eager above, because every extractor needs it. EDS is
    # LAZY: only the LRTS extractor uses it today, so a workflow that never
    # touches LRTS should not pay for an exchange — nor fail at startup because
    # an EDS_API_TOKEN it does not need is missing.

    #: Identity provider this manager's `bearer_token` belongs to. Extractors
    #: compare against their own `AUTH_PROVIDER` before syncing a token from
    #: here — see BaseExtractor.ensure_token_valid.
    AUTH_PROVIDER = "geosys"

    @property
    def eds_token(self):
        """EDS access token for this env, exchanged on first use and cached.

        Without this every LRTS extractor exchanges its own token at
        construction, so a workflow with several LRTS steps performs N exchanges
        for a credential that is valid 24 hours. Held here, one exchange serves
        the whole run.

        Returns:
            tuple: ``(access_token, expires_at)``, refreshed when within the
            expiry margin.

        Raises:
            ValueError: No EDS API token configured for this environment. Raised
                on first *use*, not at construction, so unrelated workflows are
                unaffected.
        """

        return self.refresh_eds_token_if_needed()

    def refresh_eds_token_if_needed(self):
        """Return a valid EDS token, exchanging one only when needed.

        The EDS counterpart of :meth:`refresh_token_if_needed`. Both providers
        now expose the same "give me a usable token" call, which is what
        ``@requires_token`` -> ``BaseExtractor.ensure_token_valid()`` drives.
        """
        from earthdaily.agriculture.core.identity_eds import EDSAuthenticator

        cached = getattr(self, "_eds_token", (None, None))
        eds_env = getattr(self, "eds_env", self.env)
        before = cached[0]

        self._eds_token = EDSAuthenticator.check_token(cached[1], cached[0], env=eds_env)

        if self._eds_token[0] != before:
            logger.success(f"🔐 EDS token acquired for env={eds_env} (valid until {self._eds_token[1]})")
        return self._eds_token

    def get_token(self, provider: str = "geosys"):
        """Return ``(token, expires_at)`` for an identity provider.

        The single entry point extractors use, so adding a provider does not
        mean touching every call site.

        Args:
            provider: ``"geosys"`` (default) or ``"eds"``.
        """
        if provider == "geosys":
            return self.refresh_token_if_needed()
        if provider == "eds":
            return self.eds_token
        raise ValueError(f"❌ Unknown identity provider '{provider}' — expected 'geosys' or 'eds'")

    def initialize_eds(self):
        """Exchange the EDS token now rather than on first use.

        Opt-in fail-fast, the same role ``storage="s3"`` plays for the S3 client:
        call it when you would rather a missing or wrong-environment EDS token
        surface at startup than midway through a long run.
        """
        return self.eds_token

    # ── Workflow Runner ─────────────────────────────────────────────────────

    def load_workflow(self, config_path: str) -> dict:
        """
        Load and validate a multi-step workflow YAML configuration.

        Supports two YAML formats:
        - Legacy flat format: {defaults, analytics: {name: {module, class, ...}}}
        - New step format:    {workflow: {name, settings, steps: [{name, depends_on, ...}]}}

        Stores parsed config in self.workflow_cfg.
        Builds step dependency graph and validates references.

        .. note::
            **Breaking change — ``input_from`` field.** Steps now accept an
            optional ``input_from`` field that selects the entity_list source
            (``"original"`` or another step's name). When omitted, it defaults
            to the first ``depends_on`` if present, else ``"original"``.
            Previously, every step received the ``entity_list`` passed to
            ``run_workflow()`` regardless of ``depends_on``. Existing YAMLs that
            rely on the old behavior must add ``input_from: "original"``
            explicitly. ``load_workflow()`` validates ``input_from`` references
            at load time (must be ``"original"`` or a step in an earlier
            execution level) and raises ``ValueError`` otherwise.

        Args:
            config_path: Path to the workflow YAML file.

        Returns:
            The parsed config — **unwrapped for the new step format**. The YAML is
            ``{workflow: {name, settings, steps: [...]}}``, but what comes back is
            the INNER dict, so the keys to index are ``name`` / ``settings`` /
            ``steps``, not ``workflow``.

            The shape depends on the input format, which is the trap:

            ===================  ====================================  ==============
            YAML format          Returns                               ``cfg["steps"]``
            ===================  ====================================  ==============
            new (``workflow:``)  ``raw["workflow"]`` — unwrapped       works
            legacy (``analytics:``)  ``raw`` — wrapped, as-is          n/a
            ===================  ====================================  ==============

            Reaching for ``cfg["workflow"]["steps"]`` against a valid new-format
            file therefore raises a bare ``KeyError: 'workflow'`` that points
            nowhere near the cause. The same object is also stored on the manager
            as :attr:`workflow_cfg`, and the wrapped original is available
            read-only as :attr:`workflow_cfg_raw`.

        Examples:
            Indexing the returned config (new step format)::

                cfg = manager.load_workflow("configuration/workflow.yml")

                cfg["name"]                  # workflow name
                cfg["settings"]              # settings block
                cfg["steps"]                 # list of step dicts
                cfg is manager.workflow_cfg  # True — same object

            NOT this — it raises ``KeyError: 'workflow'``::

                cfg["workflow"]["steps"]     # ✗ the wrapper is already stripped

            If you genuinely need the wrapped form (round-tripping the YAML,
            say), use the raw view rather than re-wrapping by hand::

                manager.workflow_cfg_raw["workflow"]["steps"]
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Workflow config not found: {config_path}")

        # Explicit encoding, never the locale default: on Windows that is cp1252,
        # which raises UnicodeDecodeError on any non-ASCII byte before YAML parsing
        # even begins. The project template's own banner comments use box-drawing
        # characters, so a freshly generated project failed to load its own default
        # workflow. `utf-8-sig` also tolerates a BOM, which editors add silently.
        with open(config_path, encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f)

        # Keep the wrapped original available; what this method RETURNS stays
        # unchanged (unwrapped for the new format) because callers depend on it.
        self.workflow_cfg_raw = raw

        if "workflow" in raw:
            # New step format
            self.workflow_cfg = raw["workflow"]
            # Rewrite date sentinels in-place before anything else sees the cfg.
            self._resolve_date_sentinels(self.workflow_cfg)
        elif "analytics" in raw:
            # Legacy flat format — warn but store raw config
            logger.warning(
                "Legacy flat YAML format detected. "
                "Use the new step format ({workflow: {steps: [...]}}) for run_workflow()."
            )
            self.workflow_cfg = raw
            self.workflow_steps = None
            self.step_map = None
            return raw
        else:
            raise ValueError("Invalid workflow YAML: must contain 'workflow' or 'analytics' key")

        self.workflow_steps = self.workflow_cfg.get("steps", [])

        # Build lookup: step_name -> step_config
        self.step_map = {}
        for step in self.workflow_steps:
            if "name" not in step:
                raise ValueError(f"Step missing required 'name' field: {step}")
            if step["name"] in self.step_map:
                raise ValueError(f"Duplicate step name: {step['name']}")
            if "enabled" in step and not isinstance(step["enabled"], bool):
                raise ValueError(
                    f"Step '{step['name']}': 'enabled' must be a boolean, got {type(step['enabled']).__name__}"
                )
            self.step_map[step["name"]] = step

        # Validate dependency graph (catches unknown refs and cycles)
        self._build_dependency_graph()
        levels = self._topological_order()

        # Validate input_from references — must be "original" or a step in an
        # earlier execution level (cannot reference self, parallel, or later step).
        step_to_level = {sname: lvl_idx for lvl_idx, level in enumerate(levels) for sname in level}
        for step in self.workflow_steps:
            input_from = step.get("input_from")
            if input_from is None or input_from == "original":
                continue
            if not isinstance(input_from, str):
                raise ValueError(
                    f"Step '{step['name']}': input_from must be a string "
                    f'("original" or a step name), got {type(input_from).__name__}'
                )
            if input_from not in self.step_map:
                raise ValueError(f"Step '{step['name']}': input_from='{input_from}' refers to unknown step")
            if step_to_level[input_from] >= step_to_level[step["name"]]:
                raise ValueError(
                    f"Step '{step['name']}': input_from='{input_from}' must "
                    f"reference a step in an earlier execution level "
                    f"(cannot reference self, a parallel step, or a later step)"
                )

        name = self.workflow_cfg.get("name", "Unnamed Workflow")
        logger.info(f"Loaded workflow: {name} ({len(self.workflow_steps)} steps)")
        print(f"Loaded workflow: {name} ({len(self.workflow_steps)} steps)")

        return self.workflow_cfg

    def run_workflow(
        self,
        entity_list: pd.DataFrame = None,
        run_prefix: str | None = None,
        disabled_steps: list[str] | None = None,
        *,
        generate_report: bool = False,
        report_options: dict[str, Any] | None = None,
    ) -> dict[str, dict]:
        """
        Execute all workflow steps respecting dependency order and transforms.

        .. note::
            **Breaking change — per-step input resolution.** Each step now
            receives an entity_list determined by its ``input_from`` field
            (or, when omitted, the first ``depends_on``, falling back to
            ``"original"``). Previously every step received the ``entity_list``
            argument unchanged. Steps that auto-chain from an upstream no
            longer need a no-op ``use_upstream_entities`` transform; if a step
            still requires the original ``entity_list`` despite having
            ``depends_on``, declare ``input_from: "original"`` in the YAML.

        Args:
            entity_list: Input entities. If None, uses self.sfd_list.
            run_prefix: Optional per-run tag inserted between the workflow-level
                ``settings.output_prefix`` and the step-level prefix in output
                filenames. Useful for re-running the same YAML with different
                tags (per client / per date / per scenario) without mutating
                the loaded config. Final pattern:
                ``<output_prefix>_<run_prefix>_<step_prefix>_results_<timestamp>.csv``.
                Any of the three layers can be absent — only present ones contribute.
            disabled_steps: Optional list of step names to disable for *this run only*.
                Equivalent to setting ``enabled: false`` on those steps in the YAML.
                The override is restored after the run, so the loaded YAML state is
                preserved. Unknown step names raise ``ValueError``. The headless
                counterpart of the ``select_steps_to_run()`` and
                ``interactive_workflow()`` UIs — useful for CLI / Argo / scheduled
                runs that want to skip steps without editing the YAML.
            generate_report: When ``True``, instantiate a
                :class:`~earthdaily.agriculture.reporting.WorkflowRunReporter`, populate
                it with the run context + per-step results, time the run, and
                stash the configured reporter at ``self.last_run_reporter`` so
                callers can amend it (e.g. add a ``report_path`` known only
                after a downstream export step) and render HTML / JSON
                receipts. The returned ``workflow_results`` dict is unchanged
                — this is purely additive. Default ``False`` (no reporter,
                ``self.last_run_reporter`` set to ``None``).
            report_options: Optional dict forwarded to the reporter. Accepted
                keys split between the constructor and the run-context:

                - constructor: ``include_step_table``, ``include_data_preview``,
                  ``preview_rows``, ``max_errors_shown``.
                - context: ``parameters``, ``column_mapping``,
                  ``resolved_yaml``.

                Unknown keys raise ``ValueError`` — protects against typos.
                Ignored unless ``generate_report=True``.

        Returns:
            Dict mapping step_name -> {results_df, global_errors, failed_ids}

        Flow per step:
            1. Resolve entity_list source via ``input_from`` (or its default)
            2. Run transform (if defined) — reshape entity_list into new entity_list
            3. Apply condition/filter (if defined) — subset entities
            4. Instantiate extractor (if defined) — dynamic import from YAML module/class
            5. Setup parameters — call setup_method with merged settings + step params
            6. Run extraction — call run_method with entity_list

        Transform-only steps (no extractor) store the transform output as results_df.
        """
        if self.workflow_steps is None or self.workflow_cfg is None or self.step_map is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")

        if entity_list is None:
            entity_list = self.sfd_list
        if entity_list is None or entity_list.empty:
            raise ValueError("No entities to process. Load entities first.")

        # Apply runtime disabled_steps overrides — restored in finally below so
        # the loaded YAML state isn't mutated permanently between calls.
        runtime_overrides: list[tuple[str, object]] = []
        for step_name in disabled_steps or []:
            if step_name not in self.step_map:
                raise ValueError(
                    f"disabled_steps: unknown step name '{step_name}'. Available: {list(self.step_map.keys())}"
                )
            step = self.step_map[step_name]
            # Sentinel to capture "field absent" vs "field present with value".
            runtime_overrides.append((step_name, step.get("enabled", _ENABLED_UNSET)))
            step["enabled"] = False

        name = self.workflow_cfg.get("name", "Unnamed Workflow")
        settings = self.workflow_cfg.get("settings", {})

        logger.info(f"Running workflow: {name}")
        logger.info(f"  Entities: {len(entity_list)}")
        logger.info(f"  Steps: {[s['name'] for s in self.workflow_steps]}")
        if run_prefix:
            logger.info(f"  Run prefix: {run_prefix}")
        if disabled_steps:
            logger.info(f"  Disabled (runtime): {disabled_steps}")

        self._run_prefix = run_prefix
        self.workflow_results = {}

        # Optional WorkflowRunReporter setup. Always set ``last_run_reporter``
        # so consumers can rely on the attribute existing regardless of the
        # flag — None when generate_report=False, the configured reporter
        # otherwise. Per-call ``report_options`` are split between the
        # reporter's constructor and ``set_run_context``; unknown keys raise.
        self.last_run_reporter = None
        _report_reporter = None
        _report_start_time: float | None = None
        if generate_report:
            from earthdaily.agriculture.reporting import WorkflowRunReporter

            opts = dict(report_options or {})
            ctor_kwargs = {k: opts.pop(k) for k in list(opts) if k in _REPORTER_CTOR_KEYS}
            ctx_kwargs = {k: opts.pop(k) for k in list(opts) if k in _REPORTER_CTX_KEYS}
            if opts:
                raise ValueError(
                    f"Unknown report_options keys: {sorted(opts)}. "
                    f"Accepted: {sorted(_REPORTER_CTOR_KEYS | _REPORTER_CTX_KEYS)}."
                )
            _report_reporter = WorkflowRunReporter(**ctor_kwargs)
            _report_reporter.set_run_context(
                workflow_name=name,
                prefix=settings.get("output_prefix"),
                env=self.env,
                entity_count=len(entity_list),
                **ctx_kwargs,
            )
            _report_start_time = time.monotonic()

        try:
            levels = self._topological_order()

            for level_idx, level in enumerate(levels):
                logger.info(f"\n--- Level {level_idx + 1}: {level} ---")

                if len(level) == 1:
                    step_name = level[0]
                    self.workflow_results[step_name] = self._execute_single_step(self.step_map[step_name], entity_list)
                else:
                    # Parallel execution for independent steps at the same level.
                    # Fail-fast: collect every future's outcome first (so the loop
                    # logs all errors at the same level) then re-raise the first
                    # exception so run_workflow's caller sees a real failure
                    # rather than a silent log line. Matches the single-step path.
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    max_parallel = min(len(level), settings.get("max_parallel_steps", 4))
                    parallel_errors: list[tuple[str, Exception]] = []
                    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                        futures = {
                            pool.submit(
                                self._execute_single_step,
                                self.step_map[step_name],
                                entity_list,
                            ): step_name
                            for step_name in level
                        }
                        for future in as_completed(futures):
                            step_name = futures[future]
                            try:
                                self.workflow_results[step_name] = future.result()
                            except Exception as e:
                                logger.error(f"Step '{step_name}' failed: {e}")
                                self.workflow_results[step_name] = {
                                    "results_df": pd.DataFrame(),
                                    "global_errors": [{"step": step_name, "error": str(e)}],
                                    "failed_ids": [],
                                }
                                parallel_errors.append((step_name, e))
                    if parallel_errors:
                        # The outer try/finally (lines below) still runs and
                        # restores runtime_overrides before this propagates.
                        raise parallel_errors[0][1]

            # Summary
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Workflow complete: {name}")
            for step_name, res in self.workflow_results.items():
                df = res.get("results_df", pd.DataFrame())
                status = "skipped" if res.get("skipped") else f"{len(df)} rows"
                errors = len(res.get("global_errors", []))
                logger.info(f"    {step_name}: {status}" + (f" ({errors} errors)" if errors else ""))
            logger.info(f"{'=' * 60}")
        finally:
            # Restore runtime-overridden enabled flags to their pre-call state.
            for step_name, original in runtime_overrides:
                step = self.step_map[step_name]
                if original is _ENABLED_UNSET:
                    step.pop("enabled", None)
                else:
                    step["enabled"] = original
            # Reset per-run state so a later run_workflow() call without run_prefix
            # doesn't inherit the previous tag.
            self._run_prefix = None

            # Populate the reporter (if configured) with whatever step results
            # were collected — even on exception, so a partial report is still
            # available via ``self.last_run_reporter``. Wrapped defensively so
            # a reporter bug never masks the original workflow error.
            if _report_reporter is not None:
                elapsed = time.monotonic() - _report_start_time if _report_start_time is not None else None
                step_kinds = {
                    step_name: _classify_workflow_step((self.step_map or {}).get(step_name, {}))
                    for step_name in self.workflow_results
                }
                try:
                    _report_reporter.set_step_results(
                        self.workflow_results,
                        elapsed_seconds=elapsed,
                        step_kinds=step_kinds,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"Failed to populate WorkflowRunReporter: {exc}")
                # Attach the run-results DAG so the HTML report can embed it
                # (and the modebar exposes a custom-filename PNG download).
                # Wrapped defensively — plotly is an optional install path in
                # some environments, and we never want chart rendering to
                # mask the real workflow outcome.
                try:
                    dag_fig = self.visualize_workflow_results(reporter=_report_reporter)
                    _report_reporter.set_workflow_dag(dag_fig)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"Failed to attach workflow DAG to reporter: {exc}")
                self.last_run_reporter = _report_reporter

        return self.workflow_results

    def _dag_layout(
        self,
        h_spacing: float,
        v_spacing: float,
    ) -> tuple[dict, dict, dict, list]:
        """Return ``(info, info_steps, pos, arrow_annotations)`` for DAG drawing.

        Shared between :meth:`visualize_workflow` (config view) and
        :meth:`visualize_workflow_results` (run-results view) so both stay
        in sync on layout and edge geometry.
        """
        if self.workflow_steps is None or self.step_map is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")

        info = self.inspect_workflow()
        levels = info["execution_levels"]
        info_steps = {s["name"]: s for s in info["steps"]}

        pos: dict[str, tuple[float, float]] = {}
        for col, level in enumerate(levels):
            n = len(level)
            for i, name in enumerate(level):
                pos[name] = (col * h_spacing, (i - (n - 1) / 2) * v_spacing)

        arrows: list[dict] = []
        for step in self.workflow_steps:
            deps = step.get("depends_on")
            if not deps:
                continue
            if isinstance(deps, str):
                deps = [deps]
            for dep in deps:
                if dep not in pos or step["name"] not in pos:
                    continue
                x0, y0 = pos[dep]
                x1, y1 = pos[step["name"]]
                arrows.append(
                    dict(
                        ax=x0,
                        ay=y0,
                        x=x1,
                        y=y1,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.4,
                        arrowwidth=1.8,
                        arrowcolor="#94a3b8",
                        standoff=24,
                        startstandoff=24,
                    )
                )
        return info, info_steps, pos, arrows

    def dag_export_config(self, filename: str = "edagro_workflow_dag") -> dict:
        """Return a Plotly ``config`` dict to pass to ``fig.show()`` / ``write_html``.

        Bakes in a custom PNG filename and 2× scale so the modebar download
        button produces a publication-ready image instead of ``newplot.png``.
        """
        return {
            "displaylogo": False,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "filename": filename,
                "format": "png",
                "scale": 2,
            },
        }

    def export_workflow_dag_html(
        self,
        path: str,
        *,
        mode: str = "results",
        reporter=None,
        filename_stem: str | None = None,
        **kwargs,
    ) -> str:
        """Write a standalone HTML file of the workflow DAG with export config baked in.

        Args:
            path: Output path (local or fsspec-compatible URI).
            mode: ``"results"`` for the run-results DAG (requires ``run_workflow()``
                to have produced ``self.workflow_results`` or an explicit
                ``reporter``); ``"config"`` for the static configuration DAG.
            reporter: Optional ``WorkflowRunReporter`` (only used when ``mode='results'``).
            filename_stem: Stem for the PNG-download filename surfaced by the
                modebar button. Defaults to ``<workflow>_dag_<mode>``.
            **kwargs: Forwarded to the underlying ``visualize_*`` method.

        Returns:
            The path written to.
        """
        import plotly.io as pio

        if mode == "results":
            fig = self.visualize_workflow_results(reporter=reporter, **kwargs)
        elif mode == "config":
            fig = self.visualize_workflow(**kwargs)
        else:
            raise ValueError(f"mode must be 'results' or 'config', got {mode!r}")

        wf_name = (self.workflow_cfg or {}).get("name", "workflow") if self.workflow_cfg else "workflow"
        stem = filename_stem or f"{_slugify(wf_name)}_dag_{mode}"
        config = self.dag_export_config(filename=stem)
        pio.write_html(fig, path, config=config, include_plotlyjs="cdn", full_html=True)
        return path

    def visualize_workflow(
        self,
        *,
        height: int = 560,
        h_spacing: float = 2.6,
        v_spacing: float = 1.5,
        max_value_len: int = 70,
    ):
        """
        Return an interactive Plotly DAG visualization of the loaded workflow.

        Nodes are placed left→right by topological level. Color encodes role:
        transform + extractor (orange), extractor only (green), transform-only (blue).
        Hover reveals extractor / transform setup + run params, dependencies, and
        the resolved ``input_from``.

        .. note::
            **Breaking change.** Earlier versions of this method returned a plain
            text DAG (``str``). It now returns a ``plotly.graph_objects.Figure``.
            Callers that did ``print(manager.visualize_workflow())`` should switch
            to ``manager.visualize_workflow().show()`` (or simply leave it as the
            last expression of a notebook cell to auto-render).

        Args:
            height: Figure height in pixels.
            h_spacing: Horizontal spacing between execution levels.
            v_spacing: Vertical spacing between sibling steps within a level.
            max_value_len: Max length for hover param values before truncation.

        Returns:
            plotly.graph_objects.Figure — call ``.show()`` to render in a notebook.
        """
        import plotly.graph_objects as go

        info, info_steps, pos, arrows = self._dag_layout(h_spacing, v_spacing)

        # ── Node classification + styling ──────────────────────────────────
        palette = {
            "hybrid": ("#f59e0b", "Transform + Extractor"),
            "extractor": ("#10b981", "Extractor"),
            "transform": ("#3b82f6", "Transform-only"),
        }

        def classify(meta):
            if meta.get("has_transform") and meta.get("has_extractor"):
                return "hybrid"
            if meta.get("has_extractor"):
                return "extractor"
            return "transform"

        def fmt_value(v):
            s = repr(v) if isinstance(v, str) else str(v)
            if len(s) > max_value_len:
                s = s[: max_value_len - 1] + "…"
            # Plotly hover renders HTML — escape angle brackets that may leak in.
            return s.replace("<", "&lt;").replace(">", "&gt;")

        def fmt_params(params):
            if not params:
                return ["    <i>(no params)</i>"]
            out = []
            for k, v in params.items():
                if isinstance(v, dict):
                    out.append(f"    • <b>{k}</b>:")
                    for kk, vv in v.items():
                        out.append(f"        ◦ {kk}: {fmt_value(vv)}")
                else:
                    out.append(f"    • <b>{k}</b>: {fmt_value(v)}")
            return out

        DISABLED_COLOR = "#cbd5e1"  # slate-300, grey
        node_x, node_y, colors, labels, hovers = [], [], [], [], []
        for name, (x, y) in pos.items():
            meta = info_steps[name]
            raw = (self.step_map or {})[name]
            kind = classify(meta)
            disabled = meta.get("enabled", True) is False
            node_x.append(x)
            node_y.append(y)
            colors.append(DISABLED_COLOR if disabled else palette[kind][0])
            labels.append(f"<b>{name}</b>" + (" <i>(disabled)</i>" if disabled else ""))

            kind_label = palette[kind][1] + (" — DISABLED" if disabled else "")
            lines = [
                f"<b style='font-size:14px'>{name}</b>",
                f"<i>{kind_label}</i>",
                "─────────────────────",
            ]
            if disabled:
                lines.append("<b style='color:#dc2626'>⚠ enabled: false</b> — step will be skipped at runtime.")

            deps = meta.get("depends_on")
            if deps:
                dep_str = deps if isinstance(deps, str) else ", ".join(deps)
                lines.append(f"<b>depends_on:</b> {dep_str}")

            # Resolved input_from (with default marker when inferred)
            input_from = meta.get("input_from", "original")
            explicit = meta.get("input_from_explicit", False)
            suffix = "" if explicit else " <i>(default)</i>"
            lines.append(f"<b>input_from:</b> {input_from}{suffix}")

            if raw.get("transform"):
                t = raw["transform"]
                lines.append(f"<b>Transform:</b> {t.get('module', '')}.{t.get('function', '')}")
                lines.extend(fmt_params(t.get("params")))

            if raw.get("extractor"):
                lines.append(f"<b>Extractor:</b> {raw['extractor']}")
                setup = raw.get("setup") or {}
                if setup:
                    method = setup.get("method", "")
                    lines.append(f"  <i>setup{f' — {method}' if method else ''}</i>")
                    lines.extend(fmt_params(setup.get("params")))
                run = raw.get("run") or {}
                if run:
                    method = run.get("method", "")
                    lines.append(f"  <i>run{f' — {method}' if method else ''}</i>")
                    lines.extend(fmt_params(run.get("params")))

            if raw.get("condition"):
                c = raw["condition"]
                lines.append(f"<b>filter:</b> {c.get('column')} {c.get('operator', 'eq')} {c.get('value')}")

            hovers.append("<br>".join(lines))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            textfont=dict(size=12, color="#0f172a"),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=46,
                color=colors,
                line=dict(color="white", width=3),
                symbol="circle",
            ),
            showlegend=False,
        )

        legend_traces = [
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                marker=dict(size=14, color=color),
            )
            for color, label in palette.values()
        ]

        fig = go.Figure(data=[node_trace, *legend_traces])
        fig.update_layout(
            title=dict(
                text=f"<b>{info['name']}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color="#0f172a"),
            ),
            annotations=arrows,
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=30, r=30, t=80, b=70),
            height=height,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            # Hide every axis decoration but keep the axes "alive" so the
            # scatter trace still gets a coordinate system. `visible=False`
            # would also work but combines poorly with `scaleanchor` and has
            # rendered an empty plot area in some Plotly 6 contexts.
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                ticks="",
            ),
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor="#f59e0b",
                font=dict(family="Menlo, Consolas, monospace", size=12, color="#f8fafc"),
                align="left",
            ),
        )
        return fig

    def visualize_workflow_results(
        self,
        reporter=None,
        *,
        height: int = 560,
        h_spacing: float = 2.6,
        v_spacing: float = 1.5,
    ):
        """
        Return a Plotly DAG colored by **run results** (rows / errors / skipped).

        Mirrors :meth:`visualize_workflow` layout but nodes are colored by
        outcome instead of role, and each node carries an extraction-KPI line
        (``rows • errors``). Hover shows per-step run details:
        rows, errors, failed_ids sample, skip reason, cache stats.

        Data source: a :class:`~earthdaily.agriculture.reporting.WorkflowRunReporter`.
        If ``reporter`` is None, one is built on-the-fly from
        ``self.workflow_results`` (so a notebook user who didn't construct
        a reporter explicitly still gets a chart). The on-the-fly reporter
        is **not** stored on the manager.

        Args:
            reporter: Optional pre-populated reporter
                (e.g. ``manager.last_run_reporter``). When None, derives
                from ``self.workflow_results`` (requires a prior
                ``run_workflow()`` call).
            height: Figure height in pixels.
            h_spacing: Horizontal spacing between execution levels.
            v_spacing: Vertical spacing between sibling steps within a level.

        Returns:
            plotly.graph_objects.Figure — call ``.show()`` (optionally with
            ``config=manager.dag_export_config(filename=...)`` for a
            custom-filename PNG download button) to render in a notebook.

        Raises:
            RuntimeError: if no workflow is loaded, or no reporter was passed
                and ``run_workflow()`` has not been called.
        """
        import plotly.graph_objects as go

        if reporter is None:
            if not self.workflow_results:
                raise RuntimeError(
                    "No workflow results to visualize. Either pass a WorkflowRunReporter, or run run_workflow() first."
                )
            from earthdaily.agriculture.reporting import WorkflowRunReporter

            reporter = WorkflowRunReporter(include_step_table=False)
            wf_name = (self.workflow_cfg or {}).get("name", "Workflow")
            reporter.set_run_context(
                workflow_name=wf_name,
                env=self.env,
                entity_count=int(len(self.sfd_list)) if self.sfd_list is not None else 0,
            )
            step_kinds = {
                step_name: _classify_workflow_step((self.step_map or {}).get(step_name, {}))
                for step_name in self.workflow_results
            }
            reporter.set_step_results(self.workflow_results, step_kinds=step_kinds)

        # _step_summaries is the structured KPI source; safe to read here.
        summaries_by_name = {s["name"]: s for s in reporter._step_summaries}
        elapsed = reporter._elapsed_seconds

        info, info_steps, pos, arrows = self._dag_layout(h_spacing, v_spacing)

        # Status-based palette — outcome instead of role.
        STATUS_PALETTE = {
            "success": ("#10b981", "Success"),
            "partial": ("#f59e0b", "Partial (with errors)"),
            "failed": ("#ef4444", "Failed"),
            "skipped": ("#cbd5e1", "Skipped"),
            "not_run": ("#94a3b8", "Not run"),
        }

        def classify_status(summary: dict | None) -> str:
            if summary is None:
                return "not_run"
            if summary.get("skipped"):
                return "skipped"
            rows = int(summary.get("rows", 0) or 0)
            errors = int(summary.get("errors", 0) or 0)
            if errors and rows:
                return "partial"
            if errors:
                return "failed"
            if rows:
                return "success"
            return "failed" if errors else "not_run"

        node_x: list[float] = []
        node_y: list[float] = []
        colors: list[str] = []
        labels: list[str] = []
        hovers: list[str] = []
        for name, (x, y) in pos.items():
            summary = summaries_by_name.get(name)
            meta = info_steps[name]
            status = classify_status(summary)
            color, status_label = STATUS_PALETTE[status]
            node_x.append(x)
            node_y.append(y)
            colors.append(color)

            if summary is None:
                kpi_line = "<br><i>(not run)</i>"
            else:
                rows = int(summary.get("rows", 0) or 0)
                errors = int(summary.get("errors", 0) or 0)
                kpi_line = f"<br>{rows:,} rows • {errors} errors"
            labels.append(f"<b>{name}</b>{kpi_line}")

            lines = [
                f"<b style='font-size:14px'>{name}</b>",
                f"<i>{status_label}</i>",
                "─────────────────────",
            ]
            kind = meta.get("kind") or ("extractor" if meta.get("has_extractor") else "transform")
            lines.append(f"<b>kind:</b> {kind}")
            deps = meta.get("depends_on")
            if deps:
                dep_str = deps if isinstance(deps, str) else ", ".join(deps)
                lines.append(f"<b>depends_on:</b> {dep_str}")
            input_from = meta.get("input_from", "original")
            lines.append(f"<b>input_from:</b> {input_from}")

            if summary is None:
                lines.append("<i>Step did not run.</i>")
            else:
                lines.append(f"<b>rows:</b> {int(summary.get('rows', 0) or 0):,}")
                lines.append(f"<b>errors:</b> {int(summary.get('errors', 0) or 0)}")
                failed = summary.get("failed_ids") or []
                if failed:
                    sample = ", ".join(map(str, failed[:5]))
                    more = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
                    lines.append(f"<b>failed_ids:</b> {sample}{more}")
                if summary.get("skipped"):
                    reason = summary.get("reason") or "(no reason)"
                    lines.append(f"<b>skipped:</b> {reason}")
                cache = summary.get("cache")
                if isinstance(cache, dict) and cache:
                    cache_bits = ", ".join(f"{k}={v}" for k, v in cache.items())
                    lines.append(f"<b>cache:</b> {cache_bits}")
            hovers.append("<br>".join(lines))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            textfont=dict(size=11, color="#0f172a"),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=46,
                color=colors,
                line=dict(color="white", width=3),
                symbol="circle",
            ),
            showlegend=False,
        )

        legend_traces = [
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                marker=dict(size=14, color=color),
            )
            for color, label in STATUS_PALETTE.values()
        ]

        suffix = f" — duration {elapsed:.1f}s" if elapsed else ""
        fig = go.Figure(data=[node_trace, *legend_traces])
        fig.update_layout(
            title=dict(
                text=f"<b>{info['name']} — Run Results</b>{suffix}",
                x=0.5,
                xanchor="center",
                font=dict(size=20, color="#0f172a"),
            ),
            annotations=arrows,
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=30, r=30, t=80, b=70),
            height=height,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                ticks="",
            ),
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor="#10b981",
                font=dict(family="Menlo, Consolas, monospace", size=12, color="#f8fafc"),
                align="left",
            ),
        )
        return fig

    def select_steps_to_run(self, entity_list: pd.DataFrame = None, run_prefix: str | None = None):
        """
        Render an ipywidgets-based step selector with a checkbox per step
        and a "Run workflow" button.

        Toggling a checkbox mutates ``self.step_map[name]["enabled"]`` in place,
        so the chosen subset is honored by ``run_workflow()`` (or any other path
        that reads the flag). Clicking "Run workflow" calls ``run_workflow()``
        with the given ``entity_list`` and ``run_prefix``.

        Args:
            entity_list: Entities to pass to ``run_workflow()`` when the button
                is clicked. Defaults to ``self.sfd_list``.
            run_prefix: Optional ``run_prefix`` forwarded to ``run_workflow()``.

        Returns:
            ipywidgets.VBox — display in a Jupyter notebook to interact.

        Raises:
            RuntimeError: if no workflow is loaded.
            ImportError: if ``ipywidgets`` is not available.
        """
        if self.workflow_steps is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")

        try:
            import ipywidgets as widgets
        except ImportError as e:
            raise ImportError(
                "select_steps_to_run() requires ipywidgets. Install with `pip install ipywidgets`."
            ) from e

        info = self.inspect_workflow()
        info_steps = {s["name"]: s for s in info["steps"]}

        checkboxes: dict[str, widgets.Checkbox] = {}
        rows: list[widgets.HBox] = []
        for step in self.workflow_steps:
            meta = info_steps[step["name"]]
            cb = widgets.Checkbox(
                value=meta.get("enabled", True),
                description=step["name"],
                indent=False,
                layout=widgets.Layout(width="220px"),
            )
            # Annotate with kind + dependencies so the user knows what they're toggling.
            kind_bits: list[str] = []
            if meta.get("has_extractor"):
                kind_bits.append(f"extractor: {meta['extractor']}")
            if meta.get("has_transform"):
                kind_bits.append("transform")
            deps = meta.get("depends_on")
            if deps:
                dep_str = deps if isinstance(deps, str) else ", ".join(deps)
                kind_bits.append(f"depends_on: {dep_str}")
            annotation = widgets.HTML(value=f"<i style='color:#64748b'>{' • '.join(kind_bits) or '—'}</i>")
            rows.append(widgets.HBox([cb, annotation]))
            checkboxes[step["name"]] = cb

        run_button = widgets.Button(description="Run workflow", button_style="primary", icon="play")
        output = widgets.Output()

        def _on_run(_b):
            # Sync checkbox state -> step_map.enabled before running.
            for step_name, cb in checkboxes.items():
                self.step_map[step_name]["enabled"] = cb.value
            with output:
                output.clear_output()
                self.run_workflow(entity_list=entity_list, run_prefix=run_prefix)

        run_button.on_click(_on_run)

        return widgets.VBox(
            [
                widgets.HTML("<b>Select steps to run</b>"),
                *rows,
                run_button,
                output,
            ]
        )

    def interactive_workflow(self, **kwargs):
        """
        Return a clickable Plotly DAG of the loaded workflow.

        Same layout as ``visualize_workflow()`` but returns a
        ``plotly.graph_objects.FigureWidget``. Clicking a node toggles its
        ``enabled`` state in ``self.step_map`` (and recolors it to grey).
        After toggling the desired subset, call ``run_workflow()`` to execute
        only the still-enabled steps. State persists on the manager between
        ``interactive_workflow()`` calls and into ``run_workflow()``.

        Note: cascade-skip of downstream steps is enforced at runtime by the
        existing empty-upstream branch — this widget does not pre-shade
        descendants. Disabled-node colour is the only visual cue.

        Args:
            **kwargs: Forwarded to ``visualize_workflow()`` (height, h_spacing, …).

        Returns:
            ipywidgets.HBox wrapping a ``plotly.graph_objects.FigureWidget`` —
            the wrapper makes the chart fill the cell width (FigureWidget on
            its own defaults to ~700px in VS Code / classic notebook). Clicks
            still fire on the underlying widget; display in a Jupyter
            notebook to interact (clicks won't fire on a static ``.show()``).

        Raises:
            RuntimeError: if no workflow is loaded.
            ImportError: if ``plotly`` is not available.
        """
        if self.workflow_steps is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")

        # FigureWidget click callbacks require ipywidgets >= 7 in the kernel
        # AND a Jupyter widgets-capable renderer (JupyterLab, classic Notebook
        # with widgets extension, VS Code with ipywidgets enabled). Without
        # ipywidgets the widget renders as a static Plotly figure and the
        # on_click handler never fires — defeating the purpose of this method.
        # Fail loud here with an install hint rather than silently returning a
        # non-interactive figure that *looks* like the DAG but doesn't toggle.
        try:
            import ipywidgets  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "interactive_workflow() requires ipywidgets for click handling. "
                'Install with `pip install -e ".[jupyter]"` (recommended — also '
                "pulls anywidget for VS Code) or `pip install ipywidgets`. "
                "For a non-clickable preview, call visualize_workflow() instead."
            ) from e

        import plotly.graph_objects as go

        # Reuse the static figure's layout / styling, then upgrade to FigureWidget.
        static_fig = self.visualize_workflow(**kwargs)
        fig_widget = go.FigureWidget(static_fig)
        node_trace = fig_widget.data[0]  # First trace is the node trace (see visualize_workflow).

        # Node order matches insertion order of ``pos`` in visualize_workflow,
        # which iterates self.workflow_steps in topological-level order.
        info = self.inspect_workflow()
        levels = info["execution_levels"]
        node_names: list[str] = [name for level in levels for name in level]

        # Recompute base (enabled) colors so toggling restores the original hue.
        info_steps = {s["name"]: s for s in info["steps"]}
        base_palette = {"hybrid": "#f59e0b", "extractor": "#10b981", "transform": "#3b82f6"}

        def _base_color(meta):
            if meta.get("has_transform") and meta.get("has_extractor"):
                return base_palette["hybrid"]
            if meta.get("has_extractor"):
                return base_palette["extractor"]
            return base_palette["transform"]

        DISABLED_COLOR = "#cbd5e1"
        base_colors = [_base_color(info_steps[name]) for name in node_names]

        def _on_click(trace, points, _selector):
            if not points.point_inds:
                return
            idx = points.point_inds[0]
            name = node_names[idx]
            step = self.step_map[name]
            new_enabled = not step.get("enabled", True)
            step["enabled"] = new_enabled

            # Update marker colors in-place. FigureWidget batch_update broadcasts
            # the change to the displayed widget without re-rendering the whole figure.
            current = list(trace.marker.color)
            current[idx] = base_colors[idx] if new_enabled else DISABLED_COLOR
            with fig_widget.batch_update():
                trace.marker.color = current

        node_trace.on_click(_on_click)

        # Force the widget to fill the cell width. FigureWidget's ``.layout``
        # is the plotly Layout — the ipywidget CSS layout (which actually
        # controls container width) has to come from a wrapper. Without
        # this, FigureWidget renders at its default ~700px in VS Code and
        # the classic notebook, while Figure.show() fills the container
        # via its own responsive config. Wrapping the widget in an HBox
        # with ``width="100%"`` keeps clicks routed to the underlying
        # FigureWidget while letting the container expand.
        import ipywidgets as widgets

        return widgets.HBox(
            [fig_widget],
            layout=widgets.Layout(width="100%"),
        )

    def inspect_workflow(self) -> dict:
        """Return a structured summary of the loaded workflow for programmatic inspection."""
        if self.workflow_steps is None or self.workflow_cfg is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")

        steps_info = []
        for step in self.workflow_steps:
            # Resolve effective input_from (mirrors runtime resolution)
            input_from = step.get("input_from")
            if input_from is None:
                deps = step.get("depends_on")
                if deps:
                    input_from = deps if isinstance(deps, str) else deps[0]
                else:
                    input_from = "original"
            info = {
                "name": step["name"],
                "depends_on": step.get("depends_on"),
                "input_from": input_from,
                "input_from_explicit": "input_from" in step,
                "has_transform": "transform" in step,
                "has_extractor": "extractor" in step,
                "extractor": step.get("extractor"),
                "enabled": step.get("enabled", True),
            }
            if "transform" in step:
                t = step["transform"]
                info["transform"] = f"{t['module']}.{t['function']}"
            steps_info.append(info)

        return {
            "name": self.workflow_cfg.get("name", "Unnamed Workflow"),
            "description": self.workflow_cfg.get("description", ""),
            "settings": self.workflow_cfg.get("settings", {}),
            "steps": steps_info,
            "execution_levels": self._topological_order(),
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_date_sentinels(self, workflow_cfg: dict[str, Any]) -> None:
        """Rewrite date sentinels (`today`, `yesterday`, `today-Nd`, `today-Nw`,
        `today-Nm`) inside `entity_source.params` and every step's `setup.params`
        to ISO date strings. Mutates ``workflow_cfg`` in place.

        Walks one level deep only — does not recurse into `run.params` or
        arbitrary nested dicts. Strings inside lists are out of scope.

        Timezone for "today" comes from ``workflow_cfg["settings"]["timezone"]``
        (IANA name); defaults to UTC.

        Logs each substitution at INFO level. Pass-through for non-string
        values and non-sentinel strings.
        """
        settings = workflow_cfg.get("settings") or {}
        tz = settings.get("timezone")

        def _walk(params: dict | None, path: str) -> None:
            if not params:
                return
            for k, v in list(params.items()):
                if isinstance(v, str):
                    resolved = _resolve_date_sentinel(v, tz)
                    if resolved is not None:
                        logger.info(f"🗓 {path}.{k}: {v!r} → {resolved!r}")
                        params[k] = resolved

        _walk((workflow_cfg.get("entity_source") or {}).get("params"), "entity_source.params")
        for step in workflow_cfg.get("steps") or []:
            name = step.get("name", "<unnamed>")
            _walk((step.get("setup") or {}).get("params"), f"{name}.setup.params")

    def _build_dependency_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from step depends_on fields."""
        if self.workflow_steps is None or self.step_map is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {s["name"]: 0 for s in self.workflow_steps}

        for step in self.workflow_steps:
            deps = step.get("depends_on")
            if deps is None:
                continue
            if isinstance(deps, str):
                deps = [deps]
            for dep in deps:
                if dep not in self.step_map:
                    raise ValueError(f"Step '{step['name']}' depends on unknown step '{dep}'")
                graph[dep].append(step["name"])
                in_degree[step["name"]] += 1

        return graph, in_degree

    def _topological_order(self) -> list[list[str]]:
        """
        Return steps grouped by execution level (Kahn's algorithm).
        Each inner list contains steps that can run in parallel.
        """
        if self.workflow_steps is None:
            raise RuntimeError("No workflow loaded. Call load_workflow() first.")
        graph, in_degree = self._build_dependency_graph()
        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        levels: list[list[str]] = []

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            for name in level:
                for child in graph[name]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        scheduled = sum(len(lv) for lv in levels)
        if scheduled != len(self.workflow_steps):
            raise ValueError(
                "Circular dependency detected in workflow steps. "
                f"Scheduled {scheduled}/{len(self.workflow_steps)} steps."
            )
        return levels

    def _run_transform(
        self,
        transform_conf: dict,
        entity_list: pd.DataFrame,
        upstream_results: dict[str, dict],
    ) -> pd.DataFrame:
        """
        Dynamically import and call a transform function.

        Args:
            transform_conf: YAML transform block (module, function, params).
            entity_list: Current entity DataFrame.
            upstream_results: All completed step results.

        Returns:
            Transformed entity DataFrame.
        """
        mod_path = transform_conf["module"]
        func_name = transform_conf["function"]
        params = {**transform_conf.get("params", {})}

        # Inject workflow manager reference so transforms can access auth/config
        params["_workflow_manager"] = self

        logger.info(f"  Running transform: {mod_path}.{func_name}")
        module = importlib.import_module(mod_path)
        func = getattr(module, func_name)
        result = func(entity_list, upstream_results, params)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Transform {mod_path}.{func_name} must return a DataFrame, got {type(result).__name__}")
        logger.info(f"  Transform produced {len(result)} entities")
        return result

    def _apply_condition(
        self,
        step_cfg: dict,
        entity_list: pd.DataFrame,
        upstream_results: dict[str, dict],
    ) -> pd.DataFrame:
        """
        Apply a condition block to filter entity_list based on upstream results.

        YAML example:
            condition:
              depends_on: cropid
              column: confirmation_status
              operator: eq
              value: confirmed
        """
        cond = step_cfg.get("condition")
        if cond is None:
            return entity_list

        source_step = cond["depends_on"]
        column = cond["column"]
        operator = cond.get("operator", "eq")

        settings = (self.workflow_cfg or {}).get("settings", {})

        if source_step not in upstream_results:
            raise ValueError(f"Condition references unknown step '{source_step}'")
        source_df = upstream_results[source_step].get("results_df", pd.DataFrame())
        if source_df.empty:
            logger.warning(f"  Condition source '{source_step}' is empty — skipping all entities")
            return entity_list.iloc[0:0]

        # Get entity IDs that pass the condition
        id_col = settings.get("column_mapping", {}).get("id", "id")
        entity_id_col = "entity_id" if "entity_id" in source_df.columns else id_col

        # `value` is only required for value-bearing operators. Look it up lazily
        # so value-less operators like `notnull` don't force the user to add a
        # `value: null` placeholder in YAML.
        if operator == "eq":
            if "value" not in cond:
                raise ValueError("Condition operator 'eq' requires a 'value' key")
            value = cond["value"]
            passing = source_df.loc[source_df[column] == value, entity_id_col]
        elif operator == "ne":
            if "value" not in cond:
                raise ValueError("Condition operator 'ne' requires a 'value' key")
            value = cond["value"]
            passing = source_df.loc[source_df[column] != value, entity_id_col]
        elif operator == "in":
            if "value" not in cond:
                raise ValueError("Condition operator 'in' requires a 'value' key")
            value = cond["value"]
            passing = source_df.loc[source_df[column].isin(value), entity_id_col]
        elif operator == "notnull":
            passing = source_df.loc[source_df[column].notna(), entity_id_col]
        else:
            raise ValueError(f"Unknown condition operator: {operator}")

        passing_ids = set(passing.unique())
        before = len(entity_list)
        filtered = entity_list[entity_list[id_col].isin(passing_ids)].copy()
        log_value = f" {cond['value']}" if "value" in cond else ""
        logger.info(f"  Condition ({column} {operator}{log_value}): {before} -> {len(filtered)} entities")
        return filtered

    def _execute_single_step(
        self,
        step_cfg: dict,
        entity_list: pd.DataFrame,
    ) -> dict:
        """
        Execute a single workflow step.

        Flow:
        1. Run transform (if defined) — produces modified entity_list
        2. Apply condition (if defined) — filter entities
        3. Instantiate extractor (if defined)
        4. Setup parameters
        5. Run extraction

        If the step has a transform but no extractor, the transform output
        is stored as the step result.
        """
        step_name = step_cfg["name"]
        settings = (self.workflow_cfg or {}).get("settings", {})
        logger.info(f"{'=' * 60}")
        logger.info(f"  Step: {step_name}")
        logger.info(f"{'=' * 60}")

        # Step-level enabled flag. Disabled steps return an empty result with
        # reason="disabled"; downstream steps with depends_on/input_from on this
        # one fall through the existing empty-upstream branch and cascade-skip.
        if not step_cfg.get("enabled", True):
            logger.info(f"  Step '{step_name}' disabled in YAML — skipping")
            return {
                "results_df": pd.DataFrame(),
                "global_errors": [],
                "failed_ids": [],
                "skipped": True,
                "reason": "disabled",
            }

        # Resolve input_from: explicit value > first depends_on > "original".
        # Validation of the value happens at load_workflow time.
        input_from = step_cfg.get("input_from")
        if input_from is None:
            deps = step_cfg.get("depends_on")
            if deps:
                input_from = deps if isinstance(deps, str) else deps[0]
            else:
                input_from = "original"

        if input_from != "original":
            if input_from not in self.workflow_results:
                raise ValueError(
                    f"Step '{step_name}': input_from='{input_from}' "
                    f"refers to a step that has not run yet (or does not exist)"
                )
            upstream_df = self.workflow_results[input_from].get("results_df")
            if upstream_df is None or upstream_df.empty:
                logger.warning(f"  input_from='{input_from}' produced no rows — skipping step '{step_name}'")
                return {
                    "results_df": pd.DataFrame(),
                    "global_errors": [],
                    "failed_ids": [],
                    "skipped": True,
                }
            working_entities = upstream_df.copy()
            logger.info(f"  input_from: {input_from} ({len(working_entities)} rows)")
        else:
            working_entities = entity_list.copy()

        # 1. Transform
        if "transform" in step_cfg:
            working_entities = self._run_transform(step_cfg["transform"], working_entities, self.workflow_results)
            if working_entities.empty:
                logger.warning(f"  Transform produced empty result — skipping step '{step_name}'")
                return {
                    "results_df": pd.DataFrame(),
                    "global_errors": [],
                    "failed_ids": [],
                    "skipped": True,
                }

        # 2. Condition / filter
        working_entities = self._apply_condition(step_cfg, working_entities, self.workflow_results)
        if working_entities.empty:
            logger.warning(f"  No entities after filtering — skipping step '{step_name}'")
            return {
                "results_df": pd.DataFrame(),
                "global_errors": [],
                "failed_ids": [],
                "skipped": True,
            }

        # 3. Transform-only step (no extractor)
        if "extractor" not in step_cfg:
            logger.info(f"  Transform-only step — storing {len(working_entities)} rows as result")
            return {
                "results_df": working_entities,
                "global_errors": [],
                "failed_ids": [],
            }

        # 4. Instantiate extractor
        mod = importlib.import_module(step_cfg["module"])
        cls = getattr(mod, step_cfg["extractor"])
        extractor = cls(
            self.bearer_token,
            self.token_expiration,
            config=self.config,
        )

        # Apply column_mapping from settings
        col_mapping = settings.get("column_mapping")
        if col_mapping:
            extractor.column_mapping.update(col_mapping)

        # Apply results export format. Step-level wins over workflow-level
        # settings.export_format; both override the EDAGRO_EXPORT_FORMAT default
        # already resolved in BaseExtractor.__init__. ("csv" | "parquet")
        export_format = step_cfg.get("export_format") or settings.get("export_format")
        if export_format:
            extractor.export_format = export_format

        # Apply manifest-on-export toggle (parity with export_format). Step-level
        # wins over workflow-level settings.export_manifest; both override the
        # EDAGRO_EXPORT_MANIFEST default already resolved in BaseExtractor.__init__.
        export_manifest = step_cfg.get("export_manifest")
        if export_manifest is None:
            export_manifest = settings.get("export_manifest")
        if export_manifest is not None:
            extractor.export_manifest = bool(export_manifest)

        # Apply the durable destination for postprocess="file" rasters (parity
        # with export_format). Step-level wins over workflow-level
        # settings.output_uri; both override the EDAGRO_OUTPUT_URI default
        # already resolved in BaseExtractor.__init__. Local `output_path` stays
        # the working copy — this is the copy that outlives the runner.
        output_uri = step_cfg.get("output_uri") or settings.get("output_uri")
        if output_uri:
            extractor.output_uri = output_uri
        # Apply resume mode (parity with the two above). Step-level wins over
        # workflow-level settings.retry_failed_only. `true` resumes from the newest
        # failed_ids_<prefix>_*.csv; a string names one file exactly. Deliberately
        # separate from `fail_safe`, which now only means "tolerate per-entity
        # failures" — it used to silently imply this narrowing as well.
        retry_failed_only = step_cfg.get("retry_failed_only")
        if retry_failed_only is None:
            retry_failed_only = settings.get("retry_failed_only")
        if retry_failed_only is not None:
            extractor.retry_failed_only = retry_failed_only

        # 5. Setup parameters
        setup_cfg = step_cfg.get("setup", {})
        setup_method_name = setup_cfg.get("method")
        setup_params = {**settings, **setup_cfg.get("params", {})}

        if setup_method_name:
            setup_fn = getattr(extractor, setup_method_name)
            # Filter params to only those accepted by the setup method signature
            sig = inspect.signature(setup_fn)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                filtered_params = setup_params
            else:
                accepted = set(sig.parameters.keys())
                filtered_params = {k: v for k, v in setup_params.items() if k in accepted}

            setup_fn(**filtered_params)
            logger.info(f"  Setup: {setup_method_name}({list(filtered_params.keys())})")

        # 6. Run extraction
        run_cfg = step_cfg.get("run", {})
        run_method_name = run_cfg.get("method")
        run_params = run_cfg.get("params", {}).copy()

        if not run_method_name:
            raise ValueError(f"Step '{step_name}' has extractor but no run.method")

        run_fn = getattr(extractor, run_method_name)
        max_workers = run_params.pop("max_workers", settings.get("max_workers", 10))
        prefix = run_params.pop("prefix", step_name)
        skip_export = run_params.pop("skip_export", True)
        fail_safe = run_params.pop("fail_safe", settings.get("fail_safe", False))
        generate_report = run_params.pop("generate_report", False)

        # Compose three prefix layers, each optional:
        #   <output_prefix> (workflow-level, from settings)
        #   <run_prefix>    (per-run, from run_workflow(run_prefix=...))
        #   <step_prefix>   (per-step, from run.params.prefix or step name)
        # Final filename pattern:
        #   <output_prefix>_<run_prefix>_<step_prefix>_results_<timestamp>_<suffix>.csv
        output_prefix = settings.get("output_prefix")
        prefix_parts = []
        if output_prefix:
            prefix_parts.append(_slugify(output_prefix))
        if self._run_prefix:
            prefix_parts.append(_slugify(self._run_prefix))
        prefix_parts.append(_slugify(prefix))
        prefix = "_".join(prefix_parts)

        # Forward whatever is left in run.params to the run method.
        #
        # This used to be a hardcoded kwarg list, so any option the extractor
        # genuinely accepts but this call did not name was dropped on the floor —
        # silently. A step declaring `spatial_grouping: true` ran happily, produced
        # correct data, and never grouped; `use_cache`, `page_limit`,
        # `partial_frequency`, `filter_*`, `merge_existing` and `report_options`
        # went the same way. Nothing failed, so nothing surfaced it.
        #
        # Filtered against the target signature (same approach as the setup call
        # above) so a key that matches nothing is a loud WARNING naming it —
        # `spatial_precison` should not fail silently either.
        explicitly_passed = {
            "entity_list",
            "params",
            "max_workers",
            "output_path",
            "fail_safe",
            "prefix",
            "generate_report",
            "skip_export",
        }
        run_sig = inspect.signature(run_fn)
        run_accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in run_sig.parameters.values())
        run_accepted = set(run_sig.parameters)

        passthrough, unknown = {}, []
        for key, value in run_params.items():
            if key in explicitly_passed:
                continue  # explicit kwargs win — existing behaviour is unchanged
            if run_accepts_kwargs or key in run_accepted:
                passthrough[key] = value
            else:
                unknown.append(key)

        if unknown:
            logger.warning(
                f"  Step '{step_name}': run.params key(s) {sorted(unknown)} are not accepted by "
                f"{run_method_name}() and were IGNORED — check for a typo."
            )

        logger.info(f"  Running: {run_method_name} on {len(working_entities)} entities (max_workers={max_workers})")
        if passthrough:
            logger.info(f"  Forwarding run.params: {sorted(passthrough)}")

        result = run_fn(
            entity_list=working_entities,
            params=run_params.get("params"),
            max_workers=max_workers,
            output_path=self.output_result_dir if not skip_export else None,
            fail_safe=fail_safe,
            prefix=prefix,
            generate_report=generate_report,
            skip_export=skip_export,
            **passthrough,
        )

        results_df = result.get("results_df", pd.DataFrame())
        logger.info(f"  Result: {len(results_df)} rows")

        return {
            "results_df": results_df,
            "global_errors": result.get("global_errors", []),
            "failed_ids": result.get("failed_ids", []),
        }

    def load_seasonfields(
        self,
        sowing_date_gte=None,
        sowing_date_lte=None,
        crop_id=None,
        start_date=None,
        end_date=None,
        farm_name=None,
        fields=None,
    ):
        """
        Load seasonfields from EarthDaily platform using EntityManager with optional filters.
        Delegates to EntityManager.load_seasonfields().

        Args:
            farm_name (str | list[str], optional): Filter by farm name (Field.Farm.Name).
                A single name (``"My Farm"``) or a list (``["Farm A", "Farm B"]``) —
                a list filters all named farms in ONE request via the MDM ``$in:`` operator.
            fields (str, optional): Comma-separated list of fields to retrieve
                (e.g. "id,name,geometry,sowingDate,crop.code,acreage,customerExternalId").
                If None, uses the EntityManager default.
        """
        entity_manager = EntityManager(
            bearer_token=self.bearer_token,
            token_expiration=self.token_expiration,
            config=self.config,
            workflow_ref=self,
        )
        self.sfd_list = entity_manager.load_seasonfields(
            sowing_date_gte=sowing_date_gte,
            sowing_date_lte=sowing_date_lte,
            crop_id=crop_id,
            start_date=start_date,
            end_date=end_date,
            farm_name=farm_name,
            fields=fields,
        )
        self.data_source = "api"
        self.target_dates = None
        return self.sfd_list

    def load_target_dates(self, file_path):
        """
        Load target dates from a CSV file for specific dates extraction mode.

        The CSV should have a column 'target_date' with dates in YYYY-MM-DD format.

        Args:
            file_path (str): Path to CSV file with target dates

        Returns:
            list: List of target dates as strings

        Example CSV format:
            target_date
            2024-12-02
            2024-12-15
            2024-12-21
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Target dates file not found: {file_path}")

        df = pd.read_csv(file_path)

        if "target_date" not in df.columns:
            raise ValueError("❌ CSV must contain a 'target_date' column")

        # Parse and validate dates
        dates = []
        for date_str in df["target_date"].dropna().unique():
            try:
                # Validate date format
                parsed = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
                dates.append(parsed.strftime("%Y-%m-%d"))
            except ValueError:
                print(f"⚠️ Skipping invalid date format: {date_str}")

        if not dates:
            raise ValueError("❌ No valid dates found in file")

        # Sort dates chronologically
        dates = sorted(dates)
        self.target_dates = dates

        print(f"✅ Loaded {len(dates)} target dates:")
        print(f"   From: {dates[0]} to {dates[-1]}")
        for d in dates[:5]:
            print(f"   - {d}")
        if len(dates) > 5:
            print(f"   ... and {len(dates) - 5} more")

        return dates

    def load_sfd_list(self, file_path, geometry_col="geometry", crs="EPSG:4326"):
        """
        Load seasonfields from a file (SHP, CSV, etc.)
        Automatically detects and parses target_dates column if present.

        Args:
            file_path (str): Path to the file
            geometry_col (str): Name of geometry column (mainly for CSV files)
            crs (str): Coordinate reference system

        CSV format with specific dates:
            id,name,target_dates
            2anvgxb,K-5-BE-378,2024-12-02|2024-12-15|2024-12-21
            x37bdzx,K-5-BE-385,2024-12-02|2024-12-15|2024-12-21
        """
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".csv":
            # Load CSV directly (may or may not have geometry)
            self.sfd_list = pd.read_csv(file_path)
            logger.info(f"📄 Loaded CSV file with {len(self.sfd_list)} rows")
        elif file_ext in [".shp", ".geojson", ".gpkg"]:
            # Load spatial file
            self.sfd_list = load_geodataframe(file_path, geometry_col, crs)
            logger.info(f"🗺️ Loaded spatial file with {len(self.sfd_list)} features")
        else:
            raise ValueError(f"❌ Unsupported file format: {file_ext}")

        self.data_source = "file"

        # Handle geometry if present
        if "geometry" in self.sfd_list.columns:
            # Check if it's a GeoDataFrame with actual geometry objects
            first_geom = self.sfd_list["geometry"].iloc[0]
            if first_geom is not None and hasattr(first_geom, "wkt"):
                self.sfd_list["geometry"] = self.sfd_list["geometry"].apply(lambda geom: geom.wkt if geom else None)
            print("🗺️ Geometries detected and converted to WKT")

        # Add generic ID if not present
        if "id" not in self.sfd_list.columns:
            self.sfd_list["id"] = [f"entity_{i}" for i in range(len(self.sfd_list))]
            print("🔢 Added generic IDs")

        # Add generic name if not present
        if "name" not in self.sfd_list.columns:
            self.sfd_list["name"] = [f"Field_{i}" for i in range(len(self.sfd_list))]
            print("🏷️ Added generic names")

        # Handle target_dates column if present
        if "target_dates" in self.sfd_list.columns:
            self._parse_target_dates_from_column()
        else:
            self.target_dates = None

        print(f"✅ Loaded {len(self.sfd_list)} entities from file")
        return self.sfd_list

    def _parse_target_dates_from_column(self):
        """
        Parse target_dates column (dates separated by |) and extract unique dates.
        Sets self.target_dates with the unique sorted list of dates.
        """
        all_dates = set()

        for dates_str in self.sfd_list["target_dates"].dropna():
            # Split by | separator
            dates = str(dates_str).split("|")
            for date_str in dates:
                date_str = date_str.strip()
                try:
                    # Validate date format
                    parsed = datetime.strptime(date_str, "%Y-%m-%d")
                    all_dates.add(parsed.strftime("%Y-%m-%d"))
                except ValueError:
                    logger.warning(f"Skipping invalid date: {date_str}")

        if all_dates:
            self.target_dates = sorted(list(all_dates))
            print(f"📅 Detected {len(self.target_dates)} target dates from file:")
            for d in self.target_dates[:5]:
                print(f"   - {d}")
            if len(self.target_dates) > 5:
                print(f"   ... and {len(self.target_dates) - 5} more")
        else:
            self.target_dates = None
            logger.warning("No valid target dates found in 'target_dates' column")

    def load_seasonfields_batch(self, start_date=None, end_date=None, crop_id=None, ids=None, batch_size=5000):
        """
        Load seasonfields using batch processing for improved performance.
        Delegates to EntityManager.load_seasonfields_batch().
        """
        entity_manager = EntityManager(
            bearer_token=self.bearer_token,
            token_expiration=self.token_expiration,
            config=self.config,
            workflow_ref=self,
        )
        self.sfd_list = entity_manager.load_seasonfields_batch(
            start_date=start_date, end_date=end_date, crop_id=crop_id, ids=ids, batch_size=batch_size
        )
        self.data_source = "api"
        return self.sfd_list

    def sanitize_filename(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        text = re.sub(r'[<>:"/\\|?*\s\-\(\)\[\]\{\},;\'"`~!@#$%^&+=]', "_", text)
        text = re.sub(r"_+", "_", text)
        text = text.strip("_")
        return text[:50] if len(text) > 50 else text
