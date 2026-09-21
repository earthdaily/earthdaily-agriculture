import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import pandas as pd
from loguru import logger

from earthdaily.agriculture.core.api_utils import export_results
from earthdaily.agriculture.core.logging_setup import setup_logging

# Default geohash precision for spatial grouping of point-based extractors
# (Weather / GDD / GDD-offset). 5 ≈ 4.9 km cells — matched to the coarse weather
# grid for strong dedup. Refine to 6 (~1.2 km) if Phase 3 validation shows the
# per-KPI delta exceeds tolerance.
DEFAULT_SPATIAL_PRECISION = 5

# Cap on how far a spatial group's date window may be widened by taking the union
# of its members' windows. Without a cap a single long-history field would drag
# every field sharing its cell into a decade-long pull. Groups whose union exceeds
# this are split into sub-groups (see ``BaseExtractor._resolve_group_windows``).
DEFAULT_MAX_WINDOW_DAYS = 400


def cache_single_entity(params_attr):
    """
    Decorator for process_single_entity_* methods.
    Checks cache before calling the API; stores results on success.

    Args:
        params_attr (str): Name of the instance attribute holding extraction params
            (e.g., 'coverage_params', 'disease_params'). Used as cache file key.

    The decorated method must:
        - Accept (self, row, params=None) signature
        - Return {"data": DataFrame or None, "error": ...}
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, row, params=None, **kwargs):
            # Resolve params for cache path
            cache_params = params or getattr(self, params_attr, None)

            # Only attempt cache if enabled
            if self.use_cache and self.cache_key_columns is not None:
                # Normalize row to dict for entity_id lookup
                if isinstance(row, pd.Series):
                    row_dict = row.to_dict()
                elif isinstance(row, dict):
                    row_dict = row
                else:
                    row_dict = row

                entity_id = self.get_entity_value(row_dict, "id", "unknown")

                # Cache lookup
                cached = self._lookup_entity_cache(entity_id, cache_params)
                if not cached.empty:
                    self.logger.debug(f"[Cache] Hit for entity {entity_id} in {func.__name__}")
                    return {"data": cached, "error": None}

            # Cache miss or disabled — call actual method
            result = func(self, row, params=params, **kwargs)

            # Store successful results in cache
            if (
                self.use_cache
                and self.cache_key_columns is not None
                and result.get("data") is not None
                and not result["data"].empty
            ):
                self._store_entity_cache(result["data"], cache_params)

            return result

        return wrapper

    return decorator


def requires_token(func):
    """Decorator to ensure a valid token before executing API methods.

    Calls ``self.ensure_token_valid()`` (defined on :class:`BaseExtractor`)
    before the wrapped method runs. Canonical home for the decorator that was
    previously copy-pasted into every extractor/service module.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.ensure_token_valid()  # check/refresh token before running
        return func(self, *args, **kwargs)

    return wrapper


class BaseExtractor:
    """
    Base class for all analytic extractors (Coverage, InSeason, Emergence, etc.)
    Handles token management, environment setup, shared config, and logging.

    Results export format is controlled by ``self.export_format`` — ``"csv"``
    (default) or ``"parquet"``. Set it via ``config["export_format"]`` (forwarded
    by WorkflowManager from workflow.yml ``settings.export_format``), the
    ``EDAGRO_EXPORT_FORMAT`` env var, or by assigning ``extractor.export_format``
    directly. Parquet requires ``pyarrow`` (already a dependency); error files are
    always written as CSV.
    """

    # Default column mapping for entity data fields.
    # Keys are the internal/canonical names used by extractors.
    # Values are the actual column names in the user's DataFrame.
    DEFAULT_COLUMN_MAPPING = {
        "id": "id",
        "geometry": "geometry",
        "crop": "crop",
        "start_date": "start_date",
        "end_date": "end_date",
        "sowing_date": "sowing_date",
        "emergence_date": "emergence_date",
        "batch_id": "batch_id",
        "historical_seasons": "historical_seasons",
        "historical_years": "historical_years",
        "years": "years",
        "amu_id": "amu_id",
    }

    # Fields that contain dates and should be normalized to YYYY-MM-DD
    DATE_FIELDS = {"start_date", "end_date", "sowing_date", "emergence_date"}

    #: Which identity provider this extractor's token comes from.
    #:
    #: ``"geosys"`` (the default, and every extractor but LRTS) means the token
    #: is the OAuth bearer from ``EDAuthenticator``, shared across the whole
    #: workflow. ``"eds"`` means it came from
    #: :class:`~earthdaily.agriculture.core.identity_eds.EDSAuthenticator`.
    #:
    #: This exists because :meth:`ensure_token_valid` syncs the token from
    #: ``workflow_ref`` whenever there is one — which is the normal path. Without
    #: a marker, an extractor on a second provider would have its access token
    #: silently overwritten with the Geosys one before every call, and fail with
    #: a 401 that points nowhere near the cause. Extractors that authenticate
    #: elsewhere override this and refresh through their own
    #: :meth:`get_new_token`.
    AUTH_PROVIDER = "geosys"

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        self.bearer_token = bearer_token
        self.token_expiration = token_expiration
        self.config = config
        self.workflow_ref = workflow_ref

        # 🌍 Environment
        self.env = config.get("env", "production")

        # 📂 Folder paths
        self.partial_path = config.get("partial_result_dir")
        self.output_path = config.get("output_result_dir")

        # 🔐 Credentials from environment variables (kept for backwards compatibility)
        self.client_id = os.getenv("API_CLIENT_ID")
        self.client_secret = os.getenv("API_CLIENT_SECRET")
        self.api_username = os.getenv("API_USERNAME")
        self.api_password = os.getenv("API_PASSWORD")

        # 📂 Result processing
        self.merge_existing = config.get("merge_existing", "auto")  # Default: 'auto'

        # 💾 Results export format — "csv" (default) or "parquet".
        # Precedence: config["export_format"] > EDAGRO_EXPORT_FORMAT env var > "csv".
        # WorkflowManager forwards workflow.yml `settings.export_format` here per step;
        # notebooks/scripts can also set `extractor.export_format = "parquet"` ad hoc.
        # Resolved/validated lazily at export time by api_utils.resolve_export_format.
        self.export_format = config.get("export_format") or os.getenv("EDAGRO_EXPORT_FORMAT")

        # 📄 Manifest sidecar on export (parity with export_format). When True, a
        # ``<prefix>_manifest_<ts>.json`` describing the exported dataset is written
        # next to a successful results export via ``export.generate_manifest``.
        # Resolution precedence: ``config["export_manifest"]`` (WorkflowManager
        # forwards ``settings.export_manifest``) > ``EDAGRO_EXPORT_MANIFEST`` env >
        # default ``False``. ``manifest_metadata`` carries per-run extras (e.g. a
        # region-dimension-file pointer / join key) into the manifest's metadata.
        _manifest = config.get("export_manifest")
        if _manifest is None:
            _env = os.getenv("EDAGRO_EXPORT_MANIFEST")
            _manifest = str(_env).strip().lower() in {"1", "true", "yes", "on"} if _env is not None else False
        self.export_manifest = bool(_manifest)
        self.manifest_metadata = config.get("manifest_metadata") or {}

        # 🗺️ Durable destination for `postprocess="file"` rasters (parity with
        # export_format). When set, every file the FLM / Difference / Zoning
        # writers save locally under `output_path` is ALSO written to
        # `<output_uri>/<filename>` through `_fs`, so the rasters outlive the
        # runner that produced them.
        #
        # Deliberately separate from `output_path` rather than overloading it:
        # the analysis half reads every raster back off local disk, so a
        # remote-only write would turn a seconds-long pass into thousands of
        # network reads. Local stays the working copy; this is the durable one.
        #
        # Precedence: config["output_uri"] (WorkflowManager forwards
        # workflow.yml `settings.output_uri`) > EDAGRO_OUTPUT_URI env > None
        # (local-only, today's behaviour).
        self.output_uri = config.get("output_uri") or os.getenv("EDAGRO_OUTPUT_URI") or None
        # ♻️ Resume mode — process ONLY the entities recorded in a previous run's
        # ``failed_ids_<prefix>_<ts>.csv``. Explicit and opt-in: ``True`` picks the
        # newest such file for the prefix, a string names one file exactly.
        #
        # This used to be an undeclared second meaning of ``fail_safe``: a run with
        # fail_safe=True silently narrowed its entity list to whatever a leftover
        # failed_ids file on disk contained. A 900-entity run that failed 10 left a
        # file behind, and the NEXT full run quietly processed only those 10 while
        # logging success — and since the cleanup pass only matches
        # ``<prefix>_*_partial.*``, the file was never removed, so every later run
        # stayed capped. ``fail_safe`` now means error tolerance and nothing else.
        #
        # Precedence mirrors export_format/export_manifest: config["retry_failed_only"]
        # (WorkflowManager forwards step-level then settings) > default False.
        self.retry_failed_only = config.get("retry_failed_only") or False

        # 🗂️ Column mapping (user can override via config or set_column_mapping)
        self.column_mapping = dict(self.DEFAULT_COLUMN_MAPPING)
        if "column_mapping" in config:
            self.column_mapping.update(config["column_mapping"])

        # 🗂️ Output formatting
        self.output_mapping = None  # dict: rename columns {old_name: new_name}
        # Drop geometry by default; callers overriding via configure_output must
        # include "geometry" in their list if they still want it excluded.
        self.exclude_columns = ["geometry"]
        self.output_columns = None  # list: columns to keep (whitelist, applied last)

        # 🗃️ Cache configuration (opt-in per extractor via setup methods)
        self.use_cache = config.get("use_cache", False)
        default_cache = os.path.join(
            config.get("project_root", str(Path(__file__).resolve().parent.parent.parent.parent)), "cache"
        )
        configured_cache_dir = config.get("cache_dir", default_cache)

        # Detect remote cache_dir BEFORE wrapping in Path (Path("s3://...")
        # mangles the URI on Windows). When the cache lives on an object store
        # we force-disable the cache: the local-only "atomic rename" used by
        # _update_cache has no equivalent on S3, and a per-pod cache never
        # survives a container restart anyway. See Doc 12 §4.
        from earthdaily.agriculture.core._fs import is_remote_path

        self._cache_dir_is_remote = is_remote_path(configured_cache_dir)
        if self._cache_dir_is_remote:
            self.cache_dir = configured_cache_dir  # keep as str — Path can't represent URIs cleanly
            if self.use_cache:
                # Visible warning, fired once at construction time.
                logger.warning(
                    f"⚠️  Cache disabled: cache_dir={configured_cache_dir!r} is a remote URI. "
                    "The local-only atomic-rename strategy used by _update_cache cannot run "
                    "against object stores. Set cache_dir to a local path to re-enable caching, "
                    "or accept that this run will hit the API for every entity."
                )
            self.use_cache = False
        else:
            self.cache_dir = Path(configured_cache_dir)

        self.cache_ttl_days = config.get("cache_ttl_days", 7)
        self.cache_key_columns = None  # Set by each extractor's setup method

        # 📝 Logging setup
        self._setup_logger()

    # -------------------------------
    # 📝 LOGGING SETUP
    # -------------------------------

    def _setup_logger(self):
        """
        Initialize or inherit logger from workflow.
        If standalone, sets up a new logger. If part of workflow, uses shared logger.
        """
        # Get extractor name for context (e.g., 'HarvestExtractor', 'EmergenceExtractor')
        self.extractor_name = self.__class__.__name__

        if self.workflow_ref:
            # ✅ Use shared logger from workflow
            self.logger = self.workflow_ref.logger if hasattr(self.workflow_ref, "logger") else logger
            self.logger.info(f"🔗 {self.extractor_name} initialized within workflow")
        else:
            # ✅ Standalone extractor - initialize logger if not already setup
            if not logger._core.handlers:
                # Logger hasn't been initialized yet — resolve log dir at project root
                log_config = self.config.get("logging", {})
                default_log_dir = os.path.join(
                    self.config.get("project_root", str(Path(__file__).resolve().parent.parent.parent)), "logs"
                )
                setup_logging(
                    log_dir=log_config.get("log_dir", default_log_dir),
                    log_level=log_config.get("log_level", "INFO"),
                    log_to_console=log_config.get("log_to_console", True),
                    rotation=log_config.get("rotation", "1 day"),
                    retention=log_config.get("retention", "30 days"),
                    compression=log_config.get("compression", "zip"),
                    # Container-friendly: skip the file sink. The env var
                    # EDAGRO_LOG_CONSOLE_ONLY overrides this inside setup_logging.
                    log_to_console_only=log_config.get("log_to_console_only", False),
                )

            self.logger = logger
            self.logger.info(f"🚀 {self.extractor_name} initialized (standalone mode)")

        # Log configuration details
        self.logger.debug(f"Environment: {self.env}")
        if self.output_path:
            self.logger.debug(f"Output path: {self.output_path}")
        if self.partial_path:
            self.logger.debug(f"Partial path: {self.partial_path}")
        self.logger.debug(f"Merge strategy: {self.merge_existing}")

    def get_contextualized_logger(self, context: str):
        """
        Returns a logger with additional context for specific operations.

        Args:
            context (str): Context identifier (e.g., 'API_CALL', 'BULK_PROCESSING')

        Returns:
            logger: Contextualized loguru logger
        """
        return self.logger.bind(extractor=self.extractor_name, context=context)

    # -------------------------------
    # 🗂️ COLUMN MAPPING
    # -------------------------------

    def set_column_mapping(self, mapping: dict):
        """
        Update column mapping to match user's DataFrame column names.

        Args:
            mapping (dict): Maps internal field names to user column names.
                Example: {"id": "entity_id", "geometry": "wkt", "crop": "crop_type"}
        """
        # Warn about unknown internal keys
        unknown_keys = set(mapping.keys()) - set(self.DEFAULT_COLUMN_MAPPING.keys())
        if unknown_keys:
            self.logger.warning(
                f"Unknown column mapping keys: {unknown_keys}. Known keys: {list(self.DEFAULT_COLUMN_MAPPING.keys())}"
            )
        self.column_mapping.update(mapping)
        self.logger.info(f"Column mapping updated: {mapping}")

    def validate_column_mapping(self, entity_list):
        """
        Validate that mapped columns actually exist in the entity DataFrame.
        Call this after set_column_mapping to catch mismatches early.

        Args:
            entity_list (pd.DataFrame): The DataFrame that will be used for extraction.

        Returns:
            bool: True if all mapped columns are found (or have canonical fallbacks).
        """
        if not isinstance(entity_list, pd.DataFrame) or entity_list.empty:
            self.logger.debug("Cannot validate column mapping: entity_list is not a non-empty DataFrame")
            return True

        columns = set(entity_list.columns)
        issues = []
        for internal_key, mapped_col in self.column_mapping.items():
            if mapped_col not in columns and internal_key not in columns:
                issues.append(f"  '{internal_key}' -> '{mapped_col}' (neither found in DataFrame)")
            elif mapped_col not in columns and internal_key in columns:
                self.logger.debug(
                    f"Column mapping '{internal_key}' -> '{mapped_col}' not found, "
                    f"but canonical key '{internal_key}' exists (fallback will be used)"
                )

        if issues:
            self.logger.warning(
                "Column mapping issues detected:\n" + "\n".join(issues) + f"\n  Available columns: {sorted(columns)}"
            )
            return False

        self.logger.debug("Column mapping validation passed")
        return True

    @staticmethod
    def _is_entity_provided(entity_data) -> bool:
        """
        Safely check if entity_data is provided (not None).
        Works with dict, pd.Series, pd.DataFrame without triggering
        'truth value of a Series is ambiguous' errors.

        Args:
            entity_data: Entity data to check.

        Returns:
            bool: True if entity_data is not None.
        """
        return entity_data is not None

    def validate_entity(self, entity_data, required_fields, context=""):
        """
        Validate that an entity has all required fields before making an API call.
        Uses column mapping to resolve field names, with clear error messages.

        Args:
            entity_data (dict | pd.Series): Entity data to validate.
            required_fields (list[str]): List of internal field names required
                (e.g., ['geometry', 'crop']).
            context (str, optional): Context string for error messages
                (e.g., 'LAI extraction').

        Returns:
            dict: {'valid': bool, 'missing': list[str], 'details': str}
        """
        if not self._is_entity_provided(entity_data):
            return {
                "valid": False,
                "missing": required_fields,
                "details": f"entity_data is None{f' ({context})' if context else ''}",
            }

        missing = []
        for field in required_fields:
            if not self.has_entity_field(entity_data, field):
                mapped_col = self.column_mapping.get(field, field)
                missing.append(
                    f"'{field}' (looked for column '{mapped_col}'"
                    + (f" and fallback '{field}'" if mapped_col != field else "")
                    + ")"
                )
            else:
                # Field exists but check if value is empty/None
                value = self.get_entity_value(entity_data, field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    mapped_col = self.column_mapping.get(field, field)
                    missing.append(f"'{field}' (column '{mapped_col}' exists but value is empty)")

        if missing:
            entity_id = self.get_entity_value(entity_data, "id", "unknown")
            details = (
                f"Entity {entity_id}: missing required fields"
                f"{f' for {context}' if context else ''}: " + ", ".join(missing)
            )
            self.logger.warning(details)
            return {"valid": False, "missing": missing, "details": details}

        return {"valid": True, "missing": [], "details": ""}

    @staticmethod
    def normalize_date(value):
        """
        Normalize a date value to YYYY-MM-DD string format.
        Handles ISO timestamps (e.g. '2025-04-01T00:00:00'), pd.Timestamp, and datetime objects.

        Args:
            value: Date value (str, pd.Timestamp, datetime, or None).

        Returns:
            str or None: Date in YYYY-MM-DD format, or the original value if not a recognized date.
        """
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (ValueError, TypeError):
            pass  # pd.isna() can raise on some types (e.g. lists)
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str) and "T" in value:
            return value.split("T")[0]
        return value

    def get_entity_value(self, row, key, default=None):
        """
        Resolve an entity field value using the column mapping.
        Automatically normalizes date fields to YYYY-MM-DD format.

        Args:
            row (dict | pd.Series): Entity data row.
            key (str): Internal/canonical field name (e.g., 'id', 'geometry', 'crop').
            default: Default value if the mapped column is not found.

        Returns:
            The value from the row at the mapped column name.
        """
        mapped_col = self.column_mapping.get(key, key)
        if hasattr(row, "get"):
            value = row.get(mapped_col, default)
            # Fallback to canonical key if mapped column not found
            if value is default and mapped_col != key:
                value = row.get(key, default)
        else:
            # Support pd.Series attribute access
            value = getattr(row, mapped_col, default)
            if value is default and mapped_col != key:
                value = getattr(row, key, default)

        # Normalize date fields to YYYY-MM-DD
        if key in self.DATE_FIELDS and value is not None and value != default:
            value = self.normalize_date(value)

        return value

    def has_entity_field(self, row, key) -> bool:
        """
        Check if an entity field exists in the row using the column mapping.

        Args:
            row (dict | pd.Series): Entity data row.
            key (str): Internal/canonical field name (e.g., 'id', 'geometry').

        Returns:
            bool: True if the mapped column exists in the row.
        """
        mapped_col = self.column_mapping.get(key, key)
        if isinstance(row, dict):
            # Check mapped column first, fallback to canonical key
            return mapped_col in row or (mapped_col != key and key in row)
        # pd.Series
        found = mapped_col in row.index if hasattr(row, "index") else hasattr(row, mapped_col)
        if not found and mapped_col != key:
            found = key in row.index if hasattr(row, "index") else hasattr(row, key)
        return found

    def get_mapped_column(self, key) -> str:
        """
        Get the user-facing column name for an internal field.

        Args:
            key (str): Internal field name (e.g., 'id', 'geometry').

        Returns:
            str: The mapped column name.
        """
        return self.column_mapping.get(key, key)

    def normalize_entity_row(self, row) -> dict:
        """
        Convert a row with user column names to a dict with canonical/internal names.
        Useful for passing normalized data to API call wrappers.

        Args:
            row (dict | pd.Series): Entity data row with user column names.

        Returns:
            dict: Row data with internal/canonical keys.
        """
        normalized = {}
        for internal_key, user_col in self.column_mapping.items():
            if hasattr(row, "get"):
                val = row.get(user_col)
            else:
                val = getattr(row, user_col, None)
            if val is not None:
                normalized[internal_key] = val
        return normalized

    # -------------------------------
    # 🔍 API RESPONSE VALIDATION
    # -------------------------------

    def validate_api_response(self, response, entity_id, data_type="data"):
        """
        Validate an API response in format_*_json methods.
        Returns an empty DataFrame if the response is empty/None, or None if valid.

        Args:
            response: Raw API response (list, dict, or None)
            entity_id: Entity identifier for logging
            data_type (str): Label for the data type (e.g., 'coverage', 'weather')

        Returns:
            pd.DataFrame: Empty DataFrame if response is empty/None
            None: If response is valid (caller should continue processing)
        """
        if not response:
            self.logger.warning(f"Entity {entity_id}: Empty {data_type} response")
            return pd.DataFrame()
        return None

    # -------------------------------
    # 🗂️ OUTPUT FORMATTING
    # -------------------------------

    def configure_output(self, output_mapping=None, exclude_columns=None, output_columns=None):
        """
        Configure output DataFrame formatting.
        All parameters are optional — by default everything is kept as-is (backward compatible).

        Args:
            output_mapping (dict, optional): Rename columns in output. {current_name: desired_name}
                Example: {"crop.code": "crop", "sowingDate": "sowing_date"}
            exclude_columns (list, optional): Columns to drop from output.
                Replaces the default (["geometry"]) — include "geometry" in your
                list if you want to keep the default geometry exclusion.
                Example: ["field.farm.grower.firstname", "geometry"]
            output_columns (list, optional): Whitelist of columns to keep (applied after rename).
                If set, only these columns appear in the output.
                Example: ["id", "name", "emergence_date", "confirmation_status"]
        """
        if output_mapping is not None:
            self.output_mapping = output_mapping
            self.logger.info(f"Output mapping configured: {output_mapping}")
        if exclude_columns is not None:
            self.exclude_columns = exclude_columns
            self.logger.info(f"Exclude columns configured: {exclude_columns}")
        if output_columns is not None:
            self.output_columns = output_columns
            self.logger.info(f"Output columns configured: {output_columns}")

    def apply_output_format(self, df):
        """
        Apply output formatting to a results DataFrame.
        Operations are applied in order: rename → exclude → select.
        Returns the original DataFrame unchanged if no output config is set.

        Args:
            df (pd.DataFrame): Results DataFrame to format.

        Returns:
            pd.DataFrame: Formatted DataFrame.
        """
        if df is None or df.empty:
            return df

        # Step 1: Rename columns
        if self.output_mapping:
            # Only rename columns that actually exist in the DataFrame
            valid_renames = {k: v for k, v in self.output_mapping.items() if k in df.columns}
            if valid_renames:
                df = df.rename(columns=valid_renames)
                self.logger.debug(f"Renamed {len(valid_renames)} columns: {valid_renames}")

        # Step 2: Exclude columns
        if self.exclude_columns:
            cols_to_drop = [c for c in self.exclude_columns if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.logger.debug(f"Excluded {len(cols_to_drop)} columns: {cols_to_drop}")

        # Step 3: Select columns (whitelist)
        if self.output_columns:
            cols_to_keep = [c for c in self.output_columns if c in df.columns]
            missing = set(self.output_columns) - set(df.columns)
            if missing:
                self.logger.warning(f"Output columns not found in results: {missing}")
            if cols_to_keep:
                df = df[cols_to_keep]
                self.logger.debug(f"Selected {len(cols_to_keep)} output columns")

        return df

    # -------------------------------
    # 🗃️ CACHE MANAGEMENT
    # -------------------------------

    def apply_cache_setting(self, use_cache=None):
        """
        Apply cache enable/disable override from setup methods.
        Call this from any extractor's setup_*_parameters() method.

        Args:
            use_cache (bool | None): True to enable, False to disable,
                None to keep the current setting (from config).

        When ``cache_dir`` is a remote URI (s3://, gs://, ...), this method
        silently keeps the cache disabled regardless of ``use_cache=True``.
        The construction-time warning already explained why; per-call setup
        methods don't repeat it. Disabling (``use_cache=False``) always works.
        """
        if use_cache is None:
            return
        if use_cache and getattr(self, "_cache_dir_is_remote", False):
            # Remote cache_dir → cache stays off. The __init__ warning already
            # surfaced the reason; don't spam it on every setup_*_parameters().
            self.logger.debug(
                "apply_cache_setting(use_cache=True) ignored — cache_dir is remote, cache stays disabled."
            )
            return
        self.use_cache = use_cache

    def _cache_params_hash(self, params: dict) -> str:
        """
        Generate a short hash from extraction parameters.
        Used in the cache filename so different parameter sets get separate caches.
        """
        serializable = {k: str(v) for k, v in sorted(params.items())} if params else {}
        raw = json.dumps(serializable, sort_keys=True)
        # Cache-filename discriminator only — not a security/crypto use.
        # usedforsecurity=False keeps the same digest (cache-compatible) while
        # silencing the false-positive Bandit B324 / FIPS warnings.
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:8]

    def _cache_path(self, params: dict = None) -> Path:
        """
        One parquet file per extractor class + parameter combination.

        Args:
            params (dict, optional): Extraction parameters to include in cache key.

        Returns:
            Path: Path to the cache parquet file.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        name = self.__class__.__name__.lower()
        if params:
            phash = self._cache_params_hash(params)
            return self.cache_dir / f"{name}_{phash}_cache.parquet"
        return self.cache_dir / f"{name}_cache.parquet"

    def _load_cache(self, params: dict = None) -> pd.DataFrame:
        """
        Load cached results from parquet. Returns empty DataFrame if no cache exists.

        Args:
            params (dict, optional): Extraction parameters (for cache filename).

        Returns:
            pd.DataFrame: Cached results, or empty DataFrame.
        """
        path = self._cache_path(params)
        if path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                self.logger.warning(f"[Cache] Corrupted cache file {path.name}, removing it: {e}")
                path.unlink(missing_ok=True)
                return pd.DataFrame()
            # Evict stale rows
            df = self._evict_stale(df)
            self.logger.info(f"[Cache] Loaded {len(df)} valid records from {path.name}")
            return df
        return pd.DataFrame()

    def _evict_stale(self, cached: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows older than TTL from cached DataFrame.

        Args:
            cached (pd.DataFrame): Cached results with _cached_at column.

        Returns:
            pd.DataFrame: Cached results with stale rows removed.
        """
        if cached.empty or "_cached_at" not in cached.columns:
            return cached
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.cache_ttl_days)
        fresh = cached[cached["_cached_at"] >= cutoff]
        evicted = len(cached) - len(fresh)
        if evicted > 0:
            self.logger.info(f"[Cache] Evicted {evicted} stale records (TTL={self.cache_ttl_days}d)")
        return fresh

    def _lookup_entity_cache(self, entity_id, params: dict = None) -> pd.DataFrame:
        """
        Look up cached results for a single entity.
        Returns matching rows (without _cached_at) or empty DataFrame if not found.

        Args:
            entity_id: Entity ID to look up.
            params (dict, optional): Extraction parameters (for cache filename).

        Returns:
            pd.DataFrame: Cached results for this entity, or empty DataFrame.
        """
        if not self.use_cache or self.cache_key_columns is None:
            return pd.DataFrame()

        cached = self._load_cache(params)
        if cached.empty:
            return pd.DataFrame()

        id_col = self.get_mapped_column("id")
        if id_col not in cached.columns:
            return pd.DataFrame()

        match = cached[cached[id_col] == entity_id]
        if match.empty:
            return pd.DataFrame()

        self.logger.debug(f"[Cache] Hit for entity {entity_id}: {len(match)} rows")
        if "_cached_at" in match.columns:
            match = match.drop(columns=["_cached_at"])
        return match

    def _store_entity_cache(self, result_df: pd.DataFrame, params: dict = None) -> None:
        """
        Store single-entity results in cache (if caching is enabled).

        Args:
            result_df (pd.DataFrame): Results to cache.
            params (dict, optional): Extraction parameters (for cache filename).
        """
        if not self.use_cache or self.cache_key_columns is None:
            return
        self._update_cache(result_df, params)

    def _identify_missing_entities(self, entity_list: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
        """
        Return entities from entity_list whose results are not already in the cache.
        Matching is done on the entity ID column.

        Args:
            entity_list (pd.DataFrame): Input entities to process.
            cached (pd.DataFrame): Cached results DataFrame.

        Returns:
            pd.DataFrame: Subset of entity_list that needs processing.
        """
        if cached.empty:
            return entity_list

        id_col = self.get_mapped_column("id")
        cached_ids = set(cached[id_col].unique()) if id_col in cached.columns else set()

        if not cached_ids:
            return entity_list

        missing = entity_list[~entity_list[id_col].isin(cached_ids)]
        total = len(entity_list)
        hit = total - len(missing)
        self.logger.info(f"[Cache] {hit}/{total} entities found in cache, {len(missing)} to fetch")
        return missing

    def _update_cache(self, new_data: pd.DataFrame, params: dict = None) -> None:
        """
        Append new results to cache, deduplicating on cache_key_columns.

        Args:
            new_data (pd.DataFrame): Freshly extracted results.
            params (dict, optional): Extraction parameters (for cache filename).
        """
        if new_data is None or new_data.empty:
            return

        path = self._cache_path(params)
        existing = pd.DataFrame()
        if path.exists():
            try:
                existing = pd.read_parquet(path)
            except Exception as e:
                self.logger.warning(f"[Cache] Corrupted cache file {path.name}, rebuilding: {e}")
                path.unlink(missing_ok=True)

        new_data = new_data.copy()
        new_data["_cached_at"] = pd.Timestamp.now()

        combined = pd.concat([existing, new_data], ignore_index=True)

        # Deduplicate on cache_key_columns if defined, otherwise on all columns except _cached_at
        dedup_cols = self.cache_key_columns
        if dedup_cols:
            valid_dedup = [c for c in dedup_cols if c in combined.columns]
            if valid_dedup:
                combined = combined.drop_duplicates(subset=valid_dedup, keep="last")
        else:
            data_cols = [c for c in combined.columns if c != "_cached_at"]
            combined = combined.drop_duplicates(subset=data_cols, keep="last")

        # Atomic write: write to temp file then rename to prevent corruption on interruption
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
            os.close(fd)
            combined.to_parquet(tmp_path, index=False)
            Path(tmp_path).replace(path)
        except Exception as e:
            self.logger.error(f"[Cache] Failed to write cache {path.name}: {e}")
            # Clean up temp file if it exists
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
            return
        self.logger.success(f"[Cache] Updated {path.name} -> {len(combined)} total records")

    def _filter_cached_for_request(self, cached: pd.DataFrame, entity_list: pd.DataFrame) -> pd.DataFrame:
        """
        Filter cached results to only include rows matching the requested entities.

        Args:
            cached (pd.DataFrame): Full cached results.
            entity_list (pd.DataFrame): Requested input entities.

        Returns:
            pd.DataFrame: Cached results for requested entities only.
        """
        if cached.empty:
            return cached
        id_col = self.get_mapped_column("id")
        if id_col not in cached.columns:
            return pd.DataFrame()
        requested_ids = set(entity_list[id_col].unique())
        return cached[cached[id_col].isin(requested_ids)]

    def clear_cache(self, params: dict = None):
        """
        Delete the cache file for this extractor (and optional parameter set).

        Args:
            params (dict, optional): If provided, clears only the cache for these parameters.
                If None, clears the default (un-parameterized) cache file.
        """
        path = self._cache_path(params)
        if path.exists():
            path.unlink()
            self.logger.info(f"[Cache] Cleared {path.name}")
        else:
            self.logger.info(f"[Cache] No cache file to clear at {path.name}")

    def cache_info(self, params: dict = None) -> dict:
        """
        Return information about the current cache state.

        Args:
            params (dict, optional): Extraction parameters (for cache filename).

        Returns:
            dict: Cache statistics (path, exists, records, oldest, newest).
        """
        path = self._cache_path(params)
        info = {"path": str(path), "exists": path.exists(), "records": 0}
        if path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                self.logger.warning(f"[Cache] Corrupted cache file {path.name}, removing it: {e}")
                path.unlink(missing_ok=True)
                info["exists"] = False
                info["error"] = str(e)
                return info
            info["records"] = len(df)
            if "_cached_at" in df.columns and not df.empty:
                info["oldest"] = str(df["_cached_at"].min())
                info["newest"] = str(df["_cached_at"].max())
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.cache_ttl_days)
                info["stale_records"] = int((df["_cached_at"] < cutoff).sum())
        return info

    def _bulk_with_cache(self, entity_list, bulk_method, params=None, use_cache=None, **bulk_kwargs):
        """
        Cache-aware wrapper for bulk extraction methods.
        If caching is enabled, loads cached results, identifies missing entities,
        delegates only the missing ones to the actual bulk method, then updates the cache.

        Args:
            entity_list (pd.DataFrame): Full input entity list.
            bulk_method (callable): The actual bulk extraction method to call
                (e.g., self.process_entity_coverage_bulk_parallel).
            params (dict, optional): Extraction parameters (used for cache filename keying).
            use_cache (bool, optional): Override instance-level use_cache setting.
            **bulk_kwargs: All other keyword arguments passed through to bulk_method.

        Returns:
            dict: Same return format as bulk_method, with results from cache + fresh extraction.
        """
        cache_enabled = use_cache if use_cache is not None else self.use_cache
        log = self.get_contextualized_logger("CACHE")

        if not cache_enabled or self.cache_key_columns is None:
            log.debug("Cache disabled or no cache_key_columns defined — running without cache")
            return bulk_method(entity_list=entity_list, **bulk_kwargs)

        # Step 1: Load cache and evict stale entries
        cached = self._load_cache(params)

        # Step 2: Identify entities not in cache
        missing_entities = self._identify_missing_entities(entity_list, cached)

        # Step 3: Get cached results for the requested entities
        cached_results = self._filter_cached_for_request(cached, entity_list)
        # Remove internal _cached_at column from returned results
        if "_cached_at" in cached_results.columns:
            cached_results = cached_results.drop(columns=["_cached_at"])

        # Step 4: If everything is cached, return early
        if missing_entities.empty:
            log.success(f"[Cache] Full cache hit — {len(entity_list)} entities, skipping API calls")
            print(f"🗃️ Full cache hit — {len(entity_list)} entities served from cache")
            return {
                "results_df": cached_results,
                "global_errors": [],
                "total_entities": len(entity_list),
                "total_calculations": 0,
                "successful_calculations": 0,
                "failed_calculations": 0,
                "failed_ids": [],
                "cache_hit": len(entity_list),
                "cache_miss": 0,
            }

        # Step 5: Process only missing entities
        cache_hit_count = len(entity_list) - len(missing_entities)
        print(f"🗃️ Cache: {cache_hit_count} entities from cache, {len(missing_entities)} to fetch")
        result = bulk_method(entity_list=missing_entities, **bulk_kwargs)

        # Step 6: Update cache with new results
        new_results = result.get("results_df", pd.DataFrame())
        if not new_results.empty:
            self._update_cache(new_results, params)

        # Step 7: Combine cached + new results
        if not cached_results.empty and not new_results.empty:
            combined = pd.concat([cached_results, new_results], ignore_index=True)
        elif not cached_results.empty:
            combined = cached_results
        else:
            combined = new_results

        result["results_df"] = combined
        result["cache_hit"] = cache_hit_count
        result["cache_miss"] = len(missing_entities)
        result["total_entities"] = len(entity_list)

        log.success(f"[Cache] Done — {cache_hit_count} cached + {result.get('successful_calculations', 0)} fetched")
        return result

    # ------------------------------------------------------------------
    # 🗄️ SPATIAL CACHE (geohash × date, for grouped point-based extractors)
    # ------------------------------------------------------------------
    # The entity cache above answers "have I already fetched *this field*". For
    # centroid-queried analytics that is needlessly strict: any field in the same
    # geohash cell gets the same answer. The spatial cache is therefore keyed on
    # the cell (geohash + non-date signature) × date, so one wide pull serves every
    # later run that touches that cell — whatever field set or sub-window it asks
    # for. It lives in its own parquet file and never mixes with the entity cache.

    def _spatial_cache_path(self, params: dict = None) -> Path:
        """Path of the geohash-keyed cache file for this extractor + parameter set."""
        base = self._cache_path(params)
        return base.with_name(base.name.replace("_cache.parquet", "_spatial_cache.parquet"))

    def _load_spatial_cache(self, params: dict = None) -> pd.DataFrame:
        """Load the geohash-keyed cache (TTL-evicted), or an empty frame."""
        path = self._spatial_cache_path(params)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            self.logger.warning(f"[SpatialCache] Corrupted cache file {path.name}, removing it: {e}")
            path.unlink(missing_ok=True)
            return pd.DataFrame()
        df = self._evict_stale(df)
        self.logger.info(f"[SpatialCache] Loaded {len(df)} valid records from {path.name}")
        return df

    def _serve_from_spatial_cache(self, cached, representatives, date_column="date"):
        """
        Return ``(covered_keys, rows)`` for the groups the cache can fully answer.

        A cell is a hit only when its cached dates *span* the group's whole union
        window — partial coverage is treated as a miss and refetched in full, which
        keeps the coverage test cheap and the stored series contiguous. Served rows
        are trimmed to the window and tagged with the group key so they broadcast
        exactly like freshly fetched ones.
        """
        if cached is None or cached.empty or "_cell_key" not in cached.columns:
            return set(), pd.DataFrame()
        if date_column not in cached.columns:
            return set(), pd.DataFrame()

        cached = cached.copy()
        cached["_d"] = self._naive_datetimes(cached[date_column])
        span = cached.groupby("_cell_key")["_d"].agg(["min", "max"])

        covered_keys, frames = set(), []
        for _, rep in representatives.iterrows():
            cell, gkey = rep.get("_cell_key"), rep.get("_spatial_group_key")
            win_start, win_end = rep.get("_win_start"), rep.get("_win_end")
            if cell not in span.index or pd.isna(win_start) or pd.isna(win_end):
                continue
            lo, hi = span.loc[cell, "min"], span.loc[cell, "max"]
            if pd.isna(lo) or lo > win_start or pd.isna(hi) or hi < win_end:
                continue
            in_cell = cached[cached["_cell_key"] == cell]
            rows = in_cell[(in_cell["_d"] >= win_start) & (in_cell["_d"] <= win_end)]
            if rows.empty:
                continue
            # Cached rows come from a pull that may have started *earlier* than this
            # window, so their cumulative columns are already part-way up. Carry the
            # single row before the window so the rebase has its baseline; clipping
            # drops it again once every member has been re-zeroed.
            prior = in_cell[in_cell["_d"] < win_start]
            if not prior.empty:
                rows = pd.concat([prior.nlargest(1, "_d"), rows], ignore_index=True)
            rows = rows.drop(columns=[c for c in ("_d", "_cached_at", "_cell_key") if c in rows.columns]).copy()
            rows["_spatial_group_key"] = gkey
            covered_keys.add(gkey)
            frames.append(rows)

        served = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if covered_keys:
            self.logger.success(f"[SpatialCache] {len(covered_keys)} group(s) served from cache — no API call")
        return covered_keys, served

    def _store_spatial_cache(self, rep_df, representatives, id_col, params: dict = None, date_column="date"):
        """
        Persist freshly fetched representative rows under their cell key, so any
        future run touching those cells can be served without an API call.
        Deduplicates on ``_cell_key`` × date; failures are logged, never fatal.
        """
        if rep_df is None or rep_df.empty or date_column not in rep_df.columns:
            return
        id_result_cols = [c for c in ("entity_id", id_col) if c in rep_df.columns]
        if not id_result_cols:
            return

        cell_by_id = dict(zip(representatives[id_col].astype(str), representatives["_cell_key"]))
        new_rows = rep_df.copy()
        new_rows["_cell_key"] = new_rows[id_result_cols[0]].astype(str).map(cell_by_id)
        new_rows = new_rows[new_rows["_cell_key"].notna()]
        if new_rows.empty:
            return
        new_rows["_cached_at"] = pd.Timestamp.now()

        path = self._spatial_cache_path(params)
        existing = pd.DataFrame()
        if path.exists():
            try:
                existing = pd.read_parquet(path)
            except Exception as e:
                self.logger.warning(f"[SpatialCache] Corrupted cache file {path.name}, rebuilding: {e}")
                path.unlink(missing_ok=True)

        combined = pd.concat([existing, new_rows], ignore_index=True)
        dedup = [c for c in ("_cell_key", date_column) if c in combined.columns]
        if dedup:
            combined = combined.drop_duplicates(subset=dedup, keep="last")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
            os.close(fd)
            combined.to_parquet(tmp_path, index=False)
            Path(tmp_path).replace(path)
        except Exception as e:
            self.logger.error(f"[SpatialCache] Failed to write cache {path.name}: {e}")
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
            return
        self.logger.success(f"[SpatialCache] Updated {path.name} -> {len(combined)} records")

    def clear_spatial_cache(self, params: dict = None):
        """Delete the geohash-keyed cache file for this extractor (and parameter set)."""
        path = self._spatial_cache_path(params)
        if path.exists():
            path.unlink()
            self.logger.info(f"[SpatialCache] Cleared {path.name}")
        else:
            self.logger.info(f"[SpatialCache] No cache file to clear at {path.name}")

    def spatial_cache_info(self, params: dict = None) -> dict:
        """Cells, rows and date span currently held in the geohash-keyed cache."""
        path = self._spatial_cache_path(params)
        info = {"path": str(path), "exists": path.exists(), "records": 0, "cells": 0}
        if not path.exists():
            return info
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            self.logger.warning(f"[SpatialCache] Corrupted cache file {path.name}, removing it: {e}")
            path.unlink(missing_ok=True)
            info["exists"] = False
            info["error"] = str(e)
            return info
        info["records"] = len(df)
        if "_cell_key" in df.columns:
            info["cells"] = int(df["_cell_key"].nunique())
        if "date" in df.columns and not df.empty:
            d = pd.to_datetime(df["date"], errors="coerce")
            info["date_min"], info["date_max"] = str(d.min()), str(d.max())
        return info

    # ------------------------------------------------------------------
    # 🧭 SPATIAL GROUPING (geohash dedup for point-based extractors)
    # ------------------------------------------------------------------
    # Weather / GDD / GDD-offset are queried by field *centroid*, so fields whose
    # centroids fall in the same geohash cell (and share the same date window /
    # provider) return identical values. Grouping by geohash × request-signature,
    # calling the API once per group, and broadcasting the result to every member
    # collapses thousands of near-identical calls into a few hundred — the same
    # "compute once, reuse for equivalent requests" idea as the cross-run cache
    # above, applied *within* a single run (where the moving date window means the
    # persisted cache rarely hits anyway).

    def _add_spatial_group_key(self, entity_list, signature_columns, precision):
        """
        Return a copy of ``entity_list`` with ``_geohash`` and ``_spatial_group_key``
        columns. The group key is ``geohash | sig_col=value | ...`` for every
        signature column that actually exists in the frame; columns absent from the
        frame are treated as constant across the run (they don't split groups).

        Rows whose geometry can't be encoded get a unique ``__nogeo_<n>__`` key so
        they are never merged into another field's result — they fall through to a
        normal per-field call.
        """
        df = entity_list.copy()
        geom_col = self.get_mapped_column("geometry")

        from earthdaily.agriculture.core.geometry import centroid_geohash

        def _encode(geom):
            try:
                return centroid_geohash(geom, precision=precision)
            except Exception:
                return None

        df["_geohash"] = df[geom_col].map(_encode)

        # Resolve signature columns through the column mapping; keep only present ones.
        sig_cols = []
        for key in signature_columns or []:
            col = self.get_mapped_column(key)
            if col in df.columns and col not in sig_cols:
                sig_cols.append(col)

        gh = df["_geohash"].astype("string")
        if sig_cols:
            sig_str = (
                df[sig_cols].astype("string").apply(lambda r: "|".join(f"{c}={v}" for c, v in zip(sig_cols, r)), axis=1)
            )
            df["_spatial_group_key"] = gh.str.cat(sig_str, sep="|")
        else:
            df["_spatial_group_key"] = gh

        # Give geometry-less rows their own singleton group.
        nogeo = df["_geohash"].isna()
        if nogeo.any():
            df.loc[nogeo, "_spatial_group_key"] = [f"__nogeo_{i}__" for i in range(int(nogeo.sum()))]

        # The cell key is the *date-independent* identity of a group: geohash plus
        # the non-date signature. Window bucketing may split a cell into several
        # groups, but they all share this key — which is what the spatial cache is
        # keyed on, so a cell's data is reusable whatever window a later run asks for.
        df["_cell_key"] = df["_spatial_group_key"]

        return df

    def _resolve_group_windows(self, keyed, window_columns, max_window_days):
        """
        Give every spatial group a single union date window, splitting groups whose
        union would exceed ``max_window_days``.

        With dates out of the group key, one cell can hold fields with staggered
        windows. Rather than one call per distinct window, we take ``min(start)`` /
        ``max(end)`` across the cell and make a single wider call — every member's
        window is a sub-range of it, so members are served by slicing the result
        (see ``_clip_and_rebase_members``).

        Members are sorted by start date and packed greedily into buckets whose span
        stays within the cap, so one long-history outlier gets its own call instead
        of inflating everyone else's.

        Adds ``_win_start`` / ``_win_end`` (Timestamps) and appends the bucket index
        to ``_spatial_group_key``. Rows with an unusable window are never merged.
        """
        start_col = self.get_mapped_column(window_columns[0])
        end_col = self.get_mapped_column(window_columns[1])

        df = keyed.copy()
        if start_col not in df.columns or end_col not in df.columns:
            # No per-entity window in the frame — params supply a single global
            # window, so there is nothing to widen.
            df["_win_start"] = pd.NaT
            df["_win_end"] = pd.NaT
            return df

        starts = self._naive_datetimes(df[start_col])
        ends = self._naive_datetimes(df[end_col])
        cap = pd.Timedelta(days=max_window_days) if max_window_days else None

        win_start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        win_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        bucket = pd.Series(0, index=df.index, dtype=int)
        n_split = 0

        collapsed: list[tuple[int, pd.Timedelta]] = []

        for _, idx in df.groupby("_spatial_group_key", sort=False).groups.items():
            members = sorted(idx, key=lambda i: (pd.isna(starts[i]), starts[i]))
            buckets = []  # list of [indices, start, end, smallest own-window length]
            open_b = None
            for i in members:
                s, e = starts[i], ends[i]
                if pd.isna(s) or pd.isna(e) or e < s:
                    # Unusable window: singleton bucket, no widening, no merging.
                    buckets.append([[i], s, e, None])
                    continue
                own = e - s
                if open_b is None:
                    buckets.append([[i], s, e, own])
                    open_b = len(buckets) - 1
                    continue
                idxs, bs, be, bmin = buckets[open_b]
                ns, ne = min(bs, s), max(be, e)
                # The cap bounds how far the union WIDENS a member, not the union's
                # absolute length. Comparing the raw union against the cap meant a
                # cell whose members all request the SAME window longer than the cap
                # split into one bucket per field: nobody's window got shorter, the
                # sharing was simply destroyed and dedup fell to 1.0x — the feature
                # silently doing nothing on exactly the long-history pulls it helps
                # most.
                #
                # A merge is allowed when the union is no longer than the cap OR than
                # a member's own window — that member is asking for the whole span
                # anyway, so it is not being widened. The test has to hold for EVERY
                # member, so the binding constraint is the SMALLEST own-window in the
                # bucket: taking the new member's own length alone would let a
                # long-history field swallow short-history neighbours whenever it
                # happened to arrive second.
                allowance = max(cap, min(bmin, own)) if cap is not None else None
                if allowance is not None and (ne - ns) > allowance:
                    buckets.append([[i], s, e, own])
                    open_b = len(buckets) - 1
                else:
                    buckets[open_b] = [idxs + [i], ns, ne, min(bmin, own)]
            if len(buckets) > 1:
                n_split += 1
            # Every member in its own bucket means the cap bought nothing here.
            if cap is not None and len(members) > 1 and len(buckets) == len(members):
                spans = [b[2] - b[1] for b in buckets if b[3] is not None]
                if spans:
                    collapsed.append((len(members), max(spans)))
            for bnum, (idxs, bs, be, _own) in enumerate(buckets):
                for j in idxs:
                    win_start[j] = bs
                    win_end[j] = be
                    bucket[j] = bnum

        df["_win_start"] = win_start
        df["_win_end"] = win_end
        df["_spatial_group_key"] = df["_spatial_group_key"].astype(str) + "|w" + bucket.astype(str)
        if n_split:
            self.logger.info(f"[Geohash] {n_split} cell(s) split into multiple window buckets (cap={max_window_days}d)")
        if collapsed:
            widest = max(span for _, span in collapsed)
            fields = sum(n for n, _ in collapsed)
            self.logger.warning(
                f"[Geohash] spatial_max_window_days={max_window_days} split {len(collapsed)} cell(s) "
                f"({fields} field(s)) into one bucket each — no dedup from those cells. The widest "
                f"window involved is {widest.days}d. Raise spatial_max_window_days above it to let "
                f"these fields share a call."
            )
        return df

    def _stamp_representative_windows(self, representatives, window_columns):
        """
        Overwrite each representative's start/end with its group's union window, so
        the single call it makes covers every member of the group.
        """
        start_col = self.get_mapped_column(window_columns[0])
        end_col = self.get_mapped_column(window_columns[1])
        reps = representatives.copy()
        if start_col not in reps.columns or end_col not in reps.columns:
            return reps
        has_win = reps["_win_start"].notna() & reps["_win_end"].notna()
        reps.loc[has_win, start_col] = reps.loc[has_win, "_win_start"].dt.strftime("%Y-%m-%d")
        reps.loc[has_win, end_col] = reps.loc[has_win, "_win_end"].dt.strftime("%Y-%m-%d")
        return reps

    @staticmethod
    def _naive_datetimes(values):
        """
        Parse to timezone-naive UTC timestamps.

        The weather API returns tz-aware dates (``datetime64[us, UTC]``) while entity
        windows arrive as plain date strings. Comparing the two raises, so both sides
        are normalised to naive UTC before any window arithmetic.
        """
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if isinstance(parsed, pd.Series):
            return parsed.dt.tz_localize(None)
        try:
            return parsed.tz_localize(None)
        except (AttributeError, TypeError):  # already naive, or NaT
            return parsed

    def _clip_and_rebase_members(self, merged, date_col, cumulative_columns, reset_yearly):
        """
        Cut each member's rows back to its own date window and re-zero cumulative
        columns to that member's start, so a grouped run returns exactly what an
        ungrouped run would have.

        ``merged`` carries each member's own window in the internal ``_m_start`` /
        ``_m_end`` columns. Baselines are read *before* clipping (the rows that carry
        them are about to be dropped): for each member the baseline is the cumulative
        value on the last date strictly before its start. With
        ``reset_cumulative_every_year`` the counter restarts each January, so the
        baseline is confined to the start year and later years need no correction.
        """
        if merged.empty or date_col not in merged.columns:
            return merged

        dates = self._naive_datetimes(merged[date_col])
        m_start = self._naive_datetimes(merged["_m_start"])
        m_end = self._naive_datetimes(merged["_m_end"])

        cum_cols = [c for c in (cumulative_columns or []) if c in merged.columns]
        baselines = None
        if cum_cols:
            pre_mask = dates < m_start
            if reset_yearly:
                # A value from before the January reset is not a valid baseline.
                pre_mask &= dates.dt.year == m_start.dt.year
            pre = merged.loc[pre_mask, ["_member_row_id"] + cum_cols].copy()
            if not pre.empty:
                pre["_d"] = dates[pre_mask].values
                pre = pre.sort_values("_d").groupby("_member_row_id", sort=False).last()
                baselines = pre[cum_cols]

        # Clip to the member's own window; rows without a usable window are kept.
        keep = ((dates >= m_start) & (dates <= m_end)) | m_start.isna() | m_end.isna() | dates.isna()
        out = merged.loc[keep].copy()

        if cum_cols and baselines is not None and not out.empty:
            out_dates = self._naive_datetimes(out[date_col])
            out_start = self._naive_datetimes(out["_m_start"])
            aligned = baselines.reindex(out["_member_row_id"].values)
            apply_mask = out_start.notna()
            if reset_yearly:
                apply_mask &= out_dates.dt.year == out_start.dt.year
            for col in cum_cols:
                base = pd.Series(aligned[col].values, index=out.index).fillna(0)
                out[col] = pd.to_numeric(out[col], errors="coerce") - base.where(apply_mask, 0)

        return out

    def _tag_rows_with_group_key(self, rep_df, representatives, id_col):
        """
        Tag freshly fetched representative rows with their spatial group key, via the
        representative's own id — the only link the result carries back to its group.
        Returns None if the result has no recognisable id column.
        """
        id_result_cols = [c for c in ("entity_id", id_col) if c in rep_df.columns]
        if not id_result_cols:
            return None
        gk_by_id = dict(zip(representatives[id_col].astype(str), representatives["_spatial_group_key"]))
        tagged = rep_df.copy()
        tagged["_spatial_group_key"] = tagged[id_result_cols[0]].astype(str).map(gk_by_id)
        return tagged

    # Internal columns that describe the grouping and must never reach the output.
    _GROUPING_INTERNALS = ("_geohash", "_spatial_group_key", "_cell_key", "_win_start", "_win_end")

    def _broadcast_grouped_results(
        self,
        keyed,
        representatives,
        rep_df,
        id_col,
        *,
        tagged=None,
        window_columns=None,
        cumulative_columns=None,
        reset_yearly=False,
        date_column="date",
    ):
        """
        Expand representative results back to every group member.

        Extraction columns (dates, weather params, GDD values) are shared across a
        group by construction and copied verbatim; id and per-member metadata columns
        are re-stamped from each member's own row. The output schema matches what a
        full per-field run would have produced.

        When ``window_columns`` is given the group was called over the *union* of its
        members' windows, so each member is then cut back to its own window and its
        cumulative columns re-zeroed — without which a member that asked for a later
        start would inherit the group's accumulation.

        ``tagged`` accepts rows already carrying ``_spatial_group_key`` (used when
        some groups are served from the spatial cache and have no live representative).
        """
        if tagged is None:
            if rep_df is None or rep_df.empty:
                return pd.DataFrame()
            tagged = self._tag_rows_with_group_key(rep_df, representatives, id_col)
            if tagged is None:
                self.logger.warning(
                    "[Geohash] Result has no id column — skipping broadcast, returning representatives."
                )
                return rep_df
        if tagged is None or tagged.empty:
            return pd.DataFrame()

        result_cols = [c for c in tagged.columns if c not in self._GROUPING_INTERNALS]
        orig_cols = set(keyed.columns) - set(self._GROUPING_INTERNALS)
        # id column as it appears in the result (weather uses "entity_id"; GDD keeps
        # the mapped id column via normalize_with_metadata — support either/both).
        id_result_cols = [c for c in ("entity_id", id_col) if c in result_cols]
        if not id_result_cols:
            self.logger.warning("[Geohash] Result has no id column — skipping broadcast, returning representatives.")
            return tagged[result_cols]

        meta_cols = [c for c in result_cols if c in orig_cols and c not in id_result_cols]
        extraction_cols = [c for c in result_cols if c not in id_result_cols and c not in meta_cols]

        ext = tagged[["_spatial_group_key"] + extraction_cols].copy()

        member_frame = keyed[[id_col, "_spatial_group_key"] + meta_cols].copy()
        member_frame["_member_row_id"] = range(len(member_frame))
        if window_columns:
            start_col = self.get_mapped_column(window_columns[0])
            end_col = self.get_mapped_column(window_columns[1])
            member_frame["_m_start"] = keyed[start_col].values if start_col in keyed.columns else pd.NaT
            member_frame["_m_end"] = keyed[end_col].values if end_col in keyed.columns else pd.NaT

        merged = member_frame.merge(ext, on="_spatial_group_key", how="inner")

        if window_columns:
            # Only columns the extractor explicitly declares are rebased. Guessing from
            # the name is unsafe: weather's `precipitation.cumulative` is measured to be
            # a per-day total (identical across overlapping windows), while GDD's
            # `cumulated_gdd` really does accumulate from the request start.
            merged = self._clip_and_rebase_members(merged, date_column, cumulative_columns, reset_yearly)
            merged = merged.drop(columns=[c for c in ("_m_start", "_m_end") if c in merged.columns])

        # Re-stamp the id column(s) with each member's own id.
        for col in id_result_cols:
            merged[col] = merged[id_col]

        drop = ["_spatial_group_key", "_member_row_id"]
        if id_col not in result_cols:  # e.g. weather output has entity_id but not id
            drop.append(id_col)
        merged = merged.drop(columns=[c for c in drop if c in merged.columns])

        # Preserve the representative result's column order.
        ordered = [c for c in result_cols if c in merged.columns]
        ordered += [c for c in merged.columns if c not in ordered]
        return merged[ordered]

    def _run_bulk_spatially_grouped(
        self,
        *,
        entity_list,
        run_representatives,
        signature_columns,
        precision,
        prefix,
        output_path=None,
        skip_export=False,
        generate_report=False,
        report_options=None,
        report_params=None,
        window_columns=None,
        max_window_days=DEFAULT_MAX_WINDOW_DAYS,
        cumulative_columns=None,
        reset_yearly=False,
        date_column="date",
        use_cache=None,
        cache_params=None,
    ):
        """
        Orchestrate a spatially-grouped bulk run for a point-based extractor.

        Groups entities by ``geohash(centroid) × signature_columns``, calls
        ``run_representatives(reps_df)`` on one representative per group (the callable
        must return the standard bulk result dict and must NOT export — this method
        finalizes the broadcast), broadcasts the results to all members, then handles
        export / report on the full set. Return shape matches the plain bulk methods,
        plus ``representative_calls`` / ``grouped_from`` for observability.

        When ``window_columns`` is given (extractors whose result is a date series),
        dates leave the group key: each cell is called once over the *union* of its
        members' windows and members are sliced out of that single response. The
        geohash-keyed spatial cache is then consulted first, so a cell whose window
        is already on disk costs no API call at all.
        """
        log = self.get_contextualized_logger("GEOHASH")
        id_col = self.get_mapped_column("id")

        if entity_list is None or getattr(entity_list, "empty", True):
            return run_representatives(entity_list)

        keyed = self._add_spatial_group_key(entity_list, signature_columns, precision)
        if window_columns:
            keyed = self._resolve_group_windows(keyed, window_columns, max_window_days)
        n_total = len(keyed)
        representatives = keyed.drop_duplicates("_spatial_group_key", keep="first").copy()
        if window_columns:
            representatives = self._stamp_representative_windows(representatives, window_columns)
        n_groups = len(representatives)
        ratio = n_total / n_groups if n_groups else 1.0

        log.info(
            f"[Geohash] Spatial grouping: {n_total} fields → {n_groups} group(s) "
            f"(precision={precision}, {ratio:.1f}× dedup)"
        )

        # Consult the geohash cache before deciding what still needs fetching.
        cache_enabled = (use_cache if use_cache is not None else self.use_cache) and bool(window_columns)
        covered_keys, cached_rows = set(), pd.DataFrame()
        if cache_enabled:
            covered_keys, cached_rows = self._serve_from_spatial_cache(
                self._load_spatial_cache(cache_params), representatives, date_column
            )
        to_fetch = representatives[~representatives["_spatial_group_key"].isin(covered_keys)]
        n_calls = len(to_fetch)

        print(
            f"🧭 Spatial grouping: {n_total} fields → {n_groups} group(s) → "
            f"{n_calls} API call(s) ({len(covered_keys)} from cache)"
        )

        rep_result = {"global_errors": []}
        fresh_rows = pd.DataFrame()
        if not to_fetch.empty:
            # Fetch representatives only; do not export (we finalize the broadcast below).
            rep_result = run_representatives(to_fetch.drop(columns=list(self._GROUPING_INTERNALS), errors="ignore"))
            rep_df = rep_result.get("results_df", pd.DataFrame())
            if rep_df is not None and not rep_df.empty:
                if cache_enabled:
                    self._store_spatial_cache(rep_df, to_fetch, id_col, cache_params, date_column)
                tagged = self._tag_rows_with_group_key(rep_df, to_fetch, id_col)
                fresh_rows = tagged if tagged is not None else pd.DataFrame()

        frames = [f for f in (cached_rows, fresh_rows) if f is not None and not f.empty]
        tagged_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        broadcast_df = self._broadcast_grouped_results(
            keyed,
            representatives,
            None,
            id_col,
            tagged=tagged_all,
            window_columns=window_columns,
            cumulative_columns=cumulative_columns,
            reset_yearly=reset_yearly,
            date_column=date_column,
        )

        # Recompute counts against the *full* member list (row-count invariant guard).
        if not broadcast_df.empty:
            result_id_col = "entity_id" if "entity_id" in broadcast_df.columns else id_col
            covered = set(broadcast_df[result_id_col].astype(str)) if result_id_col in broadcast_df.columns else set()
        else:
            covered = set()
        all_ids = keyed[id_col].astype(str)
        successful = int(all_ids.isin(covered).sum())
        failed_ids = sorted(set(all_ids) - covered)
        if failed_ids:
            log.warning(
                f"[Geohash] {len(failed_ids)} field(s) had no data (representative call failed for their group)"
            )

        final_df, _ = self._finalize_extraction(
            results_df=broadcast_df,
            global_errors=rep_result.get("global_errors", []),
            failed_ids=failed_ids,
            output_path=output_path,
            prefix=prefix,
            skip_export=skip_export,
            verbose=True,
            generate_report=generate_report,
            report_options=report_options,
            extraction_stats={
                "total_entities": n_total,
                "total_calculations": n_total,
                "successful": successful,
                "failed": n_total - successful,
                "elapsed_seconds": 0,
                "parameters": report_params or {},
                "entity_df": entity_list,
            },
        )

        out = {
            "results_df": final_df,
            "global_errors": rep_result.get("global_errors", []),
            "total_entities": n_total,
            "total_calculations": n_total,
            "successful_calculations": successful,
            "failed_calculations": n_total - successful,
            "failed_ids": failed_ids,
            "representative_calls": n_calls,
            "spatial_groups": n_groups,
            "grouped_from": n_total,
        }
        # Cache stats are in *group* units, not members, so grouped + cached runs
        # stay observable: how many cells the API was actually asked about.
        if cache_enabled:
            out["cache_hit"] = len(covered_keys)
            out["cache_miss"] = n_calls
        else:
            for k in ("cache_hit", "cache_miss"):
                if k in rep_result:
                    out[k] = rep_result[k]
        return out

    # -------------------------------
    # 🔐 TOKEN MANAGEMENT
    # -------------------------------

    def is_token_expired(self) -> bool:
        """Check if the bearer token is expired or about to expire."""
        if not self.token_expiration:
            self.logger.warning("No token expiration set - considering token expired")
            return True

        # Convert datetime to timestamp if needed
        if isinstance(self.token_expiration, datetime):
            exp_time = self.token_expiration.timestamp()
        else:
            exp_time = self.token_expiration

        # Refresh 2 min before expiration
        is_expired = time.time() > (exp_time - 120)

        if is_expired:
            self.logger.debug("Token expired or expiring soon")

        return is_expired

    def ensure_token_valid(self):
        """
        Ensure the current token is valid.
        - If part of a workflow, delegate refresh to workflow manager.
        - If standalone, refresh locally using get_new_token method.
        """
        # The workflow's token belongs to one identity provider. Syncing it into an
        # extractor authenticated somewhere else would overwrite a valid token with
        # a foreign one on every call — so only delegate when the providers match.
        workflow_provider = getattr(self.workflow_ref, "AUTH_PROVIDER", "geosys") if self.workflow_ref else None

        if self.workflow_ref and workflow_provider == self.AUTH_PROVIDER:
            # ✅ Use the shared refresh mechanism in the workflow
            self.logger.debug("Delegating token refresh to workflow manager")
            self.workflow_ref.refresh_token_if_needed()
            # Sync updated token and expiration
            self.bearer_token = self.workflow_ref.bearer_token
            self.token_expiration = self.workflow_ref.token_expiration
            self.logger.debug("Token synced from workflow manager")

        elif self.is_token_expired():
            # Covers both standalone extractors and those on a second identity
            # provider inside a workflow: get_new_token() is the override point,
            # and an extractor whose provider differs routes through it (which
            # may still consult workflow_ref for a shared token of ITS provider).
            self.logger.info("🔄 Token expired — refreshing locally...")
            try:
                new_token, exp_time = self.get_new_token()
                self.bearer_token = new_token
                self.token_expiration = exp_time
                self.logger.success("✅ Token refreshed successfully")
            except Exception as e:
                self.logger.error(f"❌ Token refresh failed: {e}")
                raise

    def get_new_token(self):
        """
        Default implementation of token refresh logic.
        Uses EDAuthenticator to get a new token based on environment.
        Child classes can override this if they need custom token logic.
        """
        from earthdaily.agriculture.core.identity import EDAuthenticator

        self.logger.debug(f"Requesting new token for environment: {self.env}")
        new_token, exp_time = EDAuthenticator.get_new_token(env=self.env)
        return new_token, exp_time

    # -------------------------------
    # 🔀 MERGE RESULTS
    # -------------------------------

    def _merge_with_skipped_entities(self, new_results_df, skipped_entities_df, merge_mode="auto", verbose=True):
        """
        Generic method to merge extraction results with skipped entities.
        """
        log = self.get_contextualized_logger("MERGE")

        if skipped_entities_df.empty:
            log.debug("No skipped entities to merge")
            if verbose:
                print("📋 No skipped entities to merge")
            return new_results_df

        # Only these columns are REQUIRED for entity processing (use mapped names)
        required_entity_columns = [
            self.get_mapped_column("id"),
            self.get_mapped_column("geometry"),
            self.get_mapped_column("crop"),
        ]

        # Detect extraction columns
        extraction_columns = []
        has_existing_data = False

        if not new_results_df.empty:
            extraction_columns = [
                col
                for col in new_results_df.columns
                if col in skipped_entities_df.columns and col not in required_entity_columns
            ]
            has_existing_data = len(extraction_columns) > 0
            log.debug(f"Detected {len(extraction_columns)} extraction columns in skipped entities")

        # Determine merge behavior
        if merge_mode == "auto":
            should_preserve = has_existing_data
            log.debug(f"Auto mode: {'preserving' if should_preserve else 'marking'} skipped entities")
        elif merge_mode == "preserve":
            should_preserve = True
            log.debug("Preserve mode: keeping existing data")
        else:  # 'mark'
            should_preserve = False
            log.debug("Mark mode: adding skip markers")

        # Execute merge
        if should_preserve and has_existing_data:
            log.info(f"Preserving existing data in {len(extraction_columns)} columns")
            if verbose:
                print(f"🔀 Merge mode: Preserving existing data in {len(extraction_columns)} columns")
                if len(extraction_columns) <= 5:
                    print(f"   Columns: {', '.join(extraction_columns)}")
                else:
                    print(f"   Columns: {', '.join(extraction_columns[:5])} ... (+{len(extraction_columns) - 5} more)")

            # Simple concatenation
            merged_df = pd.concat([new_results_df, skipped_entities_df], ignore_index=True)

            log.success(
                f"Merged: {len(new_results_df)} new + {len(skipped_entities_df)} preserved = {len(merged_df)} total"
            )
            if verbose:
                print(
                    f"✅ Merged: {len(new_results_df)} new + {len(skipped_entities_df)} preserved = {len(merged_df)} total rows"
                )

        elif new_results_df.empty:
            # 🚨 EDGE CASE: No new results, just return skipped entities
            log.warning("No new results to merge, returning skipped entities only")
            if verbose:
                print("⚠️ No new results to merge, returning skipped entities only")
            merged_df = skipped_entities_df.copy()

        else:
            # Legacy mode: mark skipped entities
            log.info(f"Adding {len(skipped_entities_df)} skipped entities with marker")
            if verbose:
                print(f"📋 Adding {len(skipped_entities_df)} skipped entities with marker")

            skipped_entities_df = skipped_entities_df.copy()
            skipped_entities_df["_skipped_by_filter"] = True
            merged_df = pd.concat([new_results_df, skipped_entities_df], ignore_index=True)

        return merged_df

    # -------------------------------
    # 🗺️ MAP FILE WRITER (postprocess="file")
    # -------------------------------

    def save_map_file(self, save_path, filename, data: bytes) -> str:
        """Save one raster/map file, and mirror it to ``output_uri`` when set.

        Shared by the ``postprocess="file"`` writers in FLM, Difference and
        Zoning. Routes through :mod:`earthdaily.agriculture.core._fs`, so
        ``save_path`` may itself be a remote URI (``s3://``, ``gs://``, …) and
        not just a local directory — the plain ``open(..., "wb")`` these
        writers used before treated an ``s3://`` prefix as a directory name.

        When ``self.output_uri`` is set the same bytes are written a second
        time to ``<output_uri>/<filename>`` as the durable copy. A failure on
        that write is raised, not swallowed: the caller reports it as a normal
        per-entity failure. A raster recorded in the manifest but living only
        on a runner is worse than one that failed loudly.

        Args:
            save_path: Directory (local) or prefix (remote URI) for the
                working copy.
            filename: File name, already sanitized by the caller.
            data: File contents.

        Returns:
            str: The path of the working copy — what callers put in
            ``saved_files``, and therefore what lands in the manifest. It stays
            local so the analysis step can keep opening files off disk.
        """
        from earthdaily.agriculture.core import _fs

        local_path = _fs.join_path(save_path, filename)
        _fs.write_bytes(local_path, data)

        # getattr, not self.output_uri: extractors built with __init__ bypassed
        # (the test fixtures, some notebook paths) never ran the resolution in
        # __init__. Same guard the manifest sidecar uses.
        output_uri = getattr(self, "output_uri", None)
        if output_uri:
            remote_path = _fs.join_path(output_uri, filename)
            try:
                _fs.write_bytes(remote_path, data)
                self.logger.debug(f"Mirrored to durable storage: {remote_path}")
            except Exception as e:
                # Per-entity failure — never a silent drop.
                self.logger.error(f"Durable write failed for {remote_path}: {e}")
                raise

        return local_path

    # ♻️ RESUME (retry_failed_only)
    # -------------------------------

    def _failed_ids_files(self, prefix):
        """Newest-first list of ``failed_ids_<prefix>_<ts>.csv`` under ``partial_path``.

        Uses the ``_fs`` helpers rather than ``glob.glob`` so a remote ``partial_path``
        (``s3://``, ``az://``) works — the 30 copied blocks this replaces used the
        stdlib glob and silently found nothing when partials lived in a bucket, despite
        doc 13 advertising identical resume semantics over S3.

        Filenames carry a 10-digit epoch stamp, so reverse-lexicographic ordering is
        chronological (through the year 2286).
        """
        if not self.partial_path:
            return []
        from earthdaily.agriculture.core._fs import glob_files, join_path

        return sorted(glob_files(join_path(self.partial_path, f"failed_ids_{prefix}_*.csv")), reverse=True)

    def _resolve_retry_entity_list(self, entity_list, prefix, *, fail_safe=False, log=None):
        """Apply ``retry_failed_only`` resume mode, or warn about an unused resume file.

        Replaces a block that was copy-pasted at 30 call sites, where it was gated on
        ``fail_safe`` and therefore narrowed the entity list as an undeclared side
        effect of asking for error tolerance. Resume is now explicit
        (``self.retry_failed_only``) and always announced.

        Args:
            entity_list: The already-filtered entity DataFrame.
            prefix: Export prefix — selects which ``failed_ids_<prefix>_*.csv`` to read.
            fail_safe: Only used to decide whether an *unused* resume file is worth
                warning about (these are the callers whose behaviour changed).
            log: Contextualised logger; falls back to the module logger.

        Returns:
            The entity list, narrowed only when resume was explicitly requested.

        Raises:
            FileNotFoundError: ``retry_failed_only`` was requested but no such file
                exists. Asking to resume and silently getting a full run is the same
                class of surprise this change removes, inverted.
        """
        log = log or logger
        retry = getattr(self, "retry_failed_only", False)

        if not retry:
            # Behaviour change notice: this is the call that used to be narrowed.
            if fail_safe:
                stale = self._failed_ids_files(prefix)
                if stale:
                    log.warning(
                        f"♻️ Found a previous run's failed-ID file ({stale[0]}). "
                        f"Processing ALL {len(entity_list)} entities. "
                        "fail_safe no longer implies resume — pass retry_failed_only=True "
                        "(or retry_failed_only='<path>') to process only those IDs."
                    )
            return entity_list

        from earthdaily.agriculture.core._fs import storage_options_for

        if isinstance(retry, str):
            target = retry
            if not os.path.isfile(target) and "://" not in target:
                raise FileNotFoundError(f"retry_failed_only={target!r}: file not found.")
        else:
            candidates = self._failed_ids_files(prefix)
            if not candidates:
                raise FileNotFoundError(
                    f"retry_failed_only=True but no failed_ids_{prefix}_*.csv found in "
                    f"{self.partial_path!r}. Nothing to resume — remove retry_failed_only "
                    "to process the full entity list."
                )
            target = candidates[0]

        failed_df = pd.read_csv(target, storage_options=storage_options_for(target))
        failed_ids = set(failed_df["failed_ids"].tolist())

        before = len(entity_list)
        narrowed = entity_list[entity_list[self.get_mapped_column("id")].isin(failed_ids)]

        # Loud by design, and states both counts: the old log line was log.success
        # reporting only the post-filter number, which read as a healthy full run.
        log.warning(
            f"♻️ retry_failed_only: narrowed {before} → {len(narrowed)} entities "
            f"from {target} ({len(failed_ids)} IDs recorded)."
        )
        if narrowed.empty:
            log.warning(
                f"♻️ No entity in this list matches the {len(failed_ids)} recorded ID(s) — "
                f"nothing will be processed. Is {target} from a different dataset or prefix?"
            )
        return narrowed

    # -------------------------------
    # 📦 FINALIZE EXTRACTION
    # -------------------------------

    def _finalize_extraction(
        self,
        results_df,
        global_errors,
        failed_ids,
        output_path=None,
        prefix="extraction",
        skip_export=False,
        verbose=True,
        generate_report=False,
        report_options=None,
        extraction_stats=None,
    ):
        """
        Generic finalization for extraction processes.
        Handles: output formatting, final export, failed ID storage, partial cleanup,
        and optional HTML report generation.

        Args:
            results_df (pd.DataFrame): Final results dataframe to export
            global_errors (list): List of error dictionaries
            failed_ids (list): List of failed entity IDs for fail-safe retries
            output_path (str, optional): Directory for final export. If None, uses self.output_path
            prefix (str): Prefix for output filenames (e.g., 'emergence', 'coverage')
            skip_export (bool): If True, skip final export (useful for chaining extractions)
            verbose (bool): Print status messages
            generate_report (bool): If True, generate an HTML report alongside the export
            report_options (dict, optional): Options for ExtractionReporter (include_map, preview_rows, etc.)
            extraction_stats (dict, optional): Extraction statistics to include in report:
                {
                    "total_entities": int,
                    "total_calculations": int,
                    "successful": int,
                    "failed": int,
                    "elapsed_seconds": float,
                    "parameters": dict,   # extractor-specific params
                    "entity_df": DataFrame # original entities for map
                }

        Returns:
            tuple: (formatted_results_df, finalization_status_dict)
                finalization_status: {
                    "exported": bool,
                    "export_path": str or None,
                    "failed_ids_saved": bool,
                    "partials_cleaned": bool,
                    "report_path": str or None
                }
        """
        log = self.get_contextualized_logger("FINALIZE")
        log.info(f"Starting finalization for {prefix}")

        finalization_status = {
            "exported": False,
            "export_path": None,
            "failed_ids_saved": False,
            "partials_cleaned": False,
            "report_path": None,
            "manifest_written": False,
        }

        # ========================================================================
        # 0. APPLY OUTPUT FORMATTING (rename, exclude, select)
        # ========================================================================
        results_df = self.apply_output_format(results_df)

        # ========================================================================
        # 1. FINAL EXPORT
        # ========================================================================
        if not skip_export:
            final_output_path = output_path or self.output_path

            if final_output_path:
                log.info(f"Exporting results to: {final_output_path}")
                if verbose:
                    print(f"📦 Final export to: {final_output_path}")

                try:
                    export_results(
                        results_df=results_df,
                        errors=global_errors,
                        output_path=final_output_path,
                        prefix=prefix,
                        partial=False,
                        verbose=verbose,
                        export_format=self.export_format,
                    )
                    finalization_status["exported"] = True
                    finalization_status["export_path"] = final_output_path
                    log.success(f"✅ Export completed: {len(results_df)} rows")
                except Exception as e:
                    log.error(f"❌ Export failed: {e}")
                    raise
            else:
                log.warning("No output_path defined - skipping export")
                if verbose:
                    print("⚠️ No output_path defined in arguments or config — skipping export.")
        else:
            log.info("Export skipped (chaining mode)")
            if verbose:
                print("🔄 Export skipped (chaining mode)")

        # ========================================================================
        # 1b. MANIFEST SIDECAR (optional; parity with export_format)
        # ========================================================================
        # Emit a manifest describing the exported dataset, only when a results
        # export actually happened. Non-fatal, like the HTML report: a failure
        # (e.g. a cloud output_path that generate_manifest can't write to yet)
        # warns but never breaks the extraction.
        # getattr defaults keep this robust for instances built without a full __init__
        # (tests construct extractors via __new__/bypass, so the attrs may be absent).
        if getattr(self, "export_manifest", False) and finalization_status["exported"]:
            try:
                from earthdaily.agriculture.export import generate_manifest

                entity_col = "entity_id" if "entity_id" in results_df.columns else self.get_mapped_column("id")
                metadata = {
                    **(getattr(self, "manifest_metadata", None) or {}),
                    "export_format": self.export_format or "csv",
                }
                if extraction_stats and extraction_stats.get("parameters"):
                    metadata.setdefault("parameters", extraction_stats["parameters"])
                generate_manifest(
                    source=results_df,
                    output_path=finalization_status["export_path"],
                    prefix=prefix,
                    entity_col=entity_col,
                    date_col="date",
                    metadata=metadata,
                )
                finalization_status["manifest_written"] = True
                log.success("🧾 Manifest sidecar written")
                if verbose:
                    print("🧾 Manifest sidecar written")
            except Exception as e:
                log.warning(f"Manifest generation failed (non-fatal): {e}")
                if verbose:
                    print(f"⚠️ Manifest generation failed (non-fatal): {e}")

        # ========================================================================
        # 2. STORE FAILED IDs FOR FAIL-SAFE RETRIES
        # ========================================================================
        if self.partial_path and failed_ids:
            try:
                from earthdaily.agriculture.core._fs import (
                    ensure_dir,
                    join_path,
                    storage_options_for,
                )

                failed_df = pd.DataFrame({"failed_ids": failed_ids})
                timestamp = int(time.time())
                ensure_dir(self.partial_path)
                failed_file = join_path(self.partial_path, f"failed_ids_{prefix}_{timestamp}.csv")
                failed_df.to_csv(failed_file, index=False, storage_options=storage_options_for(self.partial_path))

                log.warning(f"⚠️ {len(failed_ids)} failed entities saved to {failed_file}")
                if verbose:
                    print(f"⚠️ {len(failed_ids)} failed entities saved to {failed_file}")

                finalization_status["failed_ids_saved"] = True
            except Exception as e:
                log.error(f"Failed to save failed IDs: {e}")

        # ========================================================================
        # 3. CLEANUP PARTIALS IF EVERYTHING SUCCEEDED
        # ========================================================================
        if not global_errors and self.partial_path:
            from earthdaily.agriculture.core._fs import glob_files, join_path, remove_files

            # Match both extensions — partials follow the export format, which may
            # be parquet when EDAGRO_EXPORT_FORMAT / export_format is set.
            partial_files = []
            for _ext in ("csv", "parquet"):
                partial_files.extend(glob_files(join_path(self.partial_path, f"{prefix}_*_partial.{_ext}")))

            if partial_files:
                cleaned_count, cleanup_errors = remove_files(partial_files)
                for path, err in cleanup_errors:
                    log.warning(f"Could not delete {path}: {err}")
                    if verbose:
                        print(f"⚠️ Could not delete {path}: {err}")

                if cleaned_count > 0:
                    log.info(f"🧹 Cleaned up {cleaned_count} partial files")
                    if verbose:
                        print(f"🧹 Cleaned up {cleaned_count} partial files")

                    finalization_status["partials_cleaned"] = True

        # ========================================================================
        # 4. GENERATE HTML REPORT (optional)
        # ========================================================================
        if generate_report:
            try:
                from earthdaily.agriculture.reporting import ExtractionReporter

                opts = report_options or {}
                reporter = ExtractionReporter(
                    **{
                        k: v
                        for k, v in opts.items()
                        if k
                        in (
                            "include_map",
                            "include_data_preview",
                            "preview_rows",
                            "map_geometry_column",
                            "map_max_features",
                            "max_errors_shown",
                        )
                    }
                )

                stats = extraction_stats or {}
                reporter.set_extraction_context(
                    extractor_name=self.extractor_name,
                    prefix=prefix,
                    env=self.env,
                    parameters=stats.get("parameters", {}),
                    column_mapping=dict(self.column_mapping),
                    output_path=finalization_status.get("export_path"),
                )

                reporter.set_results(
                    results_df=results_df,
                    total_entities=stats.get("total_entities", 0),
                    total_calculations=stats.get("total_calculations", 0),
                    successful=stats.get("successful", 0),
                    failed=stats.get("failed", 0),
                    failed_ids=failed_ids,
                    global_errors=global_errors,
                    elapsed_seconds=stats.get("elapsed_seconds"),
                    finalization_status=finalization_status,
                )

                if stats.get("entity_df") is not None:
                    reporter.set_entity_data(stats["entity_df"])

                # Determine report output path (works for local + remote URIs).
                from earthdaily.agriculture.core._fs import join_path

                report_dir = output_path or self.output_path
                if report_dir:
                    report_file = join_path(report_dir, f"{prefix}_report.html")
                else:
                    report_file = f"{prefix}_report.html"

                # render_html() handles fsspec routing internally when given a
                # remote URI (see reporting/extraction_reporter.render_html).
                reporter.render_html(output_path=report_file)
                finalization_status["report_path"] = report_file
                log.info(f"HTML report generated: {report_file}")
                if verbose:
                    print(f"📊 Report generated: {report_file}")

            except Exception as e:
                log.warning(f"Report generation failed (non-blocking): {e}")

        log.success(f"Finalization completed for {prefix}")
        return results_df, finalization_status
