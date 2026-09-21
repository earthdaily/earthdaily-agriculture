"""
User Management Module for EarthDaily Agro Platform
Handles user CRUD operations via Master Data Management API v6
"""

# Standard Library
import csv
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Third-Party Libraries
import requests
from tqdm import tqdm

from earthdaily.agriculture.config.urls import agro_urls

# Internal Project Utilities
from earthdaily.agriculture.core.api_utils import (
    export_results as export_partial_results,
)
from earthdaily.agriculture.core.api_utils import (
    filter_entities,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, requires_token
from earthdaily.agriculture.core.identity import EDAuthenticator


def require_user_params(func):
    """Decorator to ensure user read parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "user_params") or self.user_params is None:
            self.logger.error("No user parameters found")
            raise RuntimeError("❌ No user parameters found. Call setup_user_parameters() first.")
        return func(self, *args, **kwargs)

    return wrapper


class UserManager(BaseExtractor):
    """
    Manages user accounts via EarthDaily MDM API.

    Handles user lifecycle operations including creation, role assignment,
    account linking, modification, and deletion. All batch operations use
    parallel processing for performance.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/

    Supported operations:
        - User creation (AGRONOMIST/GROWER) with automatic ordering
        - Role assignment (multiple roles)
        - Account linking (AGRONOMIST manages GROWER)
        - User modification (identified by ID) with link synchronization
        - User deletion with automatic link cleanup
    """

    MAX_WORKERS = 10

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        """Initialize UserManager."""
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        base = agro_urls["eda_data_management_url_fields"][self.env].rstrip("/")
        self.base_url = f"{base}/users"

        self.csv_separator = ","
        self.user_params = None
        self.logger.info(f"👤 UserManager initialized for env: {self.env}")
        self.logger.debug(f"Base URL: {self.base_url}")

    def get_new_token(self):
        """Refresh token using EDAuthenticator."""
        self.logger.debug("Refreshing token for UserManager")
        return EDAuthenticator.get_new_token(env=self.env)

    # =========================================================================
    # CSV LOADING
    # =========================================================================

    def load_csv(self, file_path: str, operation: str = "create", encoding: str = "utf-8") -> pd.DataFrame:
        """Load CSV with automatic separator detection."""
        log = self.get_contextualized_logger("CSV_LOAD")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        log.info(f"Loading CSV: {file_path} (operation: {operation})")

        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=",;\t|")
                separator = dialect.delimiter
            except csv.Error:
                separator = ","

        log.info(f"Detected separator: '{separator}'")
        print(f"📄 Detected CSV separator: '{separator}'")

        self.csv_separator = separator
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
        df.columns = df.columns.str.strip()

        # Normalize column names
        column_mapping = {
            "User_Id": "id",
            "user_id": "id",
            "userId": "id",
            "ID": "id",
            "userType.code": "User_Type",
            "usertype.code": "User_Type",
            "UserType": "User_Type",
            "Usertype": "User_Type",
            "firstname": "First_Name",
            "Firstname": "First_Name",
            "FirstName": "First_Name",
            "first_name": "First_Name",
            "lastName": "Last_Name",
            "lastname": "Last_Name",
            "LastName": "Last_Name",
            "last_name": "Last_Name",
            "email": "Email",
            "EMAIL": "Email",
            "login": "Login",
            "LOGIN": "Login",
            "companyName": "CompanyName",
            "companyname": "CompanyName",
            "country_code": "Country_code",
            "Country_Code": "Country_code",
            "culture_code": "Culture_Code",
            "CultureCode": "Culture_Code",
            "customer_id": "Customer_Id",
            "CustomerId": "Customer_Id",
            "mobilePhone": "Mobile_Phone",
            "mobile_phone": "Mobile_Phone",
            "MobilePhone": "Mobile_Phone",
            "phone": "Mobile_Phone",
        }

        for api_col, std_col in column_mapping.items():
            if api_col in df.columns and std_col not in df.columns:
                df = df.rename(columns={api_col: std_col})

        required_by_operation = {
            "create": ["User_Type", "Email", "Login", "Password", "Country_code", "First_Name", "Last_Name"],
            "modify": ["id"],
            "delete": ["id"],
        }

        required_cols = required_by_operation.get(operation, required_by_operation["create"])
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing required columns for {operation}: {missing}")

        log.success(f"✅ Loaded {len(df)} rows from CSV")
        print(f"✅ Loaded {len(df)} users from CSV")

        return df

    # NOTE: reference-data validation is deliberately NOT exposed here. This
    # module ships in the public package, and the vocabularies it would need
    # (config.reference_data) do not — see release/public_allowlist.yml.
    # Internal callers validate explicitly:
    #     from earthdaily.agriculture.core.user_validation import validate_user_dataframe
    #     errors = validate_user_dataframe(df, operation="create")

    # =========================================================================
    # API HELPERS
    # =========================================================================

    def _get_headers(self) -> Dict[str, str]:
        """Get standard API headers with Bearer token."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _build_user_body(self, row: pd.Series, include_password: bool = True, send_login_details: bool = False) -> Dict:
        """Build API request body from DataFrame row.

        ⚠️  ``email``, ``login`` and ``userType`` are ALWAYS sent, defaulting to
        ``""`` / ``""`` / ``GROWER``. On a modify that means omitting them from
        the frame **blanks them on the account** — always carry them through
        from the existing record. Every other field is omitted when absent, so
        partial updates behave as expected.

        Args:
            send_login_details: Set ``sendLoginDetailsToUser``, asking the
                platform to email the account its credentials. Off by default:
                it is an outbound email to a real person, so it should never
                fire as a side effect of a rehearsal.
        """

        def clean_value(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            if val_str.lower() in ("nan", "", "none", "null"):
                return None
            return val_str

        body = {
            "email": clean_value(row.get("Email")) or "",
            "login": clean_value(row.get("Login")) or "",
            "userType": {"code": (clean_value(row.get("User_Type")) or "GROWER").upper()},
        }

        if include_password:
            password = clean_value(row.get("Password"))
            if password:
                body["password"] = password

        if send_login_details:
            body["sendLoginDetailsToUser"] = True

        if clean_value(row.get("First_Name")):
            body["firstname"] = clean_value(row.get("First_Name"))
        if clean_value(row.get("Last_Name")):
            body["lastname"] = clean_value(row.get("Last_Name"))
        if clean_value(row.get("Country_code")):
            body["country"] = {"code": clean_value(row.get("Country_code"))}
        if clean_value(row.get("CompanyName")):
            body["companyName"] = clean_value(row.get("CompanyName"))
        if clean_value(row.get("Culture_Code")):
            body["culture"] = {"code": clean_value(row.get("Culture_Code"))}
        if clean_value(row.get("Customer_Id")):
            body["customer"] = {"id": clean_value(row.get("Customer_Id"))}
        if clean_value(row.get("Mobile_Phone")):
            body["mobilePhone"] = clean_value(row.get("Mobile_Phone"))

        return body

    # =========================================================================
    # GET USERS WITH RELATIONSHIPS (PARALLEL)
    # =========================================================================

    def _fetch_user_relationships(self, user_id: str, user_type: str) -> Dict:
        """Fetch roles and managed_by for a single user (called in parallel)."""
        result = {"id": user_id, "Role": "", "Managed_by_user": ""}
        headers = self._get_headers()

        # Fetch roles
        try:
            url = f"{self.base_url}/{user_id}/roles"
            response = requests.get(url, headers=headers, timeout=30)
            if response.ok:
                roles = response.json()
                if isinstance(roles, list):
                    result["Role"] = ";".join([r.get("id", "") for r in roles if r.get("id")])
        except Exception:
            pass

        # Fetch managed-by (GROWER only) - get logins not IDs
        if user_type.upper() == "GROWER":
            try:
                url = f"{self.base_url}/{user_id}/managed-by"
                response = requests.get(url, headers=headers, timeout=30)
                if response.ok:
                    managers = response.json()
                    if isinstance(managers, list) and managers:
                        manager_logins = []
                        for m in managers:
                            manager_id = m.get("id")
                            if manager_id:
                                try:
                                    url_user = f"{self.base_url}/{manager_id}"
                                    resp = requests.get(url_user, headers=headers, timeout=30)
                                    if resp.ok:
                                        login = resp.json().get("login", "")
                                        if login:
                                            manager_logins.append(login)
                                except Exception:
                                    pass
                        result["Managed_by_user"] = "|".join(manager_logins)
            except Exception:
                pass

        return result

    def _enrich_with_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with Role and Managed_by_user columns using parallel requests."""
        if df.empty:
            return df

        log = self.get_contextualized_logger("ENRICH")
        log.info(f"Fetching relationships for {len(df)} users (parallel, {self.MAX_WORKERS} workers)")
        print(f"🔗 Fetching relationships for {len(df)} users...")

        tasks = [(row["id"], row.get("userType.code", "")) for _, row in df.iterrows()]

        relationships = {}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(self._fetch_user_relationships, uid, utype): uid for uid, utype in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching relationships"):
                try:
                    result = future.result()
                    relationships[result["id"]] = result
                except Exception as e:
                    log.warning(f"Failed to fetch relationships: {e}")

        df = df.copy()
        df["Role"] = df["id"].map(lambda x: relationships.get(x, {}).get("Role", ""))
        df["Managed_by_user"] = df["id"].map(lambda x: relationships.get(x, {}).get("Managed_by_user", ""))

        log.success(f"✅ Relationships fetched for {len(relationships)} users")
        return df

    @requires_token
    def get_users(
        self,
        user_type: Optional[str] = None,
        login: Optional[str] = None,
        customer_id: Optional[str] = None,
        fields: str = "id,userType.code,companyName,firstname,lastname,email,login",
    ) -> pd.DataFrame:
        """
        Get users from API with roles and relationships.

        Args:
            user_type: Filter by user type code (e.g. ``"AGRONOMIST"``, ``"GROWER"``).
            login: Filter by exact login.
            customer_id: Filter by customer id — the usual back-office entry point
                for pulling one client's whole user base.
            fields: ``$fields`` projection sent to the API.

        Returns:
            pd.DataFrame with back-office column names (``User_Type``, ``First_Name``,
            ``Last_Name``, ``CompanyName``), a blank ``Password`` column positioned
            after ``login``, plus ``Role`` and ``Managed_by_user`` from the
            relationship enrichment. Shape round-trips through ``load_csv()``.
        """
        log = self.get_contextualized_logger("GET_USERS")

        params = {"$limit": "none", "$fields": fields}
        if user_type:
            params["userType.code"] = user_type.upper()
        if login:
            params["login"] = login
        if customer_id:
            params["customer.id"] = customer_id

        log.info(f"Fetching users with params: {params}")

        response = requests.get(self.base_url, headers=self._get_headers(), params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()

        if df.empty:
            log.info("No users found")
            return df

        log.info(f"Retrieved {len(df)} users")

        if "login" in df.columns:
            login_idx = df.columns.get_loc("login") + 1
            df.insert(login_idx, "Password", "")

        # Enrich first — _fetch_user_relationships keys off the raw 'userType.code'.
        df = self._enrich_with_relationships(df)

        # Then present back-office column names. load_csv() maps these back on
        # the way in, so export -> edit -> modify/delete round-trips cleanly.
        df = df.rename(
            columns={
                "userType.code": "User_Type",
                "firstname": "First_Name",
                "lastname": "Last_Name",
                "companyName": "CompanyName",
            }
        )

        return df

    @requires_token
    def get_user_by_login(self, login: str) -> Optional[Dict]:
        """Get single user by login."""
        params = {"$limit": "1", "login": login}
        response = requests.get(self.base_url, headers=self._get_headers(), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) > 0 else None

    @requires_token
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get single user by ID."""
        url = f"{self.base_url}/{user_id}"
        self.logger.debug(f"GET {url}")  # no request payload — it's a plain GET
        try:
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            self.logger.debug(f"← HTTP {response.status_code} for {url}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                self.logger.debug(f"User '{user_id}' not found (404) at {url} — returning None")
                return None
            self.logger.debug(f"HTTP {e.response.status_code} at {url}: {e.response.text[:300]}")
            raise

    @requires_token
    def get_managed_growers(self, agronomist_id: str) -> List[Dict]:
        """Get list of GROWER accounts managed by an AGRONOMIST."""
        url = f"{self.base_url}/{agronomist_id}/manage"
        response = requests.get(url, headers=self._get_headers(), params={"$limit": "none"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    def _get_current_manager_ids(self, grower_id: str) -> List[Tuple[str, str]]:
        """Get current AGRONOMIST (id, login) pairs managing a GROWER."""
        headers = self._get_headers()
        result = []

        try:
            url = f"{self.base_url}/{grower_id}/managed-by"
            response = requests.get(url, headers=headers, timeout=30)
            if response.ok:
                managers = response.json()
                if isinstance(managers, list):
                    for m in managers:
                        manager_id = m.get("id")
                        if manager_id:
                            try:
                                url_user = f"{self.base_url}/{manager_id}"
                                resp = requests.get(url_user, headers=headers, timeout=30)
                                if resp.ok:
                                    login = resp.json().get("login", "")
                                    result.append((manager_id, login))
                            except Exception:
                                result.append((manager_id, ""))
        except Exception:
            pass

        return result

    # =========================================================================
    # READ-EXTRACTION LIFECYCLE (workflow extractor surface)
    # =========================================================================
    # These methods give UserManager the standard BaseExtractor read lifecycle
    # so it can be used as a first-class `extractor` step in a YAML workflow:
    #   setup_user_parameters() -> process_single_entity_user()
    #       -> process_entity_user_bulk_parallel()
    # The per-entity API primitive is get_user_by_id(); the entity `id` column
    # (via column_mapping) is treated as the MDM user id. Read-only by design —
    # user mutations (create/modify/delete) deliberately stay off the workflow
    # surface.

    @requires_token
    def setup_user_parameters(
        self,
        fields=None,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for bulk user reads.

        Args:
            fields (list[str], optional): Subset of (flattened) user columns to
                keep in the output. None keeps every field returned by the API.
            partial_frequency (int): How often to flush partial bulk results.
            column_mapping (dict): Maps canonical entity fields to your
                DataFrame columns. The ``id`` mapping must point at the user id.
            output_mapping / exclude_columns / output_columns: Output formatting.
            use_cache (bool, optional): Per-call cache override.
        """
        if fields is not None and not isinstance(fields, (list, tuple)):
            raise ValueError(f"fields must be a list/tuple or None, got {type(fields).__name__}.")

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.configure_output(
            output_mapping=output_mapping,
            exclude_columns=exclude_columns,
            output_columns=output_columns,
        )

        self.apply_cache_setting(use_cache)

        self.user_params = {
            "fields": list(fields) if fields else None,
            "partial_frequency": partial_frequency,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        self.logger.info("User parameters configured successfully")
        print("👤 User read parameters configured:")
        for k, v in self.user_params.items():
            print(f"   {k}: {v}")

    # ----------------------------- Format ----------------------------

    def format_user_json(self, response_json, entity_data=None):
        """
        Normalise a single user record (from ``get_user_by_id``) into a one-row
        DataFrame, flattening nested objects (``userType.code``, ``country.code``, ...).

        Returns:
            pd.DataFrame | None: One-row frame, or None if the record is empty.
        """
        if not response_json:
            return None

        df = pd.json_normalize([response_json])

        fields = (self.user_params or {}).get("fields")
        if fields:
            keep = [c for c in fields if c in df.columns]
            if keep:
                df = df[keep]

        return df

    # ----------------------- Single-entity process -------------------

    def process_single_entity_user(self, row, params=None):
        """
        Fetch one user by id with retry logic.

        Args:
            row (dict | pd.Series | pd.DataFrame): Entity data; the mapped ``id``
                column holds the MDM user id.
            params (dict, optional): Override for ``self.user_params``.

        Returns:
            dict: ``{"data": DataFrame or None, "error": dict or None}``.
        """
        if params is None:
            params = self.user_params

        # Normalise row to dict
        try:
            if isinstance(row, pd.DataFrame):
                if len(row) > 1:
                    self.logger.warning(f"DataFrame with {len(row)} rows — processing first row only.")
                row = row.iloc[0].to_dict()
            elif isinstance(row, pd.Series):
                row = row.to_dict()
            elif not isinstance(row, dict):
                raise TypeError(f"Expected dict/Series/DataFrame, got {type(row).__name__}")
        except Exception as e:
            return {
                "data": None,
                "error": {"message": f"Invalid input: {e}", "entity_id": "unknown"},
            }

        entity_id = self.get_entity_value(row, "id", "unknown")
        user_id = self.get_entity_value(row, "id")

        if not user_id or str(user_id).strip().lower() in ("nan", "", "none"):
            return {
                "data": None,
                "error": {"message": "No valid user id on entity row", "entity_id": entity_id},
            }

        try:

            def _call_api():
                return self.get_user_by_id(str(user_id).strip())

            user = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            user_df = self.format_user_json(user, entity_data=row)
            if user_df is None or user_df.empty:
                return {
                    "data": None,
                    "error": {"message": f"User not found: {user_id}", "entity_id": entity_id},
                }

            result_df = normalize_with_metadata(row, user_df)
            return {"data": result_df, "error": None}

        except Exception as e:
            self.logger.error(f"Entity {entity_id}: user fetch failed — {e}")
            return {
                "data": None,
                "error": {"message": str(e), "entity_id": entity_id},
            }

    # ----------------------- Bulk parallel process -------------------

    def process_entity_user_bulk_parallel(
        self,
        entity_list,
        params=None,
        max_workers=10,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="users",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk user read with threading, partial saves, cache and merge logic.

        Conforms to the WorkflowManager run-method contract, so UserManager can
        be wired into a YAML workflow as an ``extractor`` step::

            run: {method: process_entity_user_bulk_parallel, params: {prefix: users}}

        Args:
            entity_list (pd.DataFrame): Entities whose mapped ``id`` column holds
                the MDM user ids to fetch.
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_entity_user_bulk_parallel_inner,
                entity_list=entity_list,
                params=self.user_params,
                max_workers=max_workers,
                output_path=output_path,
                partial_frequency=partial_frequency,
                fail_safe=fail_safe,
                filter_column=filter_column,
                filter_value=filter_value,
                filter_type=filter_type,
                merge_existing=merge_existing,
                skip_export=skip_export,
                prefix=prefix,
                generate_report=generate_report,
                report_options=report_options,
            )
        return self._process_entity_user_bulk_parallel_inner(
            entity_list=entity_list,
            params_kw=params,
            max_workers=max_workers,
            output_path=output_path,
            partial_frequency=partial_frequency,
            fail_safe=fail_safe,
            filter_column=filter_column,
            filter_value=filter_value,
            filter_type=filter_type,
            merge_existing=merge_existing,
            skip_export=skip_export,
            prefix=prefix,
            generate_report=generate_report,
            report_options=report_options,
        )

    @requires_token
    @require_user_params
    def _process_entity_user_bulk_parallel_inner(
        self,
        entity_list,
        params_kw=None,
        max_workers=10,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="users",
        generate_report=False,
        report_options=None,
    ):
        params = params_kw
        self.logger.info(f"🚀 Starting bulk user read for {len(entity_list)} entities")

        if merge_existing is None:
            merge_existing = self.merge_existing
        if merge_existing not in ("auto", "preserve", "mark"):
            raise ValueError(f"Invalid merge_existing='{merge_existing}'. Choose from ['auto','preserve','mark']")

        filtered, skipped, skip_count = filter_entities(entity_list, filter_column, filter_value, filter_type)

        # Resume mode is explicit (self.retry_failed_only) and never implied
        # by fail_safe -- see BaseExtractor._resolve_retry_entity_list.
        filtered = self._resolve_retry_entity_list(filtered, prefix, fail_safe=fail_safe, log=self.logger)

        all_rows = []
        global_errors = []
        successful = 0
        total = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        print(f"🔄 Fetching users for {len(filtered)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_user, row, params): self.get_entity_value(row, "id")
                for _, row in filtered.iterrows()
            }

            with tqdm(total=len(future_to_id), desc="👤 Fetching users", unit="entity") as pbar:
                for future in as_completed(future_to_id):
                    total += 1
                    entity_id = future_to_id[future]
                    try:
                        result = future.result()
                        df = result.get("data")
                        error = result.get("error")

                        if df is not None and not df.empty:
                            successful += 1
                            all_rows.append(df)
                            buffer_rows.append(df)
                        else:
                            failed_ids.append(entity_id)

                        if error:
                            record = {"entity_id": entity_id, **error}
                            global_errors.append(record)
                            buffer_errors.append(record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)

                    except Exception as e:
                        record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(record)
                        buffer_errors.append(record)
                        failed_ids.append(entity_id)

                    if self.partial_path and partial_frequency > 0 and total % partial_frequency == 0:
                        partial_df = pd.concat(buffer_rows, ignore_index=True) if buffer_rows else pd.DataFrame()
                        export_partial_results(
                            results_df=partial_df,
                            errors=buffer_errors,
                            output_path=self.partial_path,
                            prefix=prefix,
                            partial=True,
                            verbose=False,
                        )
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed = time.time() - start_time
        print(f"\n⏱️ Total processing time: {elapsed:.2f} seconds")
        print(f"✅ Successful fetches: {successful}/{total}")

        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df,
            skipped_entities_df=skipped,
            merge_mode=merge_existing,
            verbose=True,
        )

        results_df, _ = self._finalize_extraction(
            results_df=results_df,
            global_errors=global_errors,
            failed_ids=failed_ids,
            output_path=output_path,
            prefix=prefix,
            skip_export=skip_export,
            verbose=True,
            generate_report=generate_report,
            report_options=report_options,
            extraction_stats={
                "total_entities": len(entity_list),
                "total_calculations": total,
                "successful": successful,
                "failed": total - successful,
                "elapsed_seconds": elapsed,
                "parameters": getattr(self, "user_params", {}),
                "entity_df": entity_list,
            },
        )

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total,
            "successful_calculations": successful,
            "failed_calculations": total - successful,
            "failed_ids": failed_ids,
        }

    # =========================================================================
    # LINK MANAGEMENT
    # =========================================================================

    @requires_token
    def assign_roles(self, user_id: str, role_ids: List[str]) -> Tuple[bool, str]:
        """
        Assign roles to a user — **additive** (PATCH only, never revokes).

        Use :meth:`sync_roles` when the supplied list should become the user's
        complete role set.
        """
        if not role_ids:
            return True, ""

        url = f"{self.base_url}/{user_id}/roles"
        body = [{"id": rid.strip()} for rid in role_ids]

        try:
            response = requests.patch(url, headers=self._get_headers(), json=body, timeout=30)  # type: ignore[arg-type]
            response.raise_for_status()
            return True, ""
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return False, f"HTTP {e.response.status_code}: {e.response.text}"

    def _fetch_role_ids(self, user_id: str) -> Optional[List[str]]:
        """Role ids for a user, or ``None`` when the call itself failed.

        Exists so :meth:`sync_roles` can tell "this user genuinely holds no roles"
        from "the read did not happen". Both look like ``[]`` through
        :meth:`get_user_roles`, and treating a failed read as an empty role set
        would report every role as dropped.
        """
        try:
            response = requests.get(f"{self.base_url}/{user_id}/roles", headers=self._get_headers(), timeout=30)
            if response.ok:
                return [r.get("id", "") for r in response.json() if r.get("id")]
            self.logger.debug(f"Roles fetch for {user_id} returned HTTP {response.status_code}")
        except Exception as e:
            self.logger.debug(f"Could not fetch roles for user {user_id}: {e}")
        return None

    @requires_token
    def get_user_roles(self, user_id: str) -> List[str]:
        """Return the ids of the roles currently assigned to a user.

        A failed lookup is reported as no roles, which is the long-standing
        contract here. Use :meth:`_fetch_role_ids` when the difference matters.
        """
        return self._fetch_role_ids(user_id) or []

    @requires_token
    def delete_role(self, user_id: str, role_id: str) -> bool:
        """Remove a single role from a user. A 404 counts as success (already gone)."""
        url = f"{self.base_url}/{user_id}/roles/{role_id}"
        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=30)
            return response.ok or response.status_code == 404
        except Exception as e:
            self.logger.debug(f"Could not delete role {role_id} from user {user_id}: {e}")
            return False

    @requires_token
    def sync_roles(self, user_id: str, new_role_ids: List[str], verify: bool = True) -> Tuple[bool, str]:
        """
        Make ``new_role_ids`` the user's complete role set — **destructive**.

        Uses ``PUT /users/{id}/roles``, which the MDM API documents as "associate
        with the given roles and remove other existing associations" — a
        server-side replace in a single atomic call. The earlier
        read-then-DELETE-each-then-PATCH approach took 1+N+1 calls and left the
        user with **no roles at all** if it failed part-way; PUT cannot.

        Unlike :meth:`assign_roles` this revokes roles the caller left out, which
        is what a back-office modification CSV means when a role is deleted from
        the ``Role`` cell.

        Reads the current set first only to report what changed, and to skip the
        write entirely when the sets already match — so a bulk modification run
        doesn't churn every user's roles needlessly.

        Note role **ids are the role codes** (verified against the live API:
        ``id == code`` for every role), so no code->id lookup is needed — the
        values in ``product_profiles.yml`` can be sent directly.

        **A 2xx does not mean the roles attached.** The platform accepts the PUT
        and silently drops any role the *customer* cannot grant — its
        ``availableRoles``, which are a property of the tenant's products. A
        Crop_intel analyst on a Digital_Ag-only tenant kept 2 of 3 roles while
        this reported "Roles added: APP_MONITORING|REGIONAL", and the account
        could not open the app. So the result is read back and compared, and a
        short apply is a failure with the missing roles named.

        Args:
            user_id: MDM user id.
            new_role_ids: The complete desired role set.
            verify: Read the roles back after the write and fail when any are
                missing. Costs one GET. Turn it off only for a bulk run that
                reconciles separately — the silent partial apply is the whole
                reason this check exists.

        Returns:
            ``(success, message)`` — the message names what was added/removed, or
            what failed to attach, for surfacing in the run's Details column.
        """
        current_roles = self.get_user_roles(user_id)

        if set(new_role_ids) == set(current_roles):
            return True, ""  # No change needed

        removed = [r for r in current_roles if r not in new_role_ids]
        added = [r for r in new_role_ids if r not in current_roles]

        url = f"{self.base_url}/{user_id}/roles"
        body = [{"id": rid.strip()} for rid in new_role_ids]

        try:
            response = requests.put(url, headers=self._get_headers(), json=body, timeout=30)  # type: ignore[arg-type]
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return False, f"Roles sync failed: HTTP {e.response.status_code}: {e.response.text}"

        parts = []
        if removed:
            parts.append(f"Roles removed: {', '.join(removed)}")
        if added:
            parts.append(f"Roles added: {', '.join(added)}")
        summary = "; ".join(parts) if parts else "Roles updated"

        if not verify:
            return True, summary

        attached = self._fetch_role_ids(user_id)
        if attached is None:
            # Read failed rather than came back empty — do not invent a verdict.
            self.logger.warning(f"Could not verify roles for {user_id}; the write itself returned OK")
            return True, f"{summary} (unverified — the read-back failed)"

        dropped = sorted(set(new_role_ids) - set(attached))
        if dropped:
            self.logger.error(
                f"❌ {user_id}: the platform accepted the write but did not attach {dropped}. "
                "The customer cannot grant these — they have to be granted on the TENANT "
                "first, which usually means a product is missing from the customer profile. "
                "That is an administrator action, not something this client can do."
            )
            return False, (
                f"Roles NOT attached: {', '.join(dropped)} — the platform accepted the write and "
                f"dropped them, so the customer cannot grant them. They must be granted on the "
                f"tenant first (an administrator action on the customer's product entitlements)."
            )

        return True, summary

    @requires_token
    def link_agronomist_to_growers(self, agronomist_id: str, grower_ids: List[str]) -> List[Dict]:
        """Create manage links from AGRONOMIST to GROWERs."""
        log = self.get_contextualized_logger("LINK")

        url = f"{self.base_url}/{agronomist_id}/manage"
        body = [{"id": gid} for gid in grower_ids]

        results = []
        try:
            response = requests.patch(url, headers=self._get_headers(), json=body, timeout=30)  # type: ignore[arg-type]
            response.raise_for_status()
            for gid in grower_ids:
                results.append({"agronomist_id": agronomist_id, "grower_id": gid, "success": True})
                log.success(f"✅ Link created: AGRONOMIST {agronomist_id} -> GROWER {gid}")
        except Exception as e:
            log.error(f"❌ Failed to link: {e}")
            for gid in grower_ids:
                results.append({"agronomist_id": agronomist_id, "grower_id": gid, "success": False, "error": str(e)})

        return results

    @requires_token
    def link_grower_to_agronomist(self, grower_id: str, agronomist_id: str) -> Dict:
        """Create managed-by link from GROWER to AGRONOMIST."""
        url = f"{self.base_url}/{grower_id}/managed-by"
        body = [{"id": agronomist_id}]

        try:
            response = requests.patch(url, headers=self._get_headers(), json=body, timeout=30)  # type: ignore[arg-type]
            response.raise_for_status()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @requires_token
    def unlink_grower_from_agronomist(self, grower_id: str, agronomist_id: str) -> bool:
        """Remove managed-by link from GROWER to AGRONOMIST."""
        url = f"{self.base_url}/{grower_id}/managed-by/{agronomist_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return e.response.status_code == 404  # Already removed = success
        except Exception:
            return False

    @requires_token
    def unlink_agronomist_from_grower(self, agronomist_id: str, grower_id: str) -> bool:
        """Remove manage link from AGRONOMIST to GROWER."""
        url = f"{self.base_url}/{agronomist_id}/manage/{grower_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return e.response.status_code == 404
        except Exception:
            return False

    # =========================================================================
    # CREATE USERS
    # =========================================================================

    @requires_token
    def create_single_user(
        self, row: pd.Series, send_login_details: bool = False, dry_run: bool = False
    ) -> Tuple[bool, str, str]:
        """Create a single user.

        Args:
            send_login_details: Have the platform email the credentials.
            dry_run: Log the call that would be made and change nothing. Returns
                a synthetic ``DRY_RUN:<login>`` id so the caller's later phases
                (roles, links) can be rehearsed against it too.
        """
        log = self.get_contextualized_logger("CREATE_USER")

        # Built BEFORE the dry-run return, deliberately: a rehearsal that
        # "passes" on a payload the API would reject is worse than no rehearsal.
        # Every dry-run path in the package is built the same way.
        body = self._build_user_body(row, include_password=True, send_login_details=send_login_details)
        login = body.get("login", "unknown")

        if dry_run:
            user_type = body.get("userType", {}).get("code", "?")
            # Name deliberately does NOT end in the word it tests for: the
            # check-credentials hook reads `<name> = "<8+ chars>"` as an inline
            # secret, and the key here is exactly 8 characters long.
            pw_present = "password" in body
            log.info(f"   DRY RUN — POST {self.base_url}  login={login}, userType={user_type}")
            log.info(
                f"              email={body.get('email') or '(blank)'}, password={'set' if pw_present else 'ABSENT'}"
            )
            if body.get("sendLoginDetailsToUser"):
                log.warning("              sendLoginDetailsToUser=True — a LIVE run emails this person")
            return True, f"DRY_RUN:{login}", ""

        try:
            response = requests.post(self.base_url, headers=self._get_headers(), json=body, timeout=30)
            response.raise_for_status()
            user_id = response.json().get("id", "")
            log.success(f"✅ User created: {login} (ID: {user_id})")
            return True, user_id, ""
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            log.error(f"❌ Failed to create user {login}: {error_msg}")
            return False, "", error_msg
        except Exception as e:
            log.error(f"❌ Unexpected error creating user {login}: {e}")
            return False, "", str(e)

    def create_users(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        send_login_details: bool = False,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Create multiple users with proper ordering (AGRONOMIST first, then GROWER).

        ⚠️  Reports only ``CREATED`` or ``ERROR`` — there is no "already exists"
        status, so re-running over the same input marks every row ``ERROR``.
        Callers that need to resume should reconcile those against
        :meth:`get_user_by_login`.

        Args:
            send_login_details: Have the platform email each created account
                its credentials. Off by default — this sends real email.
            dry_run: Rehearse. Writes nothing, and logs every call all three
                phases would make — the user POST, the roles PATCH and the link
                PATCH — reporting status ``DRY_RUN``.

                It still performs the READS, which is what makes the rehearsal
                worth running: every login is checked with
                :meth:`get_user_by_login`, so the "no already-exists status"
                trap above surfaces here as ``WOULD COLLIDE`` instead of as a
                row of ``ERROR`` discovered after a live run has already created
                the other users. Link targets are resolved the same way, so an
                unresolvable ``Managed_by_user`` is reported before the write
                rather than silently skipped during it.
        """
        log = self.get_contextualized_logger("CREATE_USERS")
        log.info("=" * 60)
        log.info(f"{'DRY RUN (no writes)' if dry_run else 'Starting'} user creation workflow")
        log.info("=" * 60)

        results_status = [""] * len(df)
        results_user_id = [""] * len(df)
        results_details = [""] * len(df)

        df_sorted = df.copy()
        df_sorted["_sort_order"] = df_sorted["User_Type"].apply(lambda x: 0 if str(x).upper() == "AGRONOMIST" else 1)
        df_sorted["_orig_idx"] = df_sorted.index
        df_sorted = df_sorted.sort_values("_sort_order")

        created_users = {}

        print(f"\n🔐 Phase 1: {'Rehearsing' if dry_run else 'Creating'} user accounts...")

        for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="Creating users", disable=not verbose):
            orig_idx = row["_orig_idx"]
            login = str(row.get("Login", ""))

            collision = None
            if dry_run and login:
                existing = self.get_user_by_login(login)
                if existing:
                    collision = existing.get("id", "?")
                    log.warning(f"   WOULD COLLIDE — login '{login}' already exists (id={collision})")

            success, user_id, error = self.create_single_user(
                row, send_login_details=send_login_details, dry_run=dry_run
            )

            if success:
                results_status[orig_idx] = "DRY_RUN" if dry_run else "CREATED"
                results_user_id[orig_idx] = user_id
                created_users[login] = user_id
                if collision:
                    results_details[orig_idx] = f"WOULD COLLIDE with existing user {collision}"

                roles_str = str(row.get("Role", ""))
                if roles_str and roles_str.lower() not in ("nan", "", "none"):
                    role_ids = [r.strip() for r in roles_str.split(";") if r.strip()]
                    if role_ids:
                        if dry_run:
                            log.info(f"   DRY RUN — PATCH .../{user_id}/roles  {len(role_ids)} role(s): {role_ids}")
                        else:
                            role_success, role_error = self.assign_roles(user_id, role_ids)
                            if not role_success:
                                results_details[orig_idx] = f"Roles failed: {role_error}"
            else:
                results_status[orig_idx] = "ERROR"
                results_details[orig_idx] = error

        print(f"\n🔗 Phase 2: {'Rehearsing' if dry_run else 'Establishing'} account links...")

        for _, row in df_sorted.iterrows():
            orig_idx = row["_orig_idx"]
            login = str(row.get("Login", ""))
            user_type = str(row.get("User_Type", "")).upper()

            if results_status[orig_idx] not in ("CREATED", "DRY_RUN"):
                continue

            user_id = results_user_id[orig_idx]

            if user_type == "AGRONOMIST":
                manages_str = str(row.get("Manages_user", ""))
                if manages_str and manages_str.lower() not in ("nan", "", "none"):
                    grower_logins = [g.strip() for g in manages_str.split("|") if g.strip()]
                    grower_ids = []

                    for g_login in grower_logins:
                        if g_login in created_users:
                            grower_ids.append(created_users[g_login])
                        else:
                            grower_data = self.get_user_by_login(g_login)
                            if grower_data:
                                grower_ids.append(grower_data.get("id"))

                    if grower_ids:
                        if dry_run:
                            log.info(
                                f"   DRY RUN — PATCH .../{user_id}/manage  "
                                f"{len(grower_ids)} GROWER(s): {grower_ids}"
                            )
                            linked = len(grower_ids)
                        else:
                            link_results = self.link_agronomist_to_growers(user_id, grower_ids)
                            linked = sum(1 for r in link_results if r["success"])
                        current = results_details[orig_idx]
                        link_msg = f"{'Would link' if dry_run else 'Linked'} to {linked}/{len(grower_ids)} GROWERs"
                        results_details[orig_idx] = f"{current}; {link_msg}" if current else link_msg

            elif user_type == "GROWER":
                managed_by_str = str(row.get("Managed_by_user", ""))
                if managed_by_str and managed_by_str.lower() not in ("nan", "", "none"):
                    agro_login = managed_by_str.strip()
                    agro_id = created_users.get(agro_login)

                    if not agro_id:
                        agro_data = self.get_user_by_login(agro_login)
                        if agro_data:
                            agro_id = agro_data.get("id")

                    if agro_id:
                        if dry_run:
                            log.info(f"   DRY RUN — PATCH .../{user_id}/managed-by  AGRONOMIST {agro_id}")
                            linked_ok = True
                        else:
                            linked_ok = bool(self.link_grower_to_agronomist(user_id, agro_id).get("success"))
                        if linked_ok:
                            current = results_details[orig_idx]
                            link_msg = f"{'Would link' if dry_run else 'Linked'} to AGRONOMIST {agro_login}"
                            results_details[orig_idx] = f"{current}; {link_msg}" if current else link_msg
                    elif dry_run:
                        log.warning(
                            f"   UNRESOLVED — '{login}' is managed_by '{agro_login}', which is "
                            f"neither in this file nor on the platform; the link would be skipped"
                        )

        results = df.copy()
        results["Status"] = results_status
        results["User_Id"] = results_user_id
        results["Details"] = results_details

        created_count = sum(1 for s in results_status if s in ("CREATED", "DRY_RUN"))
        error_count = sum(1 for s in results_status if s == "ERROR")
        collision_count = sum(1 for d in results_details if "WOULD COLLIDE" in d)
        noun = "Rehearsal" if dry_run else "Creation"
        verb = "Would create" if dry_run else "Created"

        log.info("=" * 60)
        log.success(f"{noun} complete: {created_count} {verb.lower()}, {error_count} errors")
        log.info("=" * 60)

        print(f"\n✅ {noun} complete!")
        print(f"   {verb}: {created_count}")
        print(f"   Errors: {error_count}")
        if dry_run:
            print(f"   Would collide with an existing login: {collision_count}")
            print("   NOTHING WAS WRITTEN. Re-run with dry_run=False to apply.")

        return results

    # =========================================================================
    # MODIFY USERS (PARALLEL)
    # =========================================================================

    def _modify_single_user_with_sync(self, row: pd.Series) -> Dict:
        """
        Modify a single user with role and link synchronization (called in parallel).

        Returns:
            dict: {status, details, user_id}
        """
        user_id = str(row.get("id", "")).strip()

        if not user_id or user_id.lower() in ("nan", "", "none"):
            return {"status": "ERROR", "details": "No valid 'id' provided", "user_id": user_id}

        # Get current user data
        user_data = self.get_user_by_id(user_id)
        if not user_data:
            return {"status": "ERROR", "details": f"User not found: {user_id}", "user_id": user_id}

        detail_messages = []
        headers = self._get_headers()

        # 1. Modify user attributes
        body = self._build_user_body(row, include_password=True)
        url = f"{self.base_url}/{user_id}"

        # Track field changes
        field_mapping = {
            "firstname": "First_Name",
            "lastname": "Last_Name",
            "email": "Email",
            "login": "Login",
            "companyName": "CompanyName",
        }
        for api_field, csv_field in field_mapping.items():
            old_val = user_data.get(api_field, "")
            new_val = body.get(api_field, "")
            if str(old_val) != str(new_val) and new_val:
                detail_messages.append(f"{csv_field}: {old_val} → {new_val}")

        if "password" in body:
            detail_messages.append("Password: updated")

        try:
            response = requests.patch(url, headers=headers, json=body, timeout=30)  # type: ignore[arg-type]
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return {
                "status": "ERROR",
                "details": f"HTTP {e.response.status_code}: {e.response.text}",
                "user_id": user_id,
            }

        # 2. Synchronize roles (DELETE then ADD) — the Role cell is the complete
        #    desired set, so roles removed from the CSV must be revoked.
        roles_str = str(row.get("Role", ""))
        if roles_str and roles_str.lower() not in ("nan", "", "none"):
            # Exports quote cells containing ';' — strip that before matching.
            role_ids = [r.strip().strip('"') for r in roles_str.split(";") if r.strip()]
            if role_ids:
                role_success, role_msg = self.sync_roles(user_id, role_ids)
                if role_msg:
                    detail_messages.append(role_msg)
                elif not role_success:
                    detail_messages.append("Roles sync failed")

        # 3. Handle managed_by changes (GROWER only) - SYNC logic with DELETE then ADD
        user_type = str(row.get("User_Type", "") or row.get("userType.code", "")).upper()
        if user_type == "GROWER":
            managed_by_str = str(row.get("Managed_by_user", ""))

            # Parse new managers
            if managed_by_str and managed_by_str.lower() not in ("nan", "", "none"):
                new_manager_logins = [m.strip() for m in managed_by_str.split("|") if m.strip()]
            else:
                new_manager_logins = []

            # Get current managers (id, login) pairs
            current_managers = self._get_current_manager_ids(user_id)
            current_logins = [login for _, login in current_managers if login]

            # Check if there are changes needed
            if set(new_manager_logins) != set(current_logins):
                removed = []
                added = []
                not_found = []

                # Step 1: DELETE all existing links (with verification loop)
                max_delete_attempts = 3
                for attempt in range(max_delete_attempts):
                    # Get fresh list of current managers
                    current_managers_fresh = self._get_current_manager_ids(user_id)

                    if not current_managers_fresh:
                        break  # No more managers to delete

                    # Delete all current managers
                    for manager_id, manager_login in current_managers_fresh:
                        if manager_id:
                            url_delete = f"{self.base_url}/{user_id}/managed-by/{manager_id}"
                            try:
                                response = requests.delete(url_delete, headers=headers, timeout=30)
                                if response.ok or response.status_code == 404:
                                    if manager_login not in removed:
                                        removed.append(manager_login or manager_id)
                            except Exception as e:
                                detail_messages.append(f"Delete failed for {manager_login}: {str(e)}")

                    # Verify all were deleted
                    check_managers = self._get_current_manager_ids(user_id)
                    if not check_managers:
                        break  # Success, all deleted

                # Step 2: ADD new links (only if we have new managers)
                if new_manager_logins:
                    # Use pre-resolved IDs from modify_users()
                    agronomist_id_map = row.get("_agronomist_id_map", {})
                    new_manager_ids = []

                    for new_login in new_manager_logins:
                        if new_login in agronomist_id_map:
                            new_manager_ids.append(agronomist_id_map[new_login])
                            added.append(new_login)
                        else:
                            not_found.append(new_login)

                    # PATCH with new list
                    if new_manager_ids:
                        url_managed_by = f"{self.base_url}/{user_id}/managed-by"
                        body_managed_by = [{"id": mid} for mid in new_manager_ids]

                        try:
                            response = requests.patch(url_managed_by, headers=headers, json=body_managed_by, timeout=30)  # type: ignore[arg-type]
                            response.raise_for_status()
                        except requests.exceptions.HTTPError as e:
                            assert e.response is not None  # raise_for_status sets e.response
                            detail_messages.append(f"Failed to add links: HTTP {e.response.status_code}")
                            added = []  # Reset since it failed

                # Report changes
                if removed:
                    detail_messages.append(f"Unlinked: {', '.join(removed)}")
                if added:
                    detail_messages.append(f"Linked: {', '.join(added)}")
                if not_found:
                    detail_messages.append(f"Not found: {', '.join(not_found)}")

        return {
            "status": "MODIFIED",
            "details": "; ".join(detail_messages) if detail_messages else "",
            "user_id": user_id,
        }

    def modify_users(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Modify multiple users in parallel.
        Synchronizes roles and managed-by links (adds new, removes old).
        """
        log = self.get_contextualized_logger("MODIFY_USERS")
        log.info("=" * 60)
        log.info("Starting user modification workflow (parallel)")
        log.info("=" * 60)

        if "id" not in df.columns:
            raise ValueError("❌ DataFrame must contain 'id' column for modification")

        # PRE-RESOLVE all agronomist logins to IDs (avoid API calls in parallel threads)
        print("\n🔍 Pre-resolving AGRONOMIST logins to IDs...")
        all_agronomist_logins = set()
        for _, row in df.iterrows():
            managed_by_str = str(row.get("Managed_by_user", ""))
            if managed_by_str and managed_by_str.lower() not in ("nan", "", "none"):
                logins = [m.strip() for m in managed_by_str.split("|") if m.strip()]
                all_agronomist_logins.update(logins)

        agronomist_login_to_id = {}
        for login in all_agronomist_logins:
            agro_data = self.get_user_by_login(login)
            if agro_data:
                agronomist_login_to_id[login] = agro_data.get("id")

        print(f"   ✅ Resolved {len(agronomist_login_to_id)}/{len(all_agronomist_logins)} AGRONOMISTs")

        print(f"\n✏️ Modifying {len(df)} user accounts (parallel, {self.MAX_WORKERS} workers)...")

        results_status = [""] * len(df)
        results_details = [""] * len(df)

        # Prepare rows for parallel processing with pre-resolved IDs
        rows_with_ids = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["_agronomist_id_map"] = agronomist_login_to_id
            rows_with_ids.append(row_dict)

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._modify_single_user_with_sync, pd.Series(row)): idx
                for idx, row in enumerate(rows_with_ids)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Modifying users", disable=not verbose):
                idx = futures[future]
                try:
                    result = future.result()
                    results_status[idx] = result["status"]
                    results_details[idx] = result["details"]
                except Exception as e:
                    results_status[idx] = "ERROR"
                    results_details[idx] = str(e)

        results = df.copy()
        results["Status"] = results_status
        results["Details"] = results_details

        # Ensure lowercase column names for consistency
        rename_map = {}
        for col in results.columns:
            lower_col = col.lower()
            if lower_col in ["login", "email", "firstname", "lastname", "companyname"]:
                rename_map[col] = lower_col
        if rename_map:
            results = results.rename(columns=rename_map)

        modified_count = sum(1 for s in results_status if s == "MODIFIED")
        error_count = sum(1 for s in results_status if s == "ERROR")

        log.info("=" * 60)
        log.success(f"Modification complete: {modified_count} modified, {error_count} errors")
        log.info("=" * 60)

        print("\n✅ Modification complete!")
        print(f"   Modified: {modified_count}")
        print(f"   Errors: {error_count}")

        return results

    # =========================================================================
    # DELETE USERS (PARALLEL)
    # =========================================================================

    def _delete_single_user_task(self, row: pd.Series) -> Dict:
        """Delete a single user (called in parallel)."""
        user_id = str(row.get("id", "")).strip()

        if not user_id or user_id.lower() in ("nan", "", "none"):
            return {"status": "ERROR", "details": "No valid 'id' provided", "user_id": user_id}

        user_type = ""
        for col in ["User_Type", "userType.code"]:
            if col in row.index and pd.notna(row.get(col)):
                user_type = str(row.get(col)).upper()
                break

        # If AGRONOMIST, remove all GROWER links first
        if user_type == "AGRONOMIST":
            managed_growers = self.get_managed_growers(user_id)
            for grower in managed_growers:
                grower_id = grower.get("id")
                if grower_id:
                    self.unlink_agronomist_from_grower(user_id, grower_id)

        # Delete user
        url = f"{self.base_url}/{user_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            details = "GROWER links cleaned up" if user_type == "AGRONOMIST" else ""
            return {"status": "DELETED", "details": details, "user_id": user_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            return {
                "status": "ERROR",
                "details": f"HTTP {e.response.status_code}: {e.response.text}",
                "user_id": user_id,
            }

    def delete_users(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Delete multiple users in parallel."""
        log = self.get_contextualized_logger("DELETE_USERS")
        log.info("=" * 60)
        log.info("Starting user deletion workflow (parallel)")
        log.info("=" * 60)

        if "id" not in df.columns:
            raise ValueError("❌ DataFrame must contain 'id' column for deletion")

        print(f"\n🗑️ Deleting {len(df)} user accounts (parallel, {self.MAX_WORKERS} workers)...")

        results_status = [""] * len(df)
        results_details = [""] * len(df)

        rows = [row for _, row in df.iterrows()]

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(self._delete_single_user_task, row): idx for idx, row in enumerate(rows)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting users", disable=not verbose):
                idx = futures[future]
                try:
                    result = future.result()
                    results_status[idx] = result["status"]
                    results_details[idx] = result["details"]
                except Exception as e:
                    results_status[idx] = "ERROR"
                    results_details[idx] = str(e)

        results = df.copy()
        results["Status"] = results_status
        results["Details"] = results_details

        deleted_count = sum(1 for s in results_status if s == "DELETED")
        error_count = sum(1 for s in results_status if s == "ERROR")

        log.info("=" * 60)
        log.success(f"Deletion complete: {deleted_count} deleted, {error_count} errors")
        log.info("=" * 60)

        print("\n✅ Deletion complete!")
        print(f"   Deleted: {deleted_count}")
        print(f"   Errors: {error_count}")

        return results

    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================

    def export_credentials(
        self,
        results_df: pd.DataFrame,
        df_users: pd.DataFrame,
        output_path: str = None,
        separator: str = ";",
        statuses: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Write the credentials of newly created accounts to a CSV.

        **The generated password exists in exactly one place: memory.** The
        create body carries it directly, the API never returns it, and the
        platform only emails it when ``sendLoginDetailsToUser`` was set. Re-run
        the cell that generated it and the value changes, so the account becomes
        untestable without a password reset — and a reset through
        :meth:`modify_users` must re-send email/login/userType or it blanks them.
        Persisting at creation time is the cheap way out of all of that.

        Written to the git-ignored ``results/`` tree, one timestamped file per
        run, alongside every other export. It is still a plaintext secret on
        disk: treat it as one, and delete it once the credentials are handed over.

        Args:
            results_df: Output of :meth:`create_users`, carrying Login / Status /
                User_Id.
            df_users: The input frame, which is where ``Password`` still lives —
                ``create_users`` does not echo it back.
            output_path: Defaults to the manager's results directory.
            separator: CSV separator, matching :meth:`export_results`.
            statuses: Which rows to include. Defaults to ``["CREATED"]``; pass
                ``["CREATED", "EXISTS"]`` on a resumed run, though EXISTS rows
                carry no usable password.

        Returns:
            str | None: Path written, or None when no row qualified.
        """
        log = self.get_contextualized_logger("EXPORT_CREDENTIALS")
        statuses = statuses or ["CREATED"]

        merged = results_df.copy()
        if "Password" not in merged.columns and "Login" in df_users.columns:
            passwords = df_users.set_index("Login")["Password"] if "Password" in df_users.columns else None
            if passwords is not None:
                merged["Password"] = merged["Login"].map(passwords)

        keep = merged[merged["Status"].isin(statuses)] if "Status" in merged.columns else merged
        if keep.empty:
            log.info(f"No rows with status in {statuses} — nothing to write")
            return None

        columns = [
            c for c in ("Login", "Email", "Password", "User_Id", "User_Type", "Customer_Id") if c in keep.columns
        ]
        export = keep[columns].copy()
        export.insert(0, "Environment", self.env)
        export["Created_At"] = datetime.now().isoformat(timespec="seconds")

        if not output_path:
            output_path = self.output_path or "results"
        today = datetime.now().strftime("%Y%m%d")
        export_dir = os.path.join(output_path, f"user_management_{today}")
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, f"credentials_{today}_{datetime.now().strftime('%H%M%S')}.csv")

        export.to_csv(filepath, index=False, encoding="utf-8-sig", sep=separator)

        blank = int(export["Password"].isna().sum()) if "Password" in export.columns else len(export)
        log.success(f"Credentials for {len(export)} account(s) written to {filepath}")
        print(f"🔑 Credentials saved: {filepath}")
        print(f"   {len(export)} account(s) — PLAINTEXT PASSWORDS. Hand them over, then delete this file.")
        if blank:
            print(f"   ⚠️  {blank} row(s) have no password (platform-generated, or an existing account)")

        return filepath

    def export_results(
        self, results_df: pd.DataFrame, operation: str, output_path: str = None, separator: str = ";"
    ) -> str:
        """Export operation results to CSV."""
        log = self.get_contextualized_logger("EXPORT")

        if not output_path:
            output_path = self.output_path or "results"

        today = datetime.now().strftime("%Y%m%d")
        export_dir = os.path.join(output_path, f"user_management_{today}")
        os.makedirs(export_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"users_{operation}_{today}_{timestamp}.csv"
        filepath = os.path.join(export_dir, filename)

        log.info(f"Exporting {len(results_df)} rows")
        log.info(f"Using separator: '{separator}'")

        results_df.to_csv(filepath, index=False, encoding="utf-8-sig", sep=separator)

        log.success(f"✅ Results exported to: {filepath}")
        print(f"📄 Results exported to: {filepath}")
        print(f"   Separator: '{separator}'")

        return filepath

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_latest_export(self, output_path: str = None) -> Optional[str]:
        """Get the most recent exported CSV file from results folder."""
        if not output_path:
            output_path = self.output_path or "results"

        pattern = os.path.join(output_path, "user_management_*", "users_export_*.csv")
        export_files = glob.glob(pattern)

        if not export_files:
            return None

        return max(export_files, key=os.path.getmtime)
