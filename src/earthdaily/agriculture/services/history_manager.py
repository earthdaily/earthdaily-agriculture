"""
History API service — store and retrieve per-user solution configuration.

Wraps the EarthDaily Agro History API (``/accounts/history/v1/history-entries``),
which the platform uses to persist per-user UI state and analytic definitions.

``type`` is a free string on every method — this class does not restrict it, so
any entry type the API accepts can be read or written. The four seen in account
provisioning, all confirmed against live captures:

    - ``CUSTOM_ANALYTIC_UNIQUEAPP``      — user-defined analytic configurations
    - ``COCKPIT_UNIQUEAPP``              — dashboard layout (views + widgets)
    - ``COCKPIT_UNIQUEAPP_ACTIVE``       — which view opens first, ``{"code": ...}``
    - ``MONITORING_FAVORITES_UNIQUEAPP`` — saved map snapshots

The last two were previously described here as "out of scope, owned by other
surfaces". That was wrong for provisioning and is corrected: a cockpit built
from ``FAVORITE_MAP_WIDGET`` entries is unusable without the favourites it
references by id, and an account with no active-view pointer opens on no view.
Seeding an account writes all four.

Note ``COCKPIT_UNIQUEAPP_ACTIVE`` carries ``element_id="COCKPIT_UNIQUEAPP"`` —
the layout's element id, NOT its own type name. Easy to get wrong.

Design notes:

- **No DELETE endpoint, but POST upserts.** "Update an existing config" is
  "POST the same ``(userid, type, elementId)`` triple again" — the server
  replaces that row's ``data`` in place and keeps its ``id`` (verified preprod
  2026-08-05). Seeding is therefore idempotent. What you *cannot* undo is a row
  written under the wrong ``element_id``: it becomes an orphan no call can
  remove.
- **``data`` is opaque.** Different types carry completely different
  payload shapes; this library passes ``data`` through unmodified. Consumers
  index it directly (e.g. ``entry["data"]["configurations"]``).
- **Wire-vs-Python casing.** The wire payload mixes camelCase (``elementId``)
  with all-lowercase (``userid``). All Python-side fields use snake_case via
  a single conversion shim — see ``_envelope_from_wire`` / ``_envelope_to_wire``.
- **``userid`` is NOT the user id the rest of this codebase uses.** It is
  ``externalIds.id``, not MDM's legacy ``users.id``. Both the read filter and
  the write body take it. ``resolve_user_id`` does the translation and every
  method calls it, so callers keep passing the id they already have — see that
  method for the evidence and for why getting it wrong is quiet rather than loud.
- **Query filters are PascalCase** (``UserId`` / ``Type`` / ``ElementId`` /
  ``Id``), not ``$``-prefixed. Only pagination controls take the ``$``. The
  server ignores unknown parameters silently, so the wrong form reads as "no
  filter" rather than as an error.

Documentation: https://api.geosys-na.net/accounts/history/v1/swagger
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, TypedDict

import pandas as pd
import requests

from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.base_extractor import BaseExtractor, requires_token
from earthdaily.agriculture.core.identity import EDAuthenticator

# ---------------------------------------------------------------------------
# Typed envelope (data stays opaque; this types only the surrounding fields)
# ---------------------------------------------------------------------------


class HistoryEntry(TypedDict, total=False):
    """Lightweight typed envelope for one history-entry row.

    ``data`` is left as a free-form dict — see module docstring. All fields
    are optional because the API may return ``null`` for any of them.
    """

    id: Optional[str]
    type: Optional[str]
    element_id: Optional[str]
    user_id: Optional[str]
    data: dict[str, Any]


# Centralised so the camelCase / lowercase wire quirks don't leak everywhere.
_WIRE_TO_PYTHON = {
    "id": "id",
    "type": "type",
    "elementId": "element_id",
    "userid": "user_id",
    "data": "data",
}
_PYTHON_TO_WIRE = {v: k for k, v in _WIRE_TO_PYTHON.items()}


def _envelope_from_wire(raw: dict[str, Any]) -> HistoryEntry:
    """Convert a server-side dict into a snake_case ``HistoryEntry``."""
    entry: HistoryEntry = {}
    for wire_key, py_key in _WIRE_TO_PYTHON.items():
        if wire_key in raw:
            entry[py_key] = raw[wire_key]  # type: ignore[literal-required]
    if entry.get("data") is None:
        entry["data"] = {}
    return entry


def _envelope_to_wire(entry: HistoryEntry) -> dict[str, Any]:
    """Convert a Python-side ``HistoryEntry`` back to the wire shape.

    ``None`` values are omitted (the API's POST body schema treats every
    field as nullable, so absent == null).
    """
    out: dict[str, Any] = {}
    for py_key, wire_key in _PYTHON_TO_WIRE.items():
        value = entry.get(py_key)  # type: ignore[literal-required]
        if value is not None:
            out[wire_key] = value
    return out


# ---------------------------------------------------------------------------
# List subclass with .to_dataframe() helper
# ---------------------------------------------------------------------------


class HistoryEntryList(list):
    """Behaves like ``list[HistoryEntry]`` plus a ``.to_dataframe()`` helper."""

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten envelope into a DataFrame; ``data`` stays nested.

        Columns: ``id``, ``type``, ``element_id``, ``user_id``, ``data``.
        The ``data`` column is an object column holding each entry's
        per-type payload dict (not exploded).
        """
        rows = [
            {
                "id": e.get("id"),
                "type": e.get("type"),
                "element_id": e.get("element_id"),
                "user_id": e.get("user_id"),
                "data": e.get("data", {}),
            }
            for e in self
        ]
        return pd.DataFrame(rows, columns=["id", "type", "element_id", "user_id", "data"])


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class HistoryManager(BaseExtractor):
    """Manage per-user solution configuration via the History API.

    Documentation: https://api.geosys-na.net/accounts/history/v1/swagger

    Supported operations:
        - List history entries with filters (``UserId`` / ``Type`` / ``ElementId`` / ``Id``)
        - Get a single entry by id
        - Count entries via HEAD
        - Create a new entry (POST — also used for "updates" since the API has no DELETE)
        - ``get_user_entry`` / ``save_user_entry`` convenience methods for the
          "one config per (user, type, element_id)" invariant
    """

    DEFAULT_LIMIT = 100
    REQUEST_TIMEOUT = 30

    #: An ``externalIds.id`` is 22 characters of base62; a legacy NA id is ~7
    #: lowercase alphanumerics. Used to tell the two apart without a round-trip
    #: when a caller already passes the modern form.
    _EXTERNAL_ID_RE = re.compile(r"^[0-9A-Za-z]{20,24}$")

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)
        self.base_url = agro_urls["history_url"][self.env].rstrip("/")
        self.mdm_users_url = agro_urls["eda_data_management_url_fields"][self.env].rstrip("/") + "/users"
        # legacy id -> externalIds.id. One lookup per user per session; the
        # mapping cannot change under us.
        self._external_id_cache: dict[str, str] = {}
        self.logger.info(f"📜 HistoryManager initialized for env: {self.env}")
        self.logger.debug(f"Base URL: {self.base_url}")

    @requires_token
    def resolve_user_id(self, user_id: str) -> str:
        """Translate an MDM user id into the id the History API means by ``UserId``.

        **This API does not accept the user id everything else uses.** MDM's
        ``users.id`` (e.g. ``kxrx932``) is the *legacy* NA identifier — it comes
        back from ``UserManager.create_users`` and is what the rest of this
        codebase passes around. The History API rejects it outright::

            POST /history-entries  {"userid": "kxrx932", ...}
            404 {"code": "not_found_error",
                 "message": "'UserId' must refer to an existing user. You entered kxrx932."}

        It wants ``externalIds.id`` — the modern base62 identifier, which MDM
        only returns when explicitly selected::

            GET /users?Login=…&$fields=id,externalids
            {"id": "kxrx932",
             "externalIds": {"legacY_ID_NA": "kxrx932", "id": "1RmKq5IAZyuv3vanzjixhX"}}

        Verified on preprod 2026-08-05: the legacy id 404s for **every** user
        including the caller's own, while the external id returns 201 and the
        entry reads back under ``UserId=<externalIds.id>``.

        ⚠️  Getting this wrong does not fail loudly in the obvious place. Drop
        ``userid`` from the body and the API happily writes the entry to the
        *calling* account instead — so a provisioning run would silently seed
        the admin's own cockpit rather than the new user's.

        Args:
            user_id: A legacy MDM id or an ``externalIds.id``. The modern form is
                detected and returned unchanged, so this is safe to call twice.

        Returns:
            str: The ``externalIds.id``.

        Raises:
            ValueError: No user matches, or the user has no ``externalIds.id``.
        """
        if not user_id:
            raise ValueError("❌ user_id is required")
        if user_id in self._external_id_cache:
            return self._external_id_cache[user_id]
        if self._EXTERNAL_ID_RE.match(user_id):
            return user_id  # already the modern form

        resp = requests.get(
            self.mdm_users_url,
            params={"Id": user_id, "$limit": "1", "$fields": "id,login,externalids"},
            headers=self._auth_headers(),
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json() or []
        if not rows:
            raise ValueError(
                f"❌ No user with id '{user_id}' — cannot resolve the externalIds.id the "
                f"History API requires. Check the id came from a successful user creation."
            )

        external = ((rows[0].get("externalIds") or {}).get("id")) or ""
        if not external:
            raise ValueError(
                f"❌ User '{user_id}' has no externalIds.id, which the History API requires "
                f"as its 'userid'. Writing without one would silently target the CALLING "
                f"account instead — refusing."
            )

        self._external_id_cache[user_id] = external
        self.logger.debug(f"Resolved user id {user_id} -> externalIds.id {external}")
        return external

    def get_new_token(self):
        """Refresh token using EDAuthenticator."""
        self.logger.debug("Refreshing token for HistoryManager")
        return EDAuthenticator.get_new_token(env=self.env)

    # =====================================================================
    # Raw passthroughs — 1:1 with the OpenAPI endpoints
    # =====================================================================

    @requires_token
    def list_entries(
        self,
        *,
        user_id: Optional[str] = None,
        type: Optional[str] = None,
        element_id: Optional[str] = None,
        entry_id: Optional[str] = None,
        fields: Optional[Iterable[str]] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        sort: Optional[str] = None,
        filter_expr: Optional[str] = None,
    ) -> HistoryEntryList:
        """``GET /history-entries`` — list entries with filters.

        Named params (``user_id`` / ``type`` / ``element_id`` / ``entry_id``)
        map to the API's ``UserId`` / ``Type`` / ``ElementId`` / ``Id`` query
        parameters, exactly as the OpenAPI documents them.

        ⚠️  These were previously sent ``$``-prefixed and kebab-cased
        (``$user-id``, ``$type``, ``$element-id``), on the theory that they
        followed ``$offset`` / ``$limit`` / ``$count``. They do not, and the
        server **silently ignores unknown parameters** — so every "filtered"
        call was returning the caller's unfiltered first page. That made
        :meth:`save_user_entry` warn about a pre-existing entry on every single
        write, for entry types the user had never had. Only ``$``-prefixed
        *pagination* controls are real; the four filters are PascalCase.
        Verified on preprod 2026-08-05.

        ``user_id`` is resolved through :meth:`resolve_user_id`, so a legacy MDM
        id works here even though the API only matches ``externalIds.id``.
        ``filter_expr`` is the escape hatch for the generic ``$filter``
        expression — use only when the named params can't express what you need.
        """
        params: dict[str, Any] = {"$offset": offset, "$limit": str(limit)}
        if sort:
            params["$sort"] = sort
        if filter_expr:
            params["$filter"] = filter_expr
        if fields:
            params["$fields"] = ",".join(fields)
        if entry_id:
            params["Id"] = entry_id
        if type:
            params["Type"] = type
        if element_id:
            params["ElementId"] = element_id
        if user_id:
            params["UserId"] = self.resolve_user_id(user_id)

        resp = requests.get(
            f"{self.base_url}/history-entries",
            params=params,
            headers=self._auth_headers(),
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return HistoryEntryList(_envelope_from_wire(r) for r in (resp.json() or []))

    @requires_token
    def get_entry(self, entry_id: str, *, fields: Optional[Iterable[str]] = None) -> HistoryEntry:
        """``GET /history-entries/{id}`` — fetch a single entry by GUID."""
        params: dict[str, Any] = {}
        if fields:
            params["$fields"] = ",".join(fields)
        resp = requests.get(
            f"{self.base_url}/history-entries/{entry_id}",
            params=params or None,
            headers=self._auth_headers(),
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _envelope_from_wire(resp.json() or {})

    @requires_token
    def count_entries(
        self,
        *,
        user_id: Optional[str] = None,
        type: Optional[str] = None,
        element_id: Optional[str] = None,
        filter_expr: Optional[str] = None,
    ) -> int:
        """``HEAD /history-entries`` — return total count via response header.

        Cheap counter; doesn't return any rows. Reads the count from
        ``X-Total-Count`` (the standard convention for this server family).

        Filters are PascalCase, same as :meth:`list_entries` — an unknown
        parameter is ignored, which would silently turn a filtered count into a
        total.
        """
        params: dict[str, Any] = {"$count": "true"}
        if filter_expr:
            params["$filter"] = filter_expr
        if type:
            params["Type"] = type
        if element_id:
            params["ElementId"] = element_id
        if user_id:
            params["UserId"] = self.resolve_user_id(user_id)

        resp = requests.head(
            f"{self.base_url}/history-entries",
            params=params,
            headers=self._auth_headers(),
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return int(resp.headers.get("X-Total-Count", 0))

    @requires_token
    def create_entry(
        self,
        *,
        type: str,
        data: dict[str, Any],
        element_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> HistoryEntry:
        """``POST /history-entries`` — append a new history entry.

        Body shape: ``{type, data}`` minimum; ``elementId`` and ``userid``
        are added only when the matching kwargs are passed. ``id`` is
        never in the body — the server generates it on success.

        ⚠️  **Omitting ``user_id`` writes to the CALLING account.** That is the
        right default for "save my own config" and completely wrong for
        provisioning: it seeds the admin's cockpit instead of the new user's,
        with a 201 and no hint that anything went astray. Pass ``user_id``
        whenever the entry belongs to someone else. It is resolved through
        :meth:`resolve_user_id`, because the API matches only
        ``externalIds.id`` and 404s on the legacy MDM id.

        The API has no DELETE. POSTing an entry that matches an existing
        ``(user_id, type, element_id)`` triple **does not replace** the
        prior row; both will appear in subsequent ``list_entries`` calls.
        See ``save_user_entry`` for a wrapper that warns about this.
        """
        body: HistoryEntry = {"type": type, "data": data}
        if element_id is not None:
            body["element_id"] = element_id
        if user_id is not None:
            body["user_id"] = self.resolve_user_id(user_id)

        resp = requests.post(
            f"{self.base_url}/history-entries",
            json=_envelope_to_wire(body),
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        # OpenAPI 200 response has no schema. Some servers echo the row,
        # others return empty — handle both. Falls back to the request body.
        try:
            echoed = resp.json() or {}
            return _envelope_from_wire(echoed) if echoed else body
        except ValueError:
            return body

    # =====================================================================
    # Convenience — consumer-side invariants the API doesn't enforce
    # =====================================================================

    @requires_token
    def get_user_entry(
        self,
        user_id: str,
        type: str,
        *,
        element_id: Optional[str] = None,
    ) -> Optional[HistoryEntry]:
        """Return the first entry for ``(user_id, type[, element_id])``, or None.

        When multiple rows match (the API allows it), the first row in
        server-returned order is taken and a warning is logged. Pass an
        explicit ``sort`` via ``list_entries`` directly if you need
        different ordering semantics.
        """
        rows = self.list_entries(
            user_id=user_id,
            type=type,
            element_id=element_id,
            limit=self.DEFAULT_LIMIT,
        )
        if not rows:
            return None
        if len(rows) > 1:
            self.logger.warning(
                f"get_user_entry: {len(rows)} entries for user_id={user_id!r}, "
                f"type={type!r}, element_id={element_id!r} — returning the first. "
                f"Use list_entries() if you need them all."
            )
        return rows[0]

    @requires_token
    def save_user_entry(
        self,
        user_id: str,
        type: str,
        data: dict[str, Any],
        *,
        element_id: Optional[str] = None,
    ) -> HistoryEntry:
        """POST an entry, replacing the prior one for the same triple.

        Despite the absence of a DELETE endpoint, the server **upserts** on
        ``(userid, type, elementId)``: verified on preprod 2026-08-05, POSTing
        over an existing triple left the row's ``id`` unchanged and swapped its
        ``data``, with the row count flat. Re-running a provisioning seed is
        therefore idempotent, not accumulative.

        ⚠️  The upsert key includes ``element_id``. Two entries of the same type
        with *different* element ids are different rows and both persist — with
        no DELETE, a typo'd element id leaves an orphan that cannot be removed.
        The pre-write log line names the row about to be overwritten so a
        surprise is visible before it happens rather than after.
        """
        existing = self.list_entries(
            user_id=user_id,
            type=type,
            element_id=element_id,
            limit=1,
        )
        if existing:
            self.logger.info(
                f"save_user_entry: overwriting the existing entry for user_id={user_id!r}, "
                f"type={type!r}, element_id={element_id!r} (id={existing[0].get('id')!r}) — "
                f"the server upserts on this triple, so its data is replaced in place."
            )
        return self.create_entry(type=type, data=data, element_id=element_id, user_id=user_id)

    # =====================================================================
    # Internals
    # =====================================================================

    def _auth_headers(self) -> dict[str, str]:
        """Bearer auth header. Token freshness enforced by ``@requires_token``."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }
