"""Scaffold tests for HistoryManager.

Covers:
- Envelope conversion: camelCase / lowercase wire <-> snake_case Python
- Round-trip: from_wire(to_wire(x)) == x (and back)
- ``HistoryEntryList.to_dataframe()`` shape
- Query-param assembly in ``list_entries``
- POST body shape in ``create_entry``
- ``save_user_entry`` warns when a prior row exists

HTTP is patched at the ``requests`` module level so no network is touched.
``HistoryManager.__init__`` is bypassed via ``MagicMock`` so we don't need
to bootstrap ``BaseExtractor``'s OAuth flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earthdaily.agriculture.services.history_manager import (
    HistoryEntryList,
    HistoryManager,
    _envelope_from_wire,
    _envelope_to_wire,
)

pytestmark = pytest.mark.public

# ---------------------------------------------------------------------------
# Envelope conversion
# ---------------------------------------------------------------------------


def test_envelope_from_wire_translates_keys():
    wire = {
        "id": "abc123",
        "elementId": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "userid": "user-42",
        "data": {"version": 1, "configurations": []},
    }
    py = _envelope_from_wire(wire)
    assert py == {
        "id": "abc123",
        "element_id": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "user_id": "user-42",
        "data": {"version": 1, "configurations": []},
    }


def test_envelope_from_wire_defaults_missing_data_to_empty_dict():
    py = _envelope_from_wire({"id": "x"})
    assert py["data"] == {}


def test_envelope_from_wire_null_data_becomes_empty_dict():
    py = _envelope_from_wire({"id": "x", "data": None})
    assert py["data"] == {}


def test_envelope_to_wire_translates_back():
    py = {
        "id": "abc",
        "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "element_id": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "user_id": "user-42",
        "data": {"version": 1},
    }
    wire = _envelope_to_wire(py)
    assert wire == {
        "id": "abc",
        "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "elementId": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "userid": "user-42",
        "data": {"version": 1},
    }


def test_envelope_round_trip():
    original_wire = {
        "id": "abc",
        "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "elementId": "CUSTOM_ANALYTIC_UNIQUEAPP",
        "userid": "user-42",
        "data": {"version": 1, "configurations": [{"id": "1", "name": "x"}]},
    }
    assert _envelope_to_wire(_envelope_from_wire(original_wire)) == original_wire


def test_envelope_to_wire_skips_none_values():
    py = {"type": "X", "data": {}, "user_id": None}
    assert _envelope_to_wire(py) == {"type": "X", "data": {}}


# ---------------------------------------------------------------------------
# HistoryEntryList.to_dataframe()
# ---------------------------------------------------------------------------


def test_to_dataframe_flattens_envelope_only():
    entries = HistoryEntryList(
        [
            {
                "id": "a",
                "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
                "element_id": "CUSTOM_ANALYTIC_UNIQUEAPP",
                "user_id": "u1",
                "data": {"version": 1},
            },
            {
                "id": "b",
                "type": "MONITORING_FAVORITES_UNIQUEAPP",
                "element_id": "MONITORING_FAVORITES_UNIQUEAPP",
                "user_id": "u1",
                "data": {"favorites": []},
            },
        ]
    )
    df = entries.to_dataframe()
    assert list(df.columns) == ["id", "type", "element_id", "user_id", "data"]
    assert len(df) == 2
    assert df.iloc[0]["data"] == {"version": 1}  # nested dict preserved as object


def test_to_dataframe_handles_missing_fields_as_none():
    entries = HistoryEntryList([{"id": "x"}])
    df = entries.to_dataframe()
    assert df.iloc[0]["type"] is None
    assert df.iloc[0]["data"] == {}


# ---------------------------------------------------------------------------
# Wire-level behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """A HistoryManager bypassing BaseExtractor's auth bootstrapping."""
    with patch.object(HistoryManager, "__init__", lambda self, *a, **k: None):
        m = HistoryManager()  # type: ignore[call-arg]
    m.env = "preprod"
    m.bearer_token = "fake-token"
    m.token_expiration = 9_999_999_999
    m.logger = MagicMock()
    m.base_url = "https://api-pp.example.com/accounts/history/v1"
    m.mdm_users_url = "https://api-pp.example.com/master-data-management/v6/users"
    m._external_id_cache = {}
    m.ensure_token_valid = MagicMock()
    return m


#: A 22-char base62 id — the form the API actually matches. Pre-seeded into the
#: cache in most tests so they exercise param assembly, not id resolution.
EXTERNAL_ID = "1RmKq5IAZyuv3vanzjixhX"


def test_list_entries_builds_named_filter_params(manager):
    """PascalCase, not $-prefixed: the server ignores unknown params silently."""
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(json=lambda: [], raise_for_status=MagicMock())
        manager.list_entries(
            user_id="u1",
            type="CUSTOM_ANALYTIC_UNIQUEAPP",
            element_id="X",
            limit=5,
        )
        params = r.get.call_args.kwargs["params"]
        assert params["UserId"] == EXTERNAL_ID  # resolved, not the legacy id
        assert params["Type"] == "CUSTOM_ANALYTIC_UNIQUEAPP"
        assert params["ElementId"] == "X"
        # Only pagination controls keep the $ prefix.
        assert params["$limit"] == "5"
        assert params["$offset"] == 0
        assert not any(k.startswith("$") for k in ("UserId", "Type", "ElementId") if k in params)


def test_list_entries_converts_response_to_snake_case(manager):
    wire_response = [
        {
            "id": "a",
            "elementId": "CUSTOM_ANALYTIC_UNIQUEAPP",
            "type": "CUSTOM_ANALYTIC_UNIQUEAPP",
            "userid": "u1",
            "data": {"version": 1},
        },
    ]
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(json=lambda: wire_response, raise_for_status=MagicMock())
        rows = manager.list_entries(user_id="u1")

    assert len(rows) == 1
    assert rows[0]["element_id"] == "CUSTOM_ANALYTIC_UNIQUEAPP"
    assert rows[0]["user_id"] == "u1"
    assert "elementId" not in rows[0]


def test_create_entry_serializes_via_to_wire(manager):
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.post.return_value = MagicMock(json=lambda: {}, raise_for_status=MagicMock())
        manager.create_entry(type="X", data={"k": 1}, element_id="X", user_id="u1")
        body = r.post.call_args.kwargs["json"]
        assert body == {
            "type": "X",
            "data": {"k": 1},
            "elementId": "X",
            # The legacy id 404s; only externalIds.id is accepted.
            "userid": EXTERNAL_ID,
        }


def test_create_entry_without_user_id_omits_userid_from_body(manager):
    """Omitting user_id writes to the CALLING account — right for "save my own
    config", silently wrong for provisioning someone else. Pinned so the body
    shape stays deliberate."""
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.post.return_value = MagicMock(json=lambda: {}, raise_for_status=MagicMock())
        manager.create_entry(type="X", data={"k": 1}, element_id="X")
        body = r.post.call_args.kwargs["json"]
        assert body == {
            "type": "X",
            "data": {"k": 1},
            "elementId": "X",
        }
        assert "userid" not in body
        assert "id" not in body  # server-generated, never in POST body


def test_count_entries_reads_x_total_count_header(manager):
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.head.return_value = MagicMock(
            headers={"X-Total-Count": "42"},
            raise_for_status=MagicMock(),
        )
        count = manager.count_entries(user_id="u1")
        assert count == 42
        assert r.head.call_args.kwargs["params"]["UserId"] == EXTERNAL_ID


def test_save_user_entry_announces_the_overwrite(manager):
    """The server upserts on (userid, type, elementId) — say which row is going."""
    prior_row = {
        "id": "old",
        "userid": "u1",
        "type": "X",
        "elementId": "X",
        "data": {},
    }
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(json=lambda: [prior_row], raise_for_status=MagicMock())
        r.post.return_value = MagicMock(json=lambda: {}, raise_for_status=MagicMock())
        manager.save_user_entry("u1", "X", {"new": "data"}, element_id="X")

    logged = str(manager.logger.info.call_args_list)
    assert "overwriting" in logged and "'old'" in logged
    # Not a warning: overwriting is the intended, idempotent behaviour.
    assert manager.logger.warning.call_count == 0


def test_save_user_entry_silent_when_no_prior(manager):
    manager._external_id_cache["u1"] = EXTERNAL_ID
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(json=lambda: [], raise_for_status=MagicMock())
        r.post.return_value = MagicMock(json=lambda: {}, raise_for_status=MagicMock())
        manager.save_user_entry("u1", "X", {"new": "data"}, element_id="X")

    assert manager.logger.warning.call_count == 0
    assert not any("overwriting" in str(c) for c in manager.logger.info.call_args_list)


# ---------------------------------------------------------------------------
# User-id resolution — the History API does not accept MDM's legacy users.id
# ---------------------------------------------------------------------------


def test_resolve_user_id_looks_up_external_id(manager):
    """MDM's id is the legacy NA one; the API matches only externalIds.id."""
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(
            json=lambda: [{"id": "kxrx932", "externalIds": {"legacY_ID_NA": "kxrx932", "id": EXTERNAL_ID}}],
            raise_for_status=MagicMock(),
        )
        assert manager.resolve_user_id("kxrx932") == EXTERNAL_ID
        params = r.get.call_args.kwargs["params"]
        assert params["Id"] == "kxrx932"
        # externalids must be requested explicitly — MDM omits it by default.
        assert "externalids" in params["$fields"]


def test_resolve_user_id_passes_through_an_external_id(manager):
    """Already-modern ids must not cost a round-trip, and must be idempotent."""
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        assert manager.resolve_user_id(EXTERNAL_ID) == EXTERNAL_ID
        r.get.assert_not_called()


def test_resolve_user_id_caches(manager):
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(
            json=lambda: [{"id": "kxrx932", "externalIds": {"id": EXTERNAL_ID}}],
            raise_for_status=MagicMock(),
        )
        manager.resolve_user_id("kxrx932")
        manager.resolve_user_id("kxrx932")
        assert r.get.call_count == 1


def test_resolve_user_id_raises_when_user_missing(manager):
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(json=lambda: [], raise_for_status=MagicMock())
        with pytest.raises(ValueError, match="No user with id"):
            manager.resolve_user_id("nope123")


def test_resolve_user_id_refuses_when_external_id_absent(manager):
    """Better to fail than to fall back to a write that targets the caller."""
    with patch("earthdaily.agriculture.services.history_manager.requests") as r:
        r.get.return_value = MagicMock(
            json=lambda: [{"id": "kxrx932", "externalIds": {"legacY_ID_NA": "kxrx932"}}],
            raise_for_status=MagicMock(),
        )
        with pytest.raises(ValueError, match="no externalIds.id"):
            manager.resolve_user_id("kxrx932")


def test_resolve_user_id_rejects_empty(manager):
    with pytest.raises(ValueError, match="user_id is required"):
        manager.resolve_user_id("")
