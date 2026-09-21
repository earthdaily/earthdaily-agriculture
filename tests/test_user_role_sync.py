"""
Tests for UserManager.sync_roles() verification:
    - a 2xx PUT does not prove the roles attached; the platform silently drops
      any role the CUSTOMER cannot grant (its availableRoles, a property of the
      tenant's products)
    - a failed read-back must not be reported as "everything was dropped"

This is the failure that produced a fully "provisioned" analyst who could not open
the app: PYTHON carried Digital_Ag (APP_MONITORING|FIELDS) but not Crop_intel
(APP_MONITORING|REGIONAL), and sync_roles reported "Roles added:
APP_MONITORING|REGIONAL" while the account kept 2 of 3 roles.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.public

WANTED = ["APP_ACCESS", "ANALYTICS_PROCESSORS_USER", "APP_MONITORING|REGIONAL"]
PARTIAL = ["APP_ACCESS", "ANALYTICS_PROCESSORS_USER"]  # what the tenant could grant


def _ok_put():
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    return response


def _roles_response(role_ids):
    response = MagicMock(status_code=200, ok=True)
    response.json.return_value = [{"id": r} for r in role_ids]
    return response


class TestSyncRolesVerification:
    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_silently_dropped_role_is_a_failure(self, mock_get, mock_put, configured_user_manager):
        """The PUT succeeds, the role does not attach — that must not read as success."""
        # before: two roles; after the write: still two, |REGIONAL never attached.
        mock_get.side_effect = [_roles_response(["APP_ACCESS"]), _roles_response(PARTIAL)]
        mock_put.return_value = _ok_put()

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, message = configured_user_manager.sync_roles("usr_001", WANTED)

        assert success is False, "a partial apply must not be reported as success"
        assert "APP_MONITORING|REGIONAL" in message
        assert "NOT attached" in message
        # The message must point at the tenant-level fix — but NOT by naming
        # `BackOfficeManager.set_roles`, which it used to. That class is in
        # `private_excluded`, so a public user reading this error was sent to an
        # import they do not have. Assert the remedy, not the internal API.
        assert "tenant" in message, "the message should point at the tenant-level fix"
        assert "BackOfficeManager" not in message, "must not name a class absent from the public wheel"

    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_full_apply_still_succeeds(self, mock_get, mock_put, configured_user_manager):
        mock_get.side_effect = [_roles_response(["APP_ACCESS"]), _roles_response(WANTED)]
        mock_put.return_value = _ok_put()

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, message = configured_user_manager.sync_roles("usr_001", WANTED)

        assert success is True
        assert "NOT attached" not in message
        assert "Roles added" in message

    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_failed_read_back_is_inconclusive_not_a_drop(self, mock_get, mock_put, configured_user_manager):
        """A read that did not happen looks like [] — it must not fail the sync."""
        failed = MagicMock(status_code=503, ok=False)
        mock_get.side_effect = [_roles_response(["APP_ACCESS"]), failed]
        mock_put.return_value = _ok_put()

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, message = configured_user_manager.sync_roles("usr_001", WANTED)

        assert success is True, "the write succeeded; an unreadable verification is not a drop"
        assert "unverified" in message

    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_verify_false_skips_the_read_back(self, mock_get, mock_put, configured_user_manager):
        mock_get.side_effect = [_roles_response(["APP_ACCESS"])]  # one GET only
        mock_put.return_value = _ok_put()

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, _ = configured_user_manager.sync_roles("usr_001", WANTED, verify=False)

        assert success is True
        assert mock_get.call_count == 1, "verify=False must not issue the second GET"

    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_no_op_when_sets_already_match(self, mock_get, mock_put, configured_user_manager):
        mock_get.side_effect = [_roles_response(WANTED)]

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, message = configured_user_manager.sync_roles("usr_001", WANTED)

        assert (success, message) == (True, "")
        mock_put.assert_not_called()

    @patch("earthdaily.agriculture.services.user_management.requests.put")
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_http_failure_is_reported_without_a_read_back(self, mock_get, mock_put, configured_user_manager):
        import requests

        failing = MagicMock(status_code=400)
        failing.text = "bad role id"
        failing.raise_for_status.side_effect = requests.exceptions.HTTPError(response=failing)
        mock_get.side_effect = [_roles_response(["APP_ACCESS"])]
        mock_put.return_value = failing

        with patch.object(configured_user_manager, "ensure_token_valid"):
            success, message = configured_user_manager.sync_roles("usr_001", WANTED)

        assert success is False
        assert "400" in message and "bad role id" in message
        assert mock_get.call_count == 1, "a failed write needs no verification GET"


class TestGetUserRolesContract:
    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_failed_lookup_still_reports_no_roles(self, mock_get, configured_user_manager):
        """The public contract is unchanged: a failure reads as an empty set."""
        mock_get.return_value = MagicMock(status_code=500, ok=False)

        with patch.object(configured_user_manager, "ensure_token_valid"):
            assert configured_user_manager.get_user_roles("usr_001") == []

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_internal_fetch_distinguishes_failure_from_empty(self, mock_get, configured_user_manager):
        mock_get.return_value = MagicMock(status_code=500, ok=False)
        assert configured_user_manager._fetch_role_ids("usr_001") is None

        mock_get.return_value = _roles_response([])
        assert configured_user_manager._fetch_role_ids("usr_001") == []
