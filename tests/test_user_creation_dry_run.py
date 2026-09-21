"""`UserManager.create_users(dry_run=True)` must make no write call at all.

This is the test that matters. Everything else about a rehearsal is presentation;
the one thing it must never do is write. If `dry_run` silently created users, the
failure would be discovered on a real tenant, and back-office writes land on APIs
with no transaction and — for some artefacts — no delete route.

So these assert on the transport: `requests.post` and `requests.patch` are
patched at the module boundary and must not be called once. Reads are expected
and allowed, because the rehearsal's value comes from them (collision detection
and link resolution).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.public

MODULE = "earthdaily.agriculture.services.user_management"


@pytest.fixture
def users_df():
    """One AGRONOMIST managing two GROWERs — exercises all three write phases."""
    return pd.DataFrame(
        [
            {
                "User_Type": "AGRONOMIST",
                "Login": "agro_1",
                "Email": "agro1@example.com",
                "Password": "Welcome",
                "Country_code": "FRA",
                "First_Name": "A",
                "Last_Name": "Gro",
                "Role": "APP_ACCESS;APP_MAPANALYSIS",
                "Manages_user": "grower_1|grower_2",
                "Managed_by_user": "",
            },
            {
                "User_Type": "GROWER",
                "Login": "grower_1",
                "Email": "g1@example.com",
                "Password": "Welcome",
                "Country_code": "FRA",
                "First_Name": "G",
                "Last_Name": "One",
                "Role": "APP_ACCESS",
                "Manages_user": "",
                "Managed_by_user": "agro_1",
            },
            {
                "User_Type": "GROWER",
                "Login": "grower_2",
                "Email": "g2@example.com",
                "Password": "Welcome",
                "Country_code": "FRA",
                "First_Name": "G",
                "Last_Name": "Two",
                "Role": "APP_ACCESS",
                "Manages_user": "",
                "Managed_by_user": "agro_1",
            },
        ]
    )


@pytest.fixture
def manager(mock_logger):
    from earthdaily.agriculture.services.user_management import UserManager

    with patch.object(UserManager, "__init__", lambda self, *a, **kw: None):
        mgr = UserManager.__new__(UserManager)

    mgr.logger = mock_logger
    mgr.extractor_name = "UserManager"
    mgr.base_url = "http://fake-mdm.example.com/users"
    mgr.csv_separator = ","
    mgr.user_params = None
    mgr.bearer_token = "fake-token"
    mgr.token_expiration = 9_999_999_999
    mgr.workflow_ref = None
    mgr.env = "preprod"
    return mgr


class TestDryRunWritesNothing:
    def test_no_post_or_patch_is_issued(self, manager, users_df):
        """The whole point. A single write here is a production incident."""
        with (
            patch(f"{MODULE}.requests.post") as post,
            patch(f"{MODULE}.requests.patch") as patch_call,
            patch.object(manager, "get_user_by_login", return_value=None),
        ):
            manager.create_users(users_df, verbose=False, dry_run=True)

        post.assert_not_called()
        patch_call.assert_not_called()

    def test_helper_write_methods_are_never_reached(self, manager, users_df):
        """Belt and braces: the three write helpers must not be entered either."""
        with (
            patch.object(manager, "get_user_by_login", return_value=None),
            patch.object(manager, "assign_roles") as assign_roles,
            patch.object(manager, "link_agronomist_to_growers") as link_agro,
            patch.object(manager, "link_grower_to_agronomist") as link_grower,
        ):
            manager.create_users(users_df, verbose=False, dry_run=True)

        assign_roles.assert_not_called()
        link_agro.assert_not_called()
        link_grower.assert_not_called()

    def test_reports_dry_run_status_not_created(self, manager, users_df):
        """`CREATED` must mean created. A rehearsal has to be distinguishable."""
        with patch.object(manager, "get_user_by_login", return_value=None):
            results = manager.create_users(users_df, verbose=False, dry_run=True)

        assert set(results["Status"]) == {"DRY_RUN"}
        assert all(uid.startswith("DRY_RUN:") for uid in results["User_Id"])

    def test_existing_login_is_reported_as_a_collision(self, manager, users_df):
        """`create_users` has no already-exists status, so the rehearsal must supply one.

        Live, these rows come back ERROR — after the run has created the others.
        """
        with patch.object(manager, "get_user_by_login", return_value={"id": "existing-123"}):
            results = manager.create_users(users_df, verbose=False, dry_run=True)

        assert all("WOULD COLLIDE" in d for d in results["Details"])
        assert all("existing-123" in d for d in results["Details"])

    def test_sendlogindetails_does_not_email_during_a_rehearsal(self, manager, users_df):
        """`sendLoginDetailsToUser` is an outbound email to a real person."""
        with (
            patch(f"{MODULE}.requests.post") as post,
            patch(f"{MODULE}.requests.patch") as patch_call,
            patch.object(manager, "get_user_by_login", return_value=None),
        ):
            manager.create_users(users_df, verbose=False, send_login_details=True, dry_run=True)

        post.assert_not_called()
        patch_call.assert_not_called()

    def test_a_bad_payload_still_surfaces(self, manager):
        """The body is built BEFORE the dry-run return.

        A rehearsal that passes on a payload the API would reject is worse than
        no rehearsal — the same reasoning as BackOfficeManager.provision_customer.
        """
        df = pd.DataFrame([{"User_Type": "GROWER", "Login": "", "Email": "", "Role": ""}])

        with patch.object(manager, "get_user_by_login", return_value=None):
            results = manager.create_users(df, verbose=False, dry_run=True)

        # The blank login/email reach the logged payload rather than being hidden.
        assert results.loc[0, "Status"] == "DRY_RUN"


class TestLiveRunStillWrites:
    """The default must be unchanged — dry_run is opt-in."""

    def test_default_is_a_real_create(self, manager, users_df):
        response = MagicMock()
        response.json.return_value = {"id": "new-user-1"}
        response.raise_for_status.return_value = None

        with (
            patch(f"{MODULE}.requests.post", return_value=response) as post,
            patch.object(manager, "assign_roles", return_value=(True, "")),
            patch.object(manager, "link_agronomist_to_growers", return_value=[{"success": True}] * 2),
            patch.object(manager, "link_grower_to_agronomist", return_value={"success": True}),
            patch.object(manager, "get_user_by_login", return_value=None),
        ):
            results = manager.create_users(users_df, verbose=False)

        assert post.call_count == 3
        assert set(results["Status"]) == {"CREATED"}
