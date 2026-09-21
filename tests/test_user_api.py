"""
Tests for UserManager API call logic (read lifecycle primitive):
    - get_user_by_id() URL construction, headers, 404 handling, error propagation
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tests.conftest import FAKE_TOKEN, FAKE_USER_URL

pytestmark = pytest.mark.public

# ===================================================================
# get_user_by_id()
# ===================================================================


class TestGetUserById:
    """Tests for the per-entity API primitive used by the read lifecycle."""

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_successful_api_call(self, mock_get, configured_user_manager, sample_user_response):
        """Happy path: valid id returns the parsed JSON user record."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_user_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.get_user_by_id("usr_001")

        assert result == sample_user_response
        mock_get.assert_called_once()

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_url_targets_users_id_endpoint(self, mock_get, configured_user_manager):
        """URL should be {base_url}/{user_id}."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "usr_001"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_user_manager, "ensure_token_valid"):
            configured_user_manager.get_user_by_id("usr_001")

        actual_url = mock_get.call_args.args[0]
        assert actual_url == f"{FAKE_USER_URL}/users/usr_001"

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_authorization_header_carries_bearer_token(self, mock_get, configured_user_manager):
        """The request must send the bearer token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "usr_001"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(configured_user_manager, "ensure_token_valid"):
            configured_user_manager.get_user_by_id("usr_001")

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_404_returns_none(self, mock_get, configured_user_manager):
        """A 404 means 'user not found' → return None rather than raising."""
        err = requests.exceptions.HTTPError()
        err.response = MagicMock(status_code=404)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = err
        mock_get.return_value = mock_response

        with patch.object(configured_user_manager, "ensure_token_valid"):
            result = configured_user_manager.get_user_by_id("missing")

        assert result is None

    @patch("earthdaily.agriculture.services.user_management.requests.get")
    def test_non_404_http_error_propagates(self, mock_get, configured_user_manager):
        """A non-404 HTTP error (e.g. 500) should propagate to the caller."""
        err = requests.exceptions.HTTPError()
        err.response = MagicMock(status_code=500)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = err
        mock_get.return_value = mock_response

        with patch.object(configured_user_manager, "ensure_token_valid"):
            with pytest.raises(requests.exceptions.HTTPError):
                configured_user_manager.get_user_by_id("boom")
