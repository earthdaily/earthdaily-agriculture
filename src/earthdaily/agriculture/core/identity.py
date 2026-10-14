import base64
import os
import warnings
from datetime import datetime, timedelta
from typing import Any

import requests

from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core import storage


class EDAuthenticator:
    """
    Handles authentication and token management for ED APIs.

    Attributes:
        env (str): Environment to use ('na' or 'eu')
        bearer_token (str): Current bearer token for API access
        expiration_date (datetime): Token expiration timestamp
        s3_client (boto3.client): S3 client, created on first access from the standard AWS
            credential chain. Kept for compatibility: S3 no longer depends on this class,
            see :mod:`earthdaily.agriculture.core.storage`.
    """

    def __init__(self, env: str = "prod"):
        """
        Initialize EDAuthenticator.

        Args:
            env: Environment to use ('prod' or 'pre-prod'). Default: 'na'
        """
        self.env = env
        self.bearer_token: str | None = None
        self.expiration_date: str | None = None
        self._s3_client: Any = None  # see the `s3_client` property

        # Auto-initialize token
        self.bearer_token, self.expiration_date = self.get_new_token(env)

    @staticmethod
    def _get_env_var(env: str, var_name: str, default=None):
        """
        Get an environment variable with env prefix, falling back to non-prefixed.

        Lookup order:
        1. {ENV_PREFIX}_{var_name}  (e.g. PREPROD_API_USERNAME)
        2. {var_name}              (e.g. API_USERNAME) — backward compatibility
        3. default value
        """
        prefix = env.upper().replace("-", "")  # 'preprod' -> 'PREPROD', 'prod' -> 'PROD'
        return os.getenv(f"{prefix}_{var_name}") or os.getenv(var_name) or default

    @staticmethod
    def _get_basic_auth_header(env: str = None):
        """
        Construct Basic Auth header from client credentials.
        Encodes 'client_id:client_secret' as base64.

        Args:
            env: Environment prefix for credential lookup (e.g. 'prod', 'preprod')
        """
        if env:
            client_id = EDAuthenticator._get_env_var(env, "API_CLIENT_ID", "mapproduct_api")
            client_secret = EDAuthenticator._get_env_var(env, "API_CLIENT_SECRET", "mapproduct_api.secret")
        else:
            client_id = os.getenv("API_CLIENT_ID", "mapproduct_api")
            client_secret = os.getenv("API_CLIENT_SECRET", "mapproduct_api.secret")

        credentials = f"{client_id}:{client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    @staticmethod
    def get_new_token(env: str):
        """
        Request a fresh bearer token using environment variables.
        Reads credentials with env prefix (e.g. PREPROD_API_USERNAME) first,
        then falls back to non-prefixed vars (API_USERNAME).
        """
        identity_url = agro_urls["identity_urls"][env]
        username = EDAuthenticator._get_env_var(env, "API_USERNAME")
        password = EDAuthenticator._get_env_var(env, "API_PASSWORD")

        response = requests.post(
            identity_url,
            data={
                "grant_type": "password",
                "scope": "openid",
                "username": username,
                "password": password,
            },
            headers={
                "Authorization": EDAuthenticator._get_basic_auth_header(env),
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        result = response.json()

        bearer_token = result["access_token"]
        expiration_date = datetime.now() + timedelta(seconds=3600)
        print(f"👤 User: {username}")
        print(f"🔑 New token acquired for env={env}, valid until {expiration_date}")
        return bearer_token, expiration_date

    @staticmethod
    def check_token(
        expiration_date, bearer_token, env, client_id=None, client_secret=None, api_username=None, api_password=None
    ):
        """
        Reuse token if still valid, otherwise refresh with env variables.
        """
        if expiration_date and datetime.now() <= expiration_date - timedelta(seconds=300):
            return bearer_token, expiration_date

        identity_url = agro_urls["identity_urls"][env]
        username = EDAuthenticator._get_env_var(env, "API_USERNAME")
        password = EDAuthenticator._get_env_var(env, "API_PASSWORD")

        print(f"🔄 Renewing token at {datetime.now().strftime('%H:%M:%S')}")

        response = requests.post(
            identity_url,
            data={
                "grant_type": "password",
                "scope": "openid",
                "username": username,
                "password": password,
            },
            headers={
                "Authorization": EDAuthenticator._get_basic_auth_header(env),
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        result = response.json()

        bearer_token = result["access_token"]
        new_expiration_date = datetime.now() + timedelta(seconds=3600)

        print(f"✅ Token refreshed, valid until {new_expiration_date}")
        return bearer_token, new_expiration_date

    def refresh_token(self):
        """
        Refresh the instance's bearer token if needed.

        Returns:
            str: Current valid bearer token
        """
        self.bearer_token, self.expiration_date = self.check_token(self.expiration_date, self.bearer_token, self.env)
        return self.bearer_token

    @property
    def s3_client(self):
        """S3 client, created on first access (the shared one from :func:`storage.get_s3_client`).

        Nothing is created when the authenticator is: users without S3 pay nothing, and a
        missing or mistyped credential fails here, at first use, with the message from
        :class:`storage.MissingS3Credentials` (a ``NoCredentialsError``).
        """
        if self._s3_client is None:
            self._s3_client = storage.get_s3_client()
        return self._s3_client

    @s3_client.setter
    def s3_client(self, client):
        self._s3_client = client

    def initialize_s3_client(
        self, aws_access_key_id: str = None, aws_secret_access_key: str = None, region_name: str = None
    ):
        """Build an S3 client for this authenticator. **Deprecated**, removed in 2.8.0.

        Use :func:`earthdaily.agriculture.core.storage.create_s3_client` (explicit keys) or
        :func:`~earthdaily.agriculture.core.storage.get_s3_client` (shared client), or just
        read :attr:`s3_client`, which is created on first access.

        Credentials, first match wins: the arguments, then the standard AWS chain
        (env vars, ``~/.aws`` profile, IAM role). Region: ``region_name``, else
        ``AWS_REGION``, else ``us-east-1``. No connection probe is made: the first real
        call reports its own error, or use ``storage.check_s3_access(bucket)``.

        Returns:
            boto3.client: The new client, also stored as :attr:`s3_client`.

        Raises:
            NoCredentialsError: No credentials anywhere in the chain.
        """
        warnings.warn(
            "EDAuthenticator.initialize_s3_client() is deprecated and will be removed in 2.8.0; use "
            "earthdaily.agriculture.core.storage.create_s3_client() or get_s3_client() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._s3_client = storage.create_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
        return self._s3_client

    def ensure_s3_client(self):
        """The S3 client, created on first use. **Deprecated**: read :attr:`s3_client` instead.

        Kept working (no removal date yet) because projects outside this repo call it directly.

        Returns:
            boto3.client: S3 client instance

        Raises:
            NoCredentialsError: No credentials anywhere in the chain.
        """
        warnings.warn(
            "EDAuthenticator.ensure_s3_client() is deprecated; read `authenticator.s3_client`, or use "
            "earthdaily.agriculture.core.storage.get_s3_client().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.s3_client

    def get_token(self):
        """
        Get current valid bearer token, refreshing if needed.

        Returns:
            str: Valid bearer token
        """
        return self.refresh_token()
