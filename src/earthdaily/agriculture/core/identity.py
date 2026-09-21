import base64
import os
from datetime import datetime, timedelta
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from earthdaily.agriculture.config.urls import agro_urls


class EDAuthenticator:
    """
    Handles authentication and token management for ED APIs and AWS S3 access.

    Attributes:
        env (str): Environment to use ('na' or 'eu')
        bearer_token (str): Current bearer token for API access
        expiration_date (datetime): Token expiration timestamp
        s3_client (boto3.client): Initialized S3 client for bucket access
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
        self.s3_client: Any = None

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

    def initialize_s3_client(
        self, aws_access_key_id: str = None, aws_secret_access_key: str = None, region_name: str = None
    ):
        """
        Initialize boto3 S3 client with credentials from environment variables or parameters.

        Credentials are loaded in the following priority:
        1. Explicitly passed parameters
        2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        3. AWS credentials file (~/.aws/credentials)
        4. IAM role (if running on EC2/ECS)

        Args:
            aws_access_key_id: AWS access key ID (optional, defaults to AWS_ACCESS_KEY_ID env var)
            aws_secret_access_key: AWS secret access key (optional, defaults to AWS_SECRET_ACCESS_KEY env var)
            region_name: AWS region (optional, defaults to AWS_REGION env var or 'us-east-1')

        Returns:
            boto3.client: Configured S3 client

        Raises:
            NoCredentialsError: If no valid credentials are found
            ClientError: If there's an error initializing the client

        Environment Variables:
            AWS_ACCESS_KEY_ID: AWS access key for KA S3 bucket
            AWS_SECRET_ACCESS_KEY: AWS secret key for KA S3 bucket
            AWS_REGION: AWS region (default: us-east-1)
        """
        try:
            # Get credentials from parameters or environment
            access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            region = region_name or os.getenv("AWS_REGION", "us-east-1")

            # Build client configuration
            client_config = {"service_name": "s3", "region_name": region}

            # Add explicit credentials if provided
            if access_key and secret_key:
                client_config["aws_access_key_id"] = access_key
                client_config["aws_secret_access_key"] = secret_key
                logger.info("🔐 Using AWS credentials from environment/parameters")
            else:
                logger.info("🔐 Using default AWS credential chain")

            # Initialize S3 client
            self.s3_client = boto3.client(**client_config)

            # Test connection
            try:
                self.s3_client.list_buckets()
                logger.info("✅ Successfully initialized S3 client")
            except ClientError as e:
                logger.warning(f"⚠️ S3 client initialized but connection test failed: {str(e)}")

            return self.s3_client

        except NoCredentialsError:
            logger.error("❌ No AWS credentials found")
            raise NoCredentialsError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY in your .env file or pass them as parameters."
            )
        except ClientError as e:
            logger.error(f"❌ Error initializing S3 client: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing S3 client: {str(e)}")
            raise

    def ensure_s3_client(self):
        """
        Ensure S3 client is initialized, initialize if not.

        Returns:
            boto3.client: S3 client instance

        Raises:
            NoCredentialsError: If S3 client cannot be initialized
        """
        if self.s3_client is None:
            logger.info("🔄 S3 client not initialized, initializing now...")
            self.initialize_s3_client()
        return self.s3_client

    def get_token(self):
        """
        Get current valid bearer token, refreshing if needed.

        Returns:
            str: Valid bearer token
        """
        return self.refresh_token()
