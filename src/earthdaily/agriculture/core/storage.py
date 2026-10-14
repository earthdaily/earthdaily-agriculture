"""S3 storage helpers, independent of API authentication.

The S3 client used to live on :class:`~earthdaily.agriculture.core.identity.EDAuthenticator`,
which owns the Geosys bearer token. Storage and authentication are separate concerns, so
client creation lives here and nothing in it needs an authenticator:

- :func:`create_s3_client` builds a client from explicit keys or the standard AWS chain.
- :func:`get_s3_client` returns one shared client, created on first use.
- :func:`check_s3_access` is an opt-in ``head_bucket`` check of one bucket.
"""

from __future__ import annotations

import os
import threading

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

DEFAULT_REGION = "us-east-1"


class MissingS3Credentials(NoCredentialsError):
    """No AWS credentials could be found. Still catchable as ``NoCredentialsError``."""

    fmt = (
        "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env "
        "file (or use an AWS profile / IAM role), or pass them as parameters."
    )


def create_s3_client(aws_access_key_id: str = None, aws_secret_access_key: str = None, region_name: str = None):
    """Build a boto3 S3 client.

    Credentials, first match wins: the explicit arguments, then the standard AWS chain
    (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN`` env vars,
    ``~/.aws`` profile, IAM role). Region: ``region_name``, else ``AWS_REGION``, else
    ``us-east-1``.

    The credentials are resolved here rather than at the first request, so a missing
    setup fails at the point S3 is first needed, with a message saying what to set.
    No network probe is made.

    Raises:
        MissingS3Credentials: no credentials anywhere in the chain (a ``NoCredentialsError``).
    """
    region = region_name or os.getenv("AWS_REGION") or DEFAULT_REGION
    if aws_access_key_id and aws_secret_access_key:
        session = boto3.session.Session(
            aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region
        )
    else:
        session = boto3.session.Session(region_name=region)

    credentials = session.get_credentials()
    if credentials is None:
        logger.error("❌ No AWS credentials found")
        raise MissingS3Credentials()
    logger.info(f"🔐 S3 client using AWS credentials from: {getattr(credentials, 'method', 'parameters')}")
    return session.client("s3")


_shared_client = None
_shared_lock = threading.Lock()


def get_s3_client():
    """The shared S3 client, created on first use from the standard AWS chain.

    Thread-safe: parallel workers that all need S3 build one client between them.

    Raises:
        MissingS3Credentials: no credentials anywhere in the chain.
    """
    global _shared_client
    if _shared_client is None:
        with _shared_lock:
            if _shared_client is None:
                _shared_client = create_s3_client()
    return _shared_client


def reset_s3_client() -> None:
    """Forget the shared client, so the next :func:`get_s3_client` builds a new one (tests, rotated keys)."""
    global _shared_client
    with _shared_lock:
        _shared_client = None


def check_s3_access(bucket: str, client=None) -> bool:
    """Confirm ``client`` (default: the shared client) can reach ``bucket``, with ``head_bucket``.

    Works for a role scoped to that one bucket, unlike ``list_buckets`` (which needs
    ``s3:ListAllMyBuckets`` across the account). Opt-in: nothing calls it implicitly.

    Returns:
        True when the bucket is reachable.

    Raises:
        PermissionError: the credentials cannot access the bucket (403).
        FileNotFoundError: the bucket does not exist (404).
        ClientError: any other S3 error, unchanged.
    """
    client = client if client is not None else get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("403", "AccessDenied", "Forbidden"):
            raise PermissionError(f"S3 credentials cannot access bucket '{bucket}' (403).") from e
        if code in ("404", "NoSuchBucket", "NotFound"):
            raise FileNotFoundError(f"S3 bucket '{bucket}' does not exist (404).") from e
        raise
    return True
