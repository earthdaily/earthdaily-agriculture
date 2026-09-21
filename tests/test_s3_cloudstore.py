"""Unit tests for the ``EDA_S3_cloudstore`` write surface and typed errors.

Covers the symmetric boto3 writers (``write_to_s3`` / ``write_csv`` /
``write_parquet``), the manifest-free uploaders (``upload_file`` /
``upload_files``), the ``object_exists`` existence check, and the typed
``S3ObjectNotFound`` exception raised by ``read_from_s3``.

These are pure unit tests — the boto3 client is mocked, so no MinIO/AWS is
needed (the emulator-backed round-trips live in
tests/test_cloud_writers_integration.py and test_export_cloud_publish.py).
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from earthdaily.agriculture.services.S3 import EDA_S3_cloudstore, S3ObjectNotFound

pytestmark = pytest.mark.public


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def s3_client():
    """A mock boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def authenticator(s3_client):
    """A mock EDAuthenticator whose ensure_s3_client() returns the mock client."""
    auth = MagicMock()
    auth.ensure_s3_client.return_value = s3_client
    return auth


@pytest.fixture
def df():
    return pd.DataFrame({"id": [1, 2], "crop": ["CORN", "SOY"]})


def _client_error(code: str, operation: str = "GetObject") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


# ---------------------------------------------------------------------------
# S3ObjectNotFound
# ---------------------------------------------------------------------------
class TestS3ObjectNotFound:
    def test_is_a_valueerror_subclass(self):
        assert issubclass(S3ObjectNotFound, ValueError)

    def test_carries_bucket_and_key_and_preserves_message(self):
        e = S3ObjectNotFound("my-bucket", "state/index.csv")
        assert e.bucket_name == "my-bucket"
        assert e.file_key == "state/index.csv"
        # Original message preserved for any callers string-matching it.
        assert str(e) == "File not found in S3: state/index.csv"

    @pytest.mark.parametrize("code", ["NoSuchKey", "404"])
    def test_read_from_s3_raises_on_missing_object(self, authenticator, s3_client, code):
        s3_client.get_object.side_effect = _client_error(code)
        with pytest.raises(S3ObjectNotFound) as excinfo:
            EDA_S3_cloudstore.read_from_s3(authenticator, "bucket", "missing.csv")
        assert excinfo.value.file_key == "missing.csv"
        assert excinfo.value.bucket_name == "bucket"
        # Chained from the underlying ClientError.
        assert isinstance(excinfo.value.__cause__, ClientError)

    def test_missing_object_still_catchable_as_valueerror(self, authenticator, s3_client):
        """Backward compatibility: old ``except ValueError`` callers keep working."""
        s3_client.get_object.side_effect = _client_error("NoSuchKey")
        with pytest.raises(ValueError):
            EDA_S3_cloudstore.read_csv(authenticator, "bucket", "missing.csv")

    def test_other_client_errors_are_not_swallowed(self, authenticator, s3_client):
        s3_client.get_object.side_effect = _client_error("AccessDenied")
        with pytest.raises(ClientError):
            EDA_S3_cloudstore.read_from_s3(authenticator, "bucket", "denied.csv")


# ---------------------------------------------------------------------------
# object_exists
# ---------------------------------------------------------------------------
class TestObjectExists:
    def test_true_when_head_object_succeeds(self, authenticator, s3_client):
        s3_client.head_object.return_value = {"ContentLength": 10}
        assert EDA_S3_cloudstore.object_exists(authenticator, "bucket", "key") is True
        s3_client.head_object.assert_called_once_with(Bucket="bucket", Key="key")

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
    def test_false_when_object_missing(self, authenticator, s3_client, code):
        s3_client.head_object.side_effect = _client_error(code, "HeadObject")
        assert EDA_S3_cloudstore.object_exists(authenticator, "bucket", "missing") is False

    def test_other_client_errors_propagate(self, authenticator, s3_client):
        s3_client.head_object.side_effect = _client_error("AccessDenied", "HeadObject")
        with pytest.raises(ClientError):
            EDA_S3_cloudstore.object_exists(authenticator, "bucket", "key")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
class TestWriteToS3:
    def test_write_csv_roundtrips_with_index_false_default(self, authenticator, s3_client, df):
        uri = EDA_S3_cloudstore.write_csv(authenticator, df, "bucket", "out/state.csv")
        assert uri == "s3://bucket/out/state.csv"

        kwargs = s3_client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "bucket"
        assert kwargs["Key"] == "out/state.csv"
        roundtrip = pd.read_csv(BytesIO(kwargs["Body"]))
        pd.testing.assert_frame_equal(roundtrip, df)  # no stray index column

    def test_write_parquet_roundtrips_and_preserves_dtypes(self, authenticator, s3_client, df):
        uri = EDA_S3_cloudstore.write_parquet(authenticator, df, "bucket", "out/state.parquet")
        assert uri == "s3://bucket/out/state.parquet"

        body = s3_client.put_object.call_args.kwargs["Body"]
        roundtrip = pd.read_parquet(BytesIO(body))
        pd.testing.assert_frame_equal(roundtrip, df)

    def test_index_true_override_is_honoured(self, authenticator, s3_client, df):
        EDA_S3_cloudstore.write_csv(authenticator, df, "bucket", "x.csv", index=True)
        roundtrip = pd.read_csv(BytesIO(s3_client.put_object.call_args.kwargs["Body"]))
        assert "Unnamed: 0" in roundtrip.columns  # the written index column

    def test_auto_detects_type_from_extension(self, authenticator, s3_client, df):
        EDA_S3_cloudstore.write_to_s3(authenticator, df, "bucket", "a/b.parquet")
        body = s3_client.put_object.call_args.kwargs["Body"]
        pd.testing.assert_frame_equal(pd.read_parquet(BytesIO(body)), df)

    def test_auto_detect_failure_raises_loudly(self, authenticator, df):
        with pytest.raises(ValueError, match="Cannot auto-detect file type"):
            EDA_S3_cloudstore.write_to_s3(authenticator, df, "bucket", "no_extension")

    def test_unsupported_explicit_type_raises(self, authenticator, df):
        with pytest.raises(ValueError, match="Unsupported file type"):
            EDA_S3_cloudstore.write_to_s3(authenticator, df, "bucket", "x.txt", file_type="txt")


# ---------------------------------------------------------------------------
# Uploaders
# ---------------------------------------------------------------------------
class TestUploadFile:
    def test_uploads_existing_file_and_returns_uri(self, authenticator, s3_client, tmp_path):
        local = tmp_path / "report.html"
        local.write_text("<html></html>")

        uri = EDA_S3_cloudstore.upload_file(authenticator, "bucket", "out/report.html", str(local))

        assert uri == "s3://bucket/out/report.html"
        s3_client.upload_file.assert_called_once_with(str(local), "bucket", "out/report.html")

    def test_missing_local_file_raises_filenotfound(self, authenticator, s3_client, tmp_path):
        with pytest.raises(FileNotFoundError):
            EDA_S3_cloudstore.upload_file(authenticator, "bucket", "k", str(tmp_path / "nope.txt"))
        s3_client.upload_file.assert_not_called()


class TestUploadFiles:
    def test_uploads_under_prefix_with_basename_keys_in_order(self, authenticator, s3_client, tmp_path):
        p1 = tmp_path / "report.html"
        p2 = tmp_path / "data.csv"
        p1.write_text("a")
        p2.write_text("b")

        uris = EDA_S3_cloudstore.upload_files(authenticator, "bucket", [str(p1), str(p2)], s3_prefix="runs/2026-06-12/")

        assert uris == [
            "s3://bucket/runs/2026-06-12/report.html",
            "s3://bucket/runs/2026-06-12/data.csv",
        ]
        keys = [call.args[2] for call in s3_client.upload_file.call_args_list]
        assert keys == ["runs/2026-06-12/report.html", "runs/2026-06-12/data.csv"]

    def test_empty_prefix_uploads_to_bucket_root(self, authenticator, s3_client, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("b")

        uris = EDA_S3_cloudstore.upload_files(authenticator, "bucket", [str(p)])

        assert uris == ["s3://bucket/data.csv"]
        assert s3_client.upload_file.call_args.args[2] == "data.csv"
