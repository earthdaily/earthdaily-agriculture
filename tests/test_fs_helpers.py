"""Unit tests for ``earthdaily.agriculture.core._fs`` and the new fsspec-aware
``export_results``.

Remote/MinIO integration tests live in tests/test_cloud_writers_integration.py
so this file can run in any environment without a fixture container.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from earthdaily.agriculture.core._fs import (
    ensure_dir,
    glob_files,
    is_remote_path,
    join_path,
    remove_files,
    storage_options_for,
    write_bytes,
    write_text,
)

pytestmark = pytest.mark.public

# ---------------------------------------------------------------------------
# is_remote_path
# ---------------------------------------------------------------------------


class TestIsRemotePath:
    @pytest.mark.parametrize(
        "path",
        [
            "s3://bucket/prefix",
            "s3://bucket",
            "gs://bucket/object",
            "gcs://bucket",
            "az://container/blob",
            "azure://container/blob",
            "abfs://path",
            "abfss://path",
            "http://host/key",
            "https://host/key",
        ],
    )
    def test_remote_schemes_recognised(self, path):
        assert is_remote_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "results",
            "results/foo.csv",
            "/tmp/results",
            "C:/Users/foo/results",
            r"C:\Users\foo\results",
            "./results",
            "../results",
            # file:// is intentionally treated as local-ish today — the helper
            # leaves makedirs alone for it.
            "file:///tmp/results",
        ],
    )
    def test_local_paths_not_remote(self, path):
        assert is_remote_path(path) is False

    def test_none_returns_false(self):
        assert is_remote_path(None) is False

    def test_pathlib_object_treated_as_local(self, tmp_path):
        assert is_remote_path(tmp_path) is False


# ---------------------------------------------------------------------------
# join_path
# ---------------------------------------------------------------------------


class TestJoinPath:
    def test_local_uses_os_join(self, tmp_path):
        out = join_path(tmp_path, "a", "b.csv")
        # os.path.join honours the platform separator; just check segments are present.
        assert "a" in str(out) and "b.csv" in str(out)

    def test_remote_concatenates_with_forward_slash(self):
        assert join_path("s3://bucket/prefix", "a", "b.csv") == "s3://bucket/prefix/a/b.csv"

    def test_remote_strips_trailing_slash_on_base(self):
        assert join_path("s3://bucket/prefix/", "a.csv") == "s3://bucket/prefix/a.csv"

    def test_remote_strips_leading_slashes_on_parts(self):
        assert join_path("s3://bucket", "/a/", "/b.csv") == "s3://bucket/a/b.csv"

    def test_remote_no_parts_returns_base_without_trailing_slash(self):
        assert join_path("s3://bucket/prefix/") == "s3://bucket/prefix"

    def test_remote_skips_empty_parts(self):
        assert join_path("s3://bucket", "", "a.csv") == "s3://bucket/a.csv"


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------


class TestEnsureDir:
    def test_local_creates_missing_dir(self, tmp_path):
        target = tmp_path / "new_dir" / "nested"
        assert not target.exists()
        ensure_dir(target)
        assert target.is_dir()

    def test_local_existing_dir_is_noop(self, tmp_path):
        ensure_dir(tmp_path)  # already exists, must not raise
        assert tmp_path.is_dir()

    def test_remote_uri_does_not_attempt_filesystem_write(self):
        # No s3 client / network involved — pure no-op.
        ensure_dir("s3://bucket/prefix/should-not-touch-disk")

    def test_none_is_noop(self):
        ensure_dir(None)


# ---------------------------------------------------------------------------
# storage_options_for
# ---------------------------------------------------------------------------


class TestStorageOptionsFor:
    def test_local_returns_none(self, tmp_path):
        assert storage_options_for(tmp_path) is None
        assert storage_options_for("results/foo.csv") is None

    def test_remote_returns_empty_dict(self):
        # Empty dict is the signal pandas/fsspec needs to route through fsspec
        # while leaving credential discovery to the standard environment chain.
        assert storage_options_for("s3://bucket/prefix") == {}
        assert storage_options_for("gs://bucket") == {}

    def test_none_returns_none(self):
        assert storage_options_for(None) is None


# ---------------------------------------------------------------------------
# storage_options_for — Azure credential resolution
# ---------------------------------------------------------------------------


_AZURE_ENV_VARS = (
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_STORAGE_ACCOUNT_KEY",
)


@pytest.fixture
def _clean_azure_env(monkeypatch):
    """Remove all AZURE_STORAGE_* vars so host env never leaks into assertions."""
    for var in _AZURE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestStorageOptionsForAzure:
    @pytest.mark.parametrize("scheme", ["az", "azure", "abfs", "abfss"])
    def test_no_env_returns_empty_dict(self, _clean_azure_env, scheme):
        # Nothing configured → empty dict (adlfs falls back to its default chain).
        assert storage_options_for(f"{scheme}://container/path") == {}

    def test_connection_string_wins(self, _clean_azure_env):
        _clean_azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "CONN")
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        _clean_azure_env.setenv("AZURE_STORAGE_SAS_TOKEN", "sig=abc")
        assert storage_options_for("az://c/p") == {"connection_string": "CONN"}

    def test_account_name_plus_sas(self, _clean_azure_env):
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        _clean_azure_env.setenv("AZURE_STORAGE_SAS_TOKEN", "?sv=2026&sig=abc")
        # leading '?' is normalised off
        assert storage_options_for("az://c/p") == {"account_name": "acct", "sas_token": "sv=2026&sig=abc"}

    def test_account_name_plus_key(self, _clean_azure_env):
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_KEY", "KEY==")
        assert storage_options_for("az://c/p") == {"account_name": "acct", "account_key": "KEY=="}

    def test_sas_preferred_over_key(self, _clean_azure_env):
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        _clean_azure_env.setenv("AZURE_STORAGE_SAS_TOKEN", "sig=abc")
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_KEY", "KEY==")
        assert storage_options_for("az://c/p") == {"account_name": "acct", "sas_token": "sig=abc"}

    def test_account_name_only_uses_default_chain(self, _clean_azure_env):
        # account_name alone → adlfs uses DefaultAzureCredential (Managed Identity …)
        _clean_azure_env.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
        assert storage_options_for("az://c/p") == {"account_name": "acct"}

    def test_s3_unaffected_by_azure_env(self, _clean_azure_env):
        _clean_azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "CONN")
        assert storage_options_for("s3://bucket/prefix") == {}


# ---------------------------------------------------------------------------
# export_results — local round-trip after the fsspec refactor
# ---------------------------------------------------------------------------


class TestExportResultsLocalRoundtrip:
    """Local-path behaviour must be byte-identical to the pre-refactor version."""

    def test_writes_results_csv(self, tmp_path):
        from earthdaily.agriculture.core.api_utils import export_results

        df = pd.DataFrame({"id": ["a", "b"], "value": [1, 2]})
        results_path, errors_path = export_results(
            df, errors=[], output_path=str(tmp_path), prefix="unit", verbose=False
        )
        assert results_path is not None
        assert Path(results_path).exists()
        # Errors empty → no errors file written.
        assert errors_path is None

        readback = pd.read_csv(results_path)
        assert readback.equals(df)

    def test_writes_errors_csv_when_present(self, tmp_path):
        from earthdaily.agriculture.core.api_utils import export_results

        df = pd.DataFrame({"id": ["a"], "value": [1]})
        errors = [{"entity_id": "x", "error_message": "boom"}]
        results_path, errors_path = export_results(
            df, errors=errors, output_path=str(tmp_path), prefix="unit", verbose=False
        )
        assert errors_path is not None
        assert Path(errors_path).exists()
        readback = pd.read_csv(errors_path)
        assert readback.iloc[0]["error_message"] == "boom"

    def test_filenames_carry_prefix_and_suffix(self, tmp_path):
        from earthdaily.agriculture.core.api_utils import export_results

        df = pd.DataFrame({"x": [1]})
        results_path, _ = export_results(
            df, errors=[], output_path=str(tmp_path), prefix="my_prefix", partial=True, verbose=False
        )
        assert results_path is not None
        name = os.path.basename(results_path)
        assert name.startswith("my_prefix_results_")
        assert name.endswith("_partial.csv")

    def test_empty_results_returns_none(self, tmp_path):
        from earthdaily.agriculture.core.api_utils import export_results

        results_path, errors_path = export_results(
            pd.DataFrame(), errors=[], output_path=str(tmp_path), prefix="unit", verbose=False
        )
        assert results_path is None
        assert errors_path is None

    def test_output_path_is_created_if_missing(self, tmp_path):
        from earthdaily.agriculture.core.api_utils import export_results

        target = tmp_path / "fresh" / "subdir"
        assert not target.exists()
        df = pd.DataFrame({"x": [1]})
        results_path, _ = export_results(df, errors=[], output_path=str(target), prefix="unit", verbose=False)
        assert results_path is not None
        assert target.is_dir()


# ---------------------------------------------------------------------------
# export_results — remote path plumbing (no real S3, just intent)
# ---------------------------------------------------------------------------


class TestExportResultsRemotePathPlumbing:
    """Verifies remote URIs are routed through pandas' ``storage_options`` and
    that we do NOT call ``os.makedirs`` on them — without spinning up a real
    backend (covered in tests/test_cloud_writers_integration.py)."""

    def test_remote_path_passes_storage_options_and_skips_mkdir(self, monkeypatch):
        from earthdaily.agriculture.core import _fs, api_utils

        captured = {"to_csv_calls": []}

        def fake_to_csv(self, path, **kwargs):
            captured["to_csv_calls"].append({"path": path, **kwargs})

        # Catch any accidental makedirs on a remote URI. `api_utils` calls
        # `_fs.ensure_dir`, which is the only place `os.makedirs` could fire —
        # patch it there, not on `api_utils` (which doesn't import `os` at all).
        def boom_makedirs(path, **kwargs):
            raise AssertionError(f"os.makedirs unexpectedly called with {path!r} during remote write")

        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
        monkeypatch.setattr(_fs.os, "makedirs", boom_makedirs)

        df = pd.DataFrame({"x": [1]})
        results_path, _ = api_utils.export_results(
            df, errors=[], output_path="s3://bucket/runs", prefix="unit", verbose=False
        )

        assert results_path is not None
        assert results_path.startswith("s3://bucket/runs/unit_results_")
        # to_csv was called with storage_options={} so pandas routes through fsspec.
        assert len(captured["to_csv_calls"]) == 1
        call = captured["to_csv_calls"][0]
        assert call["storage_options"] == {}
        assert call["path"].startswith("s3://bucket/runs/unit_results_")


# ---------------------------------------------------------------------------
# write_text — local round-trip
# ---------------------------------------------------------------------------


class TestWriteText:
    def test_writes_to_existing_local_dir(self, tmp_path):
        target = tmp_path / "report.html"
        write_text(target, "<html>hi</html>")
        assert target.read_text(encoding="utf-8") == "<html>hi</html>"

    def test_creates_parent_dirs_on_local(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "report.html"
        write_text(target, "ok")
        assert target.read_text() == "ok"

    def test_remote_path_routes_through_fsspec(self, monkeypatch):
        """Verify write_text on a remote URI calls fsspec.open without touching disk."""
        captured = {"path": None, "mode": None, "content": None}

        class _FakeFh:
            def __init__(self, captured):
                self._captured = captured

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def write(self, data):
                self._captured["content"] = data

        def fake_fsspec_open(path, mode, encoding=None):  # noqa: D401 - test helper
            captured["path"] = path
            captured["mode"] = mode
            captured["encoding"] = encoding
            return _FakeFh(captured)

        # Inject a fake fsspec module so we don't need s3fs / network for this unit.
        import sys
        import types

        fake_fsspec = types.SimpleNamespace(open=fake_fsspec_open)
        monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)

        write_text("s3://bucket/report.html", "<html>remote</html>")

        assert captured["path"] == "s3://bucket/report.html"
        assert captured["mode"] == "w"
        assert captured["content"] == "<html>remote</html>"


# ---------------------------------------------------------------------------
# write_bytes — binary sibling of write_text (the raster writers' path)
# ---------------------------------------------------------------------------


class TestWriteBytes:
    def test_writes_to_existing_local_dir(self, tmp_path):
        target = tmp_path / "field.tif"
        payload = bytes([0x49, 0x49, 0x2A, 0x00]) + b"raster"  # TIFF magic + body
        write_bytes(target, payload)
        assert target.read_bytes() == payload

    def test_creates_parent_dirs_on_local(self, tmp_path):
        target = tmp_path / "tifs" / "NDVI" / "field.tif"
        write_bytes(target, b"ok")
        assert target.read_bytes() == b"ok"

    def test_round_trips_bytes_unmodified(self, tmp_path):
        """No text encoding anywhere on the path — a TIF must survive byte-exact."""
        payload = bytes(range(256))
        target = tmp_path / "all_bytes.bin"
        write_bytes(target, payload)
        assert target.read_bytes() == payload

    def test_remote_path_routes_through_fsspec(self, monkeypatch):
        """A remote URI goes to fsspec in binary mode, never to local disk."""
        captured = {}

        class _FakeFh:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def write(self, data):
                captured["content"] = data

        def fake_fsspec_open(path, mode, **kwargs):
            captured["path"] = path
            captured["mode"] = mode
            captured["kwargs"] = kwargs
            return _FakeFh()

        import sys
        import types

        monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_fsspec_open))

        write_bytes("s3://bucket/tifs/field.tif", b"raster-bytes")

        assert captured["path"] == "s3://bucket/tifs/field.tif"
        assert captured["mode"] == "wb"
        assert captured["content"] == b"raster-bytes"
        # No `encoding` kwarg may leak in — fsspec rejects it on a binary handle.
        assert "encoding" not in captured["kwargs"]

    def test_remote_path_does_not_touch_local_disk(self, tmp_path, monkeypatch):
        """The regression this whole change exists for: an s3:// prefix must not
        become a literal ./s3:/bucket/... directory."""

        class _FakeFh:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def write(self, data):
                pass

        import sys
        import types

        monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=lambda *a, **k: _FakeFh()))
        monkeypatch.chdir(tmp_path)

        write_bytes("s3://bucket/tifs/field.tif", b"x")

        assert not (tmp_path / "s3:").exists()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# glob_files / remove_files — local round-trip
# ---------------------------------------------------------------------------


class TestGlobFilesLocal:
    def test_returns_matching_files(self, tmp_path):
        (tmp_path / "a_partial.csv").write_text("x")
        (tmp_path / "b_partial.csv").write_text("y")
        (tmp_path / "c.csv").write_text("z")
        matches = glob_files(str(tmp_path / "*_partial.csv"))
        assert len(matches) == 2
        assert all("partial" in m for m in matches)

    def test_no_matches_returns_empty(self, tmp_path):
        assert glob_files(str(tmp_path / "*.bogus")) == []


class TestRemoveFilesLocal:
    def test_removes_local_files(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("x")
        b.write_text("y")
        deleted, errors = remove_files([str(a), str(b)])
        assert deleted == 2
        assert errors == []
        assert not a.exists()
        assert not b.exists()

    def test_collects_errors_on_missing_files(self, tmp_path):
        deleted, errors = remove_files([str(tmp_path / "does_not_exist.csv")])
        assert deleted == 0
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# _finalize_extraction wiring — local round-trip via a minimal extractor
# ---------------------------------------------------------------------------


class TestFinalizeExtractionLocal:
    """End-to-end check that BaseExtractor._finalize_extraction now routes its
    failed_ids CSV, partial-cleanup glob, and HTML report write through the
    new helpers — exercised on local paths so the test runs without a backend.
    """

    def _make_extractor(self, tmp_path, mocker):
        """Build a minimal extractor instance bypassing __init__."""
        from earthdaily.agriculture.core.base_extractor import BaseExtractor

        ext = BaseExtractor.__new__(BaseExtractor)
        ext.bearer_token = "fake-token"
        ext.token_expiration = None
        ext.config = {"env": "preprod"}
        ext.workflow_ref = None
        ext.env = "preprod"
        ext.partial_path = str(tmp_path / "partials")
        ext.output_path = str(tmp_path / "results")
        ext.merge_existing = "auto"
        ext.logger = mocker.MagicMock() if mocker else None
        ext.extractor_name = "Test"
        ext.column_mapping = {"id": "id", "geometry": "geometry"}
        ext.output_mapping = None
        ext.exclude_columns = None
        ext.output_columns = None
        ext.use_cache = False
        ext.cache_key_columns = None
        os.makedirs(ext.partial_path, exist_ok=True)
        os.makedirs(ext.output_path, exist_ok=True)
        return ext

    def test_failed_ids_csv_written_to_partial_path(self, tmp_path, mocker):
        ext = self._make_extractor(tmp_path, mocker)
        df = pd.DataFrame({"id": ["a"], "value": [1]})

        results_df, status = ext._finalize_extraction(
            results_df=df,
            global_errors=[{"entity_id": "x", "message": "boom"}],
            failed_ids=["x"],
            output_path=ext.output_path,
            prefix="t",
            skip_export=True,  # we're not testing export here
            verbose=False,
        )
        assert status["failed_ids_saved"] is True
        partial_files = list(Path(ext.partial_path).glob("failed_ids_t_*.csv"))
        assert len(partial_files) == 1

    def test_partial_files_cleaned_up_when_no_errors(self, tmp_path, mocker):
        ext = self._make_extractor(tmp_path, mocker)
        # Drop a couple of partial files into partial_path.
        (Path(ext.partial_path) / "t_results_20260101_partial.csv").write_text("x,y\n1,2\n")
        (Path(ext.partial_path) / "t_errors_20260101_partial.csv").write_text("err\nboom\n")
        # Plus an unrelated file that must NOT be cleaned.
        (Path(ext.partial_path) / "other_results_final.csv").write_text("a")

        df = pd.DataFrame({"id": ["a"], "value": [1]})
        _, status = ext._finalize_extraction(
            results_df=df,
            global_errors=[],
            failed_ids=[],
            output_path=ext.output_path,
            prefix="t",
            skip_export=True,
            verbose=False,
        )
        assert status["partials_cleaned"] is True
        remaining = sorted(p.name for p in Path(ext.partial_path).glob("*"))
        assert remaining == ["other_results_final.csv"]


class TestFinalizeExtractionRemotePathPlumbing:
    """Verifies _finalize_extraction routes remote partial_path / output_path
    through the helpers — without touching network or real S3."""

    def test_remote_partial_path_skips_makedirs_and_uses_storage_options(self, monkeypatch, mocker):
        from earthdaily.agriculture.core import base_extractor

        # Capture: every os.makedirs call from base_extractor should be on local
        # paths only (the partials dir is remote here, so makedirs must NOT
        # be called for it).
        makedirs_calls = []

        original_makedirs = os.makedirs

        def tracking_makedirs(path, **kwargs):
            makedirs_calls.append(str(path))
            return original_makedirs(path, **kwargs)

        monkeypatch.setattr(base_extractor.os, "makedirs", tracking_makedirs)

        captured_to_csv = []

        def fake_to_csv(self, path, **kwargs):
            captured_to_csv.append({"path": path, **kwargs})

        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

        # Build a minimal extractor with a REMOTE partial_path.
        ext = base_extractor.BaseExtractor.__new__(base_extractor.BaseExtractor)
        ext.bearer_token = "fake-token"
        ext.token_expiration = None
        ext.config = {"env": "preprod"}
        ext.workflow_ref = None
        ext.env = "preprod"
        ext.partial_path = "s3://bucket/partials"
        ext.output_path = "s3://bucket/results"
        ext.merge_existing = "auto"
        ext.logger = mocker.MagicMock()
        ext.extractor_name = "Test"
        ext.column_mapping = {"id": "id", "geometry": "geometry"}
        ext.output_mapping = None
        ext.exclude_columns = None
        ext.output_columns = None
        ext.use_cache = False
        ext.cache_key_columns = None

        df = pd.DataFrame({"id": ["a"], "value": [1]})
        ext._finalize_extraction(
            results_df=df,
            global_errors=[{"entity_id": "x", "message": "boom"}],
            failed_ids=["x"],
            output_path="s3://bucket/results",
            prefix="t",
            skip_export=True,  # don't exercise export_results here, already tested
            verbose=False,
        )

        # No makedirs was called for the s3:// partials path.
        assert all("s3://" not in p for p in makedirs_calls)
        # The failed_ids CSV was written via to_csv with an s3:// path and storage_options={}.
        assert any(c["path"].startswith("s3://bucket/partials/failed_ids_t_") for c in captured_to_csv)
        for c in captured_to_csv:
            if c["path"].startswith("s3://"):
                assert c.get("storage_options") == {}
