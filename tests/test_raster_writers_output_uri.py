"""Tests for the ``postprocess="file"`` raster writers' storage routing.

Covers ``BaseExtractor.save_map_file`` and the three writers that call it
(FLM, Difference, Zoning), which previously used ``open(..., "wb")`` directly
and so could only ever write to a local directory.

Two behaviours matter here:

* ``save_path`` may be a remote URI — an ``s3://`` prefix must reach fsspec,
  not become a literal ``./s3:/bucket/...`` directory.
* When ``output_uri`` is set, every saved file is mirrored there as the durable
  copy, while ``saved_files`` (and therefore the manifest) keeps recording the
  local working copy, because the analysis step reads rasters back off disk.
"""

from __future__ import annotations

import sys
import types
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

pytestmark = pytest.mark.public


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_response(content: bytes = b"\x89PNG-payload"):
    """A requests-like response carrying PNG bytes."""
    resp = MagicMock()
    resp.content = content
    return resp


def _tiff_zip_response(tif_bytes: bytes = b"II*\x00tif-payload"):
    """A requests-like response carrying a ZIP with one .tif inside."""
    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        zf.writestr("map.tif", tif_bytes)
    resp = MagicMock()
    resp.content = buf.getvalue()
    return resp


class _RecordingFsspec:
    """Stand-in for fsspec that records writes instead of performing them."""

    def __init__(self):
        self.writes: dict[str, bytes] = {}

    def open(self, path, mode, **kwargs):
        writes = self.writes

        class _Fh:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def write(self_inner, data):
                writes[path] = data

        return _Fh()


@pytest.fixture
def fake_fsspec(monkeypatch):
    """Install a recording fsspec so remote writes need no network or creds."""
    recorder = _RecordingFsspec()
    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=recorder.open))
    return recorder


# ---------------------------------------------------------------------------
# BaseExtractor.save_map_file
# ---------------------------------------------------------------------------


class TestSaveMapFile:
    def test_writes_local_copy_and_returns_its_path(self, configured_difference_extractor, tmp_path):
        ext = configured_difference_extractor
        path = ext.save_map_file(str(tmp_path), "field.tif", b"raster")

        assert (tmp_path / "field.tif").read_bytes() == b"raster"
        assert path == str(tmp_path / "field.tif")

    def test_creates_missing_local_subdirectories(self, configured_difference_extractor, tmp_path):
        ext = configured_difference_extractor
        target_dir = tmp_path / "tifs" / "NDVI"
        ext.save_map_file(str(target_dir), "field.tif", b"raster")

        assert (target_dir / "field.tif").read_bytes() == b"raster"

    def test_no_mirror_when_output_uri_unset(self, configured_difference_extractor, tmp_path, fake_fsspec):
        ext = configured_difference_extractor
        ext.output_uri = None
        ext.save_map_file(str(tmp_path), "field.tif", b"raster")

        assert fake_fsspec.writes == {}

    def test_missing_output_uri_attribute_is_tolerated(self, configured_difference_extractor, tmp_path):
        """Extractors built with __init__ bypassed have no output_uri at all."""
        ext = configured_difference_extractor
        del ext.output_uri
        path = ext.save_map_file(str(tmp_path), "field.tif", b"raster")

        assert (tmp_path / "field.tif").read_bytes() == b"raster"
        assert path == str(tmp_path / "field.tif")

    def test_mirrors_to_output_uri_when_set(self, configured_difference_extractor, tmp_path, fake_fsspec):
        ext = configured_difference_extractor
        ext.output_uri = "s3://bucket/prefix"
        ext.save_map_file(str(tmp_path), "field.tif", b"raster")

        assert fake_fsspec.writes == {"s3://bucket/prefix/field.tif": b"raster"}

    def test_returned_path_stays_local_when_mirroring(self, configured_difference_extractor, tmp_path, fake_fsspec):
        """The manifest must keep pointing at the copy the analysis can open."""
        ext = configured_difference_extractor
        ext.output_uri = "s3://bucket/prefix"
        path = ext.save_map_file(str(tmp_path), "field.tif", b"raster")

        assert path == str(tmp_path / "field.tif")
        assert not path.startswith("s3://")

    def test_both_copies_are_byte_identical(self, configured_difference_extractor, tmp_path, fake_fsspec):
        payload = bytes(range(256))
        ext = configured_difference_extractor
        ext.output_uri = "s3://bucket/prefix"
        ext.save_map_file(str(tmp_path), "field.tif", payload)

        assert (tmp_path / "field.tif").read_bytes() == payload
        assert fake_fsspec.writes["s3://bucket/prefix/field.tif"] == payload

    def test_remote_save_path_does_not_create_local_dir(
        self, configured_difference_extractor, tmp_path, fake_fsspec, monkeypatch
    ):
        """An s3:// save_path must not land in ./s3:/bucket/... — the original bug."""
        monkeypatch.chdir(tmp_path)
        ext = configured_difference_extractor
        ext.output_uri = None
        path = ext.save_map_file("s3://bucket/tifs", "field.tif", b"raster")

        assert path == "s3://bucket/tifs/field.tif"
        assert fake_fsspec.writes == {"s3://bucket/tifs/field.tif": b"raster"}
        assert not (tmp_path / "s3:").exists()

    def test_mirror_failure_propagates(self, configured_difference_extractor, tmp_path, monkeypatch):
        """A durable-write failure must surface, never be swallowed: a manifest
        claiming a raster that only ever existed on the runner is worse."""

        def _boom(path, mode, **kwargs):
            raise OSError("bucket unreachable")

        monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=_boom))
        ext = configured_difference_extractor
        ext.output_uri = "s3://bucket/prefix"

        with pytest.raises(OSError, match="bucket unreachable"):
            ext.save_map_file(str(tmp_path), "field.tif", b"raster")


# ---------------------------------------------------------------------------
# The writers, end to end through format_*_file
# ---------------------------------------------------------------------------


class TestWritersMirrorToOutputUri:
    def test_difference_png_mirrored(
        self, configured_difference_extractor, sample_difference_entity, tmp_path, fake_fsspec
    ):
        ext = configured_difference_extractor
        ext.difference_params["map_format"] = "png"
        ext.output_uri = "s3://bucket/diff"

        result = ext.format_difference_file(
            entity_data=sample_difference_entity,
            response=_png_response(),
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert len(fake_fsspec.writes) == 1
        remote_key = next(iter(fake_fsspec.writes))
        assert remote_key.startswith("s3://bucket/diff/")
        assert remote_key.endswith(".png")
        # Local copy is what the manifest records.
        assert result["saved_files"][0].startswith(str(tmp_path))

    def test_difference_tiff_mirrored(
        self, configured_difference_extractor, sample_difference_entity, tmp_path, fake_fsspec
    ):
        ext = configured_difference_extractor
        ext.difference_params["map_format"] = "tiff.zip"
        ext.output_uri = "s3://bucket/diff"

        result = ext.format_difference_file(
            entity_data=sample_difference_entity,
            response=_tiff_zip_response(),
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        remote_key = next(iter(fake_fsspec.writes))
        assert remote_key.endswith(".tif")
        # The unzipped raster, not the zip container.
        assert fake_fsspec.writes[remote_key] == b"II*\x00tif-payload"

    def test_local_only_writes_nothing_remote(
        self, configured_difference_extractor, sample_difference_entity, tmp_path, fake_fsspec
    ):
        """Default behaviour is unchanged: no output_uri, no remote traffic."""
        ext = configured_difference_extractor
        ext.difference_params["map_format"] = "png"
        ext.output_uri = None

        result = ext.format_difference_file(
            entity_data=sample_difference_entity,
            response=_png_response(),
            output_path=str(tmp_path),
        )

        assert result["status"] == "downloaded"
        assert fake_fsspec.writes == {}

    def test_mirror_failure_is_reported_as_entity_error(
        self, configured_difference_extractor, sample_difference_entity, tmp_path, monkeypatch
    ):
        """The writer's own error handling turns the raised failure into a
        per-entity error result rather than a crashed bulk run."""

        def _boom(path, mode, **kwargs):
            raise OSError("bucket unreachable")

        monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=_boom))
        ext = configured_difference_extractor
        ext.difference_params["map_format"] = "png"
        ext.output_uri = "s3://bucket/diff"

        result = ext.format_difference_file(
            entity_data=sample_difference_entity,
            response=_png_response(),
            output_path=str(tmp_path),
        )

        assert result["status"] == "error"
