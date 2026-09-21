"""Tests for the manifest-on-export capability (BaseExtractor.export_manifest).

Covers the resolution precedence (config > EDAGRO_EXPORT_MANIFEST env > default False)
and the _finalize_extraction wiring: a manifest sidecar is written next to a successful
export, only when enabled and not skipped, with manifest_metadata passed through.
"""

import glob
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from earthdaily.agriculture.core.base_extractor import BaseExtractor

pytestmark = pytest.mark.public


def _mk(config=None, **attrs):
    ext = BaseExtractor("token", datetime.now() + timedelta(hours=1), config=config or {})
    for k, v in attrs.items():
        setattr(ext, k, v)
    return ext


class TestResolution:
    def test_default_false(self, monkeypatch):
        monkeypatch.delenv("EDAGRO_EXPORT_MANIFEST", raising=False)
        assert _mk().export_manifest is False

    def test_config_true(self, monkeypatch):
        monkeypatch.delenv("EDAGRO_EXPORT_MANIFEST", raising=False)
        assert _mk({"export_manifest": True}).export_manifest is True

    def test_env_true(self, monkeypatch):
        monkeypatch.setenv("EDAGRO_EXPORT_MANIFEST", "true")
        assert _mk().export_manifest is True

    def test_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EDAGRO_EXPORT_MANIFEST", "true")
        assert _mk({"export_manifest": False}).export_manifest is False

    def test_metadata_passthrough(self, monkeypatch):
        monkeypatch.delenv("EDAGRO_EXPORT_MANIFEST", raising=False)
        assert _mk({"manifest_metadata": {"join_key": "amu_id"}}).manifest_metadata == {"join_key": "amu_id"}


class TestManifestOnExport:
    def _df(self):
        return pd.DataFrame(
            {
                "entity_id": [1, 1, 2],
                "date": ["2025-01-01", "2025-01-02", "2025-01-01"],
                "value": [1.0, 2.0, 3.0],
            }
        )

    def test_writes_manifest_next_to_export(self, tmp_path):
        ext = _mk(
            export_manifest=True,
            export_format="parquet",
            manifest_metadata={"join_key": "entity_id", "region_dimension_file": "regions.parquet"},
        )
        ext._finalize_extraction(
            self._df(),
            global_errors=[],
            failed_ids=[],
            output_path=str(tmp_path),
            prefix="reg_test",
            skip_export=False,
            verbose=False,
        )
        manifests = glob.glob(str(tmp_path / "reg_test_manifest_*.json"))
        assert manifests, "expected a manifest sidecar next to the export"
        m = json.load(open(manifests[0]))
        assert m["metadata"]["join_key"] == "entity_id"
        assert m["metadata"]["region_dimension_file"] == "regions.parquet"
        assert m["entities"]["count"] == 2
        assert m["date_range"]["unique_dates"] == 2

    def test_no_manifest_when_disabled(self, tmp_path):
        ext = _mk(export_manifest=False)
        ext._finalize_extraction(
            self._df(), [], [], output_path=str(tmp_path), prefix="reg_off", skip_export=False, verbose=False
        )
        assert not glob.glob(str(tmp_path / "reg_off_manifest_*.json"))

    def test_no_manifest_when_skip_export(self, tmp_path):
        ext = _mk(export_manifest=True)
        ext._finalize_extraction(
            self._df(), [], [], output_path=str(tmp_path), prefix="reg_skip", skip_export=True, verbose=False
        )
        assert not glob.glob(str(tmp_path / "reg_skip_manifest_*.json"))
