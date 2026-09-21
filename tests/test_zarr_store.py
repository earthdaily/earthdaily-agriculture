"""Tests for earthdaily.agriculture.export.ZarrStore — the zarr write/append substrate.

Offline (local tmp stores). Covers roundtrip, shape-aware + explicit chunking, idempotent
append with de-duplication, sibling-group safety, and local manifest emission.
"""

import glob

import numpy as np
import pandas as pd
import pytest

# ZarrStore needs the [cube] extra (xarray/zarr). Skip the whole module when it's
# absent — mirrors how the cloud-writer tests importorskip s3fs/adlfs, so the default
# CI (which installs only [test]) collects cleanly instead of erroring on import.
xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

from earthdaily.agriculture.export import ZarrStore, estimate_chunk_mb, n_chunks

pytestmark = pytest.mark.public


def _ds(n_date=6, n_ent=7, start="2025-01-01", base=0.0):
    dates = pd.date_range(start, periods=n_date)
    ents = np.arange(n_ent, dtype="int64")
    data = (np.arange(n_date * n_ent, dtype="float32") + base).reshape(n_date, n_ent)
    return xr.Dataset({"val": (("date", "entity"), data)}, coords={"date": dates, "entity": ents})


def _de(entities, dates, base=0.0):
    """(date, entity) cube with explicit entity + date coords."""
    ents = np.asarray(entities, dtype="int64")
    dts = pd.to_datetime(dates)
    data = (np.arange(len(dts) * len(ents), dtype="float32") + base).reshape(len(dts), len(ents))
    return xr.Dataset({"val": (("date", "entity"), data)}, coords={"date": dts, "entity": ents})


def _cube(times, y, x, var="NDVI", val=1.0):
    """(time, y, x) VI cube with explicit y/x grid coords."""
    data = np.full((len(times), len(y), len(x)), val, dtype="float32")
    return xr.Dataset(
        {var: (("time", "y", "x"), data)},
        coords={"time": np.asarray(times), "y": np.asarray(y), "x": np.asarray(x)},
    )


class TestChunkingHelpers:
    def test_default_chunks_regional(self):
        assert ZarrStore("x").default_chunks(_ds()) == {"date": 365, "entity": 512}

    def test_default_chunks_cube(self):
        ds = xr.Dataset({"NDVI": (("time", "y", "x"), np.zeros((3, 4, 5), "float32"))})
        # bounded chunk (~1-10 MB) — not a single monolithic {-1,-1,-1} chunk
        assert ZarrStore("x").default_chunks(ds) == {"time": 20, "y": 256, "x": 256}
        assert estimate_chunk_mb({"time": 20, "y": 256, "x": 256}) <= 10  # inside target

    def test_default_join_is_shape_aware(self):
        assert ZarrStore("x").default_join(_ds()) == "outer"  # (date, entity)
        assert ZarrStore("x").default_join(_cube([0], [0, 1, 2, 3], [0, 1, 2])) == "exact"  # (time, y, x)

    def test_default_chunks_unknown_shape(self):
        ds = xr.Dataset({"v": (("foo",), np.zeros(3, "float32"))})
        assert ZarrStore("x").default_chunks(ds) is None

    def test_encoding_respects_dim_order_and_caps_to_size(self):
        ds = _ds(n_date=6, n_ent=7)
        enc = ZarrStore("x")._encoding(ds, {"date": 2, "entity": 100})
        # dims are (date, entity); entity cap 100 -> min(100, 7) = 7
        assert enc["val"]["chunks"] == (2, 7)

    def test_n_chunks_and_estimate(self):
        assert n_chunks({"date": 1094, "entity": 3071}, {"date": 365, "entity": 512}) == 3 * 6
        assert round(estimate_chunk_mb({"date": 365, "entity": 512}), 2) == 0.75


class TestWriteOpen:
    def test_roundtrip(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        assert store.exists() is False
        store.write(_ds())
        assert store.exists() is True
        got = store.open()
        assert dict(got.sizes) == {"date": 6, "entity": 7}
        assert float(got["val"].isel(date=0, entity=0)) == 0.0

    def test_applies_chunks(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_ds(n_date=6, n_ent=7), chunks={"date": 2, "entity": 3})
        arr = xr.open_zarr(str(tmp_path / "cube.zarr"))["val"]
        assert tuple(arr.encoding["chunks"]) == (2, 3)

    def test_open_absent_returns_none(self, tmp_path):
        assert ZarrStore(str(tmp_path / "nope.zarr")).open() is None


class TestAppend:
    def test_append_grows_along_dim(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_ds(n_date=3, start="2025-01-01"))
        store.append(_ds(n_date=2, start="2025-01-04", base=100.0), dim="date")
        got = store.open()
        assert got.sizes["date"] == 5
        assert list(pd.to_datetime(got["date"].values).strftime("%Y-%m-%d")) == [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
        ]

    def test_append_is_idempotent(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        ds = _ds(n_date=3, start="2025-01-01")
        store.write(ds)
        store.append(ds, dim="date")  # same dates again
        assert store.open().sizes["date"] == 3  # de-duplicated, no growth

    def test_append_creates_when_absent(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.append(_ds(n_date=2), dim="date")  # no prior write
        assert store.open().sizes["date"] == 2

    def test_append_preserves_sibling_group(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_ds(n_date=3), group="weather", mode="w")
        store.write(_ds(n_date=4), group="vvi", mode="a")
        store.append(_ds(n_date=1, start="2025-02-01", base=9.0), dim="date", group="weather")
        assert store.open(group="weather").sizes["date"] == 4  # grew
        assert store.open(group="vvi").sizes["date"] == 4  # untouched


class TestManifest:
    def test_local_manifest_sidecar(self, tmp_path):
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_ds(), manifest=True, manifest_metadata={"join_key": "entity"})
        hits = glob.glob(str(tmp_path / "cube_manifest_*.json"))
        assert hits, "expected a manifest sidecar next to the local store"


class TestAppendJoin:
    def test_timeyx_shifted_grid_raises(self, tmp_path):
        """(time,y,x): a drifted pinned grid must RAISE, not silently NaN-union two footprints."""
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_cube([0], [0, 1, 2, 3], [0, 1, 2]))
        shifted = _cube([1], [2, 3, 4, 5], [0, 1, 2])  # y origin drifted by 2
        with pytest.raises(ValueError):  # xarray AlignmentError subclasses ValueError
            store.append(shifted, dim="time")  # join="auto" -> "exact"

    def test_timeyx_explicit_outer_override_unions(self, tmp_path):
        """The strict default is overridable: join='outer' restores the (lenient) union."""
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_cube([0], [0, 1, 2, 3], [0, 1, 2]))
        store.append(_cube([1], [2, 3, 4, 5], [0, 1, 2]), dim="time", join="outer")
        got = store.open()
        assert got.sizes == {"time": 2, "y": 6, "x": 3}  # y unioned 0..5

    def test_dateentity_entity_growth_is_outer_padded(self, tmp_path):
        """(date,entity): mid-season entity onboarding is an INTENDED outer-join NaN-pad."""
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(_de([0, 1, 2, 3], ["2025-01-01", "2025-01-02", "2025-01-03"], base=1.0))
        got = store.append(  # join="auto" -> "outer"
            _de([2, 3, 4, 5], ["2025-01-04", "2025-01-05"], base=100.0), dim="date"
        )
        assert dict(got.sizes) == {"date": 5, "entity": 6}  # entity union 0..5
        # entity 5 (new) has no data on an original date -> NaN there, by design
        old_date = got["date"].values[0]
        assert bool(np.isnan(float(got["val"].sel(entity=5, date=old_date))))
        # entity 2 (in both) is populated on that same old date
        assert not np.isnan(float(got["val"].sel(entity=2, date=old_date)))


class TestOpenValidate:
    def test_validate_rejects_torn_store(self, tmp_path):
        """A torn store (dims but missing coords/domain var) is reported ABSENT via validate."""
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        # simulate a run killed mid-write: dims only, no NDVI var, no y/x coords
        torn = xr.Dataset({"placeholder": (("time", "y", "x"), np.zeros((2, 4, 3), "float32"))})
        store.write(torn)

        def is_valid(ds):
            return "NDVI" in ds.data_vars and "y" in ds.coords

        assert store.open(validate=is_valid) is None  # rejected -> treated as absent
        assert store.open() is not None  # default (no validate) still returns whatever is there

    def test_append_rebuilds_over_torn_store(self, tmp_path):
        """append(validate=...) treats a torn store as absent and rebuilds from ds."""
        store = ZarrStore(str(tmp_path / "cube.zarr"))
        store.write(xr.Dataset({"placeholder": (("time", "y", "x"), np.zeros((1, 4, 3), "float32"))}))
        good = _cube([0], [0, 1, 2, 3], [0, 1, 2])
        store.append(good, dim="time", validate=lambda ds: "NDVI" in ds.data_vars)
        out = store.open()
        assert "NDVI" in out.data_vars and out.sizes["time"] == 1
