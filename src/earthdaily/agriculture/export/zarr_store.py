"""ZarrStore — package a Dataset as zarr and append to it (local or S3).

The persistence substrate for N-D cube exports. Both the harvest VI cube
(``(time, y, x)``) and the regional cube (``(date, entity)`` × parameters) need the
same dance: open-or-create a zarr store, write a Dataset with a pinned schema, and
**append along one dimension** idempotently. This factors that out so domain cube
builders stay thin — they assemble the Dataset and hand it to ``ZarrStore``.

    store = ZarrStore("s3://bucket/prefix/cube.zarr")
    store.write(ds, group="weather", chunks="auto")      # or chunks={"date":365,"entity":512}
    store.append(new_days, dim="date", group="weather")  # idempotent (dedup on the dim coord)

Design notes
------------
- **Backend** — ``s3://`` targets go through ``s3fs.S3Map``; anything else is a local
  path. ``s3fs`` / ``xarray`` are imported lazily so importing this module costs
  nothing until you use it (install the extras with ``earthdaily-agriculture[cube]``).
- **Chunking** — chunk BOTH axes, not one: entity/space chunking is what makes
  per-entity time-series reads cheap. ``chunks="auto"`` picks a shape-aware default
  (``{date:365, entity:512}`` for a regional cube; ``{time:20, y:256, x:256}`` for a
  ``(time,y,x)`` cube — a bounded ~1-10 MB chunk regardless of field size that also leaves
  room for a future ``append_dim`` region write). Applied via ``to_zarr(encoding=...)`` —
  **no dask required**. Each chunk is one S3 GET.
- **Join on append** — the concat's alignment over the NON-append dims is shape-aware
  (``default_join``): ``"outer"`` for ``(date,entity)`` (entities onboarded mid-season pad
  older dates with NaN — intended) but ``"exact"`` for ``(time,y,x)`` (a shifted pinned grid
  is corruption, not a union — RAISE, don't silently NaN-pad two incoherent footprints).
  Override via ``append(join=...)``. Setting it explicitly also silences xarray's
  ``join``-default FutureWarning (that default is changing ``outer`` -> ``exact``).
- **Append cost** — read-modify-write: each ``append`` loads the whole existing store and
  rewrites the whole combined store (``mode="w"``), so a call is O(current store size) in I/O
  and a season of N appends is ~O(N²·slab) cumulative — the dominant cost on a large, often-
  appended remote store. A ``append_dim`` region write (O(slab) per append, enabled by the
  bounded chunking above) is a tracked follow-up; until then, batch appends where possible.
  Writing to a named group means an append to one group never disturbs sibling groups.
"""

from __future__ import annotations

import math
import os

# Shape-aware chunking defaults, keyed by a frozenset of the cube's dims.
_CHUNK_DEFAULTS = {
    frozenset(("date", "entity")): {"date": 365, "entity": 512},
    # (time, y, x): a bounded ~1-10 MB chunk regardless of field size. Capping y/x at 256
    # decouples the chunk size from the spatial extent, and a finite time chunk (vs the old
    # single monolithic {-1,-1,-1} chunk, which was ~150 MB for a 500x500x150 cube and made
    # a region append_dim impossible) keeps it under target. Upper bound per chunk =
    # 20*256*256*4 B = 5.2 MB; smaller fields chunk smaller (min(cap, size) in _encoding).
    frozenset(("time", "y", "x")): {"time": 20, "y": 256, "x": 256},
}

# Shape-aware concat join for `append` over the NON-append dims (see module docstring).
_JOIN_DEFAULTS = {
    frozenset(("date", "entity")): "outer",  # entity onboarding pads old dates NaN — intended
    frozenset(("time", "y", "x")): "exact",  # grid drift is corruption — raise, don't union
}
_FALLBACK_JOIN = "outer"  # unknown shapes: preserve the historical (implicit-outer) behaviour


class ZarrStore:
    """A zarr store on a local path or S3, with write + idempotent append."""

    def __init__(self, uri: str):
        self.uri = str(uri)

    # -- backend ----------------------------------------------------------
    @property
    def is_remote(self) -> bool:
        return self.uri.startswith(("s3://", "gs://", "az://", "abfs://"))

    def _store(self):
        if self.uri.startswith("s3://"):
            import s3fs

            return s3fs.S3Map(root=self.uri[len("s3://") :], s3=s3fs.S3FileSystem(), check=False)
        return self.uri

    def exists(self) -> bool:
        if self.is_remote:
            return True  # let open() decide; a remote HEAD per call is wasteful
        return os.path.exists(self.uri)

    def open(self, group: str | None = None, *, validate=None):
        """Open the store (or one group); None if absent, unreadable, or rejected by ``validate``.

        ``validate`` is an optional ``Callable[[Dataset], bool]``. ``to_zarr`` is not atomic, so
        a run killed mid-write (CI timeout, spot eviction, OOM) can leave a store with dims but
        no coords / no domain variables. ``ZarrStore`` is dimension-agnostic and can't know that
        a ``(time,y,x)`` cube "must" carry ``field_mask``, so pass a domain check here: when it
        returns False the store is reported ABSENT (``None``), so the next ``write``/``append``
        rebuilds it instead of a torn store poisoning every later run. Default ``None`` keeps the
        previous behaviour (return whatever is found).
        """
        import xarray as xr

        try:
            ds = xr.open_zarr(self._store(), group=group)
        except Exception:
            # zarr/s3fs raise a variety of "group not found" errors; treat as absent.
            return None
        if validate is not None and not validate(ds):
            return None
        return ds

    # -- chunking ---------------------------------------------------------
    @staticmethod
    def default_chunks(ds) -> dict | None:
        """Shape-aware chunk spec for ``ds`` (None if the shape isn't recognised)."""
        return _CHUNK_DEFAULTS.get(frozenset(ds.dims))

    @staticmethod
    def default_join(ds) -> str:
        """Shape-aware concat join for ``append`` over the non-append dims (see ``_JOIN_DEFAULTS``)."""
        return _JOIN_DEFAULTS.get(frozenset(ds.dims), _FALLBACK_JOIN)

    def _encoding(self, ds, chunks):
        """Build a to_zarr ``encoding`` mapping data_vars -> chunk tuples.

        ``chunks`` is a {dim: size} dict (``-1`` / missing dim = full length) or
        ``"auto"`` (use ``default_chunks``) or None (no explicit chunking).
        """
        if chunks == "auto":
            chunks = self.default_chunks(ds)
        if not chunks:
            return {}
        enc = {}
        for name, var in ds.data_vars.items():
            sizes = ds.sizes
            tup = []
            for d in var.dims:
                c = chunks.get(d, -1)
                dim_len = int(sizes[d])
                tup.append(dim_len if c in (-1, None) else min(int(c), dim_len))
            enc[name] = {"chunks": tuple(tup)}
        return enc

    # -- write / append ---------------------------------------------------
    def write(
        self,
        ds,
        *,
        group: str | None = None,
        mode: str = "w",
        chunks="auto",
        manifest: bool = False,
        manifest_prefix: str | None = None,
        manifest_metadata: dict | None = None,
    ):
        """Write ``ds`` to the store. ``chunks`` = 'auto' | {dim: size} | None.

        With ``manifest=True`` also drops a ``generate_manifest`` sidecar next to a LOCAL
        store (non-fatal; skipped with a note for remote stores, whose sidecar write
        needs the cloud-fs path — a follow-up).
        """
        ds.to_zarr(self._store(), group=group, mode=mode, encoding=self._encoding(ds, chunks), consolidated=True)
        if manifest:
            self._emit_manifest(ds, manifest_prefix, manifest_metadata)

    def append(
        self,
        ds,
        *,
        dim: str = "time",
        dedup_coord: str | None = None,
        group: str | None = None,
        chunks="auto",
        join="auto",
        validate=None,
    ):
        """Append ``ds`` along ``dim``, de-duplicating on ``dedup_coord`` (default = ``dim``).

        Read-modify-write: reads the existing group, concats ``ds`` along ``dim``, drops
        duplicate ``dedup_coord`` values (so re-appending the same coordinates is a no-op) and
        rewrites the group (``mode="w"``). Only ``group`` is rewritten, so sibling groups are
        untouched. Creates the group if absent (or if ``validate`` rejects a torn one).

        ``join`` aligns the NON-append dims. ``"auto"`` uses the shape-aware ``default_join``:
        ``"exact"`` for ``(time,y,x)`` so a shifted pinned grid RAISES instead of silently
        NaN-unioning two incoherent footprints; ``"outer"`` for ``(date,entity)`` so mid-season
        entity onboarding pads older dates with NaN. Pass ``join="outer"``/``"exact"`` to
        override (also silences xarray's ``join``-default FutureWarning).

        ``validate`` is forwarded to :meth:`open`, so a torn existing store is treated as absent
        and rebuilt from ``ds`` rather than concatenated into.

        COST: read-modify-write ⇒ each call is O(current store size) in I/O; see the module
        docstring and batch appends where possible on large/remote stores.
        """
        import numpy as np
        import xarray as xr

        dedup_coord = dedup_coord or dim
        existing = self.open(group=group, validate=validate)
        if existing is None:
            return self.write(ds, group=group, mode="w", chunks=chunks)

        join = self.default_join(ds) if join == "auto" else join
        existing = existing.load().drop_vars("spatial_ref", errors="ignore")
        ds = ds.drop_vars("spatial_ref", errors="ignore")
        combined = xr.concat(
            [existing, ds], dim=dim, join=join, data_vars="minimal", coords="minimal", compat="override"
        )
        _, keep = np.unique(combined[dedup_coord].values, return_index=True)
        combined = combined.isel({dim: np.sort(keep)})
        self.write(combined, group=group, mode="w", chunks=chunks)
        return combined

    # -- manifest ---------------------------------------------------------
    def _emit_manifest(self, ds, prefix, metadata):
        if self.is_remote:
            return  # generate_manifest uses local file IO; cloud sidecar is a follow-up
        from earthdaily.agriculture.export.manifest import generate_manifest

        out_dir = os.path.dirname(self.uri) or "."
        prefix = prefix or os.path.splitext(os.path.basename(self.uri))[0]
        entity_col = "entity" if "entity" in ds.dims else ("amu_id" if "amu_id" in ds.dims else "id")
        date_col = "date" if "date" in ds.dims else "time"
        try:
            generate_manifest(
                source=ds,
                output_path=out_dir,
                prefix=prefix,
                entity_col=entity_col,
                date_col=date_col,
                metadata=metadata or {},
            )
        except Exception:
            pass  # non-fatal, mirrors the extractor manifest capability


def estimate_chunk_mb(chunks: dict, dtype_bytes: int = 4) -> float:
    """Rough MB per chunk for a {dim: size} spec — handy for tuning to the ~1–10 MB target."""
    n = 1
    for v in chunks.values():
        if v and v > 0:
            n *= int(v)
    return n * dtype_bytes / 1e6


def n_chunks(sizes: dict, chunks: dict) -> int:
    """Number of chunks a {dim: size} spec produces over a {dim: length} cube."""
    total = 1
    for d, length in sizes.items():
        c = chunks.get(d, -1)
        total *= 1 if c in (-1, None) else math.ceil(int(length) / int(c))
    return total
