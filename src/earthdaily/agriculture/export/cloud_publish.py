"""
Cloud publish — post-extraction hand-off from local disk to cloud storage.

This is the **local→cloud hand-off** scenario: a run that wrote results to local
disk (results/, partials/) and you want to push everything to a cloud bucket
after the fact. For the *during-extraction* scenario, just point
``WorkflowManager``'s ``output_result_dir`` / ``partial_result_dir`` at a remote
URI (``s3://``, ``az://``, ``gs://`` ...) — the writer layer (see
``earthdaily.agriculture.core._fs``) routes through fsspec directly. That path makes
this module unnecessary in the dockerization / cloud-native flow.

This module reads a manifest produced by ``generate_manifest`` (with file
entries from ``export_dataframe``), uploads each tracked file to a target cloud
prefix via fsspec, patches the manifest in place with the resulting cloud URIs,
and uploads the patched manifest itself. The backend is whatever fsspec scheme
the prefix carries (``s3://``, ``az://``, ``gs://`` ...) — credentials come from
the ambient environment per backend.

Usage::

    from earthdaily.agriculture.export import publish_local_run_to_cloud

    new_manifest = publish_local_run_to_cloud(
        manifest_path="results/coverage_manifest_2026-05-12.json",
        publish_prefix="s3://my-bucket/runs/2026-05-12/coverage",
    )

Or, wired into a YAML workflow as a transform step::

    - name: publish
      transform:
        module: earthdaily.agriculture.export.cloud_publish
        function: publish_to_cloud
        params:
          publish_prefix: s3://my-bucket/runs/2026-05-12
          output_dir: results
          publish_formats: [parquet]      # optional: filter by format
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from earthdaily.agriculture.export.manifest import load_manifest


def _join_uri(prefix: str, name: str) -> str:
    """Join a cloud prefix and a filename without doubling the slash."""
    return f"{prefix.rstrip('/')}/{name}"


def publish_local_run_to_cloud(
    manifest_path: str | Path,
    publish_prefix: str,
    *,
    publish_formats: list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    rewrite_path_to_cloud: bool = False,
) -> str:
    """Upload every file tracked in a manifest to a cloud prefix.

    Reads ``manifest_path`` (a JSON file produced by ``generate_manifest``),
    walks the ``files`` list, copies each local file to
    ``publish_prefix/<filename>`` via fsspec, patches the matching file entry
    with a ``cloud_upload`` block (``{uri, key, bucket, uploaded_at}``), adds a
    top-level ``cloud_publish`` summary, and writes the patched manifest both
    next to the original *and* to ``publish_prefix/<manifest-filename>``.

    Args:
        manifest_path: Path to the local manifest JSON.
        publish_prefix: Target cloud prefix (e.g.
            ``"s3://my-bucket/runs/2026-05-12"``, ``"az://..."``, ``"gs://..."``).
        publish_formats: Optional list of formats to upload (e.g. ``["parquet"]``).
            When set, entries whose ``format`` is not in the list are skipped.
            Default: upload every entry.
        storage_options: Forwarded to ``fsspec.open(..., **storage_options)``
            for the upload calls. Default: ``None`` (use the ambient
            backend credentials — ``AWS_*`` / ``AZURE_*`` / ``GOOGLE_*`` env,
            instance profile / SSO chain). Useful for passing
            ``client_kwargs={"endpoint_url": "..."}`` against MinIO or
            LocalStack.
        rewrite_path_to_cloud: When ``True``, every entry that is actually
            uploaded has its ``path`` rewritten to the cloud URI (the same value
            as ``cloud_upload["uri"]``) and its original local path preserved
            under ``local_path``. This lets consumers reading the manifest
            straight from cloud storage resolve ``path`` directly. Idempotent:
            an entry that already carries ``local_path`` is left untouched.
            Skipped entries (filtered by ``publish_formats`` or missing on disk)
            keep their original local ``path`` and gain no ``local_path``.
            Default: ``False`` (``path`` stays local; only ``cloud_upload`` is added).

    Returns:
        Local path of the patched manifest (overwrites ``manifest_path``).
    """
    import fsspec  # local import — keeps the export package importable without a backend extra

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    files = manifest.get("files") or []

    if not files:
        logger.warning(f"Manifest {manifest_path} has no 'files' list — nothing to publish.")
        return str(manifest_path)

    bucket = publish_prefix.split("://", 1)[1].split("/", 1)[0] if "://" in publish_prefix else ""

    uploaded = 0
    skipped = 0
    patched_entries: list[dict[str, Any]] = []
    # Explicit storage_options win; otherwise resolve per-backend credentials from the
    # environment (Azure needs account+credential passed explicitly — see core._fs).
    if storage_options is not None:
        storage_kwargs = storage_options
    else:
        from earthdaily.agriculture.core._fs import storage_options_for

        storage_kwargs = storage_options_for(publish_prefix) or {}

    for entry in files:
        fmt = entry.get("format")
        if publish_formats is not None and fmt not in publish_formats:
            logger.debug(f"Skipping {entry.get('filename')} — format {fmt!r} not in publish_formats")
            patched_entries.append(entry)
            skipped += 1
            continue

        local_path = entry.get("path")
        if not local_path or not os.path.isfile(local_path):
            logger.warning(f"Missing local file for entry {entry.get('filename')!r}: {local_path} — skipping")
            patched_entries.append(entry)
            skipped += 1
            continue

        filename = entry.get("filename") or os.path.basename(local_path)
        target_uri = _join_uri(publish_prefix, filename)

        with open(local_path, "rb") as src, fsspec.open(target_uri, "wb", **storage_kwargs) as dst:
            dst.write(src.read())

        key = target_uri.split("://", 1)[1].split("/", 1)[1] if "://" in target_uri else target_uri
        entry = {
            **entry,
            "cloud_upload": {
                "uri": target_uri,
                "key": key,
                "bucket": bucket,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        if rewrite_path_to_cloud:
            # Point `path` at the cloud URI so cloud-side consumers resolve it directly,
            # preserving the local path under `local_path`. Only stash local_path once so
            # re-publishing an already-rewritten manifest doesn't clobber the original.
            entry.setdefault("local_path", local_path)
            entry["path"] = target_uri
        patched_entries.append(entry)
        uploaded += 1
        logger.info(f"☁ uploaded {filename} → {target_uri}")

    manifest["files"] = patched_entries
    manifest["cloud_publish"] = {
        "bucket": bucket,
        "publish_prefix": publish_prefix,
        "uploaded_files": uploaded,
        "skipped_files": skipped,
        "total_files": len(files),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save the patched manifest locally (overwrites the original).
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Patched manifest saved: {manifest_path}")

    # Upload the patched manifest alongside the files so consumers reading
    # straight from cloud storage see the full picture.
    manifest_target = _join_uri(publish_prefix, manifest_path.name)
    with open(manifest_path, "rb") as src, fsspec.open(manifest_target, "wb", **storage_kwargs) as dst:
        dst.write(src.read())
    logger.info(f"☁ uploaded manifest → {manifest_target}")

    return str(manifest_path)


def publish_to_cloud(
    entity_list: pd.DataFrame,
    upstream_results: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Workflow-transform-shaped wrapper around ``publish_local_run_to_cloud``.

    Hook this into a YAML workflow as a transform step::

        - name: publish
          transform:
            module: earthdaily.agriculture.export.cloud_publish
            function: publish_to_cloud
            params:
              publish_prefix: s3://my-bucket/runs/2026-05-12
              output_dir: results
              publish_formats: [parquet]

    Finds the most recent ``*_manifest_*.json`` under ``output_dir`` (default
    ``"results"``) and uploads everything it references to ``publish_prefix``.
    Returns ``entity_list`` unchanged so downstream steps that depend on this
    one don't need to know publish ran.

    Required params: ``publish_prefix`` (target cloud prefix).
    Optional params:
        - ``output_dir``: directory holding the manifest (default ``"results"``)
        - ``publish_formats``: list[str], filter by file format
        - ``storage_options``: dict forwarded to fsspec
        - ``manifest_path``: explicit manifest to publish (overrides search)
        - ``rewrite_path_to_cloud``: bool, rewrite each uploaded entry's ``path``
          to the cloud URI (original preserved under ``local_path``). Default ``False``.
    """
    publish_prefix = params.get("publish_prefix")
    if not publish_prefix:
        raise ValueError(
            "publish_to_cloud: 'params.publish_prefix' is required (e.g. 's3://my-bucket/runs/2026-05-12')."
        )

    manifest_path = params.get("manifest_path")
    if not manifest_path:
        output_dir = Path(params.get("output_dir") or "results")
        candidates = sorted(output_dir.glob("*_manifest_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"publish_to_cloud: no *_manifest_*.json found under {output_dir} — "
                "generate one with generate_manifest first, or pass params.manifest_path."
            )
        manifest_path = candidates[-1]
        logger.info(f"publish_to_cloud: using most-recent manifest {manifest_path}")

    publish_local_run_to_cloud(
        manifest_path=manifest_path,
        publish_prefix=publish_prefix,
        publish_formats=params.get("publish_formats"),
        storage_options=params.get("storage_options"),
        rewrite_path_to_cloud=bool(params.get("rewrite_path_to_cloud", False)),
    )
    return entity_list
