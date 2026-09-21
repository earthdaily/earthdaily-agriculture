"""Filesystem-spec helpers used by writers across the package.

Lets every writer accept either a local path (e.g. ``results/foo.csv``) or a
remote URI (e.g. ``s3://bucket/runs/foo.csv``) without per-call branching.
For local paths the helpers fall back to plain ``os`` operations; for remote
paths they delegate to ``fsspec`` (which pandas / pyarrow already use under
the hood for ``to_csv`` / ``to_parquet`` URIs).

Backends recognised today: ``s3://``, ``gs://``, ``gcs://``, ``az://``,
``azure://``, ``abfs://``, ``http://``, ``https://``, ``file://``. Anything
without a ``<scheme>://`` prefix is treated as a local path.
"""

from __future__ import annotations

import os
from typing import Any

# Schemes we actively want to recognise as "remote" — i.e. the writer should
# skip os.makedirs and let fsspec handle path semantics. ``file://`` is local
# but kept out of this set because we don't want to skip mkdir for it.
_REMOTE_SCHEMES = (
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "azure://",
    "abfs://",
    "abfss://",
    "http://",
    "https://",
)


def is_remote_path(path: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` when *path* targets a remote/object-store backend."""
    if path is None:
        return False
    s = str(path)
    return any(s.startswith(scheme) for scheme in _REMOTE_SCHEMES)


def join_path(base: str | os.PathLike[str], *parts: str) -> str:
    """Path-join that works for both local paths and remote URIs.

    For local paths defers to :func:`os.path.join` so platform separators are
    correct on Windows. For remote URIs concatenates with ``/`` — never
    backslashes — so the URL stays valid on Windows callers.
    """
    base_str = str(base)
    if is_remote_path(base_str):
        head = base_str.rstrip("/")
        tail = "/".join(p.strip("/") for p in parts if p)
        return f"{head}/{tail}" if tail else head
    return os.path.join(base_str, *parts)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Create *path* if it doesn't exist; no-op for remote URIs.

    Object stores have no real "directories" — fsspec creates the prefix
    implicitly when you write a key under it, so calling ``mkdir`` on
    ``s3://bucket/prefix`` is meaningless and would error on most backends.
    """
    if path is None:
        return
    s = str(path)
    if is_remote_path(s):
        return
    os.makedirs(s, exist_ok=True)


def _azure_storage_options() -> dict[str, Any]:
    """Build fsspec/adlfs ``storage_options`` for Azure from the environment.

    Unlike AWS (where boto3 auto-discovers ``AWS_*`` env vars / IAM role from an
    empty options dict), adlfs needs the account name plus a credential passed
    explicitly. Resolution precedence, first match wins:

    1. ``AZURE_STORAGE_CONNECTION_STRING`` — carries account + credential.
    2. ``AZURE_STORAGE_ACCOUNT_NAME`` + ``AZURE_STORAGE_SAS_TOKEN``.
    3. ``AZURE_STORAGE_ACCOUNT_NAME`` + ``AZURE_STORAGE_ACCOUNT_KEY``.
    4. ``AZURE_STORAGE_ACCOUNT_NAME`` alone → adlfs falls back to its default
       credential chain (``DefaultAzureCredential``: Managed Identity, az CLI,
       env service-principal …), so MI works in ECS/Argo with no secret.

    Returns ``{}`` when nothing is configured — adlfs then attempts anonymous /
    default-chain access and surfaces its own clear error if that fails.
    """
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return {"connection_string": conn}

    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    if not account:
        return {}

    opts: dict[str, Any] = {"account_name": account}
    sas = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    if sas:
        # adlfs accepts the token with or without the leading '?'; normalise it off.
        opts["sas_token"] = sas.lstrip("?")
    elif key:
        opts["account_key"] = key
    # else: account_name only → adlfs uses DefaultAzureCredential (Managed Identity …)
    return opts


def storage_options_for(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Return the ``storage_options`` dict to pass to pandas / pyarrow / fsspec.

    - Local paths → ``None`` (pandas uses ``open`` directly).
    - ``az://`` / ``azure://`` / ``abfs(s)://`` → Azure credentials resolved from
      the environment (see :func:`_azure_storage_options`); adlfs needs the
      account name + credential passed explicitly.
    - Other remote paths (``s3://`` …) → ``{}`` so pandas/pyarrow route through
      fsspec, which picks up credentials from the environment (``AWS_ACCESS_KEY_ID``,
      IAM role, ``AWS_ENDPOINT_URL`` for MinIO/LocalStack, etc.).

    Single place for per-backend credential/endpoint tweaks to live.
    """
    if path is None or not is_remote_path(path):
        return None
    s = str(path)
    if s.startswith(("az://", "azure://", "abfs://", "abfss://")):
        return _azure_storage_options()
    return {}


def write_text(path: str | os.PathLike[str], content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path*; routes through fsspec for remote URIs.

    Local paths get the parent directory created automatically (matches
    ``Path.write_text`` ergonomics with ``parents=True``). Remote URIs
    delegate to ``fsspec.open`` so the same call works for ``s3://``,
    ``gs://``, etc. ``fsspec`` is imported lazily so local-only callers
    don't pull it in at module load.
    """
    s = str(path)
    if is_remote_path(s):
        import fsspec  # lazy: only required when writing to remote backends

        with fsspec.open(s, "w", encoding=encoding, **(storage_options_for(s) or {})) as fh:
            fh.write(content)
        return

    from pathlib import Path

    p = Path(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)


def write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write *data* to *path*; routes through fsspec for remote URIs.

    Binary sibling of :func:`write_text`, with the same semantics: local paths
    get the parent directory created automatically, remote URIs delegate to
    ``fsspec.open`` so the same call works for ``s3://``, ``gs://``, etc., and
    ``fsspec`` is imported lazily so local-only callers don't pull it in at
    module load.

    Used by the ``postprocess="file"`` raster writers (FLM, Difference,
    Zoning), which previously called ``open(..., "wb")`` directly and so
    treated an ``s3://`` prefix as a directory name rather than a URI.
    """
    s = str(path)
    if is_remote_path(s):
        import fsspec  # lazy: only required when writing to remote backends

        with fsspec.open(s, "wb", **(storage_options_for(s) or {})) as fh:
            fh.write(data)
        return

    from pathlib import Path

    p = Path(s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def glob_files(pattern: str | os.PathLike[str]) -> list[str]:
    """Return a sorted list of paths matching *pattern* (local or remote).

    For local patterns delegates to :func:`glob.glob`; for remote patterns
    uses ``fsspec``'s filesystem-level ``glob``. Remote results are
    re-prefixed with the original scheme (``s3://``) — fsspec strips it.
    """
    s = str(pattern)
    if is_remote_path(s):
        import fsspec  # lazy

        # fsspec.glob returns paths *without* the scheme prefix — re-attach it
        # so callers can pass results back to fsspec.open / rm without changes.
        scheme = s.split("://", 1)[0]
        fs, _ = fsspec.core.url_to_fs(s, **(storage_options_for(s) or {}))
        matches = fs.glob(s)
        return sorted(f"{scheme}://{m}" if not m.startswith(f"{scheme}://") else m for m in matches)

    import glob as _glob

    return sorted(_glob.glob(s))


def remove_files(paths: list[str]) -> tuple[int, list[tuple[str, Exception]]]:
    """Delete each path in *paths*. Returns ``(deleted_count, errors)``.

    Local files are removed via :func:`os.remove`; remote URIs via
    ``fsspec``'s filesystem-level ``rm``. Errors are collected (with the
    offending path and exception) rather than raised, mirroring how the
    existing partial-cleanup loop tolerates per-file failures.
    """
    deleted = 0
    errors: list[tuple[str, Exception]] = []

    # Bucket paths by remote scheme so we can reuse one filesystem instance
    # per scheme — saves a connection setup per file on s3:// / gs://.
    remote_by_scheme: dict[str, list[str]] = {}
    local_paths: list[str] = []
    for p in paths:
        if is_remote_path(p):
            scheme = p.split("://", 1)[0]
            remote_by_scheme.setdefault(scheme, []).append(p)
        else:
            local_paths.append(p)

    for p in local_paths:
        try:
            os.remove(p)
            deleted += 1
        except OSError as e:
            errors.append((p, e))

    if remote_by_scheme:
        import fsspec  # lazy

        for scheme, group in remote_by_scheme.items():
            fs, _ = fsspec.core.url_to_fs(group[0], **(storage_options_for(group[0]) or {}))
            for p in group:
                try:
                    fs.rm(p)
                    deleted += 1
                except Exception as e:  # fsspec may raise non-OSError on remote backends
                    errors.append((p, e))

    return deleted, errors
