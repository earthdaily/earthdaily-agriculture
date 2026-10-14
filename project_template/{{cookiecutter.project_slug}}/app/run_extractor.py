"""Run ONE extractor on ONE shard — the entrypoint for large stateless runs.

Large runs (100k entities and up) skip WorkflowManager: each container builds the
extractor directly, runs its bulk method on its slice of entities, writes under
EDAGRO_OUTPUT_PREFIX, prints ``REPORT_PATH=<uri>`` last and exits. Small runs keep
``app/run_pipeline.py`` and ``configuration/workflow.yml``.

The shard comes either from a file, or from the manifest written by
``app/dispatch_shards.py`` (which also gives the shard its output prefix)::

    python -m app.run_extractor --extractor CoverageExtractor \\
        --setup '{"start_date": "2025-06-01", "end_date": "2025-08-31"}' \\
        --manifest s3://bucket/runs/run_20261002 --shard 3

    python -m app.run_extractor --extractor WeatherExtractor --entities fields.parquet \\
        --setup @configuration/weather_setup.json --run '{"spatial_grouping": true}'

Retry pass (only the entities this shard recorded as failed)::

    python -m app.run_extractor ... --manifest s3://... --shard 3 --retry-failed

Exit codes: 0 = done (failed entities are recorded, not fatal) or nothing to do;
1 = no result at all; 2 = bad arguments or storage that would not survive the container.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys

from loguru import logger

import earthdaily.agriculture as edagro
from earthdaily.agriculture.core._fs import is_remote_path
from earthdaily.agriculture.core.base_extractor import BaseExtractor
from earthdaily.agriculture.core.sharding import find_report, read_manifest, read_table

_PACKAGES = ("extractors", "processors", "external", "services")


def _json_arg(value: str) -> dict:
    """A JSON object given inline, or ``@path`` to a JSON file."""
    if not value:
        return {}
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(value)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one extractor on one shard (large stateless runs).")
    p.add_argument("--extractor", required=True, help="Extractor class name, e.g. CoverageExtractor.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--entities", help="Entity table (.parquet / .csv), local or s3:// / az://.")
    src.add_argument("--manifest", help="Run base prefix (or shards.json) written by app.dispatch_shards.")
    p.add_argument("--shard", type=int, help="Shard index in the manifest (with --manifest).")
    p.add_argument("--setup", default="", help="setup_*_parameters kwargs: JSON, or @file.json.")
    p.add_argument("--run", default="", help="Extra bulk-method kwargs: JSON, or @file.json.")
    p.add_argument(
        "--method",
        help="Bulk method (default: the only one, else process_entity_<type>_bulk_parallel).",
    )
    p.add_argument("--prefix", help="Output file prefix (default: the extractor name, lower-cased).")
    p.add_argument("--max-workers", type=int, default=10, help="Threads inside this container (default 10).")
    p.add_argument("--retry-failed", action="store_true", help="Process only the entities recorded as failed.")
    p.add_argument("--env", default=os.environ.get("ENVIRONMENT", "{{ cookiecutter.environment }}"))
    return p.parse_args(argv)


def _find_extractor(name: str) -> type:
    for sub in _PACKAGES:
        pkg = importlib.import_module(f"earthdaily.agriculture.{sub}")
        for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            try:
                module = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001 — optional modules may need extras
                continue
            cls = getattr(module, name, None)
            if inspect.isclass(cls) and issubclass(cls, BaseExtractor):
                return cls
    raise SystemExit(f"Unknown extractor {name!r} (earthdaily-agriculture {edagro.__version__}).")


def _own_methods(cls: type, predicate) -> list[str]:
    return sorted(
        name
        for name, fn in inspect.getmembers(cls, inspect.isfunction)
        if predicate(name) and fn.__qualname__.split(".")[0] == cls.__name__
    )


def _pick_methods(cls: type, requested: str | None = None) -> tuple[str | None, str | None, list[str]]:
    """``(setup_method, bulk_method, candidates)`` for *cls*; a ``None`` means "ask for --method".

    With one bulk method, that one. With several, the standard
    ``process_entity_<type>_bulk_parallel`` named after ``setup_<type>_parameters`` — so
    CoverageExtractor runs ``process_entity_coverage_bulk_parallel``, not its
    ``_recalibration`` variant. Extractors whose bulk methods are steps rather than
    variants (Tillage: submit, then retrieve) have no such method and still need --method.
    """
    setups = _own_methods(cls, lambda n: n.startswith("setup_") and n.endswith("_parameters"))
    bulk = [
        m for m in _own_methods(cls, lambda n: n.startswith("process_") and "bulk" in n) if "specific_dates" not in m
    ]
    setup = setups[0] if len(setups) == 1 else None
    if requested:
        return setup, (requested if requested in bulk else None), bulk
    if len(bulk) == 1:
        return setup, bulk[0], bulk
    standard = f"process_entity_{setup[len('setup_') : -len('_parameters')]}_bulk_parallel" if setup else None
    return setup, (standard if standard in bulk else None), bulk


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # 1. Which entities, and where this shard writes.
    if args.manifest:
        if args.shard is None:
            logger.error("--manifest needs --shard.")
            return 2
        manifest = read_manifest(args.manifest)
        entry = next((s for s in manifest["shards"] if s["shard"] == args.shard), None)
        if entry is None:
            logger.warning(f"Shard {args.shard} is not in the manifest (empty shard) — nothing to do.")
            return 0
        os.environ.setdefault("EDAGRO_OUTPUT_PREFIX", entry["output_prefix"])
        entities_uri = entry["input"]
    else:
        entities_uri = args.entities

    # 2. Config + guard: a stateless container must not write to its own disk.
    from earthdaily.agriculture.core.functions_enhanced import setup_environment
    from earthdaily.agriculture.core.identity import EDAuthenticator

    config = setup_environment(args.env)
    if os.environ.get("EDAGRO_STORAGE") == "s3" and not is_remote_path(config["output_result_dir"]):
        logger.error(
            f"EDAGRO_STORAGE=s3 but results resolve to {config['output_result_dir']!r}: "
            "set EDAGRO_OUTPUT_PREFIX to an s3:// / az:// prefix (or pass --manifest)."
        )
        return 2

    entities = read_table(entities_uri)
    if entities.empty:
        logger.warning(f"No entities in {entities_uri} — nothing to do.")
        return 0

    # 3. Build and configure the extractor.
    cls = _find_extractor(args.extractor)
    setup_method, method, bulk_methods = _pick_methods(cls, args.method)
    if method is None or setup_method is None:
        logger.error(f"{cls.__name__}: pass --method, one of {bulk_methods}.")
        return 2

    token, expires = EDAuthenticator.get_new_token(env=args.env)
    extractor = cls(token, expires, config)
    getattr(extractor, setup_method)(**_json_arg(args.setup))
    if args.retry_failed:
        extractor.retry_failed_only = True

    # 4. Run. fail_safe: one bad entity is recorded, never fails the container.
    prefix = args.prefix or cls.__name__.lower().replace("extractor", "") or "extract"
    run_kwargs = {"max_workers": args.max_workers, "fail_safe": True, "prefix": prefix, **_json_arg(args.run)}
    logger.info(f"{cls.__name__}.{method}: {len(entities)} entities -> {extractor.output_path}")
    try:
        result = getattr(extractor, method)(entity_list=entities, **run_kwargs)
    except FileNotFoundError as exc:
        if not args.retry_failed:
            raise
        # The retry pass on a shard that recorded no failures: nothing to do.
        logger.info(f"Nothing to retry: {exc}")
        return 0
    failed = result.get("failed_ids") or []
    logger.info(f"Done: {len(result.get('results_df', []))} rows, {len(failed)} failed entities recorded.")

    report = find_report(str(extractor.output_path), prefix)
    if report is None:
        logger.error(f"No *_final result under {extractor.output_path}.")
        return 1
    # The stdout contract: last line, stable format.
    print(f"REPORT_PATH={report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
