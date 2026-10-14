"""Merge every shard's results into one table (the fan-in step of a large stateless run).

Reads ``<base>/shard-<i>/results/*_final.*`` for the shards in ``<base>/shards.json``
and writes ``<base>/merged/results/<prefix>_merged_results_<ts>_final.<ext>``. Where a
retry pass or a re-run shard left several result files, each entity's rows come from
the newest file that has it — nothing is lost or doubled::

    python -m app.merge_shards --base s3://bucket/runs/run_20261002 --prefix coverage

Shards with no result are listed and the merge goes ahead; ``--require-all`` refuses
to publish a partial merge instead (exit 1). Map files (FLM / Difference / Zoning
``postprocess="file"``) stay where each shard wrote them, under
``<base>/shard-<i>/results/maps``.

Prints ``REPORT_PATH=<merged result>`` last.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from earthdaily.agriculture.core.sharding import merge_shards


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge sharded results into one table.")
    p.add_argument("--base", required=True, help="Run base prefix holding shards.json and shard-<i>/.")
    p.add_argument("--prefix", help="Output prefix the shards used (app.run_extractor --prefix).")
    p.add_argument("--num-shards", type=int, help="Override the manifest's shard list (0..N-1).")
    p.add_argument("--require-all", action="store_true", help="Fail instead of merging a partial result.")
    p.add_argument("--export-format", choices=["csv", "parquet"], help="Merged file format (default: csv).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        out = merge_shards(args.base, args.num_shards, args.prefix, args.require_all, args.export_format)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    if out["missing"]:
        logger.warning(f"PARTIAL merge — shard(s) with no result: {out['missing']}")
    logger.info(f"Merged {out['shards_merged']} shard(s), {out['rows']} rows")
    print(f"REPORT_PATH={out['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
