"""Split an entity table into shards for a large stateless run (the fan-out step).

Writes ``<base>/input/shard-<i>.parquet`` and ``<base>/shards.json``; each container then
runs ``app.run_extractor --manifest <base> --shard <i>``. Size the run either with a
fixed ``--num-shards`` or from a time budget: one container's measured throughput for
the extractor (``--rate``, entities/minute) and how long a container may run::

    python -m app.dispatch_shards --entities s3://bucket/inputs/fields.parquet \\
        --base s3://bucket/runs/run_20261002 --rate 89 --target-minutes 15

Measured rates at 10 threads (2026-10-02): coverage ~960/min, GDD ~325, MRTS ~320
(historical_years=0), weather ~89. Keep ``containers running at once x --max-workers``
under the API's rate limit — that is the scheduler's parallelism setting, not this one.

Prints ``SHARDS_PATH=<base>/shards.json`` and ``NUM_SHARDS=<n>`` (shards actually
written; an empty one is skipped).
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from earthdaily.agriculture.core._fs import join_path
from earthdaily.agriculture.core.sharding import MANIFEST_NAME, plan_num_shards, read_table, split_entities


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split entities into shards for a stateless run.")
    p.add_argument("--entities", required=True, help="Entity table (.parquet / .csv), local or remote.")
    p.add_argument("--base", required=True, help="Run base prefix, e.g. s3://bucket/runs/run_20261002.")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--num-shards", type=int, help="Fixed number of shards.")
    size.add_argument("--rate", type=float, help="One container's throughput, entities/minute.")
    p.add_argument("--target-minutes", type=float, default=15.0, help="Time budget per container (default 15).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    entities = read_table(args.entities)
    num_shards = args.num_shards or plan_num_shards(len(entities), args.rate, args.target_minutes)
    manifest = split_entities(entities, args.base, num_shards)
    logger.info(f"{len(entities)} entities -> {len(manifest['shards'])} shard(s) under {args.base}")

    print(f"NUM_SHARDS={len(manifest['shards'])}")
    print(f"SHARDS_PATH={join_path(args.base, MANIFEST_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
