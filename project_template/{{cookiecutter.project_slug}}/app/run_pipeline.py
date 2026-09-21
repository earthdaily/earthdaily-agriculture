"""Headless entrypoint for {{ cookiecutter.project_name }}.

Drives ``configuration/workflow.yml`` through ``WorkflowManager`` and
prints exactly one ``REPORT_PATH=<absolute path>`` line as its last
meaningful stdout, so the CI workflow (or any orchestrator) can scrape
and forward the final artifact without parsing arbitrary log lines.

See ``docs/14 - Deployment_patterns.md`` for the contract this entrypoint
honours and the two supported deployment modes (Pattern A = GitHub Actions
cron; Pattern B = Docker container on ECS / Cloud Run / Argo).

Usage::

    python -m app.run_pipeline --prefix run_20260512
    python -m app.run_pipeline --prefix smoke --limit 5
    python -m app.run_pipeline --prefix shard_0 --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from earthdaily.agriculture.services.workflow_manager import WorkflowManager


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the {{ cookiecutter.project_name }} pipeline.")
    p.add_argument(
        "--prefix",
        required=True,
        help="Run prefix used in filenames + workflow_results['run_prefix']. "
        "Cron callers should pass run_$(date -u +%%Y%%m%%d).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test cap on the number of entities processed.",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Optional shard index for fan-out (used with --num-shards).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards in a fan-out run (used with --shard).",
    )
    p.add_argument(
        "--workflow",
        default="configuration/workflow.yml",
        help="Path to the workflow YAML (default: configuration/workflow.yml).",
    )
    return p.parse_args(argv)


def _select_shard(entity_list, shard: int, num_shards: int):
    """Pick rows where index % num_shards == shard. Pure pandas, stable order."""
    if num_shards is None or num_shards <= 1:
        return entity_list
    if shard is None or not (0 <= shard < num_shards):
        raise ValueError(f"--shard must be in [0, {num_shards}) when --num-shards={num_shards}")
    mask = entity_list.index % num_shards == shard
    return entity_list[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # When running inside a container, prefer console-only logs so the
    # orchestrator captures stdout as the log stream. The env var also
    # works — env wins if both are set.
    manager = WorkflowManager(
        env="{{ cookiecutter.environment }}",
        log_level="INFO",
        log_to_console=True,
        log_to_console_only=os.environ.get("EDAGRO_LOG_CONSOLE_ONLY") == "1",
    )

    manager.load_workflow(args.workflow)
    manager.load_seasonfields()       # or load_geodataframe(...) from inputs/

    entities = manager.sfd_list
    if args.limit:
        entities = entities.head(args.limit)
        logger.info(f"--limit {args.limit}: processing {len(entities)} entities")
    if args.num_shards:
        entities = _select_shard(entities, args.shard, args.num_shards)
        logger.info(f"--shard {args.shard}/{args.num_shards}: {len(entities)} entities")

    results = manager.run_workflow(entity_list=entities, run_prefix=args.prefix)

    # Single-step pipelines: the report is the final results CSV.
    # Multi-step pipelines: the last step's results_df is conventionally the
    # deliverable. Project owners — adjust the lookup below to point at the
    # step whose output is the user-facing artifact.
    final_step = list(results.keys())[-1]
    final_df = results[final_step].get("results_df")
    if final_df is None or final_df.empty:
        logger.error(f"Pipeline finished with no rows from step '{final_step}' — nothing to publish.")
        return 1

    # Convention: the workflow_runner already exported the file via export_results;
    # we just resolve where it landed.
    report_path = Path(manager.output_result_dir).resolve()
    # Pick the newest *_final.csv in the output dir as the canonical report.
    candidates = sorted(report_path.glob(f"*{args.prefix}*_final.csv")) or sorted(report_path.glob("*_final.csv"))
    if not candidates:
        logger.error(f"No *_final.csv found under {report_path} — workflow did not produce a report.")
        return 1
    report = candidates[-1]

    # The stdout contract — the workflow scrapes this line. Keep it last,
    # keep it on its own line, keep the format stable.
    print(f"REPORT_PATH={report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
