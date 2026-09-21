# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Project Structure

```
{{ cookiecutter.project_slug }}/
├── .env.template                      # Credential template (copy to .env)
├── .dockerignore                      # Docker build context exclusions (Pattern B)
├── .gitignore
├── requirements.txt                   # Python dependencies
├── README.md
├── CLAUDE.md                          # Project context for Claude Code
├── Dockerfile                         # Container image (Pattern B)
├── app/                               # Project-specific code
│   ├── __init__.py
│   ├── transforms.py                  # Workflow transforms
│   └── run_pipeline.py                # Headless entrypoint for cron / containers
├── configuration/                     # YAML workflow definitions
│   └── workflow.yml
├── docs/                              # Bundled EarthDaily Agriculture reference docs + showcases
├── dist/                              # Pre-built earthdaily.agriculture wheel (tracked)
├── inputs/                            # Input data
├── results/                           # Final outputs
├── partials/                          # Intermediate results
├── cache/                             # Extraction cache
├── logs/                              # Log files
├── .github/workflows/                 # GitHub Actions cron (Pattern A)
│   └── automated_extraction.yml
└── EDAgriculture_{{ cookiecutter.project_slug }}.ipynb  # Main notebook
```

## Setup

1. Copy `.env.template` to `.env` and fill in credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Place input data in `inputs/`
4. Open the notebook and run

## Usage

### Manual mode

Run each step individually in the notebook for full control.

### Workflow mode

```python
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

manager = WorkflowManager("{{ cookiecutter.environment }}")
manager.load_workflow("configuration/workflow.yml")
results = manager.run_workflow(entity_list=entities)
```

### Cloud storage (S3)

Route all writes to `s3://` by passing the paths at construction time — see `docs/13 - Cloud_storage_principles_and_usage.md` for the full guide.

```python
manager = WorkflowManager(
    "{{ cookiecutter.environment }}",
    output_result_dir="s3://my-bucket/runs/{date}/results",
    partial_result_dir="s3://my-bucket/runs/{date}/partials",
    cache_dir="s3://my-bucket/runs/{date}/cache",
)
```

Requires the `[s3]` extra: `pip install -e ".[s3]"` in the `earthdaily.agriculture` source repo, which pulls `s3fs` and `fsspec`.

### Container deploys

Set `EDAGRO_LOG_CONSOLE_ONLY=1` to skip the rotating file-sink and emit logs only to stdout (so the orchestrator can capture them the standard way). Equivalent kwarg: `WorkflowManager(..., log_to_console_only=True)`.

## Reference docs

The `docs/` folder ships with EarthDaily Agriculture's reference documentation. See `CLAUDE.md` for an indexed table of what's where.

## Pre-commit hooks

`.pre-commit-config.yaml` ships three hooks:

- **nbstripout** — strips outputs / execution counts / noisy metadata from `*.ipynb` on commit. Cells tagged `{"metadata": {"keep_output": true}}` keep their outputs.
- **ruff** (lint + format) — matches the upstream earthdaily-agriculture-internal convention.
- **refresh-ai-context** — re-runs the AI-context generator (`CLAUDE.md` + `.claude/skills/earthdaily-agriculture/`) when `dist/*.whl` or `requirements.txt` change. Requires a local earthdaily-agriculture-internal clone reachable via `EDAGRO_CLIENT_PATH` env var, the script's `--template` flag, or auto-discovery (`~/Github/earthdaily-agriculture-internal`, `~/Documents/Github/earthdaily-agriculture-internal`, …). Skips cleanly with a warning if no clone is found.

One-time setup:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks manually (no commit): `pre-commit run --all-files`.

## Deployment

Two supported patterns — see [`docs/14 - Deployment_patterns.md`](docs/14%20-%20Deployment_patterns.md) for the full decision guide.

- **Pattern A — GitHub Actions cron** (default for daily/weekly extractions): the `.github/workflows/automated_extraction.yml` skeleton runs `python -m app.run_pipeline` on a cron, scrapes the `REPORT_PATH=` stdout line, uploads the artifact, and (once secrets are wired) ships it to S3. Lowest operational surface area; pick this first.
- **Pattern B — Docker container** (for runs > 4 hrs, fan-out, VPC requirements, or sub-hourly cadence): the `Dockerfile` + `.dockerignore` produce an image that runs the same `app/run_pipeline.py` under ECS / Cloud Run / Argo / `docker run`. Set `EDAGRO_OUTPUT_PREFIX=s3://...` to route all writers to S3.

Both patterns share the same entrypoint, same YAML, same wheel — the migration is mechanical.
