---
title: Runbook — Set up a new client notebook
description: run a bulk extraction for a client — bootstrap, load entities, configure an extractor, export results. Creates nothing on the platform: provisioning a customer account is a separate, internal procedure.
visibility: public
---

# Runbook — Set up a new client notebook

Bootstrap a client-specific **extraction** notebook from scratch.

> **"Client" here means someone you extract data *for*.** This creates nothing on the platform — no customer, no users, no entitlements. Provisioning an account is a different job and a separate, internal procedure.
 The same four-step layout works for every notebook in the repo (`notebooks/`, `customers/`, `study/`, `backoffice/`).

> If you're spinning up an entire client **project** (not just one notebook inside the earthdaily-agriculture-internal repo), follow `docs/site/agriculture/01 - Quick_Start_Guide.md` instead — that uses the cookiecutter template (or `init_project.py`) to produce a self-contained project.

---

## 0. Prerequisite — credentials on disk

Do this before writing a single cell. Without it, `WorkflowManager(...)` raises
`ValueError: Missing required environment variables` in its constructor, and the
notebook cannot get past Step 1.

Create a `.env` — the package never generates or prompts for one:

```env
PROD_API_CLIENT_ID=<client_id>
PROD_API_CLIENT_SECRET=<client_secret>
PROD_API_USERNAME=<username>
PROD_API_PASSWORD=<password>
```

Use the `PREPROD_` prefix for the preprod environment; add `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` only if results go to `s3://`. Values come from your
EarthDaily contact.

**Where to put it.** Inside this repo, `src/.env` is the conventional spot and
Step 0's `init()` loads it. Outside it, put `.env` in the folder you run from —
`setup_environment()` falls back to the working directory. For a folder of your
own choosing, name it:

```python
manager = WorkflowManager("prod", project_root="/abs/path/to/that/folder")
```

That folder then also receives `results/`, `partials/`, `cache/` and `logs/`.

The full resolution order (five locations, first hit wins) and how to read the
`🔑 Loaded credentials from: …` log line are in `SKILL.md` §*Credentials*.

Never commit the file.

---

## 1. Pick the right folder

| Folder | When |
|---|---|
| `notebooks/` | Extractor class showcases, dev notebooks. Naming: `EDAgriculture_<Type>_{Processor,Service}_Function_Dev.ipynb`. |
| `customers/` | Client-specific extractions (one folder per `<Client>`). Naming: `EDAgriculture_<Client>_<Purpose>.ipynb`. |
| `study/` | Service-delivery tools (coverage check, zonal stats, custom map). Naming: `EDAgriculture_<Tool>.ipynb`. |
| `backoffice/` | User management, admin. |

---

## 2. Step 0 — bootstrap cell

**Always the first code cell.** Paste verbatim:

```python
# Bootstrap: ensure src/ is on sys.path for earthdaily.agriculture imports
import sys
from pathlib import Path

_src = str(Path().resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from earthdaily.agriculture.notebook_setup import init
init()
```

What this does:

- Inserts `<project-root>/src` into `sys.path` so `import earthdaily.agriculture` works without an editable install.
- `init()` resolves `<project-root>` (by walking up until it finds `pyproject.toml`), chdirs there, and loads `src/.env`. It prints the root it picked — check that line if credentials are not found, because the chdir changes which `.env` a working-directory lookup will see.

If you've installed the package via `pip install -e ".[jupyter]"`, the `sys.path` lines are redundant but harmless — keep them so the notebook works on every teammate's machine.

---

## 3. Step 1 — initialise `WorkflowManager`

```python
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

manager = WorkflowManager("prod", log_level="DEBUG")   # or "preprod"
```

DEBUG-level on first run shows the actual HTTP URLs, request bodies, and response shapes. Switch to `INFO` once the smoke test is clean.

Authentication, the S3 client, output paths (`results/`, `partials/`, `cache/`, `logs/`) and the loguru sink are all configured by the constructor — no further setup needed, *given* the `.env` from step 0. On the first run, check the `🔑 Loaded credentials from: …` line to confirm which file it actually read.

---

## 4. Step 2 — load entities

Three options, pick one:

**A. From the EarthDaily platform**

```python
manager.load_seasonfields(
    sowing_date_gte="2025-04-01",
    crop_id="CORN",
)
entities = manager.sfd_list
print(f"Loaded {len(entities)} entities")
print(entities.columns.tolist())   # platform returns crop.id, sowingDate, field.id, ...
```

When you load from the platform, your column names are `crop.id`, `sowingDate`, `field.id`, etc. — pass `column_mapping={"crop": "crop.id", "start_date": "sowingDate"}` to every extractor that needs `crop` or date fields.

**B. From a file (parquet / shapefile / CSV)**

```python
from earthdaily.agriculture.core.geometry import load_geodataframe

entities = load_geodataframe("inputs/client_fields.parquet")
print(entities.columns.tolist())
```

`load_geodataframe` accepts `.parquet`, `.shp`, `.geojson`, `.csv` (with a `geometry` column in WKT or WKB). It always returns a `pandas.DataFrame` with a WKT `geometry` column.

**C. Create manually for smoke tests**

```python
import pandas as pd

entities = pd.DataFrame([
    {
        "id": "smoke_001",
        "geometry": "POLYGON((-97.7 37.14, -97.69 37.14, -97.69 37.15, -97.7 37.15, -97.7 37.14))",
    },
])
```

---

## 5. Step 3 — configure & extract

```python
from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor

extractor = CoverageExtractor(
    manager.bearer_token,
    manager.token_expiration,
    config=manager.config,
    workflow_ref=manager,   # <-- delegates token refresh to the manager
)

extractor.setup_coverage_parameters(
    vegetation_index="NDVI",
    start_date="2025-01-01",
    end_date="2025-06-30",
    clear_cover_min=95,
    column_mapping={"crop": "crop.id"},   # only when entities come from the platform
)

# Smoke-test on one entity first.
smoke = extractor.get_coverage_api(entities.iloc[0])
print(smoke)

# Then run bulk.
result = extractor.process_entity_coverage_bulk_parallel(
    entity_list=entities,
    max_workers=10,
    prefix="client_coverage",
    generate_report=True,    # writes an HTML report alongside the CSV
)

df = result["results_df"]
print(f"Successful: {result['successful_calculations']}/{result['total_calculations']}")
df.head()
```

---

## 6. Save & inspect outputs

`BaseExtractor` auto-writes to `manager.output_result_dir` (default: `<project-root>/results/`). The files are:

- `<prefix>_results_<timestamp>_final.csv` — the merged final DataFrame.
- `<prefix>_report.html` — when `generate_report=True`.
- `<prefix>_failed_ids.csv` — entity IDs that errored, if any.

For S3-routed runs, pass `output_result_dir="s3://..."` to `WorkflowManager` (see `docs/site/agriculture/13 - Cloud_storage_principles_and_usage.md`).

---

## 7. (Optional) Step 4 — wire into a `WorkflowManager` YAML pipeline

For multi-step pipelines (e.g. emergence → greenness → disease on emerged fields only), define a YAML workflow under `configuration/` and load it via `manager.load_workflow(...)`. See `docs/site/agriculture/09 - Workflow_architecture.md` and `docs/site/agriculture/09b - Workflow_YAML_reference.md`.

---

## Hard rules

- Don't test `if entity_data:` on a pandas `Series` — use `self._is_entity_provided(entity_data)` (only relevant when subclassing `BaseExtractor`).
- Always pass `column_mapping` when entities come from the platform (different column names).
- LAI extraction requires `crop` on every row; the API returns empty otherwise.
- `clear_cover_min=100` is very restrictive — default to `95`.
- Notebook outputs are stripped on commit (`pre-commit` + `nbstripout`). Keep cell outputs with a `"keep_output": true` cell-metadata flag only when essential to the notebook (e.g. an HTML report preview).

---

## See also

- `docs/site/agriculture/01 - Quick_Start_Guide.md` — bootstrapping a new client *project* (cookiecutter flow).
- `docs/site/agriculture/11 - Working with EarthDaily Agriculture client.md` — day-to-day recipes and DEBUG logging tips.
- `debug_api_response.md` — what to do when a smoke test returns empty.
