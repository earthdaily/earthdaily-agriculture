# Study

Exploratory and service-delivery work for **{{ cookiecutter.project_name }}** —
the analysis that surrounds an extraction rather than the extraction itself.

This folder ships empty on purpose. It is yours to fill.

## What belongs here

Notebooks and short scripts that *investigate*: coverage checks before committing
to a pull, zonal statistics, custom maps, one-off comparisons, the chart that
answers a client question. Work that is worth keeping and re-reading, but that
is not part of the repeatable pipeline.

## What does not

- **The pipeline itself** — recurring extraction logic belongs in `app/`
  (`run_pipeline.py`, `transforms.py`) and its configuration in
  `configuration/workflow.yml`, so it can run headless.
- **Reference documentation** — the numbered guides in `docs/` are generated from
  the `earthdaily-agriculture` source at project creation. Anything you write by
  hand there will be overwritten if the project is ever re-scaffolded.
- **Outputs** — extraction results go to `results/`, intermediate bulk writes to
  `partials/`, and inputs to `inputs/`. Keep this folder to the analysis.

## Conventions

Follow the package convention for notebook names —
`EDAgriculture_<Type>_<Purpose>.ipynb` — and start each notebook with the
standard bootstrap cell so `earthdaily.agriculture` imports resolve and the
working directory is the project root:

```python
from earthdaily.agriculture.notebook_setup import init
init()
```
