---
title: Runbook — Author a workflow YAML
description: chain extractors in a workflow YAML without writing glue — depends_on for sequence, column_mapping for names, and a transform only when the grain actually changes.
visibility: public
---

# Runbook — Author a workflow YAML

Go from an empty `configuration/workflow.yml` to a multi-step run. The recurring mistake this exists to prevent: **writing a transform step to rename columns, when `column_mapping` does it declaratively.**

> For what a *single* extraction looks like, see `setup_client_notebook`. This is for chaining two or more.

---

## 0. First, the decision that saves the most time

Before adding any step, answer this. Only the last row justifies a transform.

| The next step … | Use | Not |
|---|---|---|
| calls a column something else (`crop.id` vs `crop`) | **`column_mapping`** on that step | a renaming transform |
| needs the **original** entities, not upstream results | **`input_from: original`** | a passthrough transform |
| can't find its join key because the upstream renamed or dropped it | fix the **upstream** `exclude_columns` / `output_mapping` | a transform that repairs it |
| genuinely changes grain — long→wide, aggregate, filter, join an external file | **a transform.** This is what they are for | — |

**Why the mistake is structural, not careless.** Per step the runner does:

```
1. resolve entity_list via `input_from` (or first `depends_on`, else "original")
2. transform          ← you are here when you notice the columns look wrong
3. condition / filter
4. instantiate extractor
5. setup parameters   ← column_mapping is applied HERE
6. run
```

Reasoning forwards, you meet "the columns are wrong" at step 2 and the thing that fixes it at step 5. The scaffolded `configuration/workflow.yml` reinforces it: it ships a commented-out **"Example transform step"** and never mentions `column_mapping` at all.

---

## 1. Skeleton

```yaml
workflow:
  name: My-Pipeline
  description: "One line — this appears in the run log and the report."

  settings:
    max_workers: 5
    partial_frequency: 500

    # Canonical entity field -> what YOUR frame calls it. Inherited by every
    # step; a step can add to or override individual keys (§3).
    # Delete this block if your entity frame already uses canonical names.
    # Platform-loaded entities usually need at least these two:
    column_mapping:
      crop: "crop.id"
      sowing_date: sowingDate

  steps:
    - name: coverage
      extractor: CoverageExtractor
      module: earthdaily.agriculture.extractors.coverage_function
      setup:
        method: setup_coverage_parameters
        params:
          vegetation_index: NDVI
          start_date: "2025-01-01"
          clear_cover_min: 95
      run:
        method: process_entity_coverage_bulk_parallel
        params:
          prefix: coverage
          skip_export: false

    # A second step chaining off the first. `depends_on` gives it the sequence
    # AND, by default, coverage's results_df as its entity_list.
    #
    # Zoning requires an `image_id` per entity, and coverage emits exactly that —
    # so this chain needs NO transform and NO extra column_mapping. The frame it
    # receives carries the original entity columns (id, geometry, crop, ...)
    # plus coverage's outputs.
    - name: zoning
      depends_on: coverage
      extractor: ZoningExtractor
      module: earthdaily.agriculture.extractors.zoning_functions
      setup:
        method: setup_zoning_parameters
        params:
          num_zones: 5
          postprocess: stats
      run:
        method: process_entity_zoning_bulk_parallel
        params:
          prefix: zoning
          skip_export: false
```

Four step kinds, classified by what they carry: `extractor` only, `transform` only, `transform+extractor`, or neither (generic). A transform-only step stores its output as that step's `results_df`.

**Look at what the second step does not contain.** No transform, no `input_from`, no `column_mapping` of its own. `depends_on` hands the frame over and the names already line up. **A correct chain is usually this empty** — which is the whole point of this runbook.

`column_mapping` earns its place in `settings`, reconciling *your source frame* once. A step-level override is for the narrower case where one step genuinely disagrees with the rest — §3.

---

## 2. Sequence — `depends_on` and `input_from`

**`depends_on` creates the order.** `input_from` decides which frame the step *receives*, and it defaults to the first `depends_on`:

```yaml
    - name: planted
      depends_on: emergence        # runs after emergence …
      # input_from: emergence      # … and receives its results_df. This line is redundant.
```

So an auto-chaining step needs **neither a transform nor an explicit `input_from`**. Say `input_from: original` only when a step depends on an upstream for ordering but wants the original entity frame:

```yaml
    - name: harvest
      depends_on: emergence
      input_from: original         # ordering from emergence, entities from the caller
```

Steps with no `depends_on` between them run in parallel — a real proven pipeline (`MSU-SRRI-FIELD`) runs `vegetation` and `weather` side by side for exactly that reason, because the field set was already filtered upstream.

---

## 3. Names — `column_mapping`, per step, generalised last

Extractors read canonical entity fields: `id`, `geometry`, `crop`, `start_date`, `end_date`, `sowing_date`. `column_mapping` points a canonical name at whatever your frame actually calls it.

**It works at two levels and the two MERGE.** Workflow `settings.column_mapping` seeds every step; a step's `setup.params.column_mapping` adds to or overrides individual keys and inherits the rest:

```yaml
  settings:
    column_mapping:                       # base for every step
      crop: "crop.id"
      sowing_date: sowingDate

  steps:
    - name: planted
      depends_on: emergence
      setup:
        method: setup_planted_parameters
        params:
          column_mapping:                 # merges over the base; crop/sowing_date survive
            emergence_date: emergence_date
```

### Work it out per step, then generalise

This ordering matters more than it looks:

1. For each step, write the mapping it needs **on that step**.
2. When they all work, lift the keys common to every step up into `settings`.
3. Leave the rest where they are.

**Starting from a global mapping is what makes a step look unmappable** — one mapping cannot describe both the original entity columns and an upstream's output columns, so the author concludes the YAML can't express it and writes a transform. Bottom-up never produces that dead end.

### Or normalise at the edge and skip mapping entirely

A third option, and the right one when you control how entities are loaded — build the frame with canonical names before the workflow starts. `MSU-SRRI-FIELD` does this deliberately and says so in its header:

> *"The entity frame carries the canonical columns both extractors require — `id`, `geometry` (WKT), `start_date`, `end_date`… No column_mapping is needed."*

It then renames service outputs to its own canonical set *after* the run, in Python. Mapping in the YAML, or normalising either side of it — pick one and say which in the file's header comment.

---

## 4. What a chain actually passes along

**Extractor output carries the input columns through**, so a downstream step sees the original entity fields *plus* the upstream's results. That is what makes most chains need no glue at all.

⚠️ **`exclude_columns` can silently break this.** Output formatting is applied *before* the frame is returned, so a set of `exclude_columns` chosen to make a tidy CSV also strips the column the next step needs. It surfaces as a `KeyError` two steps later or an all-NaN merge. A real workflow guards it explicitly:

```yaml
    - name: emergence
      # exclude_columns: [] keeps `geometry` in results_df so downstream steps
      # using `input_from: emergence` still have everything they need.
```

Same trap for `output_mapping`: renaming a column for presentation renames the join key out from under the next step.

---

## 5. Settings worth knowing

| Key | Scope | Notes |
|---|---|---|
| `max_workers`, `partial_frequency` | workflow | per-step `run.params` can override `max_workers` |
| `column_mapping` | workflow **+ step** | merges, see §3 |
| `export_format` | workflow + step | `csv` \| `parquet`; step wins |
| `export_manifest` | workflow + step | writes a manifest sidecar beside a successful export |
| `output_uri` | workflow + step | durable second copy for `postprocess="file"` rasters |
| `output_prefix` | workflow | one of three filename layers: `<output_prefix>_<run_prefix>_<step_prefix>_results_<ts>` |
| `fail_safe` | workflow + step | keep going past per-entity errors |

Two run-time controls that save editing the YAML — `run_workflow(run_prefix="client_a")` tags one run's filenames, and `disabled_steps=["weather"]` skips steps for that run only. Use them for per-client or per-date re-runs.

---

## 6. Run it

```python
manager = WorkflowManager("prod")
manager.load_seasonfields(crop_id="CORN")
manager.load_workflow("configuration/workflow.yml")
results = manager.run_workflow(generate_report=True)
```

⚠️ **`load_workflow()` returns the UNWRAPPED config** — index `cfg["steps"]`, never `cfg["workflow"]["steps"]`, or you get a bare `KeyError: 'workflow'`. (The legacy `analytics:` format returns the wrapped dict, so the shape follows the input.)

`results` maps step name → `{results_df, global_errors, failed_ids}`.

---

## 7. Check your graph

Run with `generate_report=True` and read the step diagram. Steps are coloured by kind — **transform+extractor orange, extractor green, transform-only blue**. A chain that is mostly blue is mostly glue: go back to §0 and check each one against the table.

---

## Failure modes

| Symptom | Likely cause |
|---|---|
| `KeyError` on a canonical field | No `column_mapping` for it, and the frame doesn't use the canonical name |
| `KeyError` two steps downstream | An upstream `exclude_columns` / `output_mapping` removed or renamed it (§4) |
| A step runs on the wrong entities | `input_from` — it defaults to the *first* `depends_on`, not the original frame |
| N API calls per entity instead of one | Long-form input into a step expecting one row per entity — a genuine grain mismatch, so a transform |
| A step silently processes nothing | Upstream returned empty; dependants cascade-skip |
| `KeyError: 'workflow'` | Indexing `cfg["workflow"]["steps"]` — see §6 |

---

## See also

- `setup_client_notebook.md` — a single extraction, before you chain anything.
- `debug_api_response.md` — when a step returns an empty DataFrame.
- `docs/site/agriculture/09 - Workflow_architecture.md` and `09b - Workflow_YAML_reference.md`.
