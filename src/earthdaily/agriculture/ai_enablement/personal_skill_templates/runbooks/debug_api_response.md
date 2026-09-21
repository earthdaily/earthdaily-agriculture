---
title: Runbook — Debug a failing or empty extraction
description: empty DataFrame, Series-truthiness, token expiry, platform-column mismatch, retry-on-400 misuse.
visibility: public
---

# Runbook — Debug a failing or empty extraction

Walk a failing extractor through the most common failure modes before opening an issue or paging an SME. The order below is roughly "cheapest signal first".

---

## 0. Reproduce with DEBUG logging

```python
manager = WorkflowManager("prod", log_level="DEBUG")
```

DEBUG prints the actual HTTP URL, the request body, the raw response shape, and the entity ID for every call. Most empty-result mysteries are visible here.

If you're inside a YAML workflow, set `log_level: DEBUG` under `settings` in the YAML and re-run.

---

## 1. Empty DataFrame from `format_*_json`

### Symptom

`process_single_entity_<type>` returns `{"success": True, "data": pd.DataFrame(), ...}` — no error, no rows.

### Common causes

| Cause | How to confirm | Fix |
|---|---|---|
| Required field missing on the entity | Print `entity_data[<col>]` for the failing row — `crop`, `start_date`, `sowing_date` are common requirements. | Provide the value, or pass a `column_mapping` that resolves it. |
| API returned `{"results": null}` instead of `[]` | Inspect `response_json` printed by DEBUG. | The extractor MUST use `response.get('key') or []` — *not* `response.get('key', [])`. The default arg only fires when the key is missing, not when its value is `None`. |
| Filter clipped every row | Look for "filtered N entities" warnings in DEBUG logs. | Loosen the filter — e.g. `clear_cover_min=95` (default for most use cases), or `clear_cover_min=80` for cloudier regions. `clear_cover_min=100` rejects almost everything. |
| LAI run with entities missing `crop` | LAI returns empty when `crop` is None / empty. | Filter the entity list to rows with a non-null `crop` first. |
| `validate_api_response` returned `False` | Check the validator's return value in DEBUG output. | The response shape changed upstream — update `format_*_json` to handle the new shape, or open an issue if the change was unintentional. |

---

## 2. "Truth value of a Series is ambiguous"

### Symptom

```
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), ...
```

### Cause

Somewhere in your custom code (or a new extractor) is `if entity_data:` where `entity_data` is a `pd.Series`. Pandas refuses to coerce a multi-element Series to a single bool.

### Fix

Use `self._is_entity_provided(entity_data)`:

```python
# wrong
if entity_data:
    ...

# right
if self._is_entity_provided(entity_data):
    ...
```

`_is_entity_provided` handles `dict`, `pd.Series`, and `pd.DataFrame` safely.

---

## 3. Token expired mid-run

### Symptom

After ~50 minutes a bulk run starts returning `401 Unauthorized` errors on every entity.

### Cause

The extractor was instantiated without `workflow_ref=manager`, so it has its own token and no auto-refresh path.

### Fix

```python
extractor = CoverageExtractor(
    manager.bearer_token,
    manager.token_expiration,
    config=manager.config,
    workflow_ref=manager,    # <-- this delegates token refresh
)
```

`@requires_token` calls `self.ensure_token_valid()`, which in turn asks `workflow_ref` for a refreshed token when it sees the expiration is near.

---

## 4. Column-mapping mismatch (platform DataFrames)

### Symptom

`KeyError: 'crop'` or "Entity has no crop field" errors, only when entities come from `manager.load_seasonfields(...)`.

### Cause

Platform DataFrames use dotted-name columns: `crop.id`, `sowingDate`, `field.id`. Your extractor expects the canonical `crop`, `sowing_date`.

### Fix

Pass `column_mapping` either via `setup_*_parameters` or via `extractor.set_column_mapping(...)`:

```python
extractor.setup_coverage_parameters(
    ...,
    column_mapping={"crop": "crop.id", "start_date": "sowingDate"},
)
```

`get_entity_value(row, "crop")` then looks for `crop.id` first and falls back to `crop`. Verify your mapping with `extractor.validate_column_mapping(entities)` — it logs warnings if any mapped column is missing.

---

## 5. Client-side 400 retried forever

### Symptom

A single entity triggers retries with the same 400 response over and over.

### Cause

The retry wrapper is not `retry_with_backoff_no_retry_on_400` — or the wrapping is broken so 400s pass through to the generic retry.

### Fix

Inside `get_*_api`, the call must be wrapped exactly:

```python
response = retry_with_backoff_no_retry_on_400(
    lambda: requests.post(url, headers=headers, json=body)
)
```

400s are terminal (client-side issue — won't get better on retry). 5xx / timeouts retry with exponential backoff + jitter.

---

## 6. Geometry validation

### Symptom

API rejects a polygon with `400 Bad geometry` or `Invalid WKT`.

### Cause

Self-intersecting polygon, wrong WKT format, or longitude/latitude swap.

### Fix

```python
from earthdaily.agriculture.core.geometry import validate_wkt

ok, msg = validate_wkt(entity_data["geometry"])
print(ok, msg)
```

`validate_wkt` returns `(bool, str)` — fix the geometry upstream (clean in QGIS, or `shapely.make_valid` it before passing).

---

## 7. S3 writes silently land in the wrong place

### Symptom

You set `EDAGRO_OUTPUT_PREFIX=s3://...` but results show up in `<project-root>/results/` instead.

### Cause

`storage="local"` is locking the writer to local mode, overriding the env var.

### Fix

Set `storage="auto"` (default) or `storage="s3"` on the `WorkflowManager` constructor. The auto-mode honours `EDAGRO_OUTPUT_PREFIX`; the s3-mode hard-requires it (raises if neither the env var nor explicit kwargs are set). See `docs/site/agriculture/13 - Cloud_storage_principles_and_usage.md`.

---

## 8. Cache hit returning stale data

### Symptom

After fixing a bug in `format_*_json`, bulk runs still show the old (buggy) output.

### Cause

`use_cache=True` is returning the cached pre-fix DataFrame.

### Fix

Either:

- Disable cache: `setup_*_parameters(..., use_cache=False)` for the next run.
- Or bump the cache version (clear `cache/` directory contents — they're parquet files keyed by entity ID + param hash).

See `docs/site/agriculture/07 - Cache_design_context.md` for the cache subsystem details.

---

## Still stuck?

- Re-introspect the new response shape: run `print(response_json)` inside `format_*_json` and compare against the actual API documented at `https://docs.earthdaily.com/agro/...` for that endpoint.
- Re-generate AI context: `cd src && python -m earthdaily.agriculture.ai_enablement.generate_ai_context` — if the existing extractor's docstring is wrong, the gap report flags it.
- Open an issue with the DEBUG-level log, the failing entity row (sanitised), and the response JSON.
