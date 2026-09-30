"""Shape guards on `agro_urls`.

Every consumer joins a base URL as ``f"{base}/<path>"``, so a base ending in ``/``
yields ``.../v6//seasonfields`` — which the MDM gateway answers with a **404**,
before auth, where the single-slash URL gets its normal 401. ``EntityManager`` built
every farm / field / seasonfield URL that way from the trailing-slash
``eda_data_management_url_fields`` key; three other services had been quietly
``.rstrip("/")``-ing the same key. The key is gone, and no base may end in ``/``.
"""

from __future__ import annotations

import pytest

from earthdaily.agriculture.config.urls import agro_urls

pytestmark = pytest.mark.public


def _walk(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{key}.{k}" if key else k)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk(v, f"{key}[{i}]")
    elif isinstance(value, str):
        yield key, value


def test_no_base_url_ends_with_a_slash():
    offenders = [f"{k} = {v}" for k, v in _walk(agro_urls) if v.endswith("/")]
    assert not offenders, "base URLs must not end in '/' (callers add it):\n" + "\n".join(offenders)


def test_mdm_has_a_single_key():
    """One service, one key — the trailing-slash duplicate caused the 404."""
    assert "eda_data_management_url_fields" not in agro_urls
    assert set(agro_urls["eda_data_management_url"]) == {"preprod", "prod"}


def test_no_two_keys_share_the_same_endpoints():
    """One service, one key. Duplicates drift: one copy gets fixed, the other doesn't.

    `eda_data_management_url_fields`, `layers_url` and `deep_resolution_field_borders_urls`
    each duplicated a live key exactly, and nothing read them. Comparing whole
    preprod/prod mappings (not single values) leaves keys that intentionally share
    one URL across environments alone — that is a separate, tracked question.
    """
    seen: dict[tuple, str] = {}
    duplicates = []
    for key, value in agro_urls.items():
        if not isinstance(value, dict):
            continue
        signature = tuple(sorted((k, str(v)) for k, v in value.items()))
        if signature in seen:
            duplicates.append(f"{key} duplicates {seen[signature]}")
        else:
            seen[signature] = key
    assert not duplicates, "\n".join(duplicates)
