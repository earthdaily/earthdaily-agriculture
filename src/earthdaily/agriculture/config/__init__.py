"""earthdaily.agriculture.config subpackage — configuration data (URLs, constants).

NOTE: ``reference_data``, ``product_profiles.yml``, ``user_profiles.yml`` and
``payloads/`` also live in this directory but are INTERNAL-ONLY (see
``release/public_allowlist.yml``). They are deliberately NOT re-exported here:
this module ships publicly, so importing them would break the public package,
where those files are absent. Import them by full path from internal code.
"""

from earthdaily.agriculture.config.urls import agro_urls

__all__ = ["agro_urls"]
