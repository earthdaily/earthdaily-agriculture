"""
Crop catalogue — the platform-wide vocabulary of crop codes.

Split out of ``config.reference_data`` because this half ships **publicly**: the
field-ingestion path (``services.entity_management.prepare_field_dataframe``)
validates crop codes against it, and that module is part of the public package.
Keeping them together shipped a public module that imported a private one, so
``pip install earthdaily-agriculture`` produced a package whose field-ingestion
path raised ``ModuleNotFoundError`` the moment it was used.

The user / role / product / unit vocabularies it used to sit beside are
account-provisioning concerns and stay INTERNAL-ONLY in ``reference_data`` —
see ``release/public_allowlist.yml``.

Like the rest of ``config/``, this holds data only: validation lives in
``core.api_utils`` alongside the other validators.

Last updated: 2026-08-04
"""

# Crop catalogue, captured from GET /crops on prod (2026-08-04).
#
# `id == code` for every crop, exactly as for roles — so these keys are the ids
# the API expects in `SeasonFieldDtoCreate.crop`, no lookup required.
#
# ⚠️ This is the PLATFORM-WIDE catalogue. The crops a given customer may
# actually use is a subset, exposed per customer as `CustomerDto.crops` and
# configured when the customer is set up. Validate against the customer's own
# list when one is available; this dict is the outer bound.
#
# `cycle_duration` is in days and comes from the API — useful for deriving a
# season window from a sowing date.
crops = {
    "2ND_CORN":          {"name": "2nd Corn",             "category": "CORN",              "cycle_duration": 200},
    "CAMELINA":          {"name": "Camelina",             "category": "CAMELINA",          "cycle_duration": 200},
    "CITRUS":            {"name": "Citrus",               "category": "OTHERS",            "cycle_duration": 365},
    "COFFEE":            {"name": "Coffee",               "category": "COFFEE",            "cycle_duration": 365},
    "CORN":              {"name": "Corn",                 "category": "CORN",              "cycle_duration": 240},
    "COTTON":            {"name": "Cotton",               "category": "COTTON",            "cycle_duration": 210},
    "DRY_BEANS":         {"name": "Dry Beans",            "category": "BEANS",             "cycle_duration": 150},
    "OTHERS":            {"name": "Others",               "category": "OTHERS",            "cycle_duration": 365},
    "RICE":              {"name": "Rice",                 "category": "RICE",              "cycle_duration": 200},
    "SORGHUM":           {"name": "Sorghum",              "category": "SORGHUM",           "cycle_duration": 210},
    "SOYBEANS":          {"name": "Soybeans",             "category": "SOYBEANS",          "cycle_duration": 150},
    "SUGARCANE":         {"name": "Sugar Cane",           "category": "SUGARCANE",         "cycle_duration": 540},
    "SUNFLOWER":         {"name": "Sunflower",            "category": "SUNFLOWER",         "cycle_duration": 150},
    "TOMATO":            {"name": "Tomato",               "category": "LEGUMES",           "cycle_duration": 365},
    "WINTER_BARLEY":     {"name": "Winter Barley",        "category": "BARLEY_WINTER",     "cycle_duration": 300},
    "WINTER_OSR":        {"name": "Winter Oil Seed rape", "category": "OSR_WINTER",        "cycle_duration": 300},
    "WINTER_SOFT_WHEAT": {"name": "Soft Winter Wheat",    "category": "WHEAT_WINTER_SOFT", "cycle_duration": 330},
    "WINTER_WHEAT":      {"name": "Winter Wheat",         "category": "WINTER_WHEAT",      "cycle_duration": 300},
}  # fmt: skip
