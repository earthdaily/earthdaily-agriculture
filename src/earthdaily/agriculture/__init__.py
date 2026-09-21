"""
earthdaily.agriculture — Utilities for analytics bulk extraction from EarthDaily Agriculture API.

Subpackages:
    core        — Base classes, auth, API utils, geometry, logging, URLs
    extractors  — Service-level extractors (coverage, disease, FLM, VTS, weather, ...)
    processors  — Processor-based extractors (emergence, harvest, greenness, ...)
    services    — Workflow manager, user/entity management, S3, layer service
    analysis    — Data analysis, raster processing, predictive coverage
"""

from earthdaily.agriculture.core.api_utils import (
    filter_entities,
)
from earthdaily.agriculture.core.functions_enhanced import (
    print_summary,
    save_error_reports,
    setup_environment,
)
from earthdaily.agriculture.core.geometry import load_geodataframe, validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator
from earthdaily.agriculture.processors.processor_change_index_functions import (
    ChangeIndexExtractor,
)
from earthdaily.agriculture.processors.processor_inseason_monitoring_functions import (
    InSeasonMonitoringExtractor,
)
from earthdaily.agriculture.services.entity_management import get_seasonfield_list

# Backwards-compatible alias (was the explicit "API" name when both forms
# were exported via different paths).
api_get_seasonfield_list = get_seasonfield_list

__version__ = "2.6.0"
__author__ = "EarthDaily Agriculture KA Team"
__description__ = "Utilities for analytics bulk extraction, weather data, change index and In season Monitoring from EarthDaily Agriculture API"

__all__ = [
    "EDAuthenticator",
    "setup_environment",
    "get_seasonfield_list",
    "save_error_reports",
    "print_summary",
    "validate_wkt",
    "load_geodataframe",
    "api_get_seasonfield_list",
    "filter_entities",
    "ChangeIndexExtractor",
    "InSeasonMonitoringExtractor",
]
