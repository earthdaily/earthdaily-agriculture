"""
Shared fixtures for extractor unit tests (Greenness, Baresoil, ChangeIndex).

Strategy:
    - Patch BaseExtractor.__init__ to skip real initialization (logging, env vars, etc.)
    - Manually set the attributes that the extractor and its methods rely on.
    - This isolates tests from external dependencies (identity server, file system, loguru).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
from earthdaily.agriculture.extractors.cropid_functions import cropidExtractor
from earthdaily.agriculture.extractors.difference_functions import DifferenceExtractor
from earthdaily.agriculture.extractors.FLM_functions import FLMExtractor
from earthdaily.agriculture.extractors.gdd_functions import GDDExtractor
from earthdaily.agriculture.extractors.location_based_border_functions import LocationBasedBorderExtractor
from earthdaily.agriculture.extractors.regional_ts_extractor import RegionalExtractor
from earthdaily.agriculture.extractors.VTS_functions import MRTSExtractor, VegationTsExtractor
from earthdaily.agriculture.extractors.weather_functions import WeatherExtractor
from earthdaily.agriculture.extractors.zoning_functions import ZoningExtractor
from earthdaily.agriculture.processors.processor_baresoil_function import BaresoilExtractor
from earthdaily.agriculture.processors.processor_change_index_functions import ChangeIndexExtractor
from earthdaily.agriculture.processors.processor_disease_risk_functions import DiseaseExtractor
from earthdaily.agriculture.processors.processor_emergence_functions import EmergenceExtractor
from earthdaily.agriculture.processors.processor_greenness_functions import GreennessExtractor
from earthdaily.agriculture.processors.processor_harvest_functions import HarvestExtractor
from earthdaily.agriculture.processors.processor_inseason_monitoring_functions import (
    InSeasonMonitoringExtractor,
)
from earthdaily.agriculture.processors.processor_plantedarea_functions import PlantedExtractor
from earthdaily.agriculture.processors.processor_score_functions import (
    HistoricalScoreExtractor,
    InseasonScoreExtractor,
)
from earthdaily.agriculture.services.user_management import UserManager

# StandingCrop and Covercrop are private (non-public) modules. Guard the imports
# so this same conftest works in both the full private tree and the stripped
# public subset (where these modules are absent). Their fixtures below are only
# requested by their own tests, which never ship to the public package.
try:
    from earthdaily.agriculture.processors.processor_standing_crop_functions import StandingCropExtractor
except ModuleNotFoundError:
    StandingCropExtractor = None

try:
    from earthdaily.agriculture.processors.processor_covercrop_function import CovercropExtractor
except ModuleNotFoundError:
    CovercropExtractor = None

# ---------------------------------------------------------------------------
# Reusable test constants
# ---------------------------------------------------------------------------

FAKE_TOKEN = "fake-bearer-token-abc123"
FAKE_EXPIRATION = datetime.now() + timedelta(hours=1)

FAKE_CONFIG = {
    "env": "preprod",
    "partial_result_dir": "/tmp/partials",
    "output_result_dir": "/tmp/output",
    "merge_existing": "auto",
}

VALID_WKT = "POLYGON((-50 -15, -49 -15, -49 -14, -50 -14, -50 -15))"

FAKE_GREENNESS_URL = "https://fake-greenness-api.example.com"

DEFAULT_GREENNESS_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": 2025,
    "sowing_date": "2025-04-01",
    "data_source": "LR",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_entity(entity_id="entity_001", crop="CORN", geometry=None, sowing_date=None):
    """Helper to build a minimal entity dict."""
    entity = {
        "id": entity_id,
        "crop": crop,
        "geometry": geometry or VALID_WKT,
    }
    if sowing_date:
        entity["sowing_date"] = sowing_date
    return entity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger():
    """A MagicMock that stands in for a loguru logger (with .bind(), .success(), etc.)."""
    logger = MagicMock()
    # .bind() returns another mock logger so chained calls like log.info() work
    logger.bind.return_value = logger
    return logger


@pytest.fixture
def greenness_extractor(mock_logger):
    """
    Build a GreennessExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(GreennessExtractor, "__init__", lambda self, *a, **kw: None):
        ext = GreennessExtractor.__new__(GreennessExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "GreennessExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes (used by cache_single_entity decorator)
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by GreennessExtractor.__init__
    ext.greenness_params = None
    ext.greenness_url = FAKE_GREENNESS_URL

    return ext


@pytest.fixture
def configured_extractor(greenness_extractor):
    """A GreennessExtractor that already has greenness_params set (ready for API calls)."""
    greenness_extractor.greenness_params = DEFAULT_GREENNESS_PARAMS.copy()
    greenness_extractor.cache_key_columns = ["id"]
    return greenness_extractor


@pytest.fixture
def sample_entity():
    """A single valid entity dict."""
    return _make_entity()


@pytest.fixture
def sample_entity_list():
    """A small pandas DataFrame of entities (for bulk tests)."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_entity("ent_001", "CORN"),
            _make_entity("ent_002", "SOYBEANS"),
            _make_entity("ent_003", "COTTON"),
        ]
    )


@pytest.fixture
def sample_api_response():
    """A realistic API response for a single entity."""
    return {
        "id": "entity_001",
        "data": {"greenness_score": 0.85, "greenness_date": "2025-06-15", "potential_score": 72.3, "status": "GREEN"},
    }


@pytest.fixture
def sample_api_response_list():
    """An API response where data is a list of records."""
    return {
        "id": "entity_001",
        "data": [
            {"greenness_score": 0.85, "date": "2025-06-15", "status": "GREEN"},
            {"greenness_score": 0.72, "date": "2025-06-22", "status": "GREEN"},
        ],
    }


# ---------------------------------------------------------------------------
# Baresoil fixtures
# ---------------------------------------------------------------------------

FAKE_BARESOIL_URL = "https://fake-baresoil-api.example.com"

DEFAULT_BARESOIL_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": 2025,
    "filter": "summary",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_baresoil_entity(entity_id="entity_001", geometry=None):
    """Helper to build a minimal baresoil entity dict (id + geometry only)."""
    return {
        "id": entity_id,
        "geometry": geometry or VALID_WKT,
    }


@pytest.fixture
def baresoil_extractor(mock_logger):
    """
    Build a BaresoilExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(BaresoilExtractor, "__init__", lambda self, *a, **kw: None):
        ext = BaresoilExtractor.__new__(BaresoilExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "BaresoilExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes (used by setup + cache_single_entity decorator)
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by BaresoilExtractor.__init__
    ext.baresoil_params = None
    ext.baresoil_url = FAKE_BARESOIL_URL

    return ext


@pytest.fixture
def configured_baresoil_extractor(baresoil_extractor):
    """A BaresoilExtractor that already has baresoil_params set."""
    baresoil_extractor.baresoil_params = DEFAULT_BARESOIL_PARAMS.copy()
    baresoil_extractor.cache_key_columns = ["id"]
    return baresoil_extractor


@pytest.fixture
def sample_baresoil_entity():
    """A single valid baresoil entity dict."""
    return _make_baresoil_entity()


@pytest.fixture
def sample_baresoil_entity_list():
    """A small pandas DataFrame of baresoil entities (for bulk tests)."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_baresoil_entity("ent_001"),
            _make_baresoil_entity("ent_002"),
            _make_baresoil_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_baresoil_response_summary():
    """A realistic baresoil API response with a baresoil-detected season."""
    return {
        "id": "entity_001",
        "data": {
            "year": 2025,
            "duration": 120,
            "startDay": 1,
            "startMonth": 4,
            "baresoilDays": 48,
            "baresoilPeriods": [
                {"start": "2025-04-15", "end": "2025-05-10", "periodLength": 25},
                {"start": "2025-06-01", "end": "2025-06-23", "periodLength": 23},
            ],
        },
    }


@pytest.fixture
def sample_baresoil_response_no_baresoil():
    """A baresoil response where no baresoil days were detected."""
    return {
        "id": "entity_002",
        "data": {
            "year": 2025,
            "duration": 120,
            "startDay": 1,
            "startMonth": 4,
            "baresoilDays": 0,
            "baresoilPeriods": [],
        },
    }


# ---------------------------------------------------------------------------
# ChangeIndex fixtures
# ---------------------------------------------------------------------------

FAKE_CHANGE_INDEX_URL = "https://fake-change-index-api.example.com"

# Real geometry used in the EDAgro_ChangeIndex_Processor_Function_Dev notebook
CHANGE_INDEX_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_CHANGE_INDEX_PARAMS = {
    "map_type": "NDVI",
    "collections": ["Sentinel-2"],
    "max_period_reference": 7,
    "max_period_previous": 15,
    "min_period_previous": 5,
    "same_sensor": False,
    "parameter_profile": "change_index_v1",
    "partial_frequency": 50,
    "publish_af": False,
}


def _make_change_index_entity(
    entity_id="z361x33",
    geometry=None,
    crop="OTHERS",
    sowing_date="2025-04-01",
    reference_date="2025-06-15",
):
    """Helper to build a change index entity dict matching the notebook schema."""
    entity = {
        "id": entity_id,
        "geometry": geometry or CHANGE_INDEX_WKT,
    }
    if crop is not None:
        entity["crop"] = crop
    if sowing_date is not None:
        entity["sowing_date"] = sowing_date
    if reference_date is not None:
        entity["reference_date"] = reference_date
    return entity


@pytest.fixture
def change_index_extractor(mock_logger):
    """ChangeIndexExtractor with BaseExtractor.__init__ bypassed."""
    with patch.object(ChangeIndexExtractor, "__init__", lambda self, *a, **kw: None):
        ext = ChangeIndexExtractor.__new__(ChangeIndexExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "ChangeIndexExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    # ChangeIndex uses a non-default field on the entity row
    ext.column_mapping["reference_date"] = "reference_date"
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.change_index_params = None
    ext.change_index_url = FAKE_CHANGE_INDEX_URL

    return ext


@pytest.fixture
def configured_change_index_extractor(change_index_extractor):
    """ChangeIndexExtractor with change_index_params already configured."""
    change_index_extractor.change_index_params = DEFAULT_CHANGE_INDEX_PARAMS.copy()
    change_index_extractor.cache_key_columns = ["id", "reference_date"]
    return change_index_extractor


@pytest.fixture
def sample_change_index_entity():
    """Single notebook-shaped entity (id, geometry, crop, sowing_date, reference_date)."""
    return _make_change_index_entity()


@pytest.fixture
def sample_change_index_entity_list():
    """Small DataFrame of notebook-shaped entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_change_index_entity("z361x33"),
            _make_change_index_entity("7e5gwem"),
            _make_change_index_entity("x3vbx3q"),
        ]
    )


@pytest.fixture
def sample_change_index_response():
    """
    Realistic Change Index API response captured from the dev notebook
    (cell 17, EDAgro_ChangeIndex_Processor_Function_Dev.ipynb). The processor
    stamps `id` and `reference_date` onto the response before returning, so
    this fixture mirrors the post-stamp shape that format_change_index_json
    will receive.
    """
    return {
        "id": "z361x33",
        "status": "Processor done and metrics pushed.",
        "data": {
            "CurrentMapAverage": 0.8999999761581421,
            "CurrentMapMinimum": 0.03999999910593033,
            "CurrentMapMaximum": 0.9100000262260437,
            "CurrentMapStddev": 0.039000000804662704,
            "CurrentMapVariance": 0.001500000013038516,
            "ReferenceMapAverage": 0.8899999856948853,
            "ReferenceMapMinimum": 0.05000000074505806,
            "ReferenceMapMaximum": 0.9100000262260437,
            "ReferenceMapStddev": 0.039000000804662704,
            "ReferenceMapVariance": 0.001500000013038516,
            "SPAEFIndex": 0.71,
            "ChangeIndex": 1.43,
            "SPAEFIndexV2": 0.52,
            "ChangeIndexV2": 2.41,
            "PearsonCoefficient": 0.9900000095367432,
            "Covariance": 1.0,
            "HistogramMatching": 0.52,
            "date_ref": "2025-06-12T14:16:31Z",
            "sensor_ref": "SENTINEL_2",
            "date_current": "2025-06-07T14:16:54Z",
            "sensor_current": "SENTINEL_2",
            "status": "Processor done and metrics pushed.",
        },
        "reference_date": "2025-06-15",
    }


# ---------------------------------------------------------------------------
# Coverage fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by CoverageExtractor (map_products_url)
FAKE_COVERAGE_URL = "http://fake-coverage-api.example.com/field-level-maps/v5"

# Notebook test entity (cell 15)
COVERAGE_PARIS_WKT = "POLYGON((2.2945 48.8584, 2.2955 48.8584, 2.2955 48.8594, 2.2945 48.8594, 2.2945 48.8584))"

# Notebook test entity for process_single_entity (cell 23)
COVERAGE_BRAZIL_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_COVERAGE_PARAMS = {
    "vegetation_index": "NDVI",
    "start_date": "2025-01-01",
    "end_date": None,
    "clear_cover_min": 80,
    "clear_cover_max": 100,
    "use_specific_date": False,
    "filter": "none",
    "delay": 3,
    "mask": "auto",
    "partial_frequency": 20,
    "recalibration": False,
    "historical_seasons": None,
}


def _make_coverage_entity(entity_id="test_001", geometry=None, crop=None):
    """Helper to build a minimal coverage entity dict (id + geometry, optional crop)."""
    entity = {"id": entity_id, "geometry": geometry or COVERAGE_PARIS_WKT}
    if crop is not None:
        entity["crop"] = crop
    return entity


@pytest.fixture
def coverage_extractor(mock_logger):
    """
    Build a CoverageExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(CoverageExtractor, "__init__", lambda self, *a, **kw: None):
        ext = CoverageExtractor.__new__(CoverageExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "CoverageExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by CoverageExtractor.__init__
    ext.coverage_params = None
    ext.map_products_url = FAKE_COVERAGE_URL

    return ext


@pytest.fixture
def configured_coverage_extractor(coverage_extractor):
    """A CoverageExtractor that already has coverage_params set (ready for API calls)."""
    coverage_extractor.coverage_params = DEFAULT_COVERAGE_PARAMS.copy()
    coverage_extractor.cache_key_columns = ["id", "image_id", "mask"]
    return coverage_extractor


@pytest.fixture
def sample_coverage_entity():
    """Single notebook-shaped coverage entity (id + geometry only)."""
    return _make_coverage_entity()


@pytest.fixture
def sample_coverage_entity_list():
    """Small DataFrame of notebook-shaped coverage entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_coverage_entity("ent_001"),
            _make_coverage_entity("ent_002"),
            _make_coverage_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_coverage_response_list():
    """
    Realistic coverage API response shape from the notebook (cell 17 output).
    The catalog-imagery endpoint returns a JSON list of records, each with
    'coveragePercent', 'image' (id/date/sensor/spatialResolution), and 'mask'.
    """
    return [
        {
            "coveragePercent": 100.0,
            "image": {
                "id": "sentinel-2-c1-l2a|S2C_T31UDQ_20260429T105026_L2A",
                "spatialResolution": 10.0,
                "date": "2026-04-29T10:57:29Z",
                "sensor": "SENTINEL_2",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 95.0,
            "image": {
                "id": "landsat-c2l2-sr|LC09_L2SP_199026_20260426_20260427_02_T1_SR",
                "spatialResolution": 30.0,
                "date": "2026-04-26T10:40:34Z",
                "sensor": "LANDSAT_9",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 88.0,
            "image": {
                "id": "sentinel-2-c1-l2a|S2A_T31UDQ_20260424T110355_L2A",
                "spatialResolution": 10.0,
                "date": "2026-04-24T11:07:41Z",
                "sensor": "SENTINEL_2",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 75.0,  # below default clear_cover_min=80, should be filtered out
            "image": {
                "id": "landsat-c2l2-sr|LC08_L2SP_199026_20260317_20260406_02_T1_SR",
                "spatialResolution": 30.0,
                "date": "2026-03-17T10:40:37Z",
                "sensor": "LANDSAT_8",
            },
            "mask": "ML",
        },
    ]


@pytest.fixture
def sample_coverage_response_no_sensor():
    """
    Coverage response where the 'sensor' field is missing from image metadata —
    the formatter should infer it from the image id (sentinel/landsat substring).
    """
    return [
        {
            "coveragePercent": 100.0,
            "image": {
                "id": "sentinel-2-c1-l2a|S2C_T31UDQ_20260429T105026_L2A",
                "spatialResolution": 10.0,
                "date": "2026-04-29T10:57:29Z",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 95.0,
            "image": {
                "id": "landsat-c2l2-sr|LC09_L2SP_199026_20260426_20260427_02_T1_SR",
                "spatialResolution": 30.0,
                "date": "2026-04-26T10:40:34Z",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 90.0,
            "image": {
                "id": "landsat-c2l2-sr|LC08_L2SP_199026_20260317_20260406_02_T1_SR",
                "spatialResolution": 30.0,
                "date": "2026-03-17T10:40:37Z",
            },
            "mask": "ML",
        },
        {
            "coveragePercent": 85.0,
            "image": {
                "id": "unknown-sensor|XYZ_20260315_AB",
                "spatialResolution": 10.0,
                "date": "2026-03-15T10:00:00Z",
            },
            "mask": "ML",
        },
    ]


# ---------------------------------------------------------------------------
# Covercrop fixtures
# ---------------------------------------------------------------------------

FAKE_COVERCROP_URL = "https://fake-covercrop-api.example.com/processors/cover-crop/v1"

# Notebook test entity (cells 15 & 24)
COVERCROP_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_COVERCROP_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": "2025",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_covercrop_entity(entity_id="z361x33", geometry=None):
    """Helper to build a minimal covercrop entity dict (id + geometry only)."""
    return {"id": entity_id, "geometry": geometry or COVERCROP_WKT}


@pytest.fixture
def covercrop_extractor(mock_logger):
    """
    Build a CovercropExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(CovercropExtractor, "__init__", lambda self, *a, **kw: None):
        ext = CovercropExtractor.__new__(CovercropExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "CovercropExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by CovercropExtractor.__init__
    ext.covercrop_params = None
    ext.covercrop_url = FAKE_COVERCROP_URL

    return ext


@pytest.fixture
def configured_covercrop_extractor(covercrop_extractor):
    """A CovercropExtractor that already has covercrop_params set."""
    covercrop_extractor.covercrop_params = DEFAULT_COVERCROP_PARAMS.copy()
    covercrop_extractor.cache_key_columns = ["id"]
    return covercrop_extractor


@pytest.fixture
def sample_covercrop_entity():
    """Single notebook-shaped covercrop entity (id + geometry only)."""
    return _make_covercrop_entity()


@pytest.fixture
def sample_covercrop_entity_list():
    """Small DataFrame of notebook-shaped covercrop entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_covercrop_entity("ent_001"),
            _make_covercrop_entity("ent_002"),
            _make_covercrop_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_covercrop_response_false():
    """
    Realistic cover crop API response from the notebook (cell 17 output):
    {'cover_crop': 'False', 'year': 2025}.
    Note: 'cover_crop' is returned as a string and normalized to bool by the formatter.
    """
    return {"cover_crop": "False", "year": 2025}


@pytest.fixture
def sample_covercrop_response_true():
    """Cover crop API response with cover crop detected (string 'True')."""
    return {"cover_crop": "True", "year": 2025}


# ---------------------------------------------------------------------------
# Cropid fixtures
# ---------------------------------------------------------------------------

FAKE_CROPID_URL = "https://fake-cropid-api.example.com/cropmasks/v1/cropmasks/crops"

# Notebook test entity (cells 15 & 27)
CROPID_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_CROPID_PARAMS = {
    "begin_year": 2020,
    "end_year": 2025,
    "mask_type": "EndSeason",
    "limit_nb_crop": 1,
    "crop_mask_percent": 75,
    "mode": "history",
    "partial_frequency": 50,
}


def _make_cropid_entity(entity_id="z361x33", geometry=None, crop="SOYBEANS"):
    """Helper to build a minimal cropid entity dict (id, geometry, crop)."""
    return {
        "id": entity_id,
        "geometry": geometry or CROPID_WKT,
        "crop": crop,
    }


@pytest.fixture
def cropid_extractor(mock_logger):
    """
    Build a cropidExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(cropidExtractor, "__init__", lambda self, *a, **kw: None):
        ext = cropidExtractor.__new__(cropidExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "cropidExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by cropidExtractor.__init__
    ext.cropid_params = None
    ext.cropid_url = FAKE_CROPID_URL

    return ext


@pytest.fixture
def configured_cropid_extractor(cropid_extractor):
    """A cropidExtractor that already has cropid_params set (history mode by default)."""
    cropid_extractor.cropid_params = DEFAULT_CROPID_PARAMS.copy()
    cropid_extractor.cache_key_columns = ["id", "year"]
    return cropid_extractor


@pytest.fixture
def sample_cropid_entity():
    """Notebook-shaped cropid entity (id, geometry, crop=SOYBEANS)."""
    return _make_cropid_entity()


@pytest.fixture
def sample_cropid_entity_list():
    """Small DataFrame of notebook-shaped cropid entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_cropid_entity("ent_001", crop="SOYBEANS"),
            _make_cropid_entity("ent_002", crop="CORN"),
            _make_cropid_entity("ent_003", crop="COTTON"),
        ]
    )


@pytest.fixture
def sample_cropid_response():
    """
    Realistic cropid API response from the notebook (cell 17 output):
    resultsByYear with 6 years (2020-2025), alternating SOYBEANS/COTTON.
    """
    return {
        "resultsByYear": {
            "2020": {
                "crops": [
                    {
                        "cropName": "Soybean",
                        "rawCropName": "soybeans",
                        "cropMaskPercent": 97,
                        "edaCropCode": "SOYBEANS",
                    }
                ]
            },
            "2021": {
                "crops": [
                    {
                        "cropName": "Cotton",
                        "rawCropName": "cotton",
                        "cropMaskPercent": 97,
                        "edaCropCode": "COTTON",
                    }
                ]
            },
            "2022": {
                "crops": [
                    {
                        "cropName": "Soybean",
                        "rawCropName": "soybeans",
                        "cropMaskPercent": 97,
                        "edaCropCode": "SOYBEANS",
                    }
                ]
            },
            "2023": {
                "crops": [
                    {
                        "cropName": "Cotton",
                        "rawCropName": "cotton",
                        "cropMaskPercent": 97,
                        "edaCropCode": "COTTON",
                    }
                ]
            },
            "2024": {
                "crops": [
                    {
                        "cropName": "Soybean",
                        "rawCropName": "soybeans",
                        "cropMaskPercent": 97,
                        "edaCropCode": "SOYBEANS",
                    }
                ]
            },
            "2025": {
                "crops": [
                    {
                        "cropName": "Soybean",
                        "rawCropName": "soybeans",
                        "cropMaskPercent": 97,
                        "edaCropCode": "SOYBEANS",
                    }
                ]
            },
        }
    }


@pytest.fixture
def sample_cropid_response_empty():
    """Cropid response with no resultsByYear (empty dict)."""
    return {"resultsByYear": {}}


# ---------------------------------------------------------------------------
# Difference fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by DifferenceExtractor (map_products_url)
FAKE_DIFFERENCE_URL = "http://fake-diff-api.example.com/field-level-maps/v5"

# Notebook test entity (cell 11)
DIFFERENCE_WKT = (
    "POLYGON ((-97.70066562 37.14062335, -97.69927729 37.14227539, "
    "-97.69935777 37.14233954, -97.70004188 37.14248389, "
    "-97.70008212 37.14359058, -97.69969534 37.14450741, "
    "-97.70262024 37.14450741, -97.70254625 37.14062335, "
    "-97.70066562 37.14062335))"
)

# Image IDs (sentinel-2 catalog format with one '|' separator)
DIFFERENCE_IMG_1 = "sentinel-2-c1-l2a|S2A_T14SLF_20250515T172859_L2A"
DIFFERENCE_IMG_2 = "sentinel-2-c1-l2a|S2C_T14SLF_20250730T172901_L2A"

DEFAULT_DIFFERENCE_PARAMS = {
    "product": "DIFFERENCE_NDVI",
    "output_epsg": 4326,
    "postprocess": "stats",
    "map_format": None,
    "output_path": None,
    "skip_existing": True,
    "directLinks": False,
    "partial_frequency": 50,
}


def _make_difference_entity(
    entity_id="test_001",
    geometry=None,
    image_id_1=None,
    image_id_2=None,
    name=None,
):
    """Helper to build a minimal difference entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or DIFFERENCE_WKT,
        "image_id_1": image_id_1 or DIFFERENCE_IMG_1,
        "image_id_2": image_id_2 or DIFFERENCE_IMG_2,
    }
    if name is not None:
        entity["name"] = name
    return entity


@pytest.fixture
def difference_extractor(mock_logger):
    """
    Build a DifferenceExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(DifferenceExtractor, "__init__", lambda self, *a, **kw: None):
        ext = DifferenceExtractor.__new__(DifferenceExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "DifferenceExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by DifferenceExtractor.__init__
    ext.difference_params = None
    ext.map_products_url = FAKE_DIFFERENCE_URL

    return ext


@pytest.fixture
def configured_difference_extractor(difference_extractor):
    """A DifferenceExtractor with stats-mode params already set."""
    difference_extractor.difference_params = DEFAULT_DIFFERENCE_PARAMS.copy()
    difference_extractor.cache_key_columns = ["id", "image_id_1", "image_id_2"]
    return difference_extractor


@pytest.fixture
def sample_difference_entity():
    """Notebook-shaped difference entity (id, geometry, image_id_1, image_id_2)."""
    return _make_difference_entity()


@pytest.fixture
def sample_difference_entity_list():
    """Small DataFrame of difference entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_difference_entity("ent_001"),
            _make_difference_entity("ent_002"),
            _make_difference_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_difference_stats_response():
    """
    Realistic difference-map API response in stats mode. Mirrors the EarthDaily
    Field Level Maps API contract: legend.stat (max/mean/min) and legend.ranges
    (per-bucket pixel/area/color breakdowns).
    """
    return {
        "legend": {
            "stat": {"max": 0.42, "mean": 0.05, "min": -0.31},
            "ranges": [
                {
                    "minValue": -0.5,
                    "maxValue": -0.2,
                    "numberOfPixels": 120,
                    "area": 1200.0,
                    "color": {"r": 200, "g": 50, "b": 50},
                },
                {
                    "minValue": -0.2,
                    "maxValue": 0.0,
                    "numberOfPixels": 340,
                    "area": 3400.0,
                    "color": {"r": 200, "g": 200, "b": 50},
                },
                {
                    "minValue": 0.0,
                    "maxValue": 0.5,
                    "numberOfPixels": 540,
                    "area": 5400.0,
                    "color": {"r": 50, "g": 200, "b": 50},
                },
            ],
        }
    }


@pytest.fixture
def sample_difference_links_response():
    """
    Realistic difference-map API response in links mode (directLinks=True).
    Includes _links, worldFile, mapSize, bBox, and seasonField.
    """
    return {
        "_links": {
            "image:image/png": "https://api.example.com/diff/test_001.png",
            "worldfile": "https://api.example.com/diff/test_001.pgw",
            "thumbnail": "https://api.example.com/diff/test_001_thumb.png",
        },
        "worldFile": {
            "a": 1e-5,
            "b": 0.0,
            "c": -97.7,
            "d": 0.0,
            "e": -1e-5,
            "f": 37.14,
        },
        "mapSize": {"width": 512, "height": 512},
        "bBox": {"xMin": -97.703, "xMax": -97.699, "yMin": 37.140, "yMax": 37.145},
        "seasonField": {"id": "test_001"},
    }


# ---------------------------------------------------------------------------
# Disease fixtures
# ---------------------------------------------------------------------------

FAKE_DISEASE_URL = "https://fake-disease-api.example.com/v1"

# Notebook test entity (cell 17)
DISEASE_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_DISEASE_PARAMS = {
    "start_date": "2025-06-01",
    "end_date": "2025-10-01",
    "partial_frequency": 50,
}


def _make_disease_entity(
    entity_id="z361x33",
    geometry=None,
    crop="SOYBEANS",
    start_date="2025-06-01",
    end_date="2025-10-01",
):
    """Helper to build a notebook-shaped disease entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or DISEASE_WKT,
    }
    if crop is not None:
        entity["crop"] = crop
    if start_date is not None:
        entity["start_date"] = start_date
    if end_date is not None:
        entity["end_date"] = end_date
    return entity


@pytest.fixture
def disease_extractor(mock_logger):
    """
    Build a DiseaseExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(DiseaseExtractor, "__init__", lambda self, *a, **kw: None):
        ext = DiseaseExtractor.__new__(DiseaseExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "DiseaseExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by DiseaseExtractor.__init__
    ext.disease_params = None
    ext.disease_url = FAKE_DISEASE_URL

    return ext


@pytest.fixture
def configured_disease_extractor(disease_extractor):
    """A DiseaseExtractor that already has disease_params set (ready for API calls)."""
    disease_extractor.disease_params = DEFAULT_DISEASE_PARAMS.copy()
    disease_extractor.cache_key_columns = ["id", "date"]
    return disease_extractor


@pytest.fixture
def sample_disease_entity():
    """Notebook-shaped disease entity (id, geometry, crop, start_date, end_date)."""
    return _make_disease_entity()


@pytest.fixture
def sample_disease_entity_list():
    """Small DataFrame of notebook-shaped disease entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_disease_entity("ent_001"),
            _make_disease_entity("ent_002"),
            _make_disease_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_disease_response_list():
    """
    Realistic disease API response (list of daily records). Mirrors the columns
    seen in the notebook bulk run (cell 32):
    frogeye_leaf_spot, giberella_ear_rot, gray_leaf_spot, tar_spot,
    white_mold_dry, white_mold_irr_15, white_mold_irr_30.
    """
    return [
        {
            "date": "2025-06-01",
            "frogeye_leaf_spot": 0.4658,
            "giberella_ear_rot": 0.000012,
            "gray_leaf_spot": 0.999894,
            "tar_spot": 0.023187,
            "white_mold_dry": 0.175166,
            "white_mold_irr_15": 0.922087,
            "white_mold_irr_30": 0.522747,
        },
        {
            "date": "2025-06-02",
            "frogeye_leaf_spot": 0.4823,
            "giberella_ear_rot": 0.000035,
            "gray_leaf_spot": 0.999905,
            "tar_spot": 0.024779,
            "white_mold_dry": 0.180752,
            "white_mold_irr_15": 0.928874,
            "white_mold_irr_30": 0.547238,
        },
        {
            "date": "2025-06-03",
            "frogeye_leaf_spot": 0.4877,
            "giberella_ear_rot": 0.000091,
            "gray_leaf_spot": 0.999911,
            "tar_spot": 0.027693,
            "white_mold_dry": 0.185563,
            "white_mold_irr_15": 0.932651,
            "white_mold_irr_30": 0.561721,
        },
    ]


@pytest.fixture
def sample_disease_response_with_nested():
    """
    Disease response with nested dicts — exercise the flattening branch in
    format_disease_json (key.sub_key column naming).
    """
    return [
        {
            "date": "2025-06-01",
            "frogeye_leaf_spot": 0.466,
            "metadata": {"sensor": "S2", "confidence": 0.9},
        },
        {
            "date": "2025-06-02",
            "frogeye_leaf_spot": 0.482,
            "metadata": {"sensor": "S2", "confidence": 0.85},
        },
    ]


# ---------------------------------------------------------------------------
# Emergence fixtures
# ---------------------------------------------------------------------------

FAKE_EMERGENCE_URL = "https://fake-emergence-api.example.com"

# Notebook test entity (cell 15)
EMERGENCE_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_EMERGENCE_PARAMS = {
    "emergence_type": "INSEASON",
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": 2025,
    "data_source": "LR",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_emergence_entity(entity_id="z361x33", crop="CORN", geometry=None, sowing_date=None):
    """Helper to build a notebook-shaped emergence entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or EMERGENCE_WKT,
    }
    if crop is not None:
        entity["crop"] = crop
    if sowing_date is not None:
        entity["sowing_date"] = sowing_date
    return entity


@pytest.fixture
def emergence_extractor(mock_logger):
    """
    Build an EmergenceExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(EmergenceExtractor, "__init__", lambda self, *a, **kw: None):
        ext = EmergenceExtractor.__new__(EmergenceExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "EmergenceExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes (used by cache_single_entity decorator)
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by EmergenceExtractor.__init__
    ext.emergence_params = None
    ext.emergence_url = FAKE_EMERGENCE_URL

    return ext


@pytest.fixture
def configured_emergence_extractor(emergence_extractor):
    """An EmergenceExtractor that already has emergence_params set (INSEASON mode by default)."""
    emergence_extractor.emergence_params = DEFAULT_EMERGENCE_PARAMS.copy()
    emergence_extractor.cache_key_columns = ["id"]
    return emergence_extractor


@pytest.fixture
def sample_emergence_entity():
    """Notebook-shaped emergence entity (id, geometry, crop=CORN)."""
    return _make_emergence_entity()


@pytest.fixture
def sample_emergence_entity_list():
    """Small DataFrame of notebook-shaped emergence entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_emergence_entity("ent_001", "CORN"),
            _make_emergence_entity("ent_002", "SOYBEANS"),
            _make_emergence_entity("ent_003", "COTTON"),
        ]
    )


@pytest.fixture
def sample_emergence_inseason_response():
    """
    Realistic INSEASON emergence API response — single record with EmergenceDate,
    EmergenceStatus, ConfirmationStatus.
    """
    return {
        "id": "z361x33",
        "data": {
            "EmergenceDate": "2025-04-18",
            "EmergenceStatus": "CONFIRMED",
            "ConfirmationStatus": "VALIDATED",
        },
    }


@pytest.fixture
def sample_emergence_historical_response():
    """
    Realistic HISTORICAL emergence API response — 5 prior years' emergence dates plus
    historical average (in MM-DD format).
    """
    return {
        "id": "z361x33",
        "data": {
            "Emergence_year-1": "2024-04-15",
            "Emergence_year-2": "2023-04-22",
            "Emergence_year-3": "2022-04-12",
            "Emergence_year-4": "2021-04-20",
            "Emergence_year-5": "2020-04-18",
            "Hist_avg_emergence": "04-17",
        },
    }


@pytest.fixture
def sample_emergence_delay_response():
    """
    Realistic DELAY emergence API response — current emergence date,
    historical average, and delay in days.
    """
    return {
        "id": "z361x33",
        "data": {
            "EmergenceDate": "2025-04-25",
            "AverageEmergenceDate": "04-17",
            "EmergenceDelay": 8,
        },
    }


# ---------------------------------------------------------------------------
# FLM fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by FLMExtractor (map_products_url)
FAKE_FLM_URL = "http://fake-flm-api.example.com/field-level-maps/v5"

# Notebook test entity (cell 15)
FLM_WKT = (
    "POLYGON ((-97.70066562 37.14062335, -97.69927729 37.14227539, "
    "-97.69935777 37.14233954, -97.70004188 37.14248389, "
    "-97.70008212 37.14359058, -97.69969534 37.14450741, "
    "-97.70262024 37.14450741, -97.70254625 37.14062335, "
    "-97.70066562 37.14062335))"
)

# Image ID from the notebook (cell 15)
FLM_IMAGE_ID = "sentinel-2-c1-l2a|S2B_T14SPG_20251010T172020_L2A"

DEFAULT_FLM_PARAMS = {
    "vegetation_index": "NDVI",
    "map_format": None,
    "output_epsg": 4326,
    "postprocess": "stats",
    "skip_existing": True,
    "output_path": None,
    "extract_stats": False,
    "partial_frequency": 50,
    "directLinks": False,
    "clipping": "FieldBorder",
    "buffer": 0,
    "number_bins": None,
    "legendType": None,
}


def _make_flm_entity(entity_id="test_001", geometry=None, name=None, crop=None):
    """Helper to build a notebook-shaped FLM entity dict (id + geometry, optional name + crop)."""
    entity = {"id": entity_id, "geometry": geometry or FLM_WKT, "image_id": FLM_IMAGE_ID}
    if name is not None:
        entity["name"] = name
    if crop is not None:
        entity["crop"] = crop
    return entity


@pytest.fixture
def flm_extractor(mock_logger):
    """
    Build an FLMExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(FLMExtractor, "__init__", lambda self, *a, **kw: None):
        ext = FLMExtractor.__new__(FLMExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "FLMExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by FLMExtractor.__init__
    ext.flm_params = None
    ext.map_products_url = FAKE_FLM_URL

    return ext


@pytest.fixture
def configured_flm_extractor(flm_extractor):
    """An FLMExtractor with stats-mode params already set (matches notebook's bulk stats setup)."""
    flm_extractor.flm_params = DEFAULT_FLM_PARAMS.copy()
    flm_extractor.cache_key_columns = ["id"]
    return flm_extractor


@pytest.fixture
def sample_flm_entity():
    """Notebook-shaped FLM entity (id, geometry, image_id, name)."""
    return _make_flm_entity(entity_id="toto", name="Test_Field")


@pytest.fixture
def sample_flm_entity_list():
    """Small DataFrame of notebook-shaped FLM entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_flm_entity("ent_001"),
            _make_flm_entity("ent_002"),
            _make_flm_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_flm_stats_response():
    """
    Realistic stats-mode FLM API response. Mirrors the EarthDaily Field Level Maps
    contract: legend.stat (max/mean/min) + legend.ranges (per-bucket pixel counts)
    + seasonField with id.
    """
    return {
        "seasonField": {"id": "test_001"},
        "legend": {
            "stat": {"max": 0.85, "mean": 0.62, "min": 0.21},
            "ranges": [
                {"minValue": 0.0, "maxValue": 0.3, "numberOfPixels": 120},
                {"minValue": 0.3, "maxValue": 0.6, "numberOfPixels": 340},
                {"minValue": 0.6, "maxValue": 0.9, "numberOfPixels": 540},
            ],
        },
    }


@pytest.fixture
def sample_flm_histogram_response():
    """
    Realistic histogram-mode FLM API response. The histogram block carries a
    global stat plus an `items` list with per-bucket valueMin/valueMax/numberOfPixel/area.
    """
    return {
        "seasonField": {"id": "test_001"},
        "histogram": {
            "min": 0.21,
            "max": 0.85,
            "mean": 0.62,
            "items": [
                {"valueMin": 0.21, "valueMax": 0.40, "numberOfPixel": 120, "area": 1200.0},
                {"valueMin": 0.40, "valueMax": 0.60, "numberOfPixel": 340, "area": 3400.0},
                {"valueMin": 0.60, "valueMax": 0.85, "numberOfPixel": 540, "area": 5400.0},
            ],
        },
    }


@pytest.fixture
def sample_flm_links_response():
    """
    Realistic links-mode FLM API response (directLinks=True). Includes _links,
    worldFile, mapSize, bBox, and seasonField.
    """
    return {
        "_links": {
            "image:image/png": "https://api.example.com/flm/test_001.png",
            "worldfile": "https://api.example.com/flm/test_001.pgw",
            "thumbnail": "https://api.example.com/flm/test_001_thumb.png",
        },
        "worldFile": {
            "a": 1e-5,
            "b": 0.0,
            "c": -97.7,
            "d": 0.0,
            "e": -1e-5,
            "f": 37.14,
        },
        "mapSize": {"width": 512, "height": 512},
        "bBox": {"xMin": -97.703, "xMax": -97.699, "yMin": 37.140, "yMax": 37.145},
        "seasonField": {"id": "test_001"},
    }


# ---------------------------------------------------------------------------
# GDD fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by GDDExtractor (weather_url)
FAKE_GDD_URL = "https://fake-weather-api.example.com"

# Notebook test entity (cell 12) — point geometry, dates fall back to params
GDD_POINT_WKT = "POINT (-58.93681679 -13.72531769)"
GDD_POLYGON_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_GDD_PARAMS = {
    "provider": "GLOBAL1",
    "lower_threshold": 10.0,
    "upper_threshold": 30.0,
    "start_date": "2023-01-17",
    "end_date": "2023-03-17",
    "reset_cumulative_every_year": False,
    "extrapolate_forecast_data": False,
    "partial_frequency": 50,
}


def _make_gdd_entity(entity_id="test_001", geometry=None, start_date=None, end_date=None):
    """Helper to build a notebook-shaped GDD entity dict."""
    entity = {"id": entity_id, "geometry": geometry or GDD_POINT_WKT}
    if start_date is not None:
        entity["start_date"] = start_date
    if end_date is not None:
        entity["end_date"] = end_date
    return entity


@pytest.fixture
def gdd_extractor(mock_logger):
    """
    Build a GDDExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(GDDExtractor, "__init__", lambda self, *a, **kw: None):
        ext = GDDExtractor.__new__(GDDExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "GDDExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by GDDExtractor.__init__
    ext.gdd_params = None
    ext.weather_url = FAKE_GDD_URL

    return ext


@pytest.fixture
def configured_gdd_extractor(gdd_extractor):
    """A GDDExtractor with gdd_params already set (matches notebook cell 10)."""
    gdd_extractor.gdd_params = DEFAULT_GDD_PARAMS.copy()
    gdd_extractor.cache_key_columns = ["id", "date"]
    return gdd_extractor


@pytest.fixture
def sample_gdd_entity():
    """Notebook-shaped GDD entity (id + point geometry, dates fall back to params)."""
    return _make_gdd_entity()


@pytest.fixture
def sample_gdd_entity_list():
    """Small DataFrame of notebook-shaped GDD entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_gdd_entity("ent_001"),
            _make_gdd_entity("ent_002"),
            _make_gdd_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_gdd_response():
    """
    Realistic GDD API response (GrowingDegreeDayResult). Three daily records
    with monotonically increasing cumulatedGrowingDegreeDay.
    """
    return {
        "elements": [
            {
                "date": "2023-01-17T00:00:00Z",
                "minimum": 18.5,
                "maximum": 28.2,
                "dailyGrowingDegreeDay": 13.35,
                "cumulatedGrowingDegreeDay": 13.35,
            },
            {
                "date": "2023-01-18T00:00:00Z",
                "minimum": 19.1,
                "maximum": 29.4,
                "dailyGrowingDegreeDay": 14.25,
                "cumulatedGrowingDegreeDay": 27.60,
            },
            {
                "date": "2023-01-19T00:00:00Z",
                "minimum": 17.8,
                "maximum": 27.6,
                "dailyGrowingDegreeDay": 12.70,
                "cumulatedGrowingDegreeDay": 40.30,
            },
        ]
    }


@pytest.fixture
def sample_gdd_response_empty_elements():
    """API response with the 'elements' key present but empty."""
    return {"elements": []}


# ---------------------------------------------------------------------------
# Harvest fixtures
# ---------------------------------------------------------------------------

FAKE_HARVEST_URL = "https://fake-harvest-api.example.com"

# Notebook test entity (cell 15)
HARVEST_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_HARVEST_PARAMS = {
    "harvest_type": "INSEASON_HARVEST",
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": 2025,
    "data_source": "LR",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_harvest_entity(entity_id="z361x33", crop="OTHERS", geometry=None, sowing_date=None):
    """Helper to build a notebook-shaped harvest entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or HARVEST_WKT,
    }
    if crop is not None:
        entity["crop"] = crop
    if sowing_date is not None:
        entity["sowing_date"] = sowing_date
    return entity


@pytest.fixture
def harvest_extractor(mock_logger):
    """
    Build a HarvestExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(HarvestExtractor, "__init__", lambda self, *a, **kw: None):
        ext = HarvestExtractor.__new__(HarvestExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "HarvestExtractor"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes (used by cache_single_entity decorator)
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by HarvestExtractor.__init__
    ext.harvest_params = None
    ext.harvest_url = FAKE_HARVEST_URL

    return ext


@pytest.fixture
def configured_harvest_extractor(harvest_extractor):
    """A HarvestExtractor with harvest_params set (INSEASON_HARVEST mode by default)."""
    harvest_extractor.harvest_params = DEFAULT_HARVEST_PARAMS.copy()
    harvest_extractor.cache_key_columns = ["id"]
    return harvest_extractor


@pytest.fixture
def sample_harvest_entity():
    """Notebook-shaped harvest entity (id, geometry, crop=OTHERS)."""
    return _make_harvest_entity()


@pytest.fixture
def sample_harvest_entity_list():
    """Small DataFrame of notebook-shaped harvest entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_harvest_entity("ent_001", "OTHERS"),
            _make_harvest_entity("ent_002", "CORN"),
            _make_harvest_entity("ent_003", "SOYBEANS"),
        ]
    )


@pytest.fixture
def sample_harvest_inseason_response():
    """
    Realistic INSEASON_HARVEST API response — single record with HarvestDate
    and HarvestStatus.
    """
    return {
        "id": "z361x33",
        "data": {
            "HarvestDate": "2025-08-22",
            "HarvestStatus": "HARVESTED",
        },
    }


@pytest.fixture
def sample_harvest_historical_response():
    """
    Realistic HISTORICAL_HARVEST API response — 5 prior years' harvest dates plus
    historical average (in MM-DD format).
    """
    return {
        "id": "z361x33",
        "data": {
            "harvest_year_1": "2024-08-15",
            "harvest_year_2": "2023-08-22",
            "harvest_year_3": "2022-08-12",
            "harvest_year_4": "2021-08-20",
            "harvest_year_5": "2020-08-18",
            "historical_harvest_average": "08-17",
        },
    }


@pytest.fixture
def sample_harvest_readiness_response():
    """
    Realistic HARVEST_READINESS API response — `date` populated when ready.
    """
    return {
        "id": "z361x33",
        "data": {"date": "2025-08-25"},
    }


@pytest.fixture
def sample_harvest_readiness_not_ready_response():
    """
    HARVEST_READINESS API response with the placeholder '0001-01-01' date,
    indicating the field is not ready for harvest.
    """
    return {
        "id": "z361x33",
        "data": {"date": "0001-01-01"},
    }


# ---------------------------------------------------------------------------
# HistoricalScore fixtures
# ---------------------------------------------------------------------------

FAKE_HISTORICAL_SCORE_URL = "https://fake-historical-score.example.com"

# Notebook test entity (cell 15)
HISTORICAL_SCORE_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_HISTORICAL_SCORE_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "threshold_start": 0.7,
    "year": 2025,
    "historical_seasons": None,
    "data_source": "LR",
    "publish_af": False,
    "partial_frequency": 50,
    "detail_level": "full",
}


def _make_historical_score_entity(entity_id="z361x33", crop="OTHERS", geometry=None, historical_seasons=None):
    """Helper to build a notebook-shaped historical_score entity dict."""
    entity = {"id": entity_id, "geometry": geometry or HISTORICAL_SCORE_WKT}
    if crop is not None:
        entity["crop"] = crop
    if historical_seasons is not None:
        entity["historical_seasons"] = historical_seasons
    return entity


@pytest.fixture
def historical_score_extractor(mock_logger):
    """
    Build a HistoricalScoreExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(HistoricalScoreExtractor, "__init__", lambda self, *a, **kw: None):
        ext = HistoricalScoreExtractor.__new__(HistoricalScoreExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "HistoricalScoreExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.historical_score_params = None
    ext.historical_score_url = FAKE_HISTORICAL_SCORE_URL

    return ext


@pytest.fixture
def configured_historical_score_extractor(historical_score_extractor):
    """A HistoricalScoreExtractor with historical_score_params already set."""
    historical_score_extractor.historical_score_params = DEFAULT_HISTORICAL_SCORE_PARAMS.copy()
    historical_score_extractor.cache_key_columns = ["id"]
    return historical_score_extractor


@pytest.fixture
def sample_historical_score_entity():
    """Notebook-shaped historical_score entity (id, geometry, crop=OTHERS)."""
    return _make_historical_score_entity()


@pytest.fixture
def sample_historical_score_entity_list():
    """Small DataFrame of notebook-shaped historical_score entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_historical_score_entity("ent_001", "OTHERS"),
            _make_historical_score_entity("ent_002", "CORN"),
            _make_historical_score_entity("ent_003", "SOYBEANS"),
        ]
    )


@pytest.fixture
def sample_historical_score_response():
    """
    Realistic full-detail historical score response. PotentialScores and SeasonBreaks
    are JSON-encoded strings — the formatter parses them and pivots into
    potential_score_<season> / season_break_<season> columns.
    """
    return {
        "id": "z361x33",
        "data": {
            "AveragePotentialScore": 0.78,
            "OlympicMeanPotentialScore": 0.80,
            "StandardDeviation": 0.05,
            "RiskScore": 0.22,
            "PotentialScores": (
                '[{"season":"2020","potentialScore":0.75},'
                '{"season":"2021","potentialScore":0.81},'
                '{"season":"2022","potentialScore":0.78},'
                '{"season":"2023","potentialScore":0.82},'
                '{"season":"2024","potentialScore":0.76}]'
            ),
            "SeasonBreaks": (
                '[{"season":"2020","seasonBreak":"2020-04-15"},'
                '{"season":"2021","seasonBreak":"2021-04-12"},'
                '{"season":"2022","seasonBreak":"2022-04-18"},'
                '{"season":"2023","seasonBreak":"2023-04-10"},'
                '{"season":"2024","seasonBreak":"2024-04-14"}]'
            ),
        },
    }


# ---------------------------------------------------------------------------
# InseasonScore fixtures
# ---------------------------------------------------------------------------

FAKE_INSEASON_SCORE_URL = "https://fake-inseason-score.example.com"

# Notebook test entity (cell 15)
INSEASON_SCORE_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_INSEASON_SCORE_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "nb_historical_year": 1,
    "threshold_start": 0.7,
    "historical_seasons": None,
    "data_source": "LR",
    "publish_af": False,
    "partial_frequency": 50,
    "detail_level": "full",
}


def _make_inseason_score_entity(
    entity_id="z361x33",
    crop="OTHERS",
    geometry=None,
    sowing_date="2025-10-25",
    end_date=None,
    historical_seasons=None,
):
    """Helper to build a notebook-shaped inseason_score entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or INSEASON_SCORE_WKT,
        "crop": crop,
        "sowing_date": sowing_date,
    }
    if end_date is not None:
        entity["end_date"] = end_date
    if historical_seasons is not None:
        entity["historical_seasons"] = historical_seasons
    return entity


@pytest.fixture
def inseason_score_extractor(mock_logger):
    """
    Build an InseasonScoreExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(InseasonScoreExtractor, "__init__", lambda self, *a, **kw: None):
        ext = InseasonScoreExtractor.__new__(InseasonScoreExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "InseasonScoreExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.inseason_score_params = None
    ext.inseason_score_url = FAKE_INSEASON_SCORE_URL

    return ext


@pytest.fixture
def configured_inseason_score_extractor(inseason_score_extractor):
    """An InseasonScoreExtractor with inseason_score_params already set."""
    inseason_score_extractor.inseason_score_params = DEFAULT_INSEASON_SCORE_PARAMS.copy()
    inseason_score_extractor.cache_key_columns = ["id"]
    return inseason_score_extractor


@pytest.fixture
def sample_inseason_score_entity():
    """Notebook-shaped inseason_score entity (id, geometry, crop, sowing_date)."""
    return _make_inseason_score_entity()


@pytest.fixture
def sample_inseason_score_entity_list():
    """Small DataFrame of notebook-shaped inseason_score entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_inseason_score_entity("ent_001", "OTHERS"),
            _make_inseason_score_entity("ent_002", "CORN"),
            _make_inseason_score_entity("ent_003", "SOYBEANS"),
        ]
    )


@pytest.fixture
def sample_inseason_score_response():
    """Realistic in-season score API response (three score metrics)."""
    return {
        "id": "z361x33",
        "data": {
            "historical_potential_score": 0.78,
            "inseason_potential_score": 0.65,
            "relative_potential_score": -0.13,
        },
    }


# ---------------------------------------------------------------------------
# InSeasonMonitoring fixtures
# ---------------------------------------------------------------------------

FAKE_ISM_URL = "https://fake-inseason-monitoring.example.com/v1"

# Notebook test entity (cell 15)
ISM_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

# Notebook cell 13 passes data_source='LR' (a string) — the source's default
# of ["LR", "MR"] is unhashable and would fail validation; tests follow the
# notebook's actual usage.
DEFAULT_ISM_PARAMS = {
    "season_duration": 120,
    "season_start_month": 4,
    "season_start_day": 1,
    "year": "2025",
    "data_source": "LR",
    "partial_frequency": 50,
}


def _make_ism_entity(entity_id="z361x33", crop="OTHERS", geometry=None):
    """Helper to build a notebook-shaped in-season monitoring entity dict."""
    return {"id": entity_id, "geometry": geometry or ISM_WKT, "crop": crop}


@pytest.fixture
def ism_extractor(mock_logger):
    """
    Build an InSeasonMonitoringExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(InSeasonMonitoringExtractor, "__init__", lambda self, *a, **kw: None):
        ext = InSeasonMonitoringExtractor.__new__(InSeasonMonitoringExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "InSeasonMonitoringExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.inseason_monitoring_params = None
    ext.inseason_monitoring_url = FAKE_ISM_URL

    return ext


@pytest.fixture
def configured_ism_extractor(ism_extractor):
    """An InSeasonMonitoringExtractor with inseason_monitoring_params already set."""
    ism_extractor.inseason_monitoring_params = DEFAULT_ISM_PARAMS.copy()
    ism_extractor.cache_key_columns = ["id"]
    return ism_extractor


@pytest.fixture
def sample_ism_entity():
    """Notebook-shaped ISM entity (id, geometry, crop=OTHERS)."""
    return _make_ism_entity()


@pytest.fixture
def sample_ism_entity_list():
    """Small DataFrame of notebook-shaped ISM entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_ism_entity("ent_001", "OTHERS"),
            _make_ism_entity("ent_002", "CORN"),
            _make_ism_entity("ent_003", "SOYBEANS"),
        ]
    )


@pytest.fixture
def sample_ism_response():
    """
    Realistic ISM API response — 3 daily monitoring records with vegetation index,
    cumulative metrics, emergence info, and historical comparison fields.
    """
    return {
        "id": "z361x33",
        "data": [
            {
                "Season": "2025",
                "RequestDate": "2025-04-15",
                "VegetationIndexValue": 0.32,
                "CumulativeVegetationIndex": 0.32,
                "EmergenceDate": "2025-04-12",
                "EmergenceStatus": "CONFIRMED",
                "DaysSinceEmergence": 3,
                "CumulativeVegetationComparedToAverage": "BELOW",
                "HistoricalAverageCumulativeVegetationIndex": 0.45,
                "Delta": -0.13,
            },
            {
                "Season": "2025",
                "RequestDate": "2025-04-22",
                "VegetationIndexValue": 0.51,
                "CumulativeVegetationIndex": 0.83,
                "EmergenceDate": "2025-04-12",
                "EmergenceStatus": "CONFIRMED",
                "DaysSinceEmergence": 10,
                "CumulativeVegetationComparedToAverage": "BELOW",
                "HistoricalAverageCumulativeVegetationIndex": 1.10,
                "Delta": -0.27,
            },
            {
                "Season": "2025",
                "RequestDate": "2025-04-29",
                "VegetationIndexValue": 0.68,
                "CumulativeVegetationIndex": 1.51,
                "EmergenceDate": "2025-04-12",
                "EmergenceStatus": "CONFIRMED",
                "DaysSinceEmergence": 17,
                "CumulativeVegetationComparedToAverage": "ON_TRACK",
                "HistoricalAverageCumulativeVegetationIndex": 1.55,
                "Delta": -0.04,
            },
        ],
    }


@pytest.fixture
def sample_ism_response_empty():
    """API response with the 'data' key present but empty — formatter returns empty DataFrame."""
    return {"id": "z361x33", "data": []}


# ---------------------------------------------------------------------------
# MRTS fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by MRTSExtractor (map_products_url)
FAKE_MRTS_URL = "http://fake-flm-api.example.com/field-level-maps/v5"

# Notebook-style polygon geometry
MRTS_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_MRTS_PARAMS = {
    "start_date": "2025-05-01",
    "end_date": "2025-10-15",
    "sensors": None,  # None → omit from payload, use all available sensors
    "vegetation_index": "NDVI",
    "aggregation": "average",
    "smoothing_method": "Whittaker",
    "apply_denoiser": True,
    "apply_end_of_curve": True,
    "clear_cover_min": 100,
    "mask": "Auto",
    "output_saturation": True,
    "extract_raw_datasets": True,
    "compute_temporal_consistency": True,
    "temporal_consistency_threshold": {"Ndvi": 0.06, "Lai": 0.3, "S2Rep": 2.5},
    "mode": "full",
    "historical_years": 10,
    "partial_frequency": 50,
    "kpi_filter": None,
}


def _make_mrts_entity(entity_id="ent_001", crop=None, geometry=None, start_date=None, end_date=None):
    """Helper to build a notebook-shaped MRTS entity dict."""
    entity = {"id": entity_id, "geometry": geometry or MRTS_WKT}
    if crop is not None:
        entity["crop"] = crop
    if start_date is not None:
        entity["start_date"] = start_date
    if end_date is not None:
        entity["end_date"] = end_date
    return entity


@pytest.fixture
def mrts_extractor(mock_logger):
    """
    Build an MRTSExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(MRTSExtractor, "__init__", lambda self, *a, **kw: None):
        ext = MRTSExtractor.__new__(MRTSExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "MRTSExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    # Class-level enums normally set in MRTSExtractor.__init__
    ext.available_indices = {"NDVI", "EVI", "CVI", "GNDVI", "NDWI", "LAI", "NDRE", "NDMI", "S2REP"}
    ext.available_sensors = {"Sentinel_2", "Landsat_8", "Landsat_9", "HJ2B_CCD4", "GAOFEN_6_WFV3"}
    ext.available_smoothing = {"Whittaker", "SavitzkyGolay", "None"}
    ext.available_aggregation = {"average", "median", "max", "min", "accumulation", "std"}
    ext.available_masks = {"native": "Native", "acm": "ACM", "auto": "Auto", "ml": "ML", "mlcirrus": "MLCirrus"}

    ext.vegetation_ts_params = None
    ext.mrts_params = None
    ext.mrts_url = FAKE_MRTS_URL

    return ext


@pytest.fixture
def configured_mrts_extractor(mrts_extractor):
    """An MRTSExtractor with mrts_params already set."""
    mrts_extractor.mrts_params = DEFAULT_MRTS_PARAMS.copy()
    mrts_extractor.cache_key_columns = ["id", "date"]
    return mrts_extractor


@pytest.fixture
def sample_mrts_entity():
    """Notebook-shaped MRTS entity (id, geometry)."""
    return _make_mrts_entity()


@pytest.fixture
def sample_mrts_entity_list():
    """Small DataFrame of MRTS entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_mrts_entity("ent_001"),
            _make_mrts_entity("ent_002"),
            _make_mrts_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_mrts_response_full():
    """
    Realistic MRTS API response with both rawData and smoothedData.
    The formatter merges them on date in 'full' mode (and renames `value` to
    `raw_value` / `smoothed_value`).
    """
    return {
        "rawData": [
            {
                "date": "2025-05-10",
                "value": 0.42,
                "noised": False,
                "temporalConsistencyCheck": "OK",
                "mask": "ML",
                "coveragePercent": 100.0,
                "image": {"id": "sentinel-2-c1-l2a|S2A_T20JNT_20250510T134221_L2A"},
            },
            {
                "date": "2025-05-25",
                "value": 0.55,
                "noised": False,
                "temporalConsistencyCheck": "OK",
                "mask": "ML",
                "coveragePercent": 95.0,
                "image": {"id": "sentinel-2-c1-l2a|S2B_T20JNT_20250525T134219_L2A"},
            },
            {
                "date": "2025-06-09",
                "value": 0.71,
                "noised": False,
                "temporalConsistencyCheck": "OK",
                "mask": "ML",
                "coveragePercent": 100.0,
                "image": {"id": "sentinel-2-c1-l2a|S2A_T20JNT_20250609T134221_L2A"},
            },
        ],
        "smoothedData": [
            {"date": "2025-05-10", "value": 0.40},
            {"date": "2025-05-25", "value": 0.56},
            {"date": "2025-06-09", "value": 0.70},
        ],
    }


@pytest.fixture
def sample_mrts_response_raw_only():
    """MRTS response with rawData only, no smoothedData."""
    return {
        "rawData": [
            {
                "date": "2025-05-10",
                "value": 0.42,
                "noised": False,
                "temporalConsistencyCheck": "OK",
                "mask": "ML",
                "coveragePercent": 100.0,
                "image": {"id": "sentinel-2-c1-l2a|S2A_T20JNT_20250510T134221_L2A"},
            },
            {
                "date": "2025-05-25",
                "value": 0.55,
                "noised": False,
                "temporalConsistencyCheck": "OK",
                "mask": "ML",
                "coveragePercent": 95.0,
                "image": {"id": "sentinel-2-c1-l2a|S2B_T20JNT_20250525T134219_L2A"},
            },
        ],
        "smoothedData": [],
    }


@pytest.fixture
def sample_mrts_response_empty():
    """API response where both rawData and smoothedData are empty."""
    return {"rawData": [], "smoothedData": []}


# ---------------------------------------------------------------------------
# Planted Area fixtures
# ---------------------------------------------------------------------------

FAKE_PLANTED_URL = "https://fake-planted-area.example.com"

# Notebook test entity (cell 15)
PLANTED_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_PLANTED_PARAMS = {
    "processor_mode": "PLANTED_AREA",
    "emergence_date": "2025-04-01",
    "threshold": 30,
    "control_threshold": 4,
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_planted_entity(
    entity_id="z361x33",
    geometry=None,
    crop="OTHERS",
    emergence_date=None,
):
    """Helper to build a notebook-shaped planted entity dict."""
    entity = {"id": entity_id, "geometry": geometry or PLANTED_WKT, "crop": crop}
    if emergence_date is not None:
        entity["emergence_date"] = emergence_date
    return entity


@pytest.fixture
def planted_extractor(mock_logger):
    """
    Build a PlantedExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(PlantedExtractor, "__init__", lambda self, *a, **kw: None):
        ext = PlantedExtractor.__new__(PlantedExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "PlantedExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.planted_params = None
    ext.planted_url = FAKE_PLANTED_URL

    return ext


@pytest.fixture
def configured_planted_extractor(planted_extractor):
    """A PlantedExtractor with planted_params already set (PLANTED_AREA mode)."""
    planted_extractor.planted_params = DEFAULT_PLANTED_PARAMS.copy()
    planted_extractor.cache_key_columns = ["id"]
    return planted_extractor


@pytest.fixture
def sample_planted_entity():
    """Notebook-shaped planted entity (id, geometry, crop, emergence_date)."""
    return _make_planted_entity(emergence_date="2025-04-02")


@pytest.fixture
def sample_planted_entity_list():
    """Small DataFrame of notebook-shaped planted entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_planted_entity("ent_001", crop="OTHERS", emergence_date="2025-04-02"),
            _make_planted_entity("ent_002", crop="CORN", emergence_date="2025-04-05"),
            _make_planted_entity("ent_003", crop="SOYBEANS", emergence_date="2025-04-10"),
        ]
    )


@pytest.fixture
def sample_planted_response():
    """Realistic PLANTED_AREA API response with planted_area + planted_percentage."""
    return {
        "planted_area": 14523.5,
        "planted_percentage": 92.4,
    }


@pytest.fixture
def sample_planted_control_response():
    """Realistic CONTROL mode response — difference + control_threshold + result."""
    return {
        "difference": 0.025,
        "control_threshold": 0.04,
        "result": True,
    }


# ---------------------------------------------------------------------------
# VegationTsExtractor (VTS) fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by VegationTsExtractor (vts_urls)
FAKE_VTS_URL = "http://fake-vts-api.example.com/vegetation-time-series/v1"

# Notebook test entity (cell 20) — Uruguay polygon used in the dev notebook
VTS_WKT = (
    "POLYGON ((-57.17400567 -33.70070656, -57.17404544 -33.70089983, "
    "-57.17426497 -33.70085284, -57.1751335 -33.700639620000004, "
    "-57.17621637 -33.70034084, -57.176741660000005 -33.70013531, "
    "-57.17750935 -33.69988044, -57.17758905 -33.69976989, "
    "-57.179569210000004 -33.69782629, -57.17962191 -33.69772377, "
    "-57.17902293 -33.6971412, -57.17789108 -33.69631739, "
    "-57.17602474 -33.69811988, -57.175453260000005 -33.69861127, "
    "-57.17495038 -33.69896085, -57.17434281 -33.69944557, "
    "-57.17369835 -33.6998999, -57.17379511 -33.70024369, "
    "-57.17389311 -33.70048781, -57.17400567 -33.70070656))"
)

DEFAULT_VTS_PARAMS = {
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
    "vegetation_index": "NDVI",
    "is_extrapolated": True,
    "limit": 3000,
    "historical_years": 10,
    "partial_frequency": 50,
    "extraction_mode": "period",
    "target_dates": None,
    "kpi_filter": None,
}


def _make_vts_entity(entity_id="z361x33", geometry=None, start_date=None, end_date=None, years=None):
    """Helper to build a notebook-shaped VTS entity dict."""
    entity = {"id": entity_id, "geometry": geometry or VTS_WKT}
    if start_date is not None:
        entity["start_date"] = start_date
    if end_date is not None:
        entity["end_date"] = end_date
    if years is not None:
        entity["years"] = years
    return entity


@pytest.fixture
def vts_extractor(mock_logger):
    """
    Build a VegationTsExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(VegationTsExtractor, "__init__", lambda self, *a, **kw: None):
        ext = VegationTsExtractor.__new__(VegationTsExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "VegationTsExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.vegetation_ts_params = None
    ext.vegetation_ts_url = FAKE_VTS_URL

    return ext


@pytest.fixture
def configured_vts_extractor(vts_extractor):
    """A VegationTsExtractor with vegetation_ts_params already set (period mode)."""
    vts_extractor.vegetation_ts_params = DEFAULT_VTS_PARAMS.copy()
    vts_extractor.cache_key_columns = ["id", "date"]
    return vts_extractor


@pytest.fixture
def sample_vts_entity():
    """Notebook-shaped VTS entity (id, geometry)."""
    return _make_vts_entity()


@pytest.fixture
def sample_vts_entity_list():
    """Small DataFrame of VTS entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_vts_entity("ent_001"),
            _make_vts_entity("ent_002"),
            _make_vts_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_vts_response():
    """
    Realistic VTS API response — list of dicts with date + value, ordered descending
    by date (matches the API's `$sort=-date`).
    """
    return [
        {"date": "2025-08-15T00:00:00Z", "value": 0.78},
        {"date": "2025-07-15T00:00:00Z", "value": 0.62},
        {"date": "2025-06-15T00:00:00Z", "value": 0.45},
        {"date": "2025-05-15T00:00:00Z", "value": 0.31},
    ]


@pytest.fixture
def sample_vts_response_empty():
    """Empty list response — formatter should return an empty DataFrame."""
    return []


# ---------------------------------------------------------------------------
# Weather fixtures
# ---------------------------------------------------------------------------

FAKE_WEATHER_URL = "https://fake-weather-api.example.com"

# Notebook test entity (cell 17)
WEATHER_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.730314100000001, "
    "-58.93159922 -13.71888551, -58.94540508 -13.72028589))"
)

DEFAULT_WEATHER_PARAMS = {
    "weather_type": "HISTORICAL_DAILY",
    "weather_parameters": "Temperature.standardmax",
    "historical_years": 0,
    "partial_frequency": 50,
    "kpi_filter": None,
}


def _make_weather_entity(
    entity_id="z361x33",
    geometry=None,
    crop="SOYBEANS",
    start_date="2025-06-01",
    end_date="2025-10-01",
    years=None,
):
    """Helper to build a notebook-shaped weather entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or WEATHER_WKT,
        "crop": crop,
        "start_date": start_date,
        "end_date": end_date,
    }
    if years is not None:
        entity["years"] = years
    return entity


@pytest.fixture
def weather_extractor(mock_logger):
    """
    Build a WeatherExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(WeatherExtractor, "__init__", lambda self, *a, **kw: None):
        ext = WeatherExtractor.__new__(WeatherExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "WeatherExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.weather_params = None
    ext.weather_url = FAKE_WEATHER_URL

    return ext


@pytest.fixture
def configured_weather_extractor(weather_extractor):
    """A WeatherExtractor with weather_params already set (matches notebook cell 15)."""
    weather_extractor.weather_params = DEFAULT_WEATHER_PARAMS.copy()
    weather_extractor.cache_key_columns = ["id", "date"]
    return weather_extractor


@pytest.fixture
def sample_weather_entity():
    """Notebook-shaped weather entity (id, geometry, crop, start/end dates)."""
    return _make_weather_entity()


@pytest.fixture
def sample_weather_entity_list():
    """Small DataFrame of notebook-shaped weather entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_weather_entity("ent_001"),
            _make_weather_entity("ent_002"),
            _make_weather_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_weather_response():
    """
    Realistic weather API response — list of daily records. The 'Temperature' field
    is nested (the formatter flattens it to dotted column names).
    """
    return [
        {
            "date": "2025-06-01T00:00:00Z",
            "Temperature": {"standard": 18.5, "standardMin": 12.3, "standardMax": 24.1},
            "precipitation": {"cumulative": 0.0},
            "wind": 3.2,
        },
        {
            "date": "2025-06-02T00:00:00Z",
            "Temperature": {"standard": 19.1, "standardMin": 13.0, "standardMax": 25.3},
            "precipitation": {"cumulative": 2.5},
            "wind": 4.1,
        },
        {
            "date": "2025-06-03T00:00:00Z",
            "Temperature": {"standard": 17.8, "standardMin": 11.5, "standardMax": 23.6},
            "precipitation": {"cumulative": 0.8},
            "wind": 2.8,
        },
    ]


@pytest.fixture
def sample_weather_response_empty():
    """Empty list — formatter returns an empty DataFrame."""
    return []


# ---------------------------------------------------------------------------
# Regional fixtures
# ---------------------------------------------------------------------------

FAKE_REGIONAL_URL = "https://fake-regional-api.example.com"

# Mirrors the notebook setup (cell 8): Germany Districts block, idpixeltype=1, VVI.
DEFAULT_REGIONAL_PARAMS = {
    "index": "vegetation-vigor-index",
    "start_date": "2018-01-01",
    "end_date": "2026-12-31",
    "fillyeargap": False,
    "idblock": 281,
    "idpixeltype": 1,
    "indicatorTypeIds": [1],
    "partial_frequency": 50,
}


def _make_regional_entity(amu_id=2432528):
    """Helper to build a notebook-shaped regional entity dict (amu_id only)."""
    return {"amu_id": amu_id}


@pytest.fixture
def regional_extractor(mock_logger):
    """
    Build a RegionalExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(RegionalExtractor, "__init__", lambda self, *a, **kw: None):
        ext = RegionalExtractor.__new__(RegionalExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "RegionalExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled so cache_single_entity decorator is a no-op
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by RegionalExtractor.__init__
    ext.regional_params = None
    ext.regional_url = FAKE_REGIONAL_URL

    return ext


@pytest.fixture
def configured_regional_extractor(regional_extractor):
    """A RegionalExtractor with regional_params set (ready for API calls)."""
    regional_extractor.regional_params = DEFAULT_REGIONAL_PARAMS.copy()
    regional_extractor.cache_key_columns = ["id", "date"]
    return regional_extractor


@pytest.fixture
def sample_regional_entity():
    """Notebook-shaped regional entity (amu_id only — known good ID from notebook cell 11)."""
    return _make_regional_entity()


@pytest.fixture
def sample_regional_entity_list():
    """Small DataFrame of notebook-shaped regional entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_regional_entity(2432528),
            _make_regional_entity(2121564),
            _make_regional_entity(2121566),
        ]
    )


@pytest.fixture
def sample_regional_response():
    """
    Realistic regional API response with both observedMeasures and dailyAverage.
    - observedMeasures: ISO-time daily readings.
    - dailyAverage: dayOfYear in 1MMDD format (climatology).
    """
    return {
        "observedMeasures": [
            {"time": "2025-03-15T00:00:00Z", "dayId": 3, "indicatorTypeId": 1, "value": 0.55},
            {"time": "2025-01-15T00:00:00Z", "dayId": 1, "indicatorTypeId": 1, "value": 0.42},
            {"time": "2025-02-01T00:00:00Z", "dayId": 2, "indicatorTypeId": 1, "value": 0.45},
        ],
        "dailyAverage": [
            {"dayOfYear": 10315, "value": 0.50},
            {"dayOfYear": 10115, "value": 0.40},
            {"dayOfYear": 10201, "value": 0.43},
        ],
    }


@pytest.fixture
def sample_regional_response_observed_only():
    """Response with only observedMeasures populated."""
    return {
        "observedMeasures": [
            {"time": "2025-01-15T00:00:00Z", "dayId": 1, "indicatorTypeId": 1, "value": 0.42},
            {"time": "2025-02-01T00:00:00Z", "dayId": 2, "indicatorTypeId": 1, "value": 0.45},
        ],
        "dailyAverage": [],
    }


@pytest.fixture
def sample_regional_response_daily_avg_only():
    """Response with only dailyAverage populated."""
    return {
        "observedMeasures": [],
        "dailyAverage": [
            {"dayOfYear": 10115, "value": 0.40},
            {"dayOfYear": 10201, "value": 0.43},
        ],
    }


@pytest.fixture
def sample_regional_response_empty():
    """Both observedMeasures and dailyAverage empty."""
    return {"observedMeasures": [], "dailyAverage": []}


# ---------------------------------------------------------------------------
# Zoning (SAMZ) fixtures
# ---------------------------------------------------------------------------

# Match the prod URL shape used by ZoningExtractor (map_products_url)
FAKE_ZONING_URL = "http://fake-zoning-api.example.com/field-level-maps/v5"

# Notebook test entity (cells 11 & 22) — Kansas field
ZONING_WKT = (
    "POLYGON ((-97.70066562 37.14062335, -97.69927729 37.14227539, "
    "-97.69935777 37.14233954, -97.70004188 37.14248389, "
    "-97.70008212 37.14359058, -97.69969534 37.14450741, "
    "-97.70262024 37.14450741, -97.70254625 37.14062335, "
    "-97.70066562 37.14062335))"
)

# Single image_id and a list of three (notebook cell 11 returns top-3)
ZONING_IMAGE_ID = "sentinel-2-c1-l2a|S2B_T14SPG_20251010T172020_L2A"
ZONING_IMAGE_ID_LIST = [
    "sentinel-2-c1-l2a|S2B_T14SPG_20251010T172020_L2A",
    "sentinel-2-c1-l2a|S2A_T14SPG_20251005T172020_L2A",
    "landsat-c2l2-sr|LC09_L2SP_028033_20251002_20251003_02_T1_SR",
]

# Mirrors notebook cell 14 setup (stats mode)
DEFAULT_ZONING_PARAMS = {
    "num_zones": 5,
    "output_epsg": 4326,
    "postprocess": "stats",
    "map_format": None,
    "output_path": None,
    "skip_existing": True,
    "directLinks": False,
    "partial_frequency": 50,
}


def _make_zoning_entity(entity_id="test_001", geometry=None, image_id=None, name=None):
    """Helper to build a notebook-shaped zoning entity dict (id + geometry + image_id, optional name)."""
    entity = {
        "id": entity_id,
        "geometry": geometry or ZONING_WKT,
        "image_id": image_id if image_id is not None else ZONING_IMAGE_ID,
    }
    if name is not None:
        entity["name"] = name
    return entity


@pytest.fixture
def zoning_extractor(mock_logger):
    """
    Build a ZoningExtractor with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the class are set manually.
    """
    with patch.object(ZoningExtractor, "__init__", lambda self, *a, **kw: None):
        ext = ZoningExtractor.__new__(ZoningExtractor)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "ZoningExtractor"

    # Column mapping & output formatting
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled by default for zoning
    ext.use_cache = False
    ext.cache_key_columns = None

    # Attributes set by ZoningExtractor.__init__
    ext.zoning_params = None
    ext.map_products_url = FAKE_ZONING_URL

    return ext


@pytest.fixture
def configured_zoning_extractor(zoning_extractor):
    """A ZoningExtractor with stats-mode params set (mirrors notebook cell 14)."""
    zoning_extractor.zoning_params = DEFAULT_ZONING_PARAMS.copy()
    zoning_extractor.cache_key_columns = ["id", "image_id"]
    return zoning_extractor


@pytest.fixture
def sample_zoning_entity():
    """Notebook-shaped zoning entity (id, geometry, image_id, name)."""
    return _make_zoning_entity(entity_id="test_001", name="Test_Field")


@pytest.fixture
def sample_zoning_entity_list():
    """Small DataFrame of notebook-shaped zoning entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_zoning_entity("ent_001"),
            _make_zoning_entity("ent_002"),
            _make_zoning_entity("ent_003"),
        ]
    )


@pytest.fixture
def sample_zoning_stats_response():
    """
    Realistic stats-mode SAMZ API response. Mirrors the legend.stat / legend.ranges
    contract: field-level stats + per-zone area/productivity/variability.
    """
    return {
        "seasonField": {"id": "test_001"},
        "legend": {
            "stat": {
                "fieldVariability": "MEDIUM",
                "mostVariableZone": "3",
                "highestInterZoneVariability": ["1", "5"],
                "fieldProductivityIndex": 0.62,
                "fieldVariabilityIndex": 0.18,
            },
            "ranges": [
                {"name": "1", "fieldAreaPercent": 12.5, "productivityIndex": 0.32, "variabilityIndex": 0.05},
                {"name": "2", "fieldAreaPercent": 22.0, "productivityIndex": 0.48, "variabilityIndex": 0.08},
                {"name": "3", "fieldAreaPercent": 30.5, "productivityIndex": 0.62, "variabilityIndex": 0.20},
                {"name": "4", "fieldAreaPercent": 22.0, "productivityIndex": 0.74, "variabilityIndex": 0.10},
                {"name": "5", "fieldAreaPercent": 13.0, "productivityIndex": 0.85, "variabilityIndex": 0.06},
            ],
        },
    }


@pytest.fixture
def sample_zoning_stats_geo_response():
    """
    Realistic stats_geo-mode SAMZ API response. Combines legend.ranges (productivity/
    variability) with zones[*].segments (geometry) and zones[*].stats (mean/max/min/area).
    Includes one zone with multiple segments to exercise the GEOMETRYCOLLECTION wrap.
    """
    return {
        "seasonField": {"id": "test_001"},
        "legend": {
            "stat": {
                "fieldVariability": "HIGH",
                "mostVariableZone": "2",
                "highestInterZoneVariability": ["1", "3"],
                "fieldProductivityIndex": 0.55,
                "fieldVariabilityIndex": 0.25,
            },
            "ranges": [
                {
                    "name": "1",
                    "fieldAreaPercent": 30.0,
                    "productivityIndex": 0.30,
                    "variabilityIndex": 0.08,
                    "numberOfPixels": 1200,
                },
                {
                    "name": "2",
                    "fieldAreaPercent": 40.0,
                    "productivityIndex": 0.55,
                    "variabilityIndex": 0.15,
                    "numberOfPixels": 1600,
                },
                {
                    "name": "3",
                    "fieldAreaPercent": 30.0,
                    "productivityIndex": 0.80,
                    "variabilityIndex": 0.06,
                    "numberOfPixels": 1200,
                },
            ],
        },
        "zones": [
            {
                "id": 1,
                "segments": [
                    {"geometry": "POLYGON((-97.70 37.14, -97.69 37.14, -97.69 37.15, -97.70 37.15, -97.70 37.14))"}
                ],
                "stats": {"mean": 0.30, "max": 0.40, "min": 0.20, "area": 1500.0},
            },
            {
                # Two segments → should be wrapped in GEOMETRYCOLLECTION
                "id": 2,
                "segments": [
                    {"geometry": "POLYGON((-97.70 37.14, -97.695 37.14, -97.695 37.145, -97.70 37.145, -97.70 37.14))"},
                    {
                        "geometry": "POLYGON((-97.695 37.145, -97.69 37.145, -97.69 37.15, -97.695 37.15, -97.695 37.145))"
                    },
                ],
                "stats": {"mean": 0.55, "max": 0.65, "min": 0.45, "area": 2000.0},
            },
            {
                "id": 3,
                "segments": [
                    {"geometry": "POLYGON((-97.695 37.14, -97.69 37.14, -97.69 37.145, -97.695 37.145, -97.695 37.14))"}
                ],
                "stats": {"mean": 0.80, "max": 0.90, "min": 0.70, "area": 1500.0},
            },
        ],
    }


@pytest.fixture
def sample_zoning_links_response():
    """Realistic links-mode SAMZ response with _links + worldFile + mapSize + bBox."""
    return {
        "seasonField": {"id": "test_001"},
        "_links": {
            "image:image/png": "https://api.example.com/samz/test_001.png",
            "worldfile": "https://api.example.com/samz/test_001.pgw",
            "thumbnail": "https://api.example.com/samz/test_001_thumb.png",
        },
        "worldFile": {"a": 1e-5, "b": 0.0, "c": -97.7, "d": 0.0, "e": -1e-5, "f": 37.14},
        "mapSize": {"width": 512, "height": 512},
        "bBox": {"xMin": -97.703, "xMax": -97.699, "yMin": 37.140, "yMax": 37.145},
    }


# ---------------------------------------------------------------------------
# LocationBasedBorder fixtures
# ---------------------------------------------------------------------------

FAKE_LOCATION_BORDER_URL = "http://fake-borders-api.example.com/field-borders/v1/AutomaticBoundary"

# Single Point WKT (lon, lat) — Iowa farmland.
LOCATION_BORDER_POINT_WKT = "POINT (-93.6 41.5)"

# A polygon the fake API will return for the point above.
LOCATION_BORDER_POLYGON_WKT = (
    "POLYGON ((-93.601 41.499, -93.599 41.499, -93.599 41.501, -93.601 41.501, -93.601 41.499))"
)

DEFAULT_LOCATION_BORDER_PARAMS = {
    "simplified_geom": True,
    "partial_frequency": 50,
}


def _make_location_border_entity(entity_id="loc_001", geometry=None, name=None):
    """Helper to build a minimal location-based-border entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or LOCATION_BORDER_POINT_WKT,
    }
    if name is not None:
        entity["name"] = name
    return entity


@pytest.fixture
def location_based_border_extractor(mock_logger):
    """Build a LocationBasedBorderExtractor with BaseExtractor.__init__ bypassed."""
    with patch.object(LocationBasedBorderExtractor, "__init__", lambda self, *a, **kw: None):
        ext = LocationBasedBorderExtractor.__new__(LocationBasedBorderExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "LocationBasedBorderExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.location_based_border_params = None
    ext.location_based_border_url = FAKE_LOCATION_BORDER_URL

    return ext


@pytest.fixture
def configured_location_based_border_extractor(location_based_border_extractor):
    """A LocationBasedBorderExtractor with default params already set."""
    location_based_border_extractor.location_based_border_params = DEFAULT_LOCATION_BORDER_PARAMS.copy()
    location_based_border_extractor.cache_key_columns = ["id", "geometry"]
    return location_based_border_extractor


@pytest.fixture
def sample_location_border_entity():
    """Notebook-shaped entity (id, geometry as a Point WKT)."""
    return _make_location_border_entity()


@pytest.fixture
def sample_location_border_entity_list():
    """Small DataFrame of point entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_location_border_entity("loc_001"),
            _make_location_border_entity("loc_002", geometry="POINT (-93.61 41.51)"),
            _make_location_border_entity("loc_003", geometry="POINT (-93.62 41.52)"),
        ]
    )


# ---------------------------------------------------------------------------
# StandingCrop fixtures
# ---------------------------------------------------------------------------

FAKE_STANDING_CROP_URL = "http://fake-standing-crop.example.com/v1"

STANDING_CROP_WKT = (
    "POLYGON ((-58.94540508 -13.72028589, -58.942163 -13.73172321, "
    "-58.928124090000004 -13.73031410000000, -58.93060599 -13.71867078, "
    "-58.94540508 -13.72028589))"
)

DEFAULT_STANDING_CROP_PARAMS = {
    "reference_date": "2025-06-15",
    "crop": None,
    "threshold": 30,
    "index": "NDVI",
    "publish_af": False,
    "partial_frequency": 50,
}


def _make_standing_crop_entity(entity_id="sc_001", geometry=None, reference_date=None, crop=None):
    """Helper to build a minimal StandingCrop entity dict."""
    entity = {
        "id": entity_id,
        "geometry": geometry or STANDING_CROP_WKT,
    }
    if reference_date is not None:
        entity["reference_date"] = reference_date
    if crop is not None:
        entity["crop"] = crop
    return entity


@pytest.fixture
def standing_crop_extractor(mock_logger):
    """Build a StandingCropExtractor with BaseExtractor.__init__ bypassed."""
    with patch.object(StandingCropExtractor, "__init__", lambda self, *a, **kw: None):
        ext = StandingCropExtractor.__new__(StandingCropExtractor)

    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "StandingCropExtractor"

    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.column_mapping["reference_date"] = "reference_date"
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    ext.use_cache = False
    ext.cache_key_columns = None

    ext.standing_crop_params = None
    ext.standing_crop_url = FAKE_STANDING_CROP_URL

    return ext


@pytest.fixture
def configured_standing_crop_extractor(standing_crop_extractor):
    """A StandingCropExtractor with default params set."""
    standing_crop_extractor.standing_crop_params = DEFAULT_STANDING_CROP_PARAMS.copy()
    standing_crop_extractor.cache_key_columns = ["id", "reference_date"]
    return standing_crop_extractor


@pytest.fixture
def sample_standing_crop_entity():
    """Notebook-shaped StandingCrop entity (id, geometry, optional reference_date)."""
    return _make_standing_crop_entity()


@pytest.fixture
def sample_standing_crop_entity_list():
    """Small DataFrame of StandingCrop entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_standing_crop_entity("sc_001"),
            _make_standing_crop_entity("sc_002"),
            _make_standing_crop_entity("sc_003"),
        ]
    )


# ---------------------------------------------------------------------------
# User (UserManager read-lifecycle) fixtures
# ---------------------------------------------------------------------------

FAKE_USER_URL = "https://fake-mdm-api.example.com"


def _make_user_entity(entity_id="usr_001"):
    """Minimal user-read entity — the mapped ``id`` column holds the MDM user id."""
    return {"id": entity_id}


@pytest.fixture
def user_manager(mock_logger):
    """
    Build a UserManager with BaseExtractor.__init__ fully bypassed.
    All attributes needed by the read lifecycle are set manually.
    """
    with patch.object(UserManager, "__init__", lambda self, *a, **kw: None):
        ext = UserManager.__new__(UserManager)

    # Attributes normally set by BaseExtractor.__init__
    ext.bearer_token = FAKE_TOKEN
    ext.token_expiration = FAKE_EXPIRATION
    ext.config = FAKE_CONFIG
    ext.workflow_ref = None
    ext.env = FAKE_CONFIG["env"]
    ext.partial_path = FAKE_CONFIG["partial_result_dir"]
    ext.output_path = FAKE_CONFIG["output_result_dir"]
    ext.output_uri = None  # durable raster mirror; None = local-only
    ext.merge_existing = FAKE_CONFIG["merge_existing"]
    ext.logger = mock_logger
    ext.extractor_name = "UserManager"

    # Column mapping & output formatting (set by BaseExtractor.__init__)
    from earthdaily.agriculture.core.base_extractor import BaseExtractor

    ext.column_mapping = dict(BaseExtractor.DEFAULT_COLUMN_MAPPING)
    ext.output_mapping = None
    ext.exclude_columns = None
    ext.output_columns = None

    # Cache attributes — disabled
    ext.use_cache = False
    ext.cache_dir = None
    ext.cache_key_columns = None

    # Attributes set by UserManager.__init__
    ext.csv_separator = ","
    ext.user_params = None
    ext.base_url = f"{FAKE_USER_URL}/users"

    return ext


@pytest.fixture
def configured_user_manager(user_manager):
    """A UserManager with user_params already set (read lifecycle ready)."""
    user_manager.user_params = {"fields": None, "partial_frequency": 50}
    user_manager.cache_key_columns = ["id"]
    return user_manager


@pytest.fixture
def sample_user_entity():
    """A single user-read entity (id only)."""
    return _make_user_entity()


@pytest.fixture
def sample_user_entity_list():
    """Small DataFrame of user-read entities for bulk tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            _make_user_entity("usr_001"),
            _make_user_entity("usr_002"),
            _make_user_entity("usr_003"),
        ]
    )


@pytest.fixture
def sample_user_response():
    """Realistic MDM user record as returned by get_user_by_id (nested objects)."""
    return {
        "id": "usr_001",
        "login": "jdoe",
        "email": "jdoe@example.com",
        "firstname": "Jane",
        "lastname": "Doe",
        "companyName": "Acme Farms",
        "userType": {"code": "GROWER"},
        "country": {"code": "US"},
    }
