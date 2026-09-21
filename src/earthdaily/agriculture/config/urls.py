"""
- `eda_data_management_url`: url to get business entity like fields
- `layers_url`: geosys field border API urls to extract field borders from a given location, based on field layer produced by Geosys.
- `deep_resolution_field_borders_urls`: geosys field border API urls to extract field borders from a given location, based on field layer produced by Digifarm.
- `location_based_border_url`: alias for the AutomaticBoundary endpoint (`/field-borders/v1/AutomaticBoundary`) used by `LocationBasedBorderExtractor` to derive a polygon from a point WKT.
- `crop_id_url`: url to fetch crop information for a given geometry
- `change_index_url`: url for change index calculations between NDVI images
"""

agro_urls = {
    "eda_data_management_url": {
        "preprod": "https://api-pp.geosys-na.net/master-data-management/v6",
        "prod": "https://api.geosys-na.net/master-data-management/v6",
    },
    "eda_data_management_url_fields": {
        "preprod": "https://api-pp.geosys-na.net/master-data-management/v6/",
        "prod": "https://api.geosys-na.net/master-data-management/v6/",
    },
    "layers_url": {"preprod": "https://api-pp.geosys-na.net/layers/v1", "prod": "https://api.geosys-na.net/layers/v1"},
    "deep_resolution_field_borders_urls": {
        "preprod": "https://api-pp.geosys-na.net/field-borders/v1/AutomaticBoundary",
        "prod": "https://api.geosys-na.net/field-borders/v1/AutomaticBoundary",
    },
    # Same endpoint as `deep_resolution_field_borders_urls` — kept as a clearer
    # alias for the new LocationBasedBorderExtractor (point WKT -> polygon WKT).
    "location_based_border_url": {
        "preprod": "https://api-pp.geosys-na.net/field-borders/v1/AutomaticBoundary",
        "prod": "https://api.geosys-na.net/field-borders/v1/AutomaticBoundary",
    },
    "crop_id_url": {
        "preprod": "https://api-pp.geosys-na.net/cropmasks/v1/cropmasks/crops",
        "prod": "https://api.geosys-na.net/cropmasks/v1/cropmasks/crops",
    },
    "identity_urls": {
        "preprod": "https://identity.preprod.geosys-na.com/v2.1/connect/token",
        "prod": "https://identity.geosys-na.com/v2.1/connect/token",
    },
    "vts_urls": {
        "preprod": "https://api-pp.geosys-na.net/vegetation-time-series/v1",
        "prod": "https://api.geosys-na.net/vegetation-time-series/v1",
    },
    "vts_urls_values": {
        "preprod": "https://api-pp.geosys-na.net/vegetation-time-series/v1/season-fields/values",
        "prod": "https://api.geosys-na.net/vegetation-time-series/v1/season-fields/values",
    },
    "map_products_url": {
        "preprod": "https://api-pp.geosys-na.net/field-level-maps/v5",
        "prod": "https://api.geosys-na.net/field-level-maps/v5",
    },
    "base_url": {"preprod": "https://api-pp.geosys-na.net", "prod": "https://api.geosys-na.net"},
    "change_index_url": {
        "preprod": "https://change-index.aws-dev.geosys.com",
        "prod": "https://change-index.aws.geosys.com",
    },
    "inseason_monitoring_url": {
        "preprod": "https://inseason-monitoring.aws-dev.geosys.com/v1",
        "prod": "https://5gciu5p2msxwjrt54fks3kvvxu0oxdyr.lambda-url.us-east-1.on.aws/v1",
    },
    "emergence_url": {
        "preprod": "https://emergence-detection.aws-dev.geosys.com",
        "prod": "https://emergence-detection.aws.geosys.com",
    },
    "weather_url": {
        "preprod": "https://api-pp.geosys-na.net/Weather/v1",
        "prod": "https://api.geosys-na.net/Weather/v1",
    },
    "harvest_url": {
        "preprod": "https://harvest-detection.aws-dev.geosys.com",
        "prod": "https://harvest-detection.aws.geosys.com",
    },
    "historical_score_url": {
        "preprod": "https://historical-potential-risk-score.aws-dev.geosys.com",
        "prod": "https://historical-potential-risk-score.aws.geosys.com",
    },
    "inseason_score_url": {
        "preprod": "https://inseason-potential-score.aws-dev.geosys.com",
        "prod": "https://inseason-potential-score.aws.geosys.com",
    },
    "layer_service_url": {
        "preprod": "https://api-pp.geosys-na.net/layers/v1",
        "prod": "https://api.geosys-na.net/layers/v1",
    },
    "cover_crop_processor_url": {
        "preprod": "https://api-pp.geosys.com/processors/cover-crop/v1",
        "prod": "https://api.geosys.com/processors/cover-crop/v1",
    },
    "tillage_processor_url": {
        "preprod": "https://api-pp.geosys.com/processors/tillage/v1/prod/detection",
        "prod": "https://api.geosys.com/processors/tillage/v1/prod/detection",
    },
    "regional_url": {
        "preprod": "https://api-pp.geosys-na.net/Agriquest/Geosys.AgriQuest.CropMonitoring.WebApi/v0",
        "prod": "https://api.geosys-na.net/Agriquest/Geosys.AgriQuest.CropMonitoring.WebApi/v0",
    },
    "zarc_url": {
        "preprod": "https://zarc.aws-dev.geosys.com",
        "prod": "https://zvjihjkwfwohudwcnbhjbpsudq0jhszp.lambda-url.us-east-1.on.aws",
    },
    "disease_url": {
        "preprod": "https://k6j6vnyqscc3xjg3dwl6ltewua0zckzq.lambda-url.us-east-1.on.aws/v1",
        "prod": "https://t7izeqf7q5t4xseagqap5ev7xm0xpmmh.lambda-url.us-east-1.on.aws/v1",
    },
    "digifarm_url": {
        "preprod": "https://api.digifarm.io/development/delineated-fields/polygon",
        "prod": "https://api.digifarm.io/development/delineated-fields/polygon",
    },
    "environmental_compliance_url": {
        "preprod": "https://api-pp.geosys-na.net/reporting/environmentalcompliance/v1",
        "prod": "https://api.geosys-na.net/reporting/environmentalcompliance/v1",
    },
    "baresoil_processor_url": {
        "preprod": "https://avuqeoz2lrpi2s5qovww5k4vca0itlyy.lambda-url.us-east-1.on.aws/v1",
        "prod": "https://avuqeoz2lrpi2s5qovww5k4vca0itlyy.lambda-url.us-east-1.on.aws/v1",
    },
    "medium_resolution_time_series_radar_urls": {
        "preprod": "https://api-pp.geosys-na.net/field-level-maps/v5/time-serie-radar",
        "prod": "https://api.geosys-na.net/field-level-maps/v5/time-serie-radar",
    },
    "planted_urls": {
        "preprod": "https://planted-area.aws-dev.geosys.com/",
        "prod": "https://planted-area.aws.geosys.com",
    },
    "greenness_urls": {
        "preprod": "https://zn6hzsqoyoe3qgaoau4ssgpq440vtmpa.lambda-url.us-east-1.on.aws",
        "prod": "https://zn6hzsqoyoe3qgaoau4ssgpq440vtmpa.lambda-url.us-east-1.on.aws",
    },
    # LRTS moved to EarthDaily Data Studio. Both hosts authenticate with the EDS
    # token exchange (core/identity_eds.py), NOT the Geosys bearer — the old
    # preprod host lrts-api.aws-dev.geosys.com answers
    # 401 "Invalid or missing API key" to a Geosys token and is superseded.
    #
    # Verified 2026-08-06 with the same EDS token against both:
    #   BLEEDING -> 200, 90 NDVI observations
    #   prod     -> 403 Forbidden (account not subscribed to /lrts on prod yet)
    # So preprod is the working environment today; keep prod pointed at the real
    # host so it starts working the moment the entitlement lands.
    "lrts_urls": {
        "preprod": "https://bleeding-api.test-internal.earthdaily.com/lrts",
        "prod": "https://api.earthdaily.com/lrts",
    },
    "analytics_url": {
        "preprod": "https://api-pp.geosys-na.net/analytics",
        "prod": "https://api.geosys-na.net/analytics",
    },
    "standing_crop_url": {
        "preprod": "https://standing-crop.aws-dev.geosys.com/v1",
        "prod": "https://standing-crop.aws-dev.geosys.com/v1",
    },
    "history_url": {
        "preprod": "https://api-pp.geosys-na.net/accounts/history/v1",
        "prod": "https://api.geosys-na.net/accounts/history/v1",
    },
}
