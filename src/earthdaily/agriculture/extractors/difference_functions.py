#  Standard Library
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

#  Third-Party Libraries
import requests
from tqdm.auto import tqdm

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.api_utils import (
    export_results,
    filter_entities,
    normalize_with_metadata,
    retry_with_backoff_no_retry_on_400,
)
from earthdaily.agriculture.core.base_extractor import BaseExtractor, cache_single_entity, requires_token
from earthdaily.agriculture.core.geometry import validate_wkt
from earthdaily.agriculture.core.identity import EDAuthenticator

# Supported difference map products — use short index name as key
DIFFERENCE_PRODUCTS = {
    "NDVI": "DIFFERENCE_NDVI",
    "EVI": "DIFFERENCE_EVI",
    "GNDVI": "DIFFERENCE_GNDVI",
    "NDRE": "DIFFERENCE_NDRE",
    "CVI": "DIFFERENCE_CVI",
    "CVIN": "DIFFERENCE_CVIN",
    "LAI": "DIFFERENCE_LAI",
    "NDWI": "DIFFERENCE_NDWI",
    "NDMI": "DIFFERENCE_NDMI",
    "S2REP": "DIFFERENCE_S2REP",
}


def require_difference_params(func):
    """Decorator to ensure difference parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "difference_params") or self.difference_params is None:
            error_msg = "No difference parameters found. Call setup_difference_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class DifferenceExtractor(BaseExtractor):
    """
    Extracts difference maps between two satellite images for agricultural entities.

    Calls the Difference Map API to compute pixel-level vegetation index differences
    between two dates. Returns field-level statistics (max, mean, min) and per-range
    breakdowns with pixel counts and area.

    Documentation: https://docs.earthdaily.com/agro/library/Field%20Level%20Maps/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_difference.ipynb

    Args (setup_difference_parameters):
        product (str): Vegetation index short name (e.g. 'NDVI', 'EVI'). Default: 'NDVI'
        output_epsg (int): Output coordinate system. Default: 4326
        postprocess (str): Processing mode - 'stats', 'links', or 'file'. Default: 'stats'
        map_format (str): Output format for file mode ('png', 'tiff.zip', 'shp.zip'). Default: None
        output_path (str): Directory for file downloads. Required when postprocess='file'.
        directLinks (bool): Request direct download links from API. Default: False
        partial_frequency (int): How often to save partial results. Default: 50
        use_cache (bool): Reuse cached API responses and cache new results to avoid
            re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry, image_id_1, image_id_2 (required)

    Output columns:
        Varies by postprocess mode.
        stats: entity_id, stat_max, stat_mean, stat_min,
            range_{n}_min, range_{n}_max, range_{n}_pixels, range_{n}_area,
            range_{n}_color_r, range_{n}_color_g, range_{n}_color_b
        links: image_png_link, worldfile_link, thumbnail_link, worldfile_*, bbox_*, map_width, map_height
        file: status, map_format, file_count, total_size_bytes, saved_files

    TODO:
        - Cache support: cache is disabled by default because results depend on image_id_1
          and image_id_2. To enable cache properly, both image IDs need to be serialized
          to a stable string for cache key lookup.
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.difference_params = None
        self.map_products_url = agro_urls["map_products_url"][self.env]

        self.logger.debug(f"Map products URL configured: {self.map_products_url}")
        self.logger.info(f"DifferenceExtractor ready for {self.env} environment")

    def get_new_token(self):
        """Implements token refresh logic for DifferenceExtractor."""
        self.logger.debug("Getting new token for DifferenceExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_difference_parameters(
        self,
        product="NDVI",
        output_epsg=4326,
        postprocess="stats",
        map_format=None,
        output_path=None,
        skip_existing=True,
        directLinks=False,
        partial_frequency=50,
        column_mapping=None,
        output_mapping=None,
        exclude_columns=None,
        output_columns=None,
        use_cache=None,
    ):
        """
        Configure parameters for difference map extraction.

        Args:
            product (str): Vegetation index short name. Default: 'NDVI'
                Supported values: NDVI, EVI, GNDVI, NDRE, CVI, CVIN, LAI
                Also accepts full names (e.g. 'DIFFERENCE_NDVI')
            output_epsg (int): Output EPSG code for projection. Default: 4326
            postprocess (str): Processing mode - 'stats', 'links', or 'file'. Default: 'stats'
            map_format (str): Output format for file mode ('png', 'tiff.zip', 'shp.zip'). Default: None
            output_path (str): Directory for file downloads. Required when postprocess='file'.
            skip_existing (bool): Skip download if file already exists. Default: True
            directLinks (bool): Request direct download links. Default: False
                Note: Automatically set to True when postprocess='links'
            partial_frequency (int): How often to save partial results. Default: 50
            column_mapping (dict): Column name overrides.
            output_mapping (dict): Output column renames.
            exclude_columns (list): Columns to exclude from output.
            output_columns (list): Whitelist of output columns.
            use_cache (bool): Whether to use caching.
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring difference parameters...")

        # Validate and resolve product — accepts short name (e.g. 'NDVI') or full name (e.g. 'DIFFERENCE_NDVI')
        if product in DIFFERENCE_PRODUCTS:
            product = DIFFERENCE_PRODUCTS[product]
        elif product not in DIFFERENCE_PRODUCTS.values():
            error_msg = f"Invalid product '{product}'. Must be one of {list(DIFFERENCE_PRODUCTS.keys())} (or full name like 'DIFFERENCE_NDVI')"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Postprocess validation
        valid_postprocess = {"stats", "links", "file"}
        if postprocess not in valid_postprocess:
            error_msg = f"Invalid postprocess '{postprocess}'. Must be one of {valid_postprocess}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Map format validation
        valid_formats = {"png", "tiff.zip", "shp.zip", None}
        if map_format not in valid_formats:
            error_msg = f"Invalid map_format '{map_format}'. Must be one of {valid_formats}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Cross-validation: postprocess='links' requirements
        if postprocess == "links":
            if not directLinks:
                log.warning("postprocess='links' requires directLinks=True. Auto-correcting.")
                directLinks = True
            if map_format is not None:
                error_msg = f"postprocess='links' requires map_format=None, but got map_format='{map_format}'"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Cross-validation: postprocess='file' requirements
        if postprocess == "file":
            if map_format is None:
                error_msg = "postprocess='file' requires map_format to be set ('png', 'tiff.zip', or 'shp.zip')"
                log.error(error_msg)
                raise ValueError(error_msg)
            if output_path is None:
                error_msg = "postprocess='file' requires output_path to be set"
                log.error(error_msg)
                raise ValueError(error_msg)

        if column_mapping:
            self.set_column_mapping(column_mapping)

        self.apply_cache_setting(use_cache)

        self.difference_params = {
            "product": product,
            "output_epsg": output_epsg,
            "postprocess": postprocess,
            "map_format": map_format,
            "output_path": output_path,
            "skip_existing": skip_existing,
            "directLinks": directLinks,
            "partial_frequency": partial_frequency,
        }

        # Cache disabled by default: results depend on image_id_1 + image_id_2
        if use_cache is None:
            self.use_cache = False
        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col, "image_id_1", "image_id_2"]

        # Configure output formatting
        self.configure_output(
            output_mapping=output_mapping, exclude_columns=exclude_columns, output_columns=output_columns
        )

        log.success("Difference parameters configured successfully")
        log.debug(f"Parameters: {self.difference_params}")

        print("Difference parameters configured:")
        for k, v in self.difference_params.items():
            print(f"   {k}: {v}")

    # -------------------------------------------------------------------------
    # API CALL
    # -------------------------------------------------------------------------

    @requires_token
    @require_difference_params
    def get_difference_map_api(self, entity_data: dict):
        """
        Call the Difference Map API to compute vegetation index difference between two images.
        Returns raw Response when map_format is set (file download),
        or parsed JSON otherwise (stats/links).

        Args:
            entity_data (dict): Entity data including 'id', 'geometry', 'image_id_1', and 'image_id_2'.

        Returns:
            requests.Response (file mode) or dict (stats/links mode)
        """
        params = self.difference_params
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

        # Image ID validation.
        # `get_entity_value`, not `entity_data.get`: a plain dict lookup ignores
        # `column_mapping`, so a user who mapped these columns got no error and no
        # effect. Resolution is unchanged when no mapping is set — the lookup falls
        # back to the canonical key.
        image_id_1 = self.get_entity_value(entity_data, "image_id_1")
        image_id_2 = self.get_entity_value(entity_data, "image_id_2")

        if not image_id_1:
            error_msg = f"entity_data for {entity_id} must include an 'image_id_1' field."
            log.error(error_msg)
            raise ValueError(error_msg)

        if not image_id_2:
            error_msg = f"entity_data for {entity_id} must include an 'image_id_2' field."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Geometry validation
        if not self.has_entity_field(entity_data, "geometry"):
            error_msg = f"entity_data for {entity_id} must include a 'geometry' field in WKT format."
            log.error(error_msg)
            raise ValueError(error_msg)

        geometry = self.get_entity_value(entity_data, "geometry")
        try:
            geometry = validate_wkt(geometry)
        except ValueError as e:
            log.error(f"Invalid geometry for entity {entity_id}: {e}")
            raise

        product = params["product"]

        # API URL — append file extension if downloading
        file_extension = params.get("map_format")
        if file_extension:
            url = f"{self.map_products_url}/maps/difference-map/{product}/image.{file_extension}"
        else:
            url = f"{self.map_products_url}/maps/difference-map/{product}"

        # Query parameters
        query_params = [f"directLinks={'true' if params['directLinks'] else 'false'}", f"$epsg={params['output_epsg']}"]
        full_url = f"{url}?{'&'.join(query_params)}"

        # Payload — earliest and latest images for comparison
        payload = {
            "seasonField": {"geometry": geometry},
            "earliestImage": {"id": str(image_id_1)},
            "latestImage": {"id": str(image_id_2)},
        }

        log.info(f"Requesting difference map for entity {entity_id}, images [{image_id_1}, {image_id_2}]")
        log.debug(f"API URL: {full_url}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(
                full_url,
                headers={"Authorization": f"Bearer {self.bearer_token}", "Accept": "*/*"},
                json=payload,  # type: ignore[arg-type]
                timeout=120,
            )
            response.raise_for_status()
            log.success(f"Difference map retrieved for entity {entity_id}")

            # Return raw response for file downloads, parsed JSON otherwise
            if file_extension:
                log.debug(f"API response: binary content ({len(response.content)} bytes)")
                return response
            json_response = response.json()
            log.debug(f"API response: {json_response}")
            return json_response

        except requests.exceptions.Timeout as e:
            log.error(f"Timeout requesting difference map for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"Unexpected error for entity {entity_id}: {e}")
            raise

    def get_difference_map_api_safe(self, entity_data: dict) -> dict:
        """
        Safe wrapper around get_difference_map_api().

        Args:
            entity_data (dict): Entity data including 'id', 'geometry', 'image_id_1', 'image_id_2'

        Returns:
            dict: {"success": bool, "data": dict|None, "error": str|None, "seasonfield_id": str}
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response = self.get_difference_map_api(entity_data)
            return {"success": True, "data": response, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}
        except Exception as e:
            log.warning(f"Error for entity {entity_id}: {e}")
            return {"success": False, "data": None, "error": str(e), "seasonfield_id": entity_id}

    # -------------------------------------------------------------------------
    # FORMAT: STATS
    # -------------------------------------------------------------------------

    @require_difference_params
    def format_difference_stats_json(self, response_json):
        """
        Extract field-level statistics and per-range breakdowns from Difference Map API response.

        Args:
            response_json (dict): Parsed API response

        Returns:
            pd.DataFrame: Single-row DataFrame with stats and flattened range columns
        """
        log = self.get_contextualized_logger("FORMAT")

        if not response_json:
            log.warning("Empty response")
            return pd.DataFrame()

        # Convert Response object to JSON if needed
        if hasattr(response_json, "json"):
            try:
                response_json = response_json.json()
            except ValueError as e:
                log.error(f"Failed to parse JSON response: {e}")
                return pd.DataFrame()

        legend = response_json.get("legend", {})
        stat = legend.get("stat", {})
        ranges = legend.get("ranges", [])

        # Field-level stats
        row = {
            "stat_max": stat.get("max"),
            "stat_mean": stat.get("mean"),
            "stat_min": stat.get("min"),
        }

        # Per-range stats
        for i, rng in enumerate(ranges, start=1):
            row[f"range_{i}_min"] = rng.get("minValue")
            row[f"range_{i}_max"] = rng.get("maxValue")
            row[f"range_{i}_pixels"] = rng.get("numberOfPixels")
            row[f"range_{i}_area"] = rng.get("area")
            color = rng.get("color", {})
            row[f"range_{i}_color_r"] = color.get("r")
            row[f"range_{i}_color_g"] = color.get("g")
            row[f"range_{i}_color_b"] = color.get("b")

        log.debug(f"Extracted stats with {len(ranges)} ranges")
        return pd.DataFrame([row])

    # -------------------------------------------------------------------------
    # FORMAT: LINKS
    # -------------------------------------------------------------------------

    @require_difference_params
    def format_difference_links_json(self, response_json):
        """
        Extract direct links and metadata from Difference Map API response.

        Args:
            response_json (dict or Response): API response with _links, worldFile, etc.

        Returns:
            pd.DataFrame: Single-row DataFrame with links and spatial metadata
        """
        log = self.get_contextualized_logger("FORMAT_LINKS")

        # Convert Response object to JSON if needed
        if hasattr(response_json, "json"):
            try:
                response_json = response_json.json()
            except ValueError as e:
                log.error(f"Failed to parse JSON response: {e}")
                return pd.DataFrame()

        links = response_json.get("_links", {})
        world_file = response_json.get("worldFile", {})
        map_size = response_json.get("mapSize", {})
        bbox = response_json.get("bBox", {})
        season_field = response_json.get("seasonField", {})

        entity_id = season_field.get("id", "unknown")
        log.debug(f"Extracting links for entity {entity_id}")

        links_row = {
            # Links
            "image_png_link": links.get("image:image/png"),
            "worldfile_link": links.get("worldfile"),
            "thumbnail_link": links.get("thumbnail"),
            # WorldFile parameters (for georeferencing)
            "worldfile_a": world_file.get("a"),
            "worldfile_b": world_file.get("b"),
            "worldfile_c": world_file.get("c"),
            "worldfile_d": world_file.get("d"),
            "worldfile_e": world_file.get("e"),
            "worldfile_f": world_file.get("f"),
            # Map size
            "map_width": map_size.get("width"),
            "map_height": map_size.get("height"),
            # Bounding box
            "bbox_xmin": bbox.get("xMin"),
            "bbox_xmax": bbox.get("xMax"),
            "bbox_ymin": bbox.get("yMin"),
            "bbox_ymax": bbox.get("yMax"),
        }

        links_df = pd.DataFrame([links_row])
        log.debug(f"Created links DataFrame for entity {entity_id}, links found: {list(links.keys())}")
        return links_df

    # -------------------------------------------------------------------------
    # FORMAT: FILE
    # -------------------------------------------------------------------------

    @require_difference_params
    def format_difference_file(self, entity_data: dict, response, output_path=None):
        """
        Save difference map file response (PNG, TIFF.ZIP, or SHP.ZIP) to disk.

        Args:
            entity_data (dict): Entity data including 'id'
            response (requests.Response): Raw response from get_difference_map_api
            output_path (str, optional): Directory to save files.

        Returns:
            dict: Status and file path information
        """
        params = self.difference_params
        entity_id = self.get_entity_value(entity_data, "id")
        entity_name = entity_data.get("name")
        log = self.get_contextualized_logger("FILE_SAVE")

        image_id_1 = self.get_entity_value(entity_data, "image_id_1", "img1")
        image_id_2 = self.get_entity_value(entity_data, "image_id_2", "img2")
        product = params.get("product", "DIFFERENCE_NDVI")

        log.info(f"Saving difference map file for entity {entity_id}")

        try:
            save_path = output_path or params.get("output_path", self.output_path)
            if not save_path:
                error_msg = "No output path specified for file saving"
                log.error(error_msg)
                raise ValueError(error_msg)

            map_format = params.get("map_format", "png")

            # Construct base filename
            if entity_name:
                output_filename = f"{entity_name}_{entity_id}_{product}"
            else:
                output_filename = f"{entity_id}_{product}"

            # Sanitize filename
            for char in '<>:"|?*':
                output_filename = output_filename.replace(char, "_")
            output_filename = output_filename.replace("/", "_").replace("\\", "_")

            saved_files = []

            if map_format == "png":
                full_path = self.save_map_file(save_path, f"{output_filename}.png", response.content)
                saved_files.append(full_path)

            elif map_format == "tiff.zip":
                with ZipFile(BytesIO(response.content)) as zip_file:
                    for file_info in zip_file.namelist():
                        if file_info.endswith(".tif") or file_info.endswith(".tiff"):
                            full_path = self.save_map_file(
                                save_path, f"{output_filename}.tif", zip_file.read(file_info)
                            )
                            saved_files.append(full_path)
                            break
                if not saved_files:
                    raise ValueError("No .tif file found in the ZIP archive")

            elif map_format == "shp.zip":
                with ZipFile(BytesIO(response.content)) as zip_file:
                    for file_info in zip_file.namelist():
                        _, ext = os.path.splitext(file_info)
                        if ext:
                            new_filename = f"{output_filename}{ext}"
                            output_file_path = self.save_map_file(save_path, new_filename, zip_file.read(file_info))
                            saved_files.append(output_file_path)
                if not saved_files:
                    raise ValueError("No files found in the shapefile ZIP archive")

            else:
                raise ValueError(f"Unsupported map_format: {map_format}")

            log.success(
                f"Saved {len(saved_files)} file(s) for entity {entity_id}, total size: {len(response.content)} bytes"
            )

            return {
                "status": "downloaded",
                "entity_id": entity_id,
                "image_id_1": image_id_1,
                "image_id_2": image_id_2,
                "product": product,
                "map_format": map_format,
                "saved_files": saved_files,
                "file_count": len(saved_files),
                "total_size_bytes": len(response.content),
            }

        except Exception as e:
            log.error(f"Failed to save file for entity {entity_id}: {e}")
            return {
                "status": "error",
                "error_message": f"Failed to save file: {str(e)}",
                "entity_id": entity_id,
                "image_id_1": image_id_1,
                "image_id_2": image_id_2,
                "product": product,
                "map_format": params.get("map_format", "unknown"),
            }

    # -------------------------------------------------------------------------
    # PROCESS SINGLE ENTITY
    # -------------------------------------------------------------------------

    @requires_token
    @require_difference_params
    @cache_single_entity("difference_params")
    def process_single_entity_difference(self, row, params=None):
        """
        Process difference map extraction for a single entity with retry logic.
        Routes to stats, links, or file processing based on postprocess parameter.

        Args:
            row (dict|Series): Entity data with id, geometry, image_id_1, image_id_2
            params (dict, optional): Override parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.difference_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")

        # Validate geometry
        try:
            geometry = self.get_entity_value(row, "geometry")
            validate_wkt(geometry)
        except Exception as e:
            log.error(f"Geometry validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Validate image IDs — via get_entity_value so column_mapping applies
        image_id_1 = self.get_entity_value(row, "image_id_1")
        image_id_2 = self.get_entity_value(row, "image_id_2")

        if not image_id_1:
            log.error(f"Missing image_id_1 for entity {entity_id}")
            return {"data": None, "error": {"message": "Missing image_id_1", "entity_id": entity_id}}

        if not image_id_2:
            log.error(f"Missing image_id_2 for entity {entity_id}")
            return {"data": None, "error": {"message": "Missing image_id_2", "entity_id": entity_id}}

        # Validate image_id format
        for label, img_id in [("image_id_1", image_id_1), ("image_id_2", image_id_2)]:
            img_str = str(img_id)
            separator_count = img_str.count("|")
            if separator_count < 1 or separator_count > 3:
                log.error(f"Invalid {label} format for entity {entity_id}: {img_str}")
                return {
                    "data": None,
                    "error": {"message": f"Invalid {label} format: {img_str}", "entity_id": entity_id},
                }

            parts = img_str.split("|")
            if not all(part.strip() for part in parts):
                log.error(f"Invalid {label} format (empty parts) for entity {entity_id}: {img_str}")
                return {
                    "data": None,
                    "error": {"message": f"Invalid {label} format (empty parts): {img_str}", "entity_id": entity_id},
                }

        # API call with retry
        def _call_api():
            return self.get_difference_map_api(row)

        try:
            raw_response = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # Route by postprocess type
            postprocess_type = params.get("postprocess", "stats")
            log.debug(f"Processing with postprocess type: {postprocess_type}")

            if postprocess_type == "stats":
                # Validate API response
                validation = self.validate_api_response(raw_response, entity_id, "difference")
                if isinstance(validation, pd.DataFrame):
                    return {"data": validation, "error": {"message": "Empty API response", "entity_id": entity_id}}

                difference_df = self.format_difference_stats_json(raw_response)
                if difference_df is None or difference_df.empty:
                    return {"data": None, "error": {"message": "No stats extracted", "entity_id": entity_id}}

            elif postprocess_type == "links":
                difference_df = self.format_difference_links_json(raw_response)
                if difference_df is None or difference_df.empty:
                    return {"data": None, "error": {"message": "No links found", "entity_id": entity_id}}

            elif postprocess_type == "file":
                file_result = self.format_difference_file(
                    entity_data=row, response=raw_response, output_path=params.get("output_path")
                )
                if file_result.get("status") == "error":
                    return {
                        "data": None,
                        "error": {
                            "message": file_result.get("error_message", "File save failed"),
                            "entity_id": entity_id,
                        },
                    }
                difference_df = pd.DataFrame(
                    [
                        {
                            "status": file_result.get("status"),
                            "map_format": file_result.get("map_format"),
                            "file_count": file_result.get("file_count"),
                            "total_size_bytes": file_result.get("total_size_bytes"),
                            "saved_files": str(file_result.get("saved_files", [])),
                        }
                    ]
                )
                log.success(f"File saved for entity {entity_id}: {file_result.get('file_count')} file(s)")

            else:
                error_msg = f"Unknown postprocess type: {postprocess_type}"
                log.error(error_msg)
                return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

            # Add metadata from entity row
            difference_df = normalize_with_metadata(row, difference_df)

            log.success(f"Successfully processed entity {entity_id} with postprocess={postprocess_type}")
            return {"data": difference_df, "error": None}

        except Exception as e:
            log.error(f"Failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    # -------------------------------------------------------------------------
    # BULK PROCESSING
    # -------------------------------------------------------------------------

    def process_entity_difference_bulk_parallel(
        self,
        entity_list,
        params=None,
        max_workers=5,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="difference",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity difference maps using threads + progress bar,
        with optional fail-safe retry and filter capabilities.

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain 'id', 'geometry', 'image_id_1', and 'image_id_2'.
            params (dict, optional): Difference parameters override.
            max_workers (int, optional): Number of threads to use.
            output_path (str, optional): Directory to save final CSV results.
            partial_frequency (int, optional): How often to save partial results.
            fail_safe (bool, optional): If True, per-entity failures are captured to
                ``failed_ids`` instead of aborting the run. Error tolerance only -- it does
                NOT change which entities are processed. To reprocess just the IDs a
                previous run recorded, set ``retry_failed_only`` (see BaseExtractor).
            filter_column (str, optional): Column name to filter entities by.
            filter_value (any, optional): Value to filter in the filter_column.
            filter_type (str, optional): 'exclude' to skip rows with filter_value, 'include' to process only rows with filter_value. Defaults to 'exclude'.
            merge_existing (str, optional): Merge strategy - 'auto', 'preserve', or 'mark'. If None, uses instance default.
            skip_export (bool, optional): If True, skip final export. Default: False.
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "difference".
            generate_report (bool, optional): Generate HTML report. Default: False.
            report_options (dict, optional): Report configuration options.
            use_cache (bool, optional): Enable/disable caching for this call.

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics.
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_entity_difference_bulk_parallel_inner,
                entity_list=entity_list,
                params=self.difference_params,
                max_workers=max_workers,
                output_path=output_path,
                partial_frequency=partial_frequency,
                fail_safe=fail_safe,
                filter_column=filter_column,
                filter_value=filter_value,
                filter_type=filter_type,
                merge_existing=merge_existing,
                skip_export=skip_export,
                prefix=prefix,
                generate_report=generate_report,
                report_options=report_options,
            )
        return self._process_entity_difference_bulk_parallel_inner(
            entity_list=entity_list,
            params_kw=params,
            max_workers=max_workers,
            output_path=output_path,
            partial_frequency=partial_frequency,
            fail_safe=fail_safe,
            filter_column=filter_column,
            filter_value=filter_value,
            filter_type=filter_type,
            merge_existing=merge_existing,
            skip_export=skip_export,
            prefix=prefix,
            generate_report=generate_report,
            report_options=report_options,
        )

    @requires_token
    @require_difference_params
    def _process_entity_difference_bulk_parallel_inner(
        self,
        entity_list,
        params_kw=None,
        max_workers=5,
        output_path=None,
        partial_frequency=50,
        fail_safe=False,
        filter_column=None,
        filter_value=None,
        filter_type="exclude",
        merge_existing=None,
        skip_export=False,
        prefix="difference",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of entity difference maps using threads + progress bar.
        """
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk difference extraction: {prefix}")
        log.info("=" * 60)
        log.info(f"Total input entities: {len(entity_list)}")
        log.info(f"Max workers: {max_workers}")
        log.info(f"Partial frequency: {partial_frequency}")
        log.info(f"Fail-safe mode: {fail_safe}")

        # Use instance default if not specified
        if merge_existing is None:
            merge_existing = self.merge_existing

        # Validate merge_existing value
        valid_modes = ["auto", "preserve", "mark"]
        if merge_existing not in valid_modes:
            error_msg = f"Invalid merge_existing='{merge_existing}'. Choose from: {valid_modes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(f"Merge strategy: {merge_existing}")

        # Apply filter to skip certain entities
        log.debug("Applying entity filters...")
        filtered_entity_list, skipped_entities, skip_count = filter_entities(
            entity_list, filter_column, filter_value, filter_type
        )

        if skip_count > 0:
            log.info(f"Filtered out {skip_count} entities by {filter_column}={filter_value} ({filter_type})")

        # Resume mode is explicit (self.retry_failed_only) and never implied
        # by fail_safe -- see BaseExtractor._resolve_retry_entity_list.
        filtered_entity_list = self._resolve_retry_entity_list(
            filtered_entity_list, prefix, fail_safe=fail_safe, log=log
        )

        all_rows = []
        global_errors = []
        successful_calculations = 0
        total_calculations = 0
        buffer_rows = []
        buffer_errors = []
        failed_ids = []

        log.info(f"Processing {len(filtered_entity_list)} entities in parallel...")
        print(f"Processing {prefix.upper()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_difference, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"Processing {prefix.upper()}", unit="entity") as pbar:
                for future in as_completed(future_to_id):
                    total_calculations += 1
                    entity_id = future_to_id[future]

                    try:
                        result = future.result()
                        df = result.get("data")
                        error = result.get("error")

                        if df is not None and not df.empty:
                            successful_calculations += 1
                            all_rows.append(df)
                            buffer_rows.append(df)
                            log.debug(f"Entity {entity_id}: Success")
                        else:
                            failed_ids.append(entity_id)
                            log.debug(f"Entity {entity_id}: No data returned")

                        if error:
                            error_record = {"entity_id": entity_id, **error}
                            global_errors.append(error_record)
                            buffer_errors.append(error_record)
                            if entity_id not in failed_ids:
                                failed_ids.append(entity_id)
                            log.warning(f"Entity {entity_id}: Error - {error.get('message', 'Unknown error')}")

                    except Exception as e:
                        error_record = {
                            "entity_id": entity_id,
                            "error_message": str(e),
                            "error_code": "THREAD_ERROR",
                        }
                        global_errors.append(error_record)
                        buffer_errors.append(error_record)
                        failed_ids.append(entity_id)
                        log.error(f"Thread error for entity {entity_id}: {e}", exc_info=True)

                    # Partial export
                    if self.partial_path and partial_frequency > 0 and total_calculations % partial_frequency == 0:
                        log.debug(f"Saving partial results at {total_calculations} entities")
                        partial_df = pd.concat(buffer_rows, ignore_index=True) if buffer_rows else pd.DataFrame()
                        export_results(
                            results_df=partial_df,
                            errors=buffer_errors,
                            output_path=self.partial_path,
                            prefix=prefix,
                            partial=True,
                            verbose=False,
                        )
                        log.info(f"Partial results saved: {len(buffer_rows)} rows")
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        log.info("=" * 60)
        log.success(f"Total processing time: {elapsed_time:.2f} seconds")
        log.success(f"Successful calculations: {successful_calculations}/{total_calculations}")
        log.info(f"Failed calculations: {total_calculations - successful_calculations}/{total_calculations}")
        log.info("=" * 60)

        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        print(f"Successful calculations: {successful_calculations}/{total_calculations}")

        # Concatenate all DataFrames
        log.debug("Concatenating all results...")
        results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        log.info(f"Results DataFrame shape: {results_df.shape}")

        # Merge back skipped entities to maintain all input rows in output
        log.debug("Merging with skipped entities...")
        results_df = self._merge_with_skipped_entities(
            new_results_df=results_df, skipped_entities_df=skipped_entities, merge_mode=merge_existing, verbose=True
        )

        # Finalize extraction: export, store failed IDs, cleanup partials
        log.info("Finalizing extraction...")
        results_df, finalization_status = self._finalize_extraction(
            results_df=results_df,
            global_errors=global_errors,
            failed_ids=failed_ids,
            output_path=output_path,
            prefix=prefix,
            skip_export=skip_export,
            verbose=True,
            generate_report=generate_report,
            report_options=report_options,
            extraction_stats={
                "total_entities": len(entity_list),
                "total_calculations": total_calculations,
                "successful": successful_calculations,
                "failed": total_calculations - successful_calculations,
                "elapsed_seconds": elapsed_time,
                "parameters": getattr(self, "difference_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("Bulk difference extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
