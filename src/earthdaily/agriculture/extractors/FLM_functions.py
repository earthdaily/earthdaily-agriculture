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
from tqdm import tqdm

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


def require_flm_params(func):
    """Decorator to ensure FLM parameters are set before running a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "flm_params") or self.flm_params is None:
            error_msg = "❌ No FLM parameters found. Call setup_parameters() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        return func(self, *args, **kwargs)

    return wrapper


class FLMExtractor(BaseExtractor):
    """
    Extracts Field Level Maps (vegetation index maps) for agricultural entities.

    Generates field-level vegetation index maps (NDVI, EVI, NDRE, etc.) from satellite imagery.
    Supports map statistics extraction, direct download links, and raster file export with
    configurable clipping and buffering.

    Documentation: https://docs.earthdaily.com/agro/library/Field%20Level%20Maps/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_FLM_extraction_functions.ipynb

    Args (setup_flm_parameters):
        vegetation_index (str): Index type ('NDVI', 'EVI', 'NDRE', etc.). Default: 'NDVI'
        map_format (str): Output format (None, 'png', 'tiff'). Default: None
        output_epsg (int): Output coordinate system. Default: 4326
        postprocess (str): Post-processing mode ('stats', 'links', 'file', 'histogram'). Default: 'stats'
            'histogram' sets histogram=true on the query and returns the per-bucket
            breakdown as well as the global min/max/mean.
        extract_stats (bool): Extract map statistics. Default: False
        directLinks (bool): Return direct download links. Default: False
        clipping (str): Clipping mode ('FieldBorder', 'None'). Default: 'FieldBorder'
        buffer (int): Buffer around field border in meters. Default: 0
        number_bins (int): Number of legend bins for the rendered map. Default: None
        legendType (str): Legend style for the rendered map. Default: None
        use_cache (bool): Reuse cached API responses and cache new results to avoid re-fetching. None uses the extractor's instance default. Default: None

    Entity fields (via column_mapping):
        id, geometry (required)

    Output columns:
        Varies by postprocess mode:
        stats: stat_max, stat_mean, stat_min
        links: image_png_link, worldfile_link, thumbnail_link, bbox_*, map_width, map_height
        file: status, map_format, file_count, total_size_bytes, saved_files
        histogram: stat_min, stat_max, stat_mean, value_min_{i}, value_max_{i},
            num_pixels_{i}, area_{i} (one set of {i} columns per bucket, single row per entity)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None):
        # Initialize BaseExtractor first (sets env, paths, logger, credentials, etc.)
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        self.flm_params = None  # Will store FLM parameters after setup

        # Get resource URL
        self.map_products_url = agro_urls["map_products_url"][self.env]

        self.logger.debug(f"Map products URL configured: {self.map_products_url}")
        self.logger.info(f"🗺️ FLMExtractor ready for {self.env} environment")

    def get_new_token(self):
        """
        Implements token refresh logic for FLMExtractor.
        Called automatically by BaseExtractor.ensure_token_valid() if token is expired.
        """
        self.logger.debug("Getting new token for FLMExtractor")
        return EDAuthenticator.get_new_token(env=self.env)

    def setup_flm_parameters(
        self,
        vegetation_index="NDVI",
        map_format=None,
        output_epsg=4326,
        postprocess="stats",
        skip_existing=True,
        output_path=None,
        extract_stats=False,
        partial_frequency=50,
        directLinks=False,
        clipping="FieldBorder",
        buffer=0,
        number_bins=None,
        legendType=None,
        use_cache=None,
    ):
        """
        Configure parameters for FLM extraction

        Args:
            vegetation_index (str): Vegetation index to extract
            map_format (str): Format of the map ('png', 'tiff.zip', 'shp.zip')
            output_epsg (int): Output EPSG code for projection
            postprocess (str): Post-processing type - 'stats', 'file', 'links', or 'histogram'. Default: 'stats'
            skip_existing (bool): Skip download if file already exists
            output_path (str): Path to save downloaded maps
            extract_stats (bool): If True, extract statistics only (no file download)
            partial_frequency (int): How often to save partial results
            directLinks (bool): Whether to use direct links in API response. Default: False
                            Note: Automatically set to True when postprocess='links'
            clipping (str): Clipping method - 'FieldBorder' or 'Bbox'. Default: 'FieldBorder'
                        Note: Only applicable for COLORCOMPOSITION vegetation index
            buffer (int): Buffer distance in meters around geometry. Default: 0
            number_bins (int, optional): Number of histogram bins (1-255). When set,
                appends ``&numberOfBins=<value>`` to the API URL. Default: None
                (API uses its built-in default).
            legendType (str, optional): Legend type - one of 'Fixed', 'Dynamic', 'Common'.
                When set, appends ``&legendType=<value>`` to the API URL. Default: None
                (API uses 'Dynamic').
        """
        log = self.get_contextualized_logger("SETUP")
        log.info("Configuring FLM parameters...")

        # Vegetation index validation
        valid_indexes = {
            "NDVI",
            "EVI",
            "CVI",
            "CVIN",
            "GNDVI",
            "LAI",
            "NDWI",
            "NDMI",
            "S2REP",
            "COLORCOMPOSITION",
            "NDRE",
        }
        if vegetation_index not in valid_indexes:
            error_msg = f"Invalid vegetation_index '{vegetation_index}'. Must be one of {valid_indexes}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Map format validation
        valid_formats = {"png", "tiff.zip", "shp.zip", None}
        if map_format not in valid_formats:
            error_msg = f"Invalid map_format '{map_format}'. Must be one of {valid_formats}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Postprocess validation
        valid_postprocess = {"file", "stats", "links", "histogram"}
        if postprocess not in valid_postprocess:
            error_msg = f"Invalid postprocess '{postprocess}'. Must be one of {valid_postprocess}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # DirectLinks validation
        if not isinstance(directLinks, bool):
            error_msg = f"Invalid directLinks={directLinks}. Must be True or False."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Clipping validation
        valid_clipping = {"FieldBorder", "Bbox"}
        if clipping not in valid_clipping:
            error_msg = f"Invalid clipping '{clipping}'. Must be one of {valid_clipping}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Buffer validation
        if not isinstance(buffer, (int, float)):
            error_msg = f"Invalid buffer type '{type(buffer)}'. Must be a number."
            log.error(error_msg)
            raise ValueError(error_msg)

        if buffer < 0:
            error_msg = f"Invalid buffer={buffer}. Must be non-negative."
            log.error(error_msg)
            raise ValueError(error_msg)

        # number_bins validation (optional: int in [1, 255])
        if number_bins is not None:
            if isinstance(number_bins, bool) or not isinstance(number_bins, int):
                error_msg = f"Invalid number_bins type '{type(number_bins)}'. Must be int between 1 and 255."
                log.error(error_msg)
                raise ValueError(error_msg)
            if number_bins < 1 or number_bins > 255:
                error_msg = f"Invalid number_bins={number_bins}. Must be between 1 and 255."
                log.error(error_msg)
                raise ValueError(error_msg)

        # legendType validation (optional: one of 'Fixed', 'Dynamic', 'Common')
        if legendType is not None:
            valid_legend_types = {"Fixed", "Dynamic", "Common"}
            if legendType not in valid_legend_types:
                error_msg = f"Invalid legendType '{legendType}'. Must be one of {valid_legend_types}."
                log.error(error_msg)
                raise ValueError(error_msg)

        # ✅ Cross-validation: postprocess='links' requirements
        if postprocess == "links":
            # Auto-correct directLinks to True
            if not directLinks:
                log.warning("postprocess='links' requires directLinks=True. Auto-correcting directLinks to True.")
                print("⚠️ postprocess='links' requires directLinks=True. Auto-correcting...")
                directLinks = True

            # Validate map_format is None
            if map_format is not None:
                error_msg = f"postprocess='links' requires map_format=None, but got map_format='{map_format}'"
                log.error(error_msg)
                raise ValueError(error_msg)

            log.debug("Validated postprocess='links' requirements: directLinks=True, map_format=None")

        # ✅ Cross-validation: postprocess='file' requirements
        if postprocess == "file":
            if map_format is None:
                error_msg = "postprocess='file' requires map_format to be set ('png', 'tiff.zip', or 'shp.zip')"
                log.error(error_msg)
                raise ValueError(error_msg)

            if output_path is None:
                error_msg = "postprocess='file' requires output_path to be set"
                log.error(error_msg)
                raise ValueError(error_msg)

            log.debug(f"Validated postprocess='file' requirements: map_format={map_format}, output_path={output_path}")

        self.apply_cache_setting(use_cache)

        # ✅ Store parameters in self.flm_params
        self.flm_params = {
            "vegetation_index": vegetation_index,
            "map_format": map_format,
            "output_epsg": output_epsg,
            "postprocess": postprocess,
            "skip_existing": skip_existing,
            "output_path": output_path,
            "extract_stats": extract_stats,
            "partial_frequency": partial_frequency,
            "directLinks": directLinks,
            "clipping": clipping,
            "buffer": buffer,
            "number_bins": number_bins,
            "legendType": legendType,
        }

        id_col = self.get_mapped_column("id")
        self.cache_key_columns = [id_col]

        log.success("✅ FLM parameters configured successfully")
        log.debug(f"Parameters: {self.flm_params}")

        print("🗺️ FLM parameters configured:")
        for k, v in self.flm_params.items():
            print(f"   {k}: {v}")

    @requires_token
    @require_flm_params
    def get_flm_map(self, entity_data: dict, image_id, file_extension=None):
        """
        Process FLM map: either download file OR extract statistics OR get direct links
        Uses parameters from self.flm_params (set by setup_parameters)

        Args:
            entity_data (dict): Entity data including 'id' and 'geometry'
            image_id (str): Image ID to download
            file_extension (str, optional): File extension override

        Returns:
            requests.Response: Raw API response
        """
        params = self.flm_params
        file_extension = params["map_format"]
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API")

        # log.debug(f"Processing FLM map for entity {entity_id}, image {image_id}")

        # Step 1: Geometry validation
        if not self.has_entity_field(entity_data, "geometry"):
            error_msg = f"❌ entity_data for {entity_id} must include a 'geometry' field in WKT format."
            log.error(error_msg)
            raise ValueError(error_msg)

        geometry = self.get_entity_value(entity_data, "geometry")
        try:
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except ValueError as e:
            log.error(f"Invalid geometry for entity {entity_id}: {e}")
            raise

        # Step 2: API URL construction
        # ✅ Simple: if file_extension exists, add it to URL, otherwise just base URL
        if file_extension:
            url = f"{self.map_products_url}/maps/base-reference-map/{params['vegetation_index']}/image.{file_extension}"
            log.debug(f"Requesting file format: {file_extension}")
        else:
            url = f"{self.map_products_url}/maps/base-reference-map/{params['vegetation_index']}"
            log.debug("Requesting without file extension (stats or links)")

        # ✅ Build query parameters with dynamic directLinks + histogram flag
        include_histogram = params.get("postprocess") == "histogram"
        query_params = [
            f"histogram={'true' if include_histogram else 'false'}",
            f"directLinks={'true' if params['directLinks'] else 'false'}",
            f"$epsg={params['output_epsg']}",
        ]

        # ✅ Optional histogram bin count
        number_bins = params.get("number_bins")
        if number_bins is not None:
            query_params.append(f"numberOfBins={number_bins}")
            log.debug(f"Using numberOfBins: {number_bins}")

        # ✅ Optional legend type (Fixed / Dynamic / Common)
        legend_type = params.get("legendType")
        if legend_type is not None:
            query_params.append(f"legendType={legend_type}")
            log.debug(f"Using legendType: {legend_type}")

        # ✅ Add clipping ONLY for COLORCOMPOSITION
        if params["vegetation_index"] == "COLORCOMPOSITION":
            query_params.append(f"clipping={params['clipping']}")
            log.debug(f"Using clipping: {params['clipping']} (COLORCOMPOSITION only)")
            # Add buffer only if > 0
            if params["buffer"] > 0:
                query_params.append(f"buffer={params['buffer']}")
                log.debug(f"Using buffer: {params['buffer']}m")

        # --- Full URL ---
        full_url = f"{url}?{'&'.join(query_params)}"
        log.debug(f"API URL: {url}")

        # Build debug message for query params
        debug_params = f"directLinks={params['directLinks']}, buffer={params['buffer']}"
        if params["vegetation_index"] == "COLORCOMPOSITION":
            debug_params += f", clipping={params['clipping']}"
        log.debug(f"Query params: {debug_params}")

        # --- Payload ---
        payload = {
            "image": {"id": image_id},
            "seasonField": {"geometry": geometry, "crop": self.get_entity_value(entity_data, "crop")},
        }

        # Step 3: Request data
        log.debug(f"API URL: {full_url}")
        log.debug(f"Payload: {payload}")
        try:
            response = requests.post(
                full_url,
                headers={"Authorization": f"Bearer {self.bearer_token}", "Accept": "*/*"},
                json=payload,  # type: ignore[arg-type]
                timeout=60,
            )
            response.raise_for_status()
            if hasattr(response, "json"):
                try:
                    log.debug(f"API response: {response.json()}")
                except ValueError:
                    log.debug(f"API response: binary content ({len(response.content)} bytes)")
            return response

        except requests.exceptions.Timeout as e:
            log.error(f"⏱️ Timeout requesting FLM for entity {entity_id}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ HTTP error for entity {entity_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            log.error(f"❌ Unexpected error for entity {entity_id}: {e}")
            raise

    def get_flm_map_safe(self, entity_data: dict, image_id) -> dict:
        """
        Safe wrapper around get_flm_map().
        Returns structured response with success flag, data or error.

        Args:
            entity_data (dict): Must contain a 'geometry' key in WKT format.
            image_id (str): Image ID

        Returns:
            dict: {
                "success": bool,
                "data": dict | None,
                "error": str | None,
                "seasonfield_id": str
            }
        """
        entity_id = self.get_entity_value(entity_data, "id", "unknown")
        log = self.get_contextualized_logger("API_SAFE")

        try:
            response_json = self.get_flm_map(entity_data, image_id)
            log.debug(f"Successful safe API call for entity {entity_id}")
            return {"success": True, "data": response_json, "error": None, "seasonfield_id": entity_id}
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text if e.response is not None else str(e)
            error_msg = f"HTTP {status} - {text}"
            log.warning(f"HTTP error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}
        except Exception as e:
            error_msg = str(e)
            log.warning(f"Error for entity {entity_id}: {error_msg}")
            return {"success": False, "data": None, "error": error_msg, "seasonfield_id": entity_id}

    @require_flm_params
    def format_flm_map_stats_json(self, response_flm_map, range=False):
        """
        Extract legend statistics and metadata from Map Reference JSON.
        Converts raw response to JSON at the beginning.

        Args:
            response_flm_map (dict or Response): API response JSON with legend, mapSize, bBox, etc.
            range (bool): If True, also return ranges DataFrame. Default is False.

        Returns:
            pd.DataFrame or tuple:
                - If range=False: stats_df only
                - If range=True: (stats_df, ranges_df)
        """
        log = self.get_contextualized_logger("FORMAT")

        # Convert response to JSON
        if hasattr(response_flm_map, "json"):
            try:
                response_flm_map = response_flm_map.json()
                log.debug("Parsed JSON from response object")
            except ValueError as e:
                log.error(f"Failed to parse JSON response: {e}")
                print(f"❌ Failed to parse JSON response: {e}")
                return pd.DataFrame() if not range else (pd.DataFrame(), pd.DataFrame())

        # Extract data
        legend = response_flm_map.get("legend", {})
        stat = legend.get("stat", {})
        season_field = response_flm_map.get("seasonField", {})

        entity_id = season_field.get("id", "unknown")
        log.debug(f"Extracting stats for entity {entity_id}")

        # Stats row
        stats_row = {
            "stat_max": stat.get("max"),
            "stat_mean": stat.get("mean"),
            "stat_min": stat.get("min"),
        }

        stats_df = pd.DataFrame([stats_row])
        log.debug(f"Created stats DataFrame for entity {entity_id}")

        if not range:
            return stats_df

        # Process ranges if requested
        log.debug(f"Extracting ranges for entity {entity_id}")
        ranges_rows = []
        for range_entry in legend.get("ranges", []):
            ranges_rows.append(
                {
                    "season_field_id": entity_id,
                    "min_value": range_entry.get("minValue"),
                    "max_value": range_entry.get("maxValue"),
                    "num_pixels": range_entry.get("numberOfPixels"),
                }
            )

        ranges_df = pd.DataFrame(ranges_rows)
        log.debug(f"Created ranges DataFrame with {len(ranges_df)} rows for entity {entity_id}")

        return stats_df, ranges_df

    @require_flm_params
    def format_flm_map_histogram_json(self, response_flm_map):
        """
        Extract histogram stats and per-range values from Map Reference JSON.

        Produces a **single-row** DataFrame per entity:
            - global stats: ``stat_min``, ``stat_max``, ``stat_mean``
            - per-bucket columns (1-indexed): ``value_min_{i}``,
              ``value_max_{i}``, ``num_pixels_{i}``, ``area_{i}``

        Args:
            response_flm_map (dict or Response): API response carrying a
                ``histogram`` block (requires ``histogram=true`` on the query,
                which is set automatically when postprocess='histogram').

        Returns:
            pd.DataFrame: single-row wide DataFrame.
        """
        log = self.get_contextualized_logger("FORMAT_HISTOGRAM")

        # Convert response to JSON if needed
        if hasattr(response_flm_map, "json"):
            try:
                response_flm_map = response_flm_map.json()
                log.debug("Parsed JSON from response object")
            except ValueError as e:
                log.error(f"Failed to parse JSON response: {e}")
                print(f"❌ Failed to parse JSON response: {e}")
                return pd.DataFrame()

        histogram = response_flm_map.get("histogram") or {}
        season_field = response_flm_map.get("seasonField") or {}
        entity_id = season_field.get("id", "unknown")

        row = {
            "stat_min": histogram.get("min"),
            "stat_max": histogram.get("max"),
            "stat_mean": histogram.get("mean"),
        }

        items = histogram.get("items") or []
        log.debug(f"Extracting histogram ({len(items)} bucket(s)) for entity {entity_id}")

        for i, item in enumerate(items, start=1):
            row[f"value_min_{i}"] = item.get("valueMin")
            row[f"value_max_{i}"] = item.get("valueMax")
            row[f"num_pixels_{i}"] = item.get("numberOfPixel")
            row[f"area_{i}"] = item.get("area")

        if not items:
            log.warning(f"No histogram items for entity {entity_id}; returning stats-only row")

        df = pd.DataFrame([row])
        log.debug(f"Created histogram DataFrame ({df.shape[1]} columns) for entity {entity_id}")
        return df

    @require_flm_params
    def format_flm_map_file(self, entity_data: dict, image_id: str, response, output_path=None):
        """
        Save FLM map file response (PNG, TIFF.ZIP, or SHP.ZIP) to disk.
        Handles extraction of zipped files and proper naming.

        Args:
            entity_data (dict): Entity data including 'id' and optional 'name'
            image_id (str): Image ID for filename
            response (requests.Response): Raw response from get_flm_map
            output_path (str, optional): Directory to save files. Uses self.flm_params['output_path'] if not provided.

        Returns:
            dict: Status and file path information
        """
        params = self.flm_params
        entity_id = self.get_entity_value(entity_data, "id")
        entity_name = entity_data.get("name")  # 'name' is not a mapped field
        log = self.get_contextualized_logger("FILE_SAVE")

        log.info(f"Saving FLM map file for entity {entity_id}, image {image_id}")

        try:
            # Determine output directory
            save_path = output_path or params.get("output_path", self.output_path)

            if not save_path:
                error_msg = "No output path specified for file saving"
                log.error(error_msg)
                raise ValueError(error_msg)

            log.debug(f"Output path: {save_path}")

            # Get map format from params
            map_format = params.get("map_format", "png")
            log.debug(f"Map format: {map_format}")

            # Construct base filename
            if entity_name:
                output_filename = f"{entity_name}_{entity_id}_{image_id}_{params['vegetation_index']}"
            else:
                output_filename = f"{entity_id}_{image_id}_{params['vegetation_index']}"

            # Sanitize filename - remove invalid characters for Windows
            invalid_chars = '<>:"|?*'
            for char in invalid_chars:
                output_filename = output_filename.replace(char, "_")
            output_filename = output_filename.replace("/", "_").replace("\\", "_")

            log.debug(f"Sanitized filename: {output_filename}")

            saved_files = []

            # Handle different formats
            if map_format == "png":
                log.debug("Processing PNG format")
                full_path = self.save_map_file(save_path, f"{output_filename}.png", response.content)
                saved_files.append(full_path)
                log.debug(f"Saved PNG: {full_path}")

            elif map_format == "tiff.zip":
                log.debug("Processing TIFF.ZIP format")
                with ZipFile(BytesIO(response.content)) as zip_file:
                    for file_info in zip_file.namelist():
                        if file_info.endswith(".tif") or file_info.endswith(".tiff"):
                            full_path = self.save_map_file(
                                save_path, f"{output_filename}.tif", zip_file.read(file_info)
                            )
                            saved_files.append(full_path)
                            log.debug(f"Extracted TIFF: {full_path}")
                            break

                if not saved_files:
                    error_msg = "No .tif file found in the ZIP archive"
                    log.error(error_msg)
                    raise ValueError(error_msg)

            elif map_format == "shp.zip":
                log.debug("Processing SHP.ZIP format")
                with ZipFile(BytesIO(response.content)) as zip_file:
                    for file_info in zip_file.namelist():
                        _, ext = os.path.splitext(file_info)
                        if ext:
                            new_filename = f"{output_filename}{ext}"
                            output_file_path = self.save_map_file(save_path, new_filename, zip_file.read(file_info))
                            saved_files.append(output_file_path)
                            log.debug(f"Extracted shapefile component: {output_file_path}")

                if not saved_files:
                    error_msg = "No files found in the shapefile ZIP archive"
                    log.error(error_msg)
                    raise ValueError(error_msg)

            else:
                error_msg = f"Unsupported map_format: {map_format}"
                log.error(error_msg)
                raise ValueError(error_msg)

            log.success(
                f"✅ Saved {len(saved_files)} file(s) for entity {entity_id}, total size: {len(response.content)} bytes"
            )

            return {
                "status": "downloaded",
                "entity_id": entity_id,
                "image_id": image_id,
                "map_format": map_format,
                "saved_files": saved_files,
                "file_count": len(saved_files),
                "total_size_bytes": len(response.content),
            }

        except Exception as e:
            log.error(f"❌ Failed to save file for entity {entity_id}: {e}")
            return {
                "status": "error",
                "error_message": f"Failed to save file: {str(e)}",
                "entity_id": entity_id,
                "image_id": image_id,
                "map_format": params.get("map_format", "unknown"),
            }

    @require_flm_params
    def format_flm_links_json(self, response_flm_map):
        """
        Extract direct links and metadata from FLM API response.
        Converts raw response to JSON at the beginning.

        Args:
            response_flm_map (dict or Response): API response JSON with _links, worldFile, etc.

        Returns:
            pd.DataFrame: DataFrame containing links and metadata (one row per entity)
        """
        log = self.get_contextualized_logger("FORMAT_LINKS")

        # Convert response to JSON
        if hasattr(response_flm_map, "json"):
            try:
                response_flm_map = response_flm_map.json()
                log.debug("Parsed JSON from response object")
            except ValueError as e:
                log.error(f"Failed to parse JSON response: {e}")
                print(f"❌ Failed to parse JSON response: {e}")
                return pd.DataFrame()

        # Extract data
        links = response_flm_map.get("_links", {})
        world_file = response_flm_map.get("worldFile", {})
        map_size = response_flm_map.get("mapSize", {})
        bbox = response_flm_map.get("bBox", {})
        season_field = response_flm_map.get("seasonField", {})

        entity_id = season_field.get("id", "unknown")
        log.debug(f"Extracting links for entity {entity_id}")

        # Build row with links and metadata
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
        log.debug(f"Created links DataFrame for entity {entity_id}")
        log.debug(f"Links found: {list(links.keys())}")

        return links_df

    @requires_token
    @require_flm_params
    @cache_single_entity("flm_params")
    def process_single_entity_flm(self, row, params=None):
        """
        Process FLM extraction for a single entity with retry logic.
        Supports four postprocess modes: 'stats', 'links', 'file', or 'histogram'

        Args:
            row (dict): Entity data containing id, geometry, and image_id
            params (dict, optional): FLM parameters

        Returns:
            dict: {"data": DataFrame or None, "error": dict or None}
        """
        if params is None:
            params = self.flm_params

        entity_id = self.get_entity_value(row, "id", "unknown")
        log = self.get_contextualized_logger("SINGLE_ENTITY")
        log.debug(f"Processing entity {entity_id}")

        # Step 1: validate inputs
        try:
            geometry = self.get_entity_value(row, "geometry")
            geometry = validate_wkt(geometry)
            log.debug(f"Geometry validated for entity {entity_id}")
        except Exception as e:
            log.error(f"Geometry validation failed for entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

        # Step 2: validate image id — via get_entity_value so column_mapping applies
        image_id = self.get_entity_value(row, "image_id")

        if not image_id:
            log.error(f"Missing image_id for entity {entity_id}")
            return {"data": None, "error": {"message": "Missing image_id", "entity_id": entity_id}}

        # Validate image_id format (should have between 1 and 3 (for recalibration) | separators)
        separator_count = image_id.count("|")
        if separator_count < 1 or separator_count > 3:
            log.error(f"Invalid image_id format for entity {entity_id}: {image_id}")
            return {"data": None, "error": {"message": f"Invalid image_id format: {image_id}", "entity_id": entity_id}}

        # Optional: split and validate parts are not empty
        parts = image_id.split("|")
        if not all(part.strip() for part in parts):
            log.error(f"Invalid image_id format (empty parts) for entity {entity_id}: {image_id}")
            return {
                "data": None,
                "error": {"message": f"Invalid image_id format (empty parts): {image_id}", "entity_id": entity_id},
            }

        log.debug(f"Image ID validated for entity {entity_id}: {image_id}")

        # Step 3: define API call wrapper
        def _call_api():
            return self.get_flm_map(row, image_id)

        # Step 4: call API with retry
        try:
            # log.info(f"Calling FLM API for entity {entity_id} (with retry logic)")
            raw_response = retry_with_backoff_no_retry_on_400(_call_api, max_retries=5, base_delay=1.0, max_delay=60.0)

            # ✅ Step 5: Process based on postprocess type
            postprocess_type = params.get("postprocess", "stats")
            log.debug(f"Processing with postprocess type: {postprocess_type}")

            if postprocess_type == "links":
                # Extract direct links
                log.debug(f"Extracting links for entity {entity_id}")
                flm_df = self.format_flm_links_json(raw_response)

                if flm_df is None or flm_df.empty:
                    log.warning(f"No links found for entity {entity_id}")
                    return {"data": None, "error": {"message": "No links found", "entity_id": entity_id}}

            elif postprocess_type == "stats":
                # Extract statistics
                log.debug(f"Extracting stats for entity {entity_id}")
                flm_df = self.format_flm_map_stats_json(raw_response)

                if flm_df is None or flm_df.empty:
                    log.warning(f"No map stats found for entity {entity_id}")
                    return {"data": None, "error": {"message": "No map stats found", "entity_id": entity_id}}

            elif postprocess_type == "histogram":
                # Extract histogram (global stats + per-bucket min/max/area/pixels)
                log.debug(f"Extracting histogram for entity {entity_id}")
                flm_df = self.format_flm_map_histogram_json(raw_response)

                if flm_df is None or flm_df.empty:
                    log.warning(f"No histogram found for entity {entity_id}")
                    return {"data": None, "error": {"message": "No histogram found", "entity_id": entity_id}}

            elif postprocess_type == "file":
                # Save file to disk
                log.debug(f"Saving file for entity {entity_id}")
                file_result = self.format_flm_map_file(
                    entity_data=row, image_id=image_id, response=raw_response, output_path=params.get("output_path")
                )

                # Check if file save was successful
                if file_result.get("status") == "error":
                    log.error(f"File save failed for entity {entity_id}: {file_result.get('error_message')}")
                    return {
                        "data": None,
                        "error": {
                            "message": file_result.get("error_message", "File save failed"),
                            "entity_id": entity_id,
                        },
                    }

                # Convert file result to DataFrame for consistency
                flm_df = pd.DataFrame(
                    [
                        {
                            "status": file_result.get("status"),
                            "map_format": file_result.get("map_format"),
                            "file_count": file_result.get("file_count"),
                            "total_size_bytes": file_result.get("total_size_bytes"),
                            "saved_files": str(
                                file_result.get("saved_files", [])
                            ),  # Convert list to string for CSV compatibility
                        }
                    ]
                )

                log.success(f"✅ File saved for entity {entity_id}: {file_result.get('file_count')} file(s)")

            else:
                error_msg = f"Unknown postprocess type: {postprocess_type}"
                log.error(error_msg)
                return {"data": None, "error": {"message": error_msg, "entity_id": entity_id}}

            # Step 6: Package final response
            log.success(f"✅ Successfully processed entity {entity_id} with postprocess={postprocess_type}")
            return {"data": normalize_with_metadata(row, flm_df), "error": None}

        except Exception as e:
            log.error(f"❌ Failed to process entity {entity_id}: {e}")
            return {"data": None, "error": {"message": str(e), "entity_id": entity_id}}

    @require_flm_params
    def process_entity_flm_bulk_parallel(
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
        prefix="flm",
        generate_report=False,
        report_options=None,
        use_cache=None,
    ):
        """
        Bulk processing of entity FLM maps using threads + progress bar,
        with optional fail-safe retry and filter capabilities.

        Args:
            entity_list (pd.DataFrame): List of entities to process, must contain 'id', 'geometry', and 'image_id'.
            params (dict, optional): FLM parameters override.
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
            skip_export (bool, optional): If True, skip final export (useful when chaining extractions). Default: False.
            prefix (str, optional): Prefix for output filenames and failed IDs files. Default: "flm".
            use_cache (bool, optional): Enable/disable caching for this call.

        Returns:
            dict: Contains results DataFrame, errors list, and summary statistics.
        """
        if use_cache is not None and use_cache:
            return self._bulk_with_cache(
                bulk_method=self._process_entity_flm_bulk_parallel_inner,
                entity_list=entity_list,
                params=self.flm_params,
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
        return self._process_entity_flm_bulk_parallel_inner(
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
    @require_flm_params
    def _process_entity_flm_bulk_parallel_inner(
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
        prefix="flm",
        generate_report=False,
        report_options=None,
    ):
        """
        Inner bulk processing of entity FLM maps using threads + progress bar,
        with optional fail-safe retry and filter capabilities.
        """
        params = params_kw
        log = self.get_contextualized_logger("BULK_EXTRACTION")
        log.info("=" * 60)
        log.info(f"Starting bulk FLM extraction: {prefix}")
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
        print(f"📄 Processing {prefix.upper()} for {len(filtered_entity_list)} entities in parallel...")
        if skip_count > 0:
            print(f"   ({skip_count} entities skipped by filter)")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.process_single_entity_flm, row, params): self.get_entity_value(row, "id")
                for _, row in filtered_entity_list.iterrows()
            }

            with tqdm(total=len(future_to_id), desc=f"🗺️ Processing {prefix.upper()}", unit="entity") as pbar:
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
                        log.info(f"💾 Partial results saved: {len(buffer_rows)} rows")
                        buffer_rows.clear()
                        buffer_errors.clear()

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        log.info("=" * 60)
        log.success(f"⏱️ Total processing time: {elapsed_time:.2f} seconds")
        log.success(f"✅ Successful calculations: {successful_calculations}/{total_calculations}")
        log.info(f"❌ Failed calculations: {total_calculations - successful_calculations}/{total_calculations}")
        log.info("=" * 60)

        print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
        print(f"✅ Successful calculations: {successful_calculations}/{total_calculations}")

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
                "parameters": getattr(self, "flm_params", {}),
                "entity_df": entity_list,
            },
        )

        log.info(f"Finalization status: {finalization_status}")
        log.success("🎉 Bulk FLM extraction completed!")

        return {
            "results_df": results_df,
            "global_errors": global_errors,
            "total_entities": len(entity_list),
            "total_calculations": total_calculations,
            "successful_calculations": successful_calculations,
            "failed_calculations": total_calculations - successful_calculations,
            "failed_ids": failed_ids,
        }
