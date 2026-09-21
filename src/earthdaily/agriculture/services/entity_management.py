"""
Entity Management Module for EarthDaily Agro Platform
Handles Farm, Field, and Seasonfield CRUD operations via Master Data Management API v6

Hierarchy: Grower (user) -> Farm -> Field -> Seasonfield
"""

# Standard Library
import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, NoReturn, Optional

import pandas as pd

# Third-Party Libraries
import requests
from tqdm import tqdm

# Geospatial (optional, for SHP support)
try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls
from earthdaily.agriculture.core.base_extractor import BaseExtractor, requires_token
from earthdaily.agriculture.core.identity import EDAuthenticator
from earthdaily.agriculture.services.user_management import UserManager


def get_seasonfield_list(
    env, bearer_token, start_date=None, end_date=None, crop_id=None, sowing_date_gte=None, sowing_date_lte=None
):
    """
    Retrieve seasonfield list with optional filters.

    Args:
        env (str): Environment ('prod', 'preprod', etc.)
        bearer_token (str): Authentication token
        start_date (str, optional): Legacy parameter - alias for sowing_date_gte
        end_date (str, optional): Legacy parameter - alias for sowing_date_lte
        crop_id (str, optional): Filter by crop ID (e.g., 'CORN', 'SOYBEANS', 'WHEAT')
        sowing_date_gte (str, optional): Filter sowingDate >= this date (YYYY-MM-DD)
        sowing_date_lte (str, optional): Filter sowingDate <= this date (YYYY-MM-DD)

    Returns:
        pd.DataFrame: DataFrame with seasonfield data

    Example:
        # Get all corn fields sown after 2025-01-01
        sfd_list = get_seasonfield_list(env, token, sowing_date_gte="2025-01-01", crop_id="CORN")

        # Get soybeans fields between two dates
        sfd_list = get_seasonfield_list(env, token,
                                         sowing_date_gte="2025-01-01",
                                         sowing_date_lte="2025-06-30",
                                         crop_id="SOYBEANS")
    """
    url = agro_urls["eda_data_management_url_fields"][env]
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + bearer_token}

    # Base parameters (same as original)
    params = "/seasonfields?$limit=none&$fields=id,name,geometry,crop,field.farm.grower.companyName,field.farm.grower.firstname,sowingDate"

    # Handle legacy parameters (start_date/end_date as aliases for backward compatibility)
    if start_date and not sowing_date_gte:
        sowing_date_gte = start_date
    if end_date and not sowing_date_lte:
        sowing_date_lte = end_date

    # Add sowing date filters (NEW)
    if sowing_date_gte and sowing_date_lte:
        # Between two dates
        params += f"&sowingDate=$between:{sowing_date_gte}|{sowing_date_lte}"
    elif sowing_date_gte:
        # Greater than or equal
        params += f"&sowingDate=$gte:{sowing_date_gte}"
    elif sowing_date_lte:
        # Less than or equal
        params += f"&sowingDate=$lte:{sowing_date_lte}"

    # Add crop filter (NEW)
    if crop_id:
        params += f"&crop.id={crop_id.upper()}"

    # Make request (same as original)
    response = requests.get(url + params, headers=headers)
    response.raise_for_status()
    result = pd.json_normalize(response.json())
    return result


def get_season_fields(
    base_url: str,
    token: str,
    ids: List[str],
    batch_size: int = 5000,
    start_date: str = None,
    end_date: str = None,
    crop_id: str = None,
):
    """
    Retrieve seasonfields metadata for a given list of public IDs using batch processing.

    Args:
        base_url (str): Base URL for the API
        token (str): Bearer token for authentication
        ids (List[str]): List of public Seasonfield IDs to fetch
        batch_size (int, optional): Maximum number of IDs per request (default is 5000)
        start_date (str, optional): Start date for sowingDate filter (format: YYYY-MM-DD)
        end_date (str, optional): End date for sowingDate filter (format: YYYY-MM-DD)
        crop_id (str, optional): Crop ID to filter by

    Returns:
        pd.DataFrame: A DataFrame containing enriched seasonfield data
    """
    try:
        print(f"🌾 Fetching seasonfields for {len(ids)} IDs (batch size = {batch_size})")

        endpoint = f"{base_url}/master-data-management/V6/seasonfields"
        all_results = {}

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            sfd_id_list = "|".join(batch_ids)

            # Construction des filtres
            filters = [{"id": f"$in:{sfd_id_list}"}]

            # Ajouter les filtres de date selon le cas
            if start_date and end_date:
                filters.append({"sowingDate": f"$between:{start_date}|{end_date}"})
            elif start_date:
                filters.append({"sowingDate": f"$gte:{start_date}"})
            elif end_date:
                filters.append({"sowingDate": f"$lte:{end_date}"})

            # Ajouter le filtre sur crop.Id si fourni
            if crop_id:
                filters.append({"crop.Id": crop_id})

            payload = {
                "query": {
                    "filters": filters,
                    "limit": -1,
                },
                "fields": "id,crop.Id,geometry,centroid,sowingDate,field.farm.name,field.farm.grower.id,name",
            }

            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}

            response = requests.request("SEARCH", endpoint, json=payload, headers=headers)  # type: ignore[arg-type]
            response.raise_for_status()
            data = response.json()

            print(f"✅ Batch {i // batch_size + 1}: retrieved {len(data)} items")

            for item in data:
                crop = (item.get("crop", {}).get("id") or "").upper()
                public_id = item.get("id")
                grower_id = item.get("field", {}).get("farm", {}).get("grower", {}).get("id", {}) or ""
                farm_name = item.get("field", {}).get("farm", {}).get("name", {}) or ""

                # Handle sowing date safely
                sowing_date_raw = item.get("sowingDate", "")
                sowing_date = sowing_date_raw.split("T")[0] if sowing_date_raw else ""
                year = sowing_date.split("-")[0] if sowing_date else ""

                # Get field name, fallback to public_id if not available
                field_name = item.get("name", f"Field_{public_id}")

                all_results[public_id] = {
                    "id": public_id,
                    "name": field_name,
                    "public_id": public_id,
                    "farm_name": farm_name,
                    "crop.id": crop,
                    "field.farm.grower.companyName": farm_name,
                    "crop": crop,
                    "year": year,
                    "geometry": item.get("geometry"),
                    "centroid": item.get("centroid"),
                    "sowing_date": sowing_date,
                    "grower_id": grower_id,
                }

        print(f"✅ Total retrieved seasonfields: {len(all_results)}")

        if all_results:
            all_results_df = pd.DataFrame(all_results).T.reset_index(drop=True)
            return all_results_df
        else:
            return pd.DataFrame()

    except requests.RequestException as e:
        print(f"❌ HTTP error during seasonfield fetch: {e}")
        raise RuntimeError("Failed to retrieve seasonfields due to HTTP error.") from e
    except Exception as e:
        print(f"❌ Unexpected error while retrieving seasonfields: {e}")
        raise RuntimeError("Failed to retrieve seasonfields due to unexpected error.") from e


class EntityManager(BaseExtractor):
    """
    Manages Farm, Field, and Seasonfield entities via EarthDaily MDM API.

    Provides full CRUD operations for the agricultural entity hierarchy
    (Farm > Field > Seasonfield). Supports bulk entity loading, batch processing,
    and integration with UserManager for Grower ID resolution.

    Documentation: https://docs.earthdaily.com/agro/library/Api_reference/
    Notebook: https://github.com/earthdaily/Examples-and-showcases/blob/main/agriculture/EDAgriculture_entity_management.ipynb

    Supported operations:
        - Farm: create, get, update, delete
        - Field: create, get, update, delete
        - Seasonfield: create, get, update, delete
        - Batch loading: load_seasonfields(), load_seasonfields_batch()

    Entity fields:
        id, geometry, crop, sowing_date, name, farm_name (varies by operation)
    """

    def __init__(self, bearer_token, token_expiration, config, workflow_ref=None, user_manager: UserManager = None):
        """
        Initialize EntityManager.

        Args:
            bearer_token (str): API bearer token
            token_expiration (datetime): Token expiration time
            config (dict): Configuration dictionary
            workflow_ref: Optional workflow manager reference
            user_manager (UserManager): UserManager instance for Grower ID resolution
        """
        super().__init__(bearer_token, token_expiration, config, workflow_ref)

        # Composition: use UserManager for grower lookups
        self.user_manager = user_manager

        # API base URL
        self.mdm_url = agro_urls["eda_data_management_url_fields"][self.env]

        # Default batch size for bulk operations
        self.batch_size = 5000

        # CSV separator (auto-detected on load)
        self.csv_separator = ","

        # Farm cache: farm_name -> farm_id (to avoid duplicates)
        self._farm_cache: dict[str, str] = {}

        self.logger.info(f"🏗️ EntityManager initialized for env: {self.env}")
        if self.user_manager:
            self.logger.debug("UserManager attached for Grower resolution")

    def get_new_token(self):
        """Refresh token using EDAuthenticator."""
        self.logger.debug("Refreshing token for EntityManager")
        return EDAuthenticator.get_new_token(env=self.env)

    def _get_headers(self) -> Dict[str, str]:
        """Get standard API headers with Bearer token."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _handle_api_error(self, response: requests.Response, context: str) -> NoReturn:
        """Parse and raise meaningful API errors."""
        try:
            error_data = response.json()
            errors = error_data.get("errors", {}).get("body", {})
            error_messages = []
            for field, issues in errors.items():
                for issue in issues:
                    msg = issue.get("message", "")
                    error_messages.append(f"{field}: {msg}" if field else msg)
            detail = " | ".join(error_messages) if error_messages else response.text
        except json.JSONDecodeError:
            detail = response.text

        raise RuntimeError(f"{context} failed ({response.status_code}): {detail}")

    def clear_farm_cache(self):
        """Clear the farm name -> ID cache."""
        self._farm_cache = {}
        self.logger.debug("Farm cache cleared")

    # =========================================================================
    # FILE LOADING (CSV / Excel / SHP)
    # =========================================================================

    def load_input_file(self, file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Load input file with automatic format detection (CSV, Excel, Shapefile).

        Expected columns:
        - Grower: Grower ID (alphanumeric)
        - Farm: Farm name
        - Seasonfield: Field/Seasonfield name
        - Sowing: Sowing date (YYYY-MM-DD)
        - Crop: Crop code (e.g., CORN)
        - Geometry: WKT geometry (for CSV/Excel) or auto-extracted (for SHP)

        Args:
            file_path (str): Path to input file
            encoding (str): File encoding for CSV (default: utf-8)

        Returns:
            pd.DataFrame: Loaded and normalized data
        """
        log = self.get_contextualized_logger("LOAD_FILE")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        log.info(f"Loading file: {file_path} (format: {ext})")

        # Load based on format
        if ext == ".shp":
            df = self._load_shapefile(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = self._load_excel(file_path)
        elif ext == ".csv":
            df = self._load_csv(file_path, encoding)
        else:
            raise ValueError(f"❌ Unsupported file format: {ext}")

        # Normalize column names
        df = self._normalize_columns(df)

        # Validate required columns
        required = ["Grower", "Farm", "Seasonfield", "Sowing", "Crop", "Geometry"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing required columns: {missing}")

        log.success(f"✅ Loaded {len(df)} rows from {ext} file")
        print(f"✅ Loaded {len(df)} entities from file")

        return df

    def _load_csv(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load CSV with automatic separator detection."""
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=",;\t|")
                separator = dialect.delimiter
            except csv.Error:
                separator = ","

        self.csv_separator = separator
        print(f"📄 Detected CSV separator: '{separator}'")

        return pd.read_csv(file_path, sep=separator, encoding=encoding)

    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file."""
        print("📄 Loading Excel file")
        return pd.read_excel(file_path)

    def _load_shapefile(self, file_path: str) -> pd.DataFrame:
        """Load Shapefile and extract WKT geometry."""
        if not HAS_GEOPANDAS:
            raise ImportError("❌ geopandas required for Shapefile support. Install with: pip install geopandas")

        print("📄 Loading Shapefile")
        gdf = gpd.read_file(file_path)

        # Convert geometry to WKT
        gdf["Geometry"] = gdf["geometry"].apply(lambda g: g.wkt if g else "")

        # Drop original geometry column and convert to DataFrame
        df = pd.DataFrame(gdf.drop(columns=["geometry"]))

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        df.columns = df.columns.str.strip()

        # Column mapping (various possible names -> standard name)
        column_mapping = {
            # Grower
            "grower": "Grower",
            "grower_id": "Grower",
            "GrowerId": "Grower",
            # Farm
            "farm": "Farm",
            "farm_name": "Farm",
            "FarmName": "Farm",
            # Seasonfield / Field name
            "seasonfield": "Seasonfield",
            "field": "Seasonfield",
            "field_name": "Seasonfield",
            "FieldName": "Seasonfield",
            "parcel": "Seasonfield",
            "parcelle": "Seasonfield",
            # Sowing
            "sowing": "Sowing",
            "sowing_date": "Sowing",
            "SowingDate": "Sowing",
            "date_semis": "Sowing",
            "semis": "Sowing",
            # Crop
            "crop": "Crop",
            "crop_code": "Crop",
            "CropCode": "Crop",
            "culture": "Crop",
            # Geometry
            "geometry": "Geometry",
            "geom": "Geometry",
            "wkt": "Geometry",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        return df

    # =========================================================================
    # LOOKUP OPERATIONS
    # =========================================================================

    def get_grower_id(self, login: str) -> Optional[str]:
        """
        Get Grower ID by login using UserManager.

        Args:
            login (str): User login (username, not email)

        Returns:
            str: Grower ID (alphanumeric) or None if not found
        """
        log = self.get_contextualized_logger("GROWER")

        if not self.user_manager:
            log.error("UserManager not attached - cannot resolve Grower ID")
            raise RuntimeError(
                "UserManager required for Grower resolution. Pass user_manager to EntityManager constructor."
            )

        log.info(f"Resolving Grower ID for login: {login}")

        user_data = self.user_manager.get_user_by_login(login)

        if not user_data:
            log.warning(f"Grower not found: {login}")
            return None

        grower_id = user_data.get("id")
        log.success(f"✅ Grower ID resolved: {grower_id}")
        return grower_id

    def get_growers(self, fields: str = "id,userType.code,companyName,firstname,lastname,login") -> pd.DataFrame:
        """
        Get all GROWER users.

        Args:
            fields (str): Fields to retrieve

        Returns:
            pd.DataFrame: Growers data
        """
        if not self.user_manager:
            raise RuntimeError("UserManager required for Grower resolution.")

        return self.user_manager.get_users(user_type="GROWER", fields=fields)

    @requires_token
    def get_crops(self, fields: str = "id,cycleduration") -> pd.DataFrame:
        """
        Get all available crops.

        Args:
            fields (str): Fields to retrieve

        Returns:
            pd.DataFrame: Crops data with id and properties
        """
        log = self.get_contextualized_logger("GET_CROPS")

        url = f"{self.mdm_url}/crops"
        params = {"$limit": "none", "$fields": fields}

        log.info("Fetching crops")

        response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()

        log.info(f"Retrieved {len(df)} crops")
        return df

    # =========================================================================
    # FARM - CRUD
    # =========================================================================

    @requires_token
    def create_farm(self, grower_id: str, farm_name: str, use_cache: bool = True) -> str:
        """
        Create a Farm under a Grower. Uses cache to avoid duplicates.

        Args:
            grower_id (str): Grower user ID
            farm_name (str): Name of the farm
            use_cache (bool): Use internal cache to avoid duplicate creation

        Returns:
            str: Created or existing Farm ID
        """
        log = self.get_contextualized_logger("CREATE_FARM")

        # Check cache first
        cache_key = f"{grower_id}:{farm_name}"
        if use_cache and cache_key in self._farm_cache:
            farm_id = self._farm_cache[cache_key]
            log.debug(f"Farm '{farm_name}' found in cache: {farm_id}")
            return farm_id

        log.info(f"Creating farm '{farm_name}' for grower {grower_id}")

        url = f"{self.mdm_url}/farms"
        payload = {"name": farm_name, "grower": {"id": grower_id}}

        try:
            response = requests.post(url, headers=self._get_headers(), json=payload, timeout=60)  # type: ignore[arg-type]

            if response.status_code == 201:
                data = response.json()
                farm_id = data.get("id") or data.get("Id")
                self._farm_cache[cache_key] = farm_id
                log.success(f"✅ Farm created: {farm_id}")
                return farm_id

            # Handle "farm already exists" error
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    errors = error_data.get("errors", {}).get("body", {})
                    for field, issues in errors.items():
                        for issue in issues:
                            msg = issue.get("message", "")
                            match = re.search(r"Id\s*:\s*'?([a-zA-Z0-9]+)'?", msg)
                            if match and "already exists" in msg.lower():
                                farm_id = match.group(1)
                                self._farm_cache[cache_key] = farm_id
                                log.info(f"✅ Farm already exists: {farm_id}")
                                return farm_id
                except json.JSONDecodeError:
                    pass

            self._handle_api_error(response, "Create farm")

        except requests.exceptions.Timeout:
            log.error("⏱️ Timeout creating farm")
            raise

    @requires_token
    def get_farms(self, grower_id: str = None, fields: str = "id,name,grower.id") -> pd.DataFrame:
        """Get farms with optional grower filter."""
        log = self.get_contextualized_logger("GET_FARMS")

        url = f"{self.mdm_url}/farms"
        params = {"$limit": "none", "$fields": fields}

        if grower_id:
            params["grower.id"] = grower_id

        log.info(f"Fetching farms (grower_id={grower_id})")

        response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()

        log.info(f"Retrieved {len(df)} farms")
        return df

    @requires_token
    def get_farm_by_id(self, farm_id: str) -> Optional[Dict]:
        """Get single farm by ID."""
        url = f"{self.mdm_url}/farms/{farm_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                return None
            raise

    @requires_token
    def update_farm(self, farm_id: str, farm_name: str = None, grower_id: str = None) -> bool:
        """Update a Farm."""
        log = self.get_contextualized_logger("UPDATE_FARM")
        log.info(f"Updating farm {farm_id}")

        url = f"{self.mdm_url}/farms/{farm_id}"
        payload: dict[str, Any] = {}
        if farm_name:
            payload["name"] = farm_name
        if grower_id:
            payload["grower"] = {"id": grower_id}

        if not payload:
            log.warning("No fields to update")
            return True

        try:
            response = requests.patch(url, headers=self._get_headers(), json=payload, timeout=60)
            response.raise_for_status()
            log.success(f"✅ Farm updated: {farm_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ Failed to update farm: {e.response.text}")
            return False

    @requires_token
    def delete_farm(self, farm_id: str) -> bool:
        """Delete a Farm."""
        log = self.get_contextualized_logger("DELETE_FARM")
        log.info(f"Deleting farm {farm_id}")

        url = f"{self.mdm_url}/farms/{farm_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=60)
            response.raise_for_status()
            log.success(f"✅ Farm deleted: {farm_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                log.warning(f"Farm not found (already deleted?): {farm_id}")
                return True
            log.error(f"❌ Failed to delete farm: {e.response.text}")
            return False

    # =========================================================================
    # FIELD - CRUD
    # =========================================================================

    @requires_token
    def create_field(self, farm_id: str, field_name: str, geometry: str) -> str:
        """Create a Field under a Farm."""
        log = self.get_contextualized_logger("CREATE_FIELD")
        log.info(f"Creating field '{field_name}' for farm {farm_id}")

        url = f"{self.mdm_url}/fields"
        payload = {"name": field_name, "Geometry": geometry, "farm": {"id": farm_id}}

        try:
            response = requests.post(url, headers=self._get_headers(), json=payload, timeout=60)  # type: ignore[arg-type]

            if response.status_code == 201:
                data = response.json()
                field_id = data.get("id") or data.get("Id")
                log.success(f"✅ Field created: {field_id}")
                return field_id

            # Handle "field already exists" error
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    errors = error_data.get("errors", {}).get("body", {})
                    for field, issues in errors.items():
                        for issue in issues:
                            msg = issue.get("message", "")
                            match = re.search(r"Field Id\s*:\s*'?([a-zA-Z0-9]+)'?", msg)
                            if match:
                                field_id = match.group(1)
                                log.info(f"✅ Field already exists: {field_id}")
                                return field_id
                except json.JSONDecodeError:
                    pass

            self._handle_api_error(response, "Create field")

        except requests.exceptions.Timeout:
            log.error("⏱️ Timeout creating field")
            raise

    @requires_token
    def get_fields(self, farm_id: str = None, fields: str = "id,name,geometry,farm.id") -> pd.DataFrame:
        """Get fields with optional farm filter."""
        log = self.get_contextualized_logger("GET_FIELDS")

        url = f"{self.mdm_url}/fields"
        params = {"$limit": "none", "$fields": fields}

        if farm_id:
            params["farm.id"] = farm_id

        log.info(f"Fetching fields (farm_id={farm_id})")

        response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()

        log.info(f"Retrieved {len(df)} fields")
        return df

    @requires_token
    def get_field_by_id(self, field_id: str) -> Optional[Dict]:
        """Get single field by ID."""
        url = f"{self.mdm_url}/fields/{field_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                return None
            raise

    @requires_token
    def update_field(self, field_id: str, field_name: str = None, geometry: str = None, farm_id: str = None) -> bool:
        """Update a Field."""
        log = self.get_contextualized_logger("UPDATE_FIELD")
        log.info(f"Updating field {field_id}")

        url = f"{self.mdm_url}/fields/{field_id}"
        payload: dict[str, Any] = {}
        if field_name:
            payload["name"] = field_name
        if geometry:
            payload["Geometry"] = geometry
        if farm_id:
            payload["farm"] = {"id": farm_id}

        if not payload:
            log.warning("No fields to update")
            return True

        try:
            response = requests.patch(url, headers=self._get_headers(), json=payload, timeout=60)
            response.raise_for_status()
            log.success(f"✅ Field updated: {field_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ Failed to update field: {e.response.text}")
            return False

    @requires_token
    def delete_field(self, field_id: str) -> bool:
        """Delete a Field."""
        log = self.get_contextualized_logger("DELETE_FIELD")
        log.info(f"Deleting field {field_id}")

        url = f"{self.mdm_url}/fields/{field_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=60)
            response.raise_for_status()
            log.success(f"✅ Field deleted: {field_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                log.warning(f"Field not found (already deleted?): {field_id}")
                return True
            log.error(f"❌ Failed to delete field: {e.response.text}")
            return False

    # =========================================================================
    # SEASONFIELD - CRUD
    # =========================================================================

    @requires_token
    def create_seasonfield(
        self, field_id: str, geometry: str, sowing_date: str, crop_code: str = "CORN", name: str = None
    ) -> str:
        """Create a Seasonfield under a Field."""
        log = self.get_contextualized_logger("CREATE_SEASONFIELD")
        log.info(f"Creating seasonfield for field {field_id} (crop={crop_code}, sowing={sowing_date})")

        url = f"{self.mdm_url}/seasonfields"
        payload = {
            "geometry": geometry,
            "sowingDate": sowing_date,
            "crop": {"code": crop_code},
            "field": {"id": field_id},
        }

        if name:
            payload["name"] = name

        try:
            response = requests.post(url, headers=self._get_headers(), json=payload, timeout=60)  # type: ignore[arg-type]

            if response.status_code == 201:
                data = response.json()
                sf_id = data.get("id") or data.get("Id")
                log.success(f"✅ Seasonfield created: {sf_id}")
                return sf_id

            # Handle "seasonfield already exists" error
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    errors = error_data.get("errors", {}).get("body", {})

                    if "sowingDate" in errors:
                        for issue in errors["sowingDate"]:
                            msg = issue.get("message", "")
                            match = re.search(r"Id\s*:\s*([a-zA-Z0-9]+)", msg)
                            if match:
                                sf_id = match.group(1)
                                log.info(f"✅ Seasonfield already exists: {sf_id}")
                                return sf_id
                except json.JSONDecodeError:
                    pass

            self._handle_api_error(response, "Create seasonfield")

        except requests.exceptions.Timeout:
            log.error("⏱️ Timeout creating seasonfield")
            raise

    @requires_token
    def get_seasonfields(
        self,
        field_id: str = None,
        crop_code: str = None,
        sowing_date_gte: str = None,
        sowing_date_lte: str = None,
        farm_name: "str | list[str] | None" = None,
        external_ids: "dict | None" = None,
        fields: str = "id,name,geometry,sowingDate,crop.code,field.id,crop,field.farm.grower.companyName,field.farm.grower.firstname",
    ) -> pd.DataFrame:
        """
        Get seasonfields with optional filters.

        Args:
            field_id (str, optional): Filter by field ID
            crop_code (str, optional): Filter by crop code (e.g., 'CORN', 'SOYBEANS')
            sowing_date_gte (str, optional): Filter sowingDate >= this date (YYYY-MM-DD)
            sowing_date_lte (str, optional): Filter sowingDate <= this date (YYYY-MM-DD)
            farm_name (str | list[str], optional): Filter by farm name (Field.Farm.Name).
                Pass a single name (``"My Farm"``) or a list (``["Farm A", "Farm B"]``)
                to filter several farms in ONE request via the MDM ``$in:`` operator.
            external_ids (dict, optional): Filter by one or more externalIds systems. Maps
                an externalIds sub-key to a value or list of values, e.g.
                ``{"smbsC_ID": ["7073", "7074"]}`` -> ``externalIds.smbsC_ID=$in:7073|7074``.
                Keys may be given bare (``"smbsC_ID"``) or fully-qualified
                (``"externalIds.smbsC_ID"``). To get the externalIds back in the response,
                also include ``externalIds`` in ``fields``.
            fields (str): Comma-separated list of fields to retrieve. Include ``externalIds``
                to retrieve the nested external-id object (flattened to ``externalIds.*``
                columns on the way out).

        Returns:
            pd.DataFrame: DataFrame with seasonfield data (deduplicated on ``id``). When
                ``externalIds`` is requested its nested object is flattened to
                ``externalIds.<system>`` columns; if the native ``id`` field was not
                requested it is recovered from ``externalIds.id`` so the frame stays
                usable by the extractors.

        Example:
            mgr.get_seasonfields(farm_name=["Brazil Demo Farm", "France Demo Farm"])
            mgr.get_seasonfields(
                external_ids={"smbsC_ID": ["7073", "7074"]},
                fields="externalIds,geometry,sowingDate,customerExternalId",
            )
        """
        log = self.get_contextualized_logger("GET_SEASONFIELDS")

        url = f"{self.mdm_url}/seasonfields"
        params = {"$limit": "none", "$fields": fields}

        if field_id:
            params["field.id"] = field_id
        if crop_code:
            params["crop.code"] = crop_code.upper()
        if external_ids:
            # Each entry filters on one externalIds system. A single value is sent as a
            # bare value; multiple values use the MDM ``$in:`` operator joined on ``|``
            # (same convention as the farm_name filter above).
            for raw_key, raw_val in external_ids.items():
                if raw_key is None:
                    continue
                key = str(raw_key).strip()
                if not key:
                    continue
                param_key = key if key.startswith("externalIds.") else f"externalIds.{key}"
                if isinstance(raw_val, (list, tuple, set)):
                    vals = [str(v).strip() for v in raw_val if v is not None and str(v).strip()]
                    if len(vals) == 1:
                        params[param_key] = vals[0]
                    elif vals:
                        params[param_key] = "$in:" + "|".join(vals)
                elif raw_val is not None and str(raw_val).strip():
                    params[param_key] = str(raw_val).strip()
        if farm_name:
            # A single name is sent as a bare value (unchanged); a list is sent as a
            # single-request multi-value filter via the MDM ``$in:`` operator, joined
            # on ``|`` (same delimiter as ``$between:`` above). Verified against MDM v6
            # — it returns the union of the named farms in one call. ``requests``
            # URL-encodes spaces/special chars in the value via the ``params`` dict.
            if isinstance(farm_name, (list, tuple, set)):
                names = [str(n).strip() for n in farm_name if n is not None and str(n).strip()]
                if len(names) == 1:
                    params["field.farm.name"] = names[0]
                elif names:
                    params["field.farm.name"] = "$in:" + "|".join(names)
            else:
                params["field.farm.name"] = farm_name

        # Sowing date filters
        if sowing_date_gte and sowing_date_lte:
            params["sowingDate"] = f"$between:{sowing_date_gte}|{sowing_date_lte}"
        elif sowing_date_gte:
            params["sowingDate"] = f"$gte:{sowing_date_gte}"
        elif sowing_date_lte:
            params["sowingDate"] = f"$lte:{sowing_date_lte}"

        log.info(
            f"Fetching seasonfields (field_id={field_id}, crop={crop_code}, "
            f"sowing_gte={sowing_date_gte}, sowing_lte={sowing_date_lte}, farm_name={farm_name})"
        )

        response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()

        # externalIds comes back as a nested object; pd.json_normalize flattens it to
        # externalIds.<system> columns (e.g. externalIds.id, externalIds.smbsC_ID). When
        # the caller's $fields omits the native ``id`` but requests externalIds, the
        # native seasonfield id still arrives under externalIds.id — surface it as the
        # canonical ``id`` so dedup below and the downstream extractors work without a
        # manual column_mapping.
        if not df.empty and "id" not in df.columns and "externalIds.id" in df.columns:
            df.insert(0, "id", df["externalIds.id"])
            log.info("Recovered canonical 'id' from externalIds.id (id not in $fields)")

        # Dedup on seasonfield id so a field can never be returned twice (defensive —
        # a multi-farm $in: filter should not produce duplicates, but a custom $fields
        # set or future API behaviour might).
        if not df.empty and "id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset="id").reset_index(drop=True)
            if len(df) < before:
                log.info(f"Dropped {before - len(df)} duplicate seasonfield(s) on id")

        log.info(f"Retrieved {len(df)} seasonfields")
        return df

    def load_seasonfields(
        self,
        sowing_date_gte=None,
        sowing_date_lte=None,
        crop_id=None,
        start_date=None,
        end_date=None,
        farm_name=None,
        external_ids=None,
        fields=None,
    ) -> pd.DataFrame:
        """
        Load seasonfields from EarthDaily platform with optional filters.

        Args:
            sowing_date_gte (str, optional): Filter sowingDate >= this date (YYYY-MM-DD)
            sowing_date_lte (str, optional): Filter sowingDate <= this date (YYYY-MM-DD)
            crop_id (str, optional): Filter by crop ID (e.g., 'CORN', 'SOYBEANS', 'WHEAT')
            start_date (str, optional): Legacy alias for sowing_date_gte
            end_date (str, optional): Legacy alias for sowing_date_lte
            farm_name (str | list[str], optional): Filter by farm name (Field.Farm.Name).
                A single name or a list of names (filtered in one request via ``$in:``).
            external_ids (dict, optional): Filter by externalIds systems, e.g.
                ``{"smbsC_ID": ["7073", "7074"]}``. See get_seasonfields() for details.
            fields (str, optional): Comma-separated list of fields to retrieve.
                If None, uses get_seasonfields default. Include ``externalIds`` to get the
                external-id object back (flattened to ``externalIds.*`` columns).

        Returns:
            pd.DataFrame: Loaded seasonfields

        Example:
            entity_manager.load_seasonfields(sowing_date_gte="2025-01-01", crop_id="CORN", farm_name="My Farm")
            entity_manager.load_seasonfields(farm_name=["Farm A", "Farm B"])  # both farms, one request
            entity_manager.load_seasonfields(
                external_ids={"smbsC_ID": ["7073", "7074"]},
                fields="externalIds,geometry,sowingDate,customerExternalId",
            )
        """
        # Handle legacy parameters
        if start_date and not sowing_date_gte:
            sowing_date_gte = start_date
        if end_date and not sowing_date_lte:
            sowing_date_lte = end_date

        get_kwargs = dict(
            crop_code=crop_id,
            sowing_date_gte=sowing_date_gte,
            sowing_date_lte=sowing_date_lte,
            farm_name=farm_name,
            external_ids=external_ids,
        )
        if fields is not None:
            get_kwargs["fields"] = fields

        sfd_list = self.get_seasonfields(**get_kwargs)

        # Log filters applied
        filters_applied = []
        if sowing_date_gte:
            filters_applied.append(f"sowingDate >= {sowing_date_gte}")
        if sowing_date_lte:
            filters_applied.append(f"sowingDate <= {sowing_date_lte}")
        if crop_id:
            filters_applied.append(f"crop = {crop_id.upper()}")
        if farm_name:
            if isinstance(farm_name, (list, tuple, set)):
                filters_applied.append(f"farm in [{', '.join(str(n) for n in farm_name)}]")
            else:
                filters_applied.append(f"farm = {farm_name}")
        if external_ids:
            filters_applied.append("externalIds " + ", ".join(f"{k}={v}" for k, v in external_ids.items()))

        if filters_applied:
            print(f"🔍 Filters applied: {', '.join(filters_applied)}")

        print(f"✅ Loaded {len(sfd_list)} seasonfields")
        return sfd_list

    def load_seasonfields_batch(
        self, start_date=None, end_date=None, crop_id=None, ids=None, batch_size=5000
    ) -> pd.DataFrame:
        """
        Load seasonfields using batch processing for improved performance.

        Args:
            start_date (str, optional): Start date for sowingDate filter (format: YYYY-MM-DD)
            end_date (str, optional): End date for sowingDate filter (format: YYYY-MM-DD)
            crop_id (str, optional): Crop ID to filter by
            ids (List[str], optional): List of specific seasonfield IDs to load.
                                     If None, loads all available seasonfields.
            batch_size (int): Number of seasonfields to process per batch (default: 5000)

        Returns:
            pd.DataFrame: DataFrame containing the loaded seasonfields
        """
        base_url = agro_urls["base_url"][self.env]

        print(f"🔍 Loading seasonfields between {start_date} and {end_date}...")

        if ids is None:
            print("🔍 No specific IDs provided, getting all available seasonfields first...")
            all_seasonfields = get_seasonfield_list(
                self.env, self.bearer_token, start_date=start_date, end_date=end_date, crop_id=crop_id
            )
            ids = all_seasonfields["id"].tolist()
            print(f"📋 Found {len(ids)} seasonfields to process")

        try:
            sfd_list = get_season_fields(
                base_url=base_url,
                token=self.bearer_token,
                ids=ids,
                start_date=start_date,
                end_date=end_date,
                crop_id=crop_id,
                batch_size=batch_size,
            )

            print(f"✅ Successfully loaded {len(sfd_list)} seasonfields using batch processing")
            return sfd_list

        except Exception as e:
            print(f"❌ Error loading seasonfields with batch processing: {e}")
            print("🔄 Falling back to standard method...")
            sfd_list = get_seasonfield_list(
                self.env, self.bearer_token, start_date=start_date, end_date=end_date, crop_id=crop_id
            )
            print(f"✅ Loaded {len(sfd_list)} seasonfields using standard method")
            return sfd_list

    @requires_token
    def get_seasonfield_by_id(self, seasonfield_id: str) -> Optional[Dict]:
        """Get single seasonfield by ID."""
        url = f"{self.mdm_url}/seasonfields/{seasonfield_id}"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                return None
            raise

    @requires_token
    def update_seasonfield(
        self,
        seasonfield_id: str,
        geometry: str = None,
        sowing_date: str = None,
        crop_code: str = None,
        name: str = None,
    ) -> bool:
        """Update a Seasonfield."""
        log = self.get_contextualized_logger("UPDATE_SEASONFIELD")
        log.info(f"Updating seasonfield {seasonfield_id}")

        url = f"{self.mdm_url}/seasonfields/{seasonfield_id}"
        payload: dict[str, Any] = {}
        if geometry:
            payload["geometry"] = geometry
        if sowing_date:
            payload["sowingDate"] = sowing_date
        if crop_code:
            payload["crop"] = {"code": crop_code}
        if name:
            payload["name"] = name

        if not payload:
            log.warning("No fields to update")
            return True

        try:
            response = requests.patch(url, headers=self._get_headers(), json=payload, timeout=60)
            response.raise_for_status()
            log.success(f"✅ Seasonfield updated: {seasonfield_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            log.error(f"❌ Failed to update seasonfield: {e.response.text}")
            return False

    @requires_token
    def delete_seasonfield(self, seasonfield_id: str) -> bool:
        """Delete a Seasonfield."""
        log = self.get_contextualized_logger("DELETE_SEASONFIELD")
        log.info(f"Deleting seasonfield {seasonfield_id}")

        url = f"{self.mdm_url}/seasonfields/{seasonfield_id}"

        try:
            response = requests.delete(url, headers=self._get_headers(), timeout=60)
            response.raise_for_status()
            log.success(f"✅ Seasonfield deleted: {seasonfield_id}")
            return True
        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # raise_for_status sets e.response
            if e.response.status_code == 404:
                log.warning(f"Seasonfield not found (already deleted?): {seasonfield_id}")
                return True
            log.error(f"❌ Failed to delete seasonfield: {e.response.text}")
            return False

    # =========================================================================
    # BULK CREATION
    # =========================================================================

    #: Column names seen in client field files, lowercased, mapped to the
    #: canonical name. Extends the loader's own mapping with the geometry
    #: spellings GIS exports use.
    FIELD_COLUMN_ALIASES = {
        "geometry": "Geometry",
        "geom": "Geometry",
        "the_geom": "Geometry",
        "wkt": "Geometry",
        "wkt_geom": "Geometry",
        "wktgeometry": "Geometry",
        "shape": "Geometry",
        "boundary": "Geometry",
        "border": "Geometry",
        "seasonfield": "Seasonfield",
        "field": "Seasonfield",
        "field_name": "Seasonfield",
        "fieldname": "Seasonfield",
        "name": "Seasonfield",
        "parcel": "Seasonfield",
        "parcelle": "Seasonfield",
        "crop": "Crop",
        "crop_code": "Crop",
        "cropcode": "Crop",
        "culture": "Crop",
        "sowing": "Sowing",
        "sowing_date": "Sowing",
        "sowingdate": "Sowing",
        "date_semis": "Sowing",
        "semis": "Sowing",
        "planting_date": "Sowing",
    }

    def prepare_field_dataframe(
        self,
        df: pd.DataFrame,
        *,
        grower_id: str,
        farm_name: str,
        column_mapping: Optional[Dict[str, str]] = None,
        default_crop: str = "OTHERS",
        default_sowing_date: Optional[str] = None,
        source_epsg: Optional[Any] = None,
        min_area_ha: Optional[float] = 1.0,
        field_name_template: str = "Field {n}",
        verbose: bool = True,
    ) -> tuple:
        """Turn an arbitrary field file into rows ``create_entities_from_dataframe`` accepts.

        **Geometry is the only thing the caller must supply.** Everything else
        the entity hierarchy needs is either defaulted here or comes from the
        provisioning run, because in practice a client sends a shapefile or a
        CSV of borders and nothing else:

        =============  ==========================================================
        Grower         always ``grower_id`` — the GROWER user this run created.
                       Never read from the file: fields belong to the account
                       being provisioned, and a stale id in a spreadsheet would
                       silently file them under someone else.
        Farm           always ``farm_name`` — the farm this run creates.
        Geometry       **required**. Normalised by
                       :func:`~earthdaily.agriculture.core.geometry.normalize_field_geometry`
                       — reprojected to EPSG:4326, checked areal and valid,
                       size-floored.
        Crop           from the file when present, else ``default_crop``. An
                       unknown code also falls back, and is reported.
        Sowing         from the file when present, else ``default_sowing_date``.
        Seasonfield    from the file when present, else generated from
                       ``field_name_template``.
        =============  ==========================================================

        Nothing is dropped silently. Rows whose geometry cannot be salvaged come
        back in a second frame with the reason, so a 500-field import shows what
        it refused instead of quietly creating 480 fields.

        Args:
            df: Raw input, straight from :meth:`load_input_file` or any reader.
            grower_id: GROWER user id to file every field under.
            farm_name: Farm name to file every field under.
            column_mapping: Explicit ``{source column: canonical name}``. Applied
                before the built-in aliases, so it always wins. Canonical names
                are Geometry / Seasonfield / Crop / Sowing.
            default_crop: Crop for rows without a usable one.
            default_sowing_date: ``YYYY-MM-DD`` for rows without one. Required
                when the file has no sowing column.
            source_epsg: CRS of the input geometry. None means "assume 4326 and
                verify"; out-of-range coordinates then raise rather than guess.
            min_area_ha: Size floor. None disables it.
            field_name_template: Used when no name column exists. ``{n}`` is the
                1-based row number.
            verbose: Print the summary.

        Returns:
            tuple: ``(ready, rejected)``. ``ready`` carries the six columns
            :meth:`create_entities_from_dataframe` requires plus ``Area_ha`` and
            ``Geometry_notes``; ``rejected`` carries the original row plus
            ``Reason``.

        Raises:
            ValueError: No geometry column, or no sowing date available at all.
        """
        from shapely.wkt import loads as _load_wkt

        from earthdaily.agriculture.config.crop_reference import crops as crop_catalogue
        from earthdaily.agriculture.core.geometry import geodesic_area_ha, normalize_field_geometry

        log = self.get_contextualized_logger("PREPARE_FIELDS")
        work = df.copy()
        work.columns = [str(c).strip() for c in work.columns]

        # ── Resolve columns: explicit mapping first, then aliases ─────────────
        resolved: Dict[str, str] = {}
        for source, canonical in (column_mapping or {}).items():
            if source not in work.columns:
                raise ValueError(
                    f"❌ column_mapping points at '{source}', which is not in the file. "
                    f"Available: {list(work.columns)}"
                )
            resolved[canonical] = source
        for col in work.columns:
            # Distinct name from the loop above: that one binds a `str` from the
            # mapping, this one an `Optional[str]` from a lookup that may miss.
            alias = self.FIELD_COLUMN_ALIASES.get(col.lower())
            if alias and alias not in resolved:
                resolved[alias] = col

        if "Geometry" not in resolved:
            raise ValueError(
                f"❌ No geometry column found. Geometry is the one mandatory input.\n"
                f"   Columns present : {list(work.columns)}\n"
                f"   Recognised names: {sorted({k for k, v in self.FIELD_COLUMN_ALIASES.items() if v == 'Geometry'})}\n"
                f"   Or name it explicitly: column_mapping={{'<your column>': 'Geometry'}}"
            )

        if "Sowing" not in resolved and not default_sowing_date:
            raise ValueError(
                "❌ The file has no sowing-date column and no default_sowing_date was given. "
                "A seasonfield cannot be created without one."
            )

        # ── Row by row ────────────────────────────────────────────────────────
        ready_rows: List[Dict] = []
        rejected_rows: List[Dict] = []
        crop_fallbacks: List[str] = []

        for position, (_, row) in enumerate(work.iterrows(), start=1):
            raw_geometry = row[resolved["Geometry"]]
            try:
                if pd.isna(raw_geometry) or not str(raw_geometry).strip():
                    raise ValueError("❌ Geometry is empty")
                geometry_wkt, notes = normalize_field_geometry(
                    str(raw_geometry).strip() if isinstance(raw_geometry, str) else raw_geometry,
                    source_epsg=source_epsg,
                    min_area_ha=min_area_ha,
                )
            except Exception as exc:
                rejected_rows.append({**row.to_dict(), "Reason": str(exc)})
                continue

            crop = str(row[resolved["Crop"]]).strip().upper() if "Crop" in resolved else ""
            if crop in ("", "NAN", "NONE", "NULL"):
                crop = default_crop
            elif crop not in crop_catalogue:
                crop_fallbacks.append(crop)
                notes = notes + [f"unknown crop '{crop}' -> {default_crop}"]
                crop = default_crop

            sowing = str(row[resolved["Sowing"]]).strip() if "Sowing" in resolved else ""
            if sowing in ("", "nan", "NaN", "None", "NULL"):
                sowing = default_sowing_date or ""
            if not sowing:
                rejected_rows.append({**row.to_dict(), "Reason": "❌ No sowing date, and no default given"})
                continue

            name = str(row[resolved["Seasonfield"]]).strip() if "Seasonfield" in resolved else ""
            if name in ("", "nan", "None"):
                name = field_name_template.format(n=position)

            ready_rows.append(
                {
                    "Grower": grower_id,
                    "Farm": farm_name,
                    "Seasonfield": name,
                    "Sowing": sowing,
                    "Crop": crop,
                    "Geometry": geometry_wkt,
                    "Area_ha": round(geodesic_area_ha(_load_wkt(geometry_wkt)), 2),
                    "Geometry_notes": "; ".join(notes),
                }
            )

        ready = pd.DataFrame(
            ready_rows,
            columns=["Grower", "Farm", "Seasonfield", "Sowing", "Crop", "Geometry", "Area_ha", "Geometry_notes"],
        )
        rejected = pd.DataFrame(rejected_rows)

        if verbose:
            print(f"\n🧭 Prepared {len(ready)}/{len(work)} field(s) for '{farm_name}' (grower {grower_id})")
            if len(ready):
                repaired = int((ready["Geometry_notes"] != "").sum())
                print(f"   total area   : {ready['Area_ha'].sum():,.1f} ha")
                print(f"   crop(s)      : {dict(ready['Crop'].value_counts())}")
                if repaired:
                    print(f"   ⚠️  {repaired} geometr(ies) repaired or reprojected — see Geometry_notes")
            if crop_fallbacks:
                print(f"   ⚠️  crop code(s) not in the catalogue, set to {default_crop}: {sorted(set(crop_fallbacks))}")
            if len(rejected):
                print(f"   ❌ {len(rejected)} row(s) rejected:")
                for reason, count in rejected["Reason"].value_counts().items():
                    print(f"      {count:>3} x {str(reason)[:96]}")
        log.info(f"prepare_field_dataframe: {len(ready)} ready, {len(rejected)} rejected")

        return ready, rejected

    def create_entities_from_dataframe(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Create Farm -> Field -> Seasonfield hierarchy from DataFrame.
        Reuses existing Farm IDs when farm name repeats.

        Args:
            df (pd.DataFrame): Input data with columns [Grower, Farm, Seasonfield, Sowing, Crop, Geometry]
            verbose (bool): Print progress

        Returns:
            pd.DataFrame: Input data enriched with Farm_Id, Field_Id, Seasonfield_Id, Status, Details
        """
        log = self.get_contextualized_logger("BULK_CREATE")
        log.info("=" * 60)
        log.info("Starting bulk entity creation")
        log.info("=" * 60)
        log.info(f"Total rows: {len(df)}")

        # Clear farm cache for fresh run
        self.clear_farm_cache()

        # Result tracking
        results = df.copy()
        results["Farm_Id"] = ""
        results["Field_Id"] = ""
        results["Seasonfield_Id"] = ""
        results["Status"] = ""
        results["Details"] = ""

        print(f"\n🌾 Creating {len(df)} entities...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating entities", disable=not verbose):
            try:
                grower_id = str(row["Grower"]).strip()
                farm_name = str(row["Farm"]).strip()
                field_name = str(row["Seasonfield"]).strip()
                sowing_date = str(row["Sowing"]).strip()
                crop_code = str(row["Crop"]).strip().upper()
                geometry = str(row["Geometry"]).strip()

                # Step 1: Create/get Farm (uses cache)
                farm_id = self.create_farm(grower_id, farm_name, use_cache=True)
                results.at[idx, "Farm_Id"] = farm_id

                # Step 2: Create Field
                field_id = self.create_field(farm_id, field_name, geometry)
                results.at[idx, "Field_Id"] = field_id

                # Step 3: Create Seasonfield
                sf_id = self.create_seasonfield(field_id, geometry, sowing_date, crop_code, name=field_name)
                results.at[idx, "Seasonfield_Id"] = sf_id

                results.at[idx, "Status"] = "CREATED"

            except Exception as e:
                log.error(f"❌ Row {idx} failed: {e}")
                results.at[idx, "Status"] = "ERROR"
                results.at[idx, "Details"] = str(e)

        # Summary
        created = (results["Status"] == "CREATED").sum()
        errors = (results["Status"] == "ERROR").sum()

        log.info("=" * 60)
        log.success(f"Bulk creation complete: {created} created, {errors} errors")
        log.info("=" * 60)

        print("\n✅ Creation complete!")
        print(f"   Created: {created}")
        print(f"   Errors: {errors}")

        return results

    # =========================================================================
    # BULK DELETION
    # =========================================================================

    def delete_entities_from_dataframe(
        self, df: pd.DataFrame, delete_farms: bool = False, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Delete Seasonfields and Fields from DataFrame.
        Optionally delete Farms (careful: affects all fields under farm).

        Args:
            df (pd.DataFrame): Data with Seasonfield_Id, Field_Id, (optionally Farm_Id)
            delete_farms (bool): Also delete farms (default: False)
            verbose (bool): Print progress

        Returns:
            pd.DataFrame: Input data enriched with Status, Details
        """
        log = self.get_contextualized_logger("BULK_DELETE")
        log.info("=" * 60)
        log.info("Starting bulk entity deletion")
        log.info("=" * 60)

        results = df.copy()
        results["Status"] = ""
        results["Details"] = ""

        print(f"\n🗑️ Deleting {len(df)} entities...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Deleting entities", disable=not verbose):
            try:
                details = []

                # Delete Seasonfield
                sf_id = str(row.get("Seasonfield_Id", "")).strip()
                if sf_id and sf_id.lower() not in ("", "nan", "none"):
                    if self.delete_seasonfield(sf_id):
                        details.append(f"SF:{sf_id}")

                # Delete Field
                field_id = str(row.get("Field_Id", "")).strip()
                if field_id and field_id.lower() not in ("", "nan", "none"):
                    if self.delete_field(field_id):
                        details.append(f"Field:{field_id}")

                # Delete Farm (optional)
                if delete_farms:
                    farm_id = str(row.get("Farm_Id", "")).strip()
                    if farm_id and farm_id.lower() not in ("", "nan", "none"):
                        if self.delete_farm(farm_id):
                            details.append(f"Farm:{farm_id}")

                results.at[idx, "Status"] = "DELETED"
                results.at[idx, "Details"] = ", ".join(details)

            except Exception as e:
                log.error(f"❌ Row {idx} failed: {e}")
                results.at[idx, "Status"] = "ERROR"
                results.at[idx, "Details"] = str(e)

        # Summary
        deleted = (results["Status"] == "DELETED").sum()
        errors = (results["Status"] == "ERROR").sum()

        log.info("=" * 60)
        log.success(f"Bulk deletion complete: {deleted} deleted, {errors} errors")
        log.info("=" * 60)

        print("\n✅ Deletion complete!")
        print(f"   Deleted: {deleted}")
        print(f"   Errors: {errors}")

        return results

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_results(self, results_df: pd.DataFrame, operation: str, output_path: str = None) -> str:
        """
        Export operation results to CSV.

        Args:
            results_df (pd.DataFrame): Results DataFrame
            operation (str): Operation type (creation/modification/deletion/export)
            output_path (str, optional): Output directory

        Returns:
            str: Path to exported file
        """
        log = self.get_contextualized_logger("EXPORT")

        if not output_path:
            output_path = self.output_path or "results"

        # Create dated subfolder
        today = datetime.now().strftime("%Y%m%d")
        export_dir = os.path.join(output_path, f"entity_management_{today}")
        os.makedirs(export_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"entities_{operation}_{today}_{timestamp}.csv"
        filepath = os.path.join(export_dir, filename)

        # Export with same separator as input
        results_df.to_csv(filepath, index=False, encoding="utf-8-sig", sep=self.csv_separator)

        log.success(f"✅ Results exported to: {filepath}")
        print(f"📄 Results exported to: {filepath}")

        return filepath
