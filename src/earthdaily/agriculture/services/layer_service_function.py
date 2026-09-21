#  Standard Library

#  Third-Party Libraries
import requests

# Internal Project Utilities
from earthdaily.agriculture.config.urls import agro_urls


def query_layer_service(
    authenticator,
    env: str,
    layer_name: str = "BR_CAR_PROPERTIES",
    wkt_geometry: str = None,
    attribute_name: str = None,
    attribute_value: str = None,
    extract_fields: list = None,
    return_first_only: bool = False,
    max_retries: int = 1,
) -> dict:
    """
    Query layer service either by geometry (POST) or by attribute (GET).
    """
    if agro_urls is None:
        raise ValueError("agro_urls dictionary must be provided")

    try:
        base_url = agro_urls["layer_service_url"][env]
    except KeyError as e:
        raise ValueError(f"Invalid agro_urls structure or environment '{env}': {e}")

    # Validate query parameters
    has_geometry = wkt_geometry is not None
    has_attribute = attribute_name is not None and attribute_value is not None

    if not has_geometry and not has_attribute:
        raise ValueError("Either wkt_geometry or both attribute_name and attribute_value must be provided")

    if has_geometry and has_attribute:
        raise ValueError("Provide either wkt_geometry or attribute params, not both")

    # Clean attribute value if provided (remove whitespace)
    if attribute_value:
        attribute_value = str(attribute_value).strip()

    # Build URL
    base_endpoint = f"{base_url}/layers/{layer_name}/feature"

    if has_geometry:
        url = base_endpoint
        query_type = f"geometry query on {layer_name}"
    else:
        url = f"{base_endpoint}?attributeName={attribute_name}&attributeValue={attribute_value}"
        query_type = f"attribute query ({attribute_name}={attribute_value}) on {layer_name}"

    # Retry loop for token refresh
    for attempt in range(max_retries + 1):
        try:
            # Get current token from authenticator
            bearer_token = authenticator.get_token()  # ← FIXED: Use get_token()

            headers = {"Accept": "application/json", "Authorization": f"Bearer {bearer_token}"}

            # Execute request
            if has_geometry:
                headers["Content-Type"] = "text/plain"
                response = requests.post(url, headers=headers, data=wkt_geometry)
            else:
                response = requests.get(url, headers=headers)

            # Handle 401 - token expired
            if response.status_code == 401:
                if attempt < max_retries:
                    print(f"⚠️ Token expired, refreshing... (attempt {attempt + 1}/{max_retries})")
                    authenticator.refresh_token()
                    continue
                else:
                    error_msg = f"Authentication failed for {query_type} after {max_retries} retries"
                    print(f"❌ {error_msg}")
                    return {"error": error_msg, "status_code": 401}

            response.raise_for_status()
            result = response.json()
            print(f"✅ {query_type} successful!")

            # Process response if extract_fields is provided
            if extract_fields:
                result = _process_layer_response(result, extract_fields, return_first_only)

            return result

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error for {query_type}: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "status_code": response.status_code}

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error for {query_type}: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    return {"error": "Unexpected error in retry loop"}


def _process_layer_response(data, extract_fields: list, return_first_only: bool = False):
    """
    Process layer service response to extract specific fields.

    Args:
        data: Raw API response (dict with 'features' key OR list of features)
        extract_fields (list): List of property field names to extract.
        return_first_only (bool): If True, returns only the first feature.

    Returns:
        dict or list: Processed data with geometry and specified fields.
    """
    # Handle case where data is already a list of features
    if isinstance(data, list):
        features = data
    # Handle case where data is a dict with 'features' key
    elif isinstance(data, dict):
        features = data.get("features", [])
    else:
        return None

    if not features or len(features) == 0:
        return None

    def extract_feature_data(feature):
        """Extract specified fields from a single feature."""
        formatted_data = {"geometry": feature.get("geometry", "")}

        # Extract requested fields from properties
        properties = feature.get("properties", {})
        for field in extract_fields:
            formatted_data[field] = properties.get(field, "")

        return formatted_data

    if return_first_only:
        return extract_feature_data(features[0])
    else:
        return [extract_feature_data(feature) for feature in features]


def api_layer_properties(
    wkt_geometry: str,
    bearer_token: str,
    env: str,
    layer_name: str = "BR_CAR_PROPERTIES",
) -> dict:
    """
    Makes API call to retrieve property features for a given geometry.

    Args:
        wkt_geometry (str): Geometry in WKT format.
        bearer_token (str): API authentication token.
        env (str): Environment (e.g., 'prod', 'staging', 'dev').
        layer_name (str): Name of the layer to query. Defaults to "BR_CAR_PROPERTIES".

    Returns:
        dict: API response as JSON, or error dict if request fails.
    """

    try:
        base_url = agro_urls["layer_service_url"][env]
    except KeyError as e:
        raise ValueError(f"Invalid agro_urls structure or environment '{env}': {e}")

    url = f"{base_url}/layers/{layer_name}/feature"

    headers = {"Content-Type": "text/plain", "Authorization": f"Bearer {bearer_token}"}

    try:
        response = requests.post(url, headers=headers, data=wkt_geometry)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Request to {layer_name} successful!")

        return result

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error for {layer_name}: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg, "status_code": response.status_code}

    except requests.exceptions.RequestException as e:
        error_msg = f"Request error for {layer_name}: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
