# utils/geometry.py
"""
Geometry utilities for spatial data processing.
Handles loading, validation, transformation, and spatial filtering of geometries.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd

#  Third-Party Libraries
import pandas as pd
from loguru import logger
from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as load_wkt

# -------------------------------
# 📦 Export functions
# -------------------------------


def export_geodataframe(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
    output_crs: Optional[str] = None,
    driver: Optional[Literal["auto", "ESRI Shapefile", "GeoJSON", "GPKG", "Parquet"]] = "auto",
    overwrite: bool = True,
) -> Path:
    """
    Export GeoDataFrame to various geospatial formats with optional CRS transformation.

    Args:
        gdf: Input GeoDataFrame to export
        output_path: Path to output file (extension determines format if driver='auto')
        output_crs: Optional CRS to transform to before export (e.g., "EPSG:4326")
        driver: Output format driver. Options:
            - 'auto': Auto-detect from file extension (default)
            - 'ESRI Shapefile': Shapefile format (.shp)
            - 'GeoJSON': GeoJSON format (.geojson, .json)
            - 'GPKG': GeoPackage format (.gpkg)
            - 'Parquet': GeoParquet format (.parquet, .geoparquet)
        overwrite: Whether to overwrite existing file (default: True)

    Returns:
        Path object of the saved file

    Raises:
        ValueError: If format cannot be determined or unsupported

    Examples:
        >>> # Auto-detect format from extension
        >>> export_geodataframe(gdf, 'output/data.shp')
        >>> export_geodataframe(gdf, 'output/data.parquet')

        >>> # With CRS transformation
        >>> export_geodataframe(gdf, 'output/data.shp', output_crs="EPSG:4326")

        >>> # Specify driver explicitly
        >>> export_geodataframe(gdf, 'output/data.shp', driver='ESRI Shapefile')
    """
    logger.info("Starting GeoDataFrame export")
    logger.info(f"Input features: {len(gdf)}")
    logger.info(f"Input CRS: {gdf.crs}")

    # Convert to Path object
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.parent}")

    # Check if file exists
    if output_path.exists() and not overwrite:
        logger.error(f"❌ File already exists and overwrite=False: {output_path}")
        raise FileExistsError(f"File exists: {output_path}")

    # Create copy to avoid modifying original
    gdf_export = gdf.copy()

    # Transform CRS if requested
    if output_crs is not None and output_crs != gdf.crs:
        logger.info(f"Transforming CRS from {gdf.crs} to {output_crs}...")
        try:
            gdf_export = gdf_export.to_crs(output_crs)
            logger.success(f"✅ Transformed to {output_crs}")
        except Exception as e:
            logger.error(f"❌ Failed to transform CRS: {str(e)}")
            raise
    elif output_crs is not None:
        logger.info(f"Already in target CRS: {output_crs}")

    # Auto-detect driver from file extension. Use a separate str-typed var so
    # mypy keeps `driver` narrowed to the Literal[...] | None signature.
    resolved_driver: str = driver if driver and driver != "auto" else ""
    if driver == "auto":
        extension = output_path.suffix.lower()
        driver_map = {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".parquet": "Parquet",
            ".geoparquet": "Parquet",
        }

        if extension not in driver_map:
            logger.error(f"❌ Unsupported file extension: {extension}")
            logger.info(f"Supported extensions: {list(driver_map.keys())}")
            raise ValueError(f"Unsupported extension '{extension}'. Use one of: {list(driver_map.keys())}")

        resolved_driver = driver_map[extension]
        logger.info(f"Auto-detected driver: {resolved_driver}")

    # Export based on driver
    try:
        logger.info(f"Exporting to {resolved_driver} format...")

        if resolved_driver == "Parquet":
            # GeoParquet export
            gdf_export.to_parquet(output_path)
            logger.success(f"💾 GeoParquet saved: {output_path}")

        else:
            # Fiona-based formats (Shapefile, GeoJSON, GPKG)
            gdf_export.to_file(output_path, driver=resolved_driver)
            logger.success(f"💾 {resolved_driver} saved: {output_path}")

        # File size information
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        # Summary
        logger.info(f"""
{"=" * 60}
EXPORT SUMMARY
{"=" * 60}
Features exported:    {len(gdf_export)}
Output format:        {resolved_driver}
Output CRS:           {gdf_export.crs}
Output path:          {output_path}
File size:            {file_size_mb:.2f} MB
{"=" * 60}
        """)

        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to export: {str(e)}")
        raise


def export_multiple_formats(
    gdf: gpd.GeoDataFrame, base_path: str | Path, formats: list = None, output_crs: Optional[str] = None
) -> dict:
    """
    Export GeoDataFrame to multiple formats at once.

    Args:
        gdf: Input GeoDataFrame
        base_path: Base path without extension (e.g., 'outputs/data')
        formats: List of formats to export. Default: ['shp', 'geojson', 'parquet']
        output_crs: Optional CRS for all exports

    Returns:
        Dictionary mapping format to output path

    Example:
        >>> paths = export_multiple_formats(
        ...     gdf=gdf_buffered,
        ...     base_path='outputs/car_buffered',
        ...     formats=['shp', 'parquet', 'geojson']
        ... )
        >>> print(paths['shp'])
        >>> print(paths['parquet'])
    """
    if formats is None:
        formats = ["shp", "parquet", "geojson"]

    logger.info(f"Exporting to {len(formats)} formats: {formats}")

    base_path = Path(base_path)
    output_paths = {}

    extension_map = {
        "shp": ".shp",
        "shapefile": ".shp",
        "geojson": ".geojson",
        "json": ".geojson",
        "parquet": ".parquet",
        "geoparquet": ".geoparquet",
        "gpkg": ".gpkg",
        "geopackage": ".gpkg",
    }

    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in extension_map:
            logger.warning(f"⚠️ Unknown format '{fmt}', skipping")
            continue

        extension = extension_map[fmt_lower]
        output_path = base_path.parent / f"{base_path.stem}{extension}"

        try:
            exported_path = export_geodataframe(gdf=gdf, output_path=str(output_path), output_crs=output_crs)
            output_paths[fmt_lower] = exported_path
        except Exception as e:
            logger.error(f"❌ Failed to export {fmt}: {str(e)}")

    logger.success(f"✅ Exported to {len(output_paths)}/{len(formats)} formats")
    return output_paths


# -------------------------------
# 📦 Transform functions
# -------------------------------


def convert_df_to_geodataframe(
    df: pd.DataFrame, geometry_column: str = "geometry", crs: str = "EPSG:4326", geometry_format: str = "wkt"
) -> gpd.GeoDataFrame:
    """
    Convert DataFrame with geometry to GeoDataFrame.

    Args:
        df: Input DataFrame
        geometry_column: Name of column containing geometries (default: 'geometry')
        crs: Coordinate reference system (default: EPSG:4326)
        geometry_format: Format of geometry data ('wkt' or 'shapely')

    Returns:
        GeoDataFrame with parsed geometries

    Example:
        >>> gdf = convert_df_to_geodataframe(
        ...     df=df_results,
        ...     crs="EPSG:4326"
        ... )
    """
    logger.info("Converting DataFrame to GeoDataFrame")
    logger.info(f"Input records: {len(df)}")
    logger.info(f"Geometry format: {geometry_format}")

    # Validate geometry column exists
    if geometry_column not in df.columns:
        logger.error(f"❌ Column '{geometry_column}' not found")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{geometry_column}' not found")

    # Create copy to avoid modifying original
    df_copy = df.copy()

    # Parse geometries if needed
    if geometry_format == "wkt":
        logger.info("Parsing WKT geometries to shapely objects...")
        try:
            df_copy[geometry_column] = df_copy[geometry_column].apply(wkt.loads)
            logger.success(f"✅ Parsed {len(df_copy)} WKT geometries")
        except Exception as e:
            logger.error(f"❌ Failed to parse WKT geometries: {str(e)}")
            raise

    # Create GeoDataFrame
    logger.info(f"Creating GeoDataFrame with CRS: {crs}")
    try:
        gdf = gpd.GeoDataFrame(df_copy, geometry=geometry_column, crs=crs)

        # Validate geometries
        invalid_count = (~gdf.is_valid).sum()
        if invalid_count > 0:
            logger.warning(f"⚠️ {invalid_count} invalid geometries detected")

        empty_count = gdf.is_empty.sum()
        if empty_count > 0:
            logger.warning(f"⚠️ {empty_count} empty geometries detected")

        logger.success(f"✅ GeoDataFrame created with {len(gdf)} features")

        return gdf

    except Exception as e:
        logger.error(f"❌ Failed to create GeoDataFrame: {str(e)}")
        raise


def apply_buffer_and_dissolve(
    gdf: gpd.GeoDataFrame,
    buffer_distance: float = -300,
    dissolve_column: str = "cod_imovel",
    target_crs: str = "EPSG:32723",
    calculate_area: bool = True,
) -> gpd.GeoDataFrame:
    """
    Apply buffer and dissolve operations to GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        buffer_distance: Buffer distance in meters (negative = inward buffer)
        dissolve_column: Column to dissolve/group by
        target_crs: Target metric CRS for buffering (default: EPSG:32723 - UTM 23S)
        calculate_area: Whether to calculate area statistics (default: True)

    Returns:
        GeoDataFrame with buffered and dissolved geometries

    Example:
        >>> gdf_buffered = apply_buffer_and_dissolve(
        ...     gdf=gdf,
        ...     buffer_distance=-300,
        ...     dissolve_column='cod_imovel'
        ... )
    """
    logger.info("Starting buffer and dissolve operation")
    logger.info(f"Input features: {len(gdf)}")
    logger.info(f"Buffer distance: {buffer_distance}m")

    # Step 1: Reproject to metric CRS if needed
    source_crs = gdf.crs
    if source_crs != target_crs:
        logger.info(f"Step 1/3: Reprojecting from {source_crs} to {target_crs}...")
        try:
            gdf_metric = gdf.to_crs(target_crs)
            logger.success(f"✅ Reprojected to {target_crs}")
        except Exception as e:
            logger.error(f"❌ Failed to reproject: {str(e)}")
            raise
    else:
        logger.info(f"Step 1/3: Already in target CRS ({target_crs})")
        gdf_metric = gdf.copy()

    # Step 2: Apply buffer
    logger.info(f"Step 2/3: Applying {buffer_distance}m buffer...")
    try:
        gdf_metric["geometry"] = gdf_metric.geometry.buffer(buffer_distance)

        # Check for invalid geometries after buffer
        invalid_count = (~gdf_metric.is_valid).sum()
        if invalid_count > 0:
            logger.warning(f"⚠️ {invalid_count} invalid geometries after buffer, fixing...")
            gdf_metric["geometry"] = gdf_metric.geometry.buffer(0)

        # Remove empty geometries
        empty_count = gdf_metric.is_empty.sum()
        if empty_count > 0:
            logger.warning(f"⚠️ {empty_count} empty geometries after buffer (removed)")
            gdf_metric = gdf_metric[~gdf_metric.is_empty]

        logger.success(f"✅ Buffer applied to {len(gdf_metric)} geometries")

    except Exception as e:
        logger.error(f"❌ Failed to apply buffer: {str(e)}")
        raise

    # Step 3: Dissolve by column
    logger.info(f"Step 3/3: Dissolving by '{dissolve_column}'...")
    try:
        # Check if dissolve column exists
        if dissolve_column not in gdf_metric.columns:
            logger.error(f"❌ Column '{dissolve_column}' not found")
            logger.info(f"Available columns: {gdf_metric.columns.tolist()}")
            raise ValueError(f"Column '{dissolve_column}' not found")

        # Dissolve geometries
        gdf_dissolved = gdf_metric.dissolve(by=dissolve_column)

        logger.success(f"✅ Dissolved into {len(gdf_dissolved)} unique features")
        logger.info(f"   Original features: {len(gdf_metric)}")
        logger.info(f"   Dissolved features: {len(gdf_dissolved)}")

    except Exception as e:
        logger.error(f"❌ Failed to dissolve: {str(e)}")
        raise

    # Calculate area statistics if requested
    if calculate_area:
        logger.info("Calculating area statistics...")
        gdf_dissolved["area_m2"] = gdf_dissolved.geometry.area
        gdf_dissolved["area_ha"] = gdf_dissolved["area_m2"] / 10000

        total_area_ha = gdf_dissolved["area_ha"].sum()
        avg_area_ha = gdf_dissolved["area_ha"].mean()

        logger.info(f"   Total area: {total_area_ha:,.2f} ha")
        logger.info(f"   Average area: {avg_area_ha:,.2f} ha")
        logger.info(f"   Min area: {gdf_dissolved['area_ha'].min():,.2f} ha")
        logger.info(f"   Max area: {gdf_dissolved['area_ha'].max():,.2f} ha")

    # Summary
    logger.info(f"""
{"=" * 60}
BUFFER & DISSOLVE SUMMARY
{"=" * 60}
Input features:       {len(gdf)}
Output features:      {len(gdf_dissolved)}
Buffer distance:      {buffer_distance} m
Dissolve column:      {dissolve_column}
Target CRS:           {target_crs}
{"=" * 60}
    """)

    return gdf_dissolved


# -------------------------------
# 📦 Load data functions
# -------------------------------


def load_geodataframe(
    file_path: str, geometry_col: str = "geometry", crs: str = "EPSG:4326", verbose: bool = False, sep: str = ","
) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from various file formats.

    Supports: Shapefile (.shp), GeoParquet (.parquet, .gpq, .geoparquet),
              GeoJSON (.geojson, .json), GeoPackage (.gpkg), CSV with WKT geometry

    Args:
        file_path: Path to the file
        geometry_col: Name of the geometry column (for CSV files)
        crs: Coordinate reference system (for CSV files). Default: "EPSG:4326"
        verbose: Whether to log progress messages
        sep: Column separator for CSV files. Default: ","

    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame

    Raises:
        ValueError: If file format is unsupported or geometry column is missing
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> gdf = load_geodataframe('data/fields.shp', verbose=True)
        >>> gdf = load_geodataframe('data/fields.csv', geometry_col='wkt', verbose=True)
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if verbose:
        logger.info(f"📂 Loading geodataframe from: {file_path}")
        logger.info(f"   Format: {ext}")

    try:
        if ext == ".shp":
            gdf = gpd.read_file(file_path)
            if verbose:
                logger.success(f"✅ Loaded Shapefile: {len(gdf)} features")

        elif ext in [".parquet", ".gpq", ".geoparquet"]:
            gdf = gpd.read_parquet(file_path)
            if verbose:
                logger.success(f"✅ Loaded GeoParquet: {len(gdf)} features")

        elif ext in [".geojson", ".json", ".gpkg"]:
            gdf = gpd.read_file(file_path)
            if verbose:
                logger.success(f"✅ Loaded {ext.upper()}: {len(gdf)} features")

        elif ext == ".csv":
            if verbose:
                logger.info("📄 Loading CSV with WKT geometries...")

            df = pd.read_csv(file_path, sep=sep)

            if geometry_col not in df.columns:
                logger.error(f"❌ CSV file must have a '{geometry_col}' column")
                raise ValueError(f"CSV file must have a '{geometry_col}' column.")

            if verbose:
                logger.info(f"   Converting '{geometry_col}' column to geometry...")

            df[geometry_col] = df[geometry_col].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, geometry=geometry_col)
            gdf.set_crs(crs, inplace=True)

            if verbose:
                logger.success(f"✅ Loaded CSV with {len(gdf)} features")
                logger.info(f"   CRS set to: {crs}")
        else:
            logger.error(f"❌ Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        if verbose:
            logger.info(f"   Columns: {list(gdf.columns)}")
            logger.info(f"   CRS: {gdf.crs}")
            logger.info(f"   Geometry type(s): {gdf.geometry.geom_type.unique().tolist()}")

        return gdf

    except Exception as e:
        logger.error(f"❌ Error loading geodataframe: {str(e)}")
        raise


def safe_wkt_fix(poly_wkt: str, verbose: bool = False) -> Optional[Polygon]:
    """
    Attempt to fix common WKT polygon issues (e.g., unclosed rings).

    Args:
        poly_wkt: WKT string representing a polygon
        verbose: Whether to log progress messages

    Returns:
        Polygon or None: Fixed polygon geometry, or None if fixing failed

    Examples:
        >>> wkt_str = "POLYGON((0 0, 1 0, 1 1, 0 1))"  # Unclosed
        >>> fixed = safe_wkt_fix(wkt_str, verbose=True)
    """
    try:
        if verbose:
            logger.debug("🔧 Attempting to fix WKT polygon...")

        geom = wkt.loads(poly_wkt)

        if geom.geom_type == "Polygon":
            # Force ring to be closed
            coords = list(geom.exterior.coords)

            if coords[0] != coords[-1]:
                if verbose:
                    logger.info("   Closing polygon ring...")
                coords.append(coords[0])
                geom = Polygon(coords)

                if verbose:
                    logger.success("✅ Polygon ring closed successfully")
            else:
                if verbose:
                    logger.debug("   Polygon ring already closed")

        return geom

    except Exception as e:
        logger.error(f"❌ WKT Load Error: {e}")
        if verbose:
            logger.debug(f"   WKT: {poly_wkt}")
        return None


# -------------------------------
# 📦 Utilities functions
# -------------------------------
def validate_wkt(geometry: Union[str, "BaseGeometry", None], verbose: bool = False) -> str:
    """
    Validate and normalize geometry to a WKT string.

    Accepts WKT strings or shapely geometry objects. Shapely objects
    (e.g., from a GeoDataFrame's geometry column) are automatically
    converted to WKT.

    Args:
        geometry: WKT geometry string or shapely geometry object
        verbose: Whether to log progress messages

    Returns:
        str: Validated WKT string

    Raises:
        ValueError: If geometry is None, empty, or invalid
        TypeError: If geometry is not a string or shapely geometry

    Examples:
        >>> validate_wkt("POINT(0 0)", verbose=True)
        >>> validate_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", verbose=True)
        >>> from shapely.geometry import Point
        >>> validate_wkt(Point(0, 0))  # returns "POINT (0 0)"
    """
    if verbose:
        logger.debug("🔍 Validating WKT geometry...")

    if geometry is None:
        logger.error("❌ Geometry must be provided in WKT format")
        raise ValueError("❌ Geometry must be provided in WKT format.")

    # Handle shapely geometry objects — convert to WKT
    if isinstance(geometry, BaseGeometry):
        if geometry.is_empty:
            logger.error("❌ Shapely geometry is empty")
            raise ValueError("❌ Shapely geometry is empty.")
        if verbose:
            logger.info(f"🔄 Converting shapely {geometry.geom_type} to WKT")
        geometry = geometry.wkt

    if not isinstance(geometry, str):
        logger.error(f"❌ Geometry must be a WKT string or shapely geometry, got {type(geometry)} instead")
        raise TypeError(f"❌ Geometry must be a WKT string or shapely geometry, got {type(geometry)} instead.")

    try:
        geom_obj = load_wkt(geometry)

        if geom_obj.is_empty:
            logger.error("❌ Geometry WKT is empty or invalid")
            raise ValueError("❌ Geometry WKT is empty or invalid.")

        if verbose:
            logger.success(f"✅ Valid WKT geometry ({geom_obj.geom_type})")

        return geometry

    except (ShapelyError, Exception) as e:
        logger.error(f"❌ Invalid WKT geometry: {e}")
        raise ValueError(f"❌ Invalid WKT geometry: {e}")


def get_centroid_wkt(geometry_wkt: str, verbose: bool = False) -> str:
    """
    Compute the centroid of a given geometry (in WKT) and return it as WKT.
    Performs validation before processing.

    Args:
        geometry_wkt: Geometry in WKT format (Polygon, MultiPolygon, etc.)
        verbose: Whether to log progress messages

    Returns:
        str: Centroid geometry as WKT (POINT)

    Raises:
        ValueError: If geometry is invalid or empty, or if centroid computation fails

    Examples:
        >>> polygon_wkt = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"
        >>> centroid = get_centroid_wkt(polygon_wkt, verbose=True)
        >>> print(centroid)  # "POINT(2 2)"
    """
    if verbose:
        logger.info("📍 Computing centroid...")

    # Validate and normalize geometry (handles both WKT strings and shapely objects)
    geometry_wkt = validate_wkt(geometry_wkt, verbose=verbose)

    # Parse geometry
    geom = wkt.loads(geometry_wkt)

    if verbose:
        logger.info(f"   Input geometry: {geom.geom_type}")
        logger.info(f"   Bounds: {geom.bounds}")

    # Compute centroid
    centroid: Point = geom.centroid

    # Ensure valid centroid
    if centroid.is_empty:
        logger.error("❌ Centroid computation failed — empty geometry returned")
        raise ValueError("❌ Centroid computation failed — empty geometry returned.")

    if verbose:
        logger.success(f"✅ Centroid computed: ({centroid.x:.6f}, {centroid.y:.6f})")

    return centroid.wkt


def centroid_geohash(geometry, precision: int = 6, verbose: bool = False) -> str:
    """
    Encode the centroid of a geometry as a geohash string.

    Point-based analytics (Weather, GDD, GDD-offset) query the API by field
    **centroid**, so fields whose centroids land in the same geohash cell return
    identical values. The geohash is therefore the natural spatial bucket key for
    grouped/deduplicated extraction (see ``BaseExtractor._run_bulk_spatially_grouped``).

    Precision → approximate cell size (equatorial):
        4 ≈ 39 km, 5 ≈ 4.9 km, 6 ≈ 1.2 km, 7 ≈ 153 m.

    Args:
        geometry: Geometry in WKT format (or a shapely object). Any type accepted
            by :func:`get_centroid_wkt` — the centroid is computed first.
        precision: Geohash length. Default 6 (~1.2 km); callers typically pass
            ``DEFAULT_SPATIAL_PRECISION`` (5, ~4.9 km) matched to the weather grid.
        verbose: Whether to log progress messages.

    Returns:
        str: Geohash string of length ``precision``.

    Raises:
        ValueError: If the geometry is invalid or the centroid cannot be computed.
    """
    # Imported lazily so the package still imports if the (declared) dependency is
    # missing in a stale environment — the error then points at the real fix.
    try:
        import pygeohash
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "centroid_geohash requires the 'pygeohash' package. Install it with "
            "`pip install pygeohash` or reinstall the pinned requirements."
        ) from exc

    centroid = get_centroid_wkt(geometry, verbose=verbose)
    point: Point = wkt.loads(centroid)
    return pygeohash.encode(point.y, point.x, precision=precision)


def get_area_sqm(geometry) -> float | None:
    """
    Return the area in square meters of a WKT string or Shapely geometry in EPSG:4326.
    Returns None if the computation fails for any reason.
    """
    import numpy as np
    from shapely import get_coordinates

    try:
        # Parse WKT string if needed
        geom = geometry if isinstance(geometry, BaseGeometry) else wkt.loads(geometry)

        # Basic validity checks
        if geom is None or geom.is_empty:
            return None
        coords = get_coordinates(geom)
        if not np.isfinite(coords).all():
            return None

        # Reproject to UTM and compute area
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
        return gdf_projected.geometry.area.iloc[0]

    except Exception:
        return None


def add_area_column(
    df: pd.DataFrame,
    geometry_column: str = "geometry",
    area_column: str = "area_sqm",
) -> pd.DataFrame:
    """
    Add an area column (in square meters) to a DataFrame containing WKT geometries.

    Converts to GeoDataFrame once, reprojects to UTM, computes area vectorized,
    and returns the original DataFrame with the new column appended.
    Rows with invalid or unparseable WKT are coerced to None (area = NaN).

    Args:
        df: Input DataFrame with a WKT geometry column.
        geometry_column: Name of the column containing WKT strings.
        area_column: Name of the output area column.

    Returns:
        DataFrame with the new area column added (NaN for invalid geometries).
    """
    import numpy as np

    result = df.copy()

    # Parse WKT strings safely — coerce failures to None
    def _safe_parse(val):
        try:
            geom = wkt.loads(str(val))
            if geom is None or geom.is_empty or not geom.is_valid:
                return None
            return geom
        except Exception:
            return None

    geometries = result[geometry_column].apply(_safe_parse)
    valid_mask = geometries.notna()
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        logger.warning(f"⚠️ {n_invalid} invalid geometries coerced to NaN in area computation")

    # Default all to NaN, then compute area only for valid rows
    result[area_column] = np.nan
    if valid_mask.any():
        gdf = gpd.GeoDataFrame(geometry=geometries[valid_mask].tolist(), crs="EPSG:4326")
        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        result.loc[valid_mask, area_column] = gdf_utm.geometry.area.values

    return result


# -------------------------------
# 📦 Filtered read functions
# -------------------------------
def filter_gdf_by_geometry(gdf: gpd.GeoDataFrame, wkt_geometry: str, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame to get rows intersecting with a WKT geometry.
    Convenience wrapper for filter_gdf_by_spatial_relation with 'intersects' operation.

    Args:
        gdf: Input GeoDataFrame
        wkt_geometry: WKT string of the geometry to intersect with
        verbose: Whether to log progress messages

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with intersecting rows

    Examples:
        >>> polygon = "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))"
        >>> filtered = filter_gdf_by_geometry(gdf, polygon, verbose=True)
    """
    return filter_gdf_by_spatial_relation(
        gdf=gdf, wkt_geometry=wkt_geometry, operation="intersects", buffer_distance=0, verbose=verbose
    )


def filter_gdf_by_spatial_relation(
    gdf: gpd.GeoDataFrame,
    wkt_geometry: str,
    operation: Literal["intersects", "contains", "within", "overlaps", "touches"] = "intersects",
    buffer_distance: float = 0,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame based on spatial relationship with a WKT geometry.

    Args:
        gdf: Input GeoDataFrame
        wkt_geometry: WKT string of the geometry
        operation: Spatial operation to use:
            - 'intersects': Any intersection (default)
            - 'contains': Filter geometry contains gdf geometries
            - 'within': gdf geometries within filter geometry
            - 'overlaps': Geometries overlap
            - 'touches': Geometries touch but don't overlap
        buffer_distance: Optional buffer around filter geometry (in CRS units)
        verbose: Whether to log progress messages

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame

    Raises:
        ValueError: If operation is unsupported or WKT is invalid

    Examples:
        >>> # Filter by intersection
        >>> polygon = "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))"
        >>> filtered = filter_gdf_by_spatial_relation(gdf, polygon, verbose=True)

        >>> # Filter with buffer
        >>> point = "POINT(2.5 2.5)"
        >>> filtered = filter_gdf_by_spatial_relation(
        ...     gdf, point, operation='intersects', buffer_distance=1.0, verbose=True
        ... )

        >>> # Filter for features within polygon
        >>> filtered = filter_gdf_by_spatial_relation(
        ...     gdf, polygon, operation='within', verbose=True
        ... )
    """
    try:
        # Parse WKT geometry
        if verbose:
            logger.info("📐 Parsing WKT geometry...")

        filter_geom = wkt.loads(wkt_geometry)

        # Apply buffer if specified
        if buffer_distance > 0:
            if verbose:
                logger.info(f"📏 Applying buffer of {buffer_distance} units")
            filter_geom = filter_geom.buffer(buffer_distance)

        if verbose:
            logger.info(f"   Geometry type: {filter_geom.geom_type}")
            logger.info(f"   Geometry bounds: {filter_geom.bounds}")

        # Ensure GeoDataFrame has a CRS
        if gdf.crs is None:
            logger.warning("⚠️  GeoDataFrame has no CRS, assuming EPSG:4326")
            gdf = gdf.set_crs(epsg=4326)

        # Apply spatial filter
        if verbose:
            logger.info(f"🔍 Applying '{operation}' operation on {len(gdf)} rows...")

        if operation == "intersects":
            mask = gdf.geometry.intersects(filter_geom)
        elif operation == "contains":
            mask = filter_geom.contains(gdf.geometry)
        elif operation == "within":
            mask = gdf.geometry.within(filter_geom)
        elif operation == "overlaps":
            mask = gdf.geometry.overlaps(filter_geom)
        elif operation == "touches":
            mask = gdf.geometry.touches(filter_geom)
        else:
            logger.error(f"❌ Unsupported operation: {operation}")
            raise ValueError(f"Unsupported operation: {operation}")

        filtered_gdf = gdf[mask].copy()

        if verbose:
            logger.success(f"✅ Found {len(filtered_gdf)} rows using '{operation}'")
            if len(filtered_gdf) > 0:
                logger.info(f"   Coverage: {(len(filtered_gdf) / len(gdf) * 100):.2f}% of total rows")
            else:
                logger.warning("⚠️  No features found matching the spatial criteria")

        return filtered_gdf

    except Exception as e:
        logger.error(f"❌ Error filtering GeoDataFrame: {str(e)}")
        raise


def get_geometry_stats(gdf: gpd.GeoDataFrame, verbose: bool = True) -> dict:
    """
    Get summary statistics about geometries in a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        verbose: Whether to log statistics

    Returns:
        dict: Dictionary with geometry statistics

    Examples:
        >>> stats = get_geometry_stats(gdf, verbose=True)
    """
    stats = {
        "total_features": len(gdf),
        "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": gdf.total_bounds.tolist() if len(gdf) > 0 else None,
        "has_null_geometries": gdf.geometry.isna().sum(),
        "has_empty_geometries": gdf.geometry.is_empty.sum(),
    }

    if verbose:
        logger.info("📊 GeoDataFrame Statistics:")
        logger.info(f"   Total features: {stats['total_features']}")
        logger.info(f"   CRS: {stats['crs']}")
        logger.info(f"   Geometry types: {stats['geometry_types']}")
        if stats["bounds"]:
            logger.info(f"   Bounds: {stats['bounds']}")
        if stats["has_null_geometries"] > 0:
            logger.warning(f"   ⚠️  Null geometries: {stats['has_null_geometries']}")
        if stats["has_empty_geometries"] > 0:
            logger.warning(f"   ⚠️  Empty geometries: {stats['has_empty_geometries']}")

    return stats


# -------------------------------
# 🧭 CRS + polygon normalisation
# -------------------------------
#
# The platform stores field borders as WGS84 (EPSG:4326) WKT polygons. Input
# files rarely arrive that way: a shapefile carries its own projected CRS, and a
# CSV of WKT carries no CRS at all. These helpers turn "whatever the client
# sent" into "a valid EPSG:4326 polygon", loudly.


#: Bare lat/lon bounds. Coordinates outside these cannot be EPSG:4326, which is
#: the only way to detect a projected CSV that arrived with no CRS metadata.
_WGS84_BOUNDS = (-180.0, -90.0, 180.0, 90.0)


def looks_like_lonlat(geometry: "BaseGeometry") -> bool:
    """True when every coordinate falls inside WGS84 bounds.

    A CSV of WKT has no CRS metadata, so this is the only available signal that
    the coordinates are *not* degrees. It cannot prove they are — a small field
    in a metre-based CRS near the origin would pass — but it reliably catches
    the common case of UTM / state-plane / Web Mercator eastings and northings,
    which are orders of magnitude out of range.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    return (
        _WGS84_BOUNDS[0] <= min_x <= _WGS84_BOUNDS[2]
        and _WGS84_BOUNDS[0] <= max_x <= _WGS84_BOUNDS[2]
        and _WGS84_BOUNDS[1] <= min_y <= _WGS84_BOUNDS[3]
        and _WGS84_BOUNDS[1] <= max_y <= _WGS84_BOUNDS[3]
    )


def reproject_geometry(geometry: "BaseGeometry", source_epsg: Union[int, str], target_epsg: int = 4326):
    """Reproject a shapely geometry between CRSs.

    Args:
        geometry: Any shapely geometry.
        source_epsg: EPSG code the coordinates are currently in.
        target_epsg: EPSG code to convert to. Defaults to 4326, which is what
            every EarthDaily entity API expects.

    Returns:
        The reprojected geometry, or the original when source == target.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    if str(source_epsg).lstrip("EPSG:").strip() == str(target_epsg):
        return geometry

    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def close_wkt_rings(wkt_string: str) -> str:
    """Close any unclosed rings in a polygon WKT, textually.

    Shapely 2 refuses to *parse* a ring whose last coordinate differs from its
    first, so an unclosed ring cannot be repaired after loading — and unclosed
    rings are one of the most common defects in client-supplied files. This
    operates on the string, before parsing, and is a no-op on anything already
    well-formed.

    Only innermost coordinate groups are touched, so polygons with holes and
    multipolygons are handled without special-casing.
    """
    import re

    def _close(match: "re.Match") -> str:
        body = match.group(1).strip()
        coords = [c.strip() for c in body.split(",") if c.strip()]
        # Only coordinate lists — skip anything that isn't "x y [z]" pairs.
        if len(coords) < 3 or not all(len(c.split()) in (2, 3) for c in coords):
            return match.group(0)
        try:
            for c in coords:
                [float(v) for v in c.split()]
        except ValueError:
            return match.group(0)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return "(" + ", ".join(coords) + ")"

    return re.sub(r"\(([^()]*)\)", _close, wkt_string)


def normalize_field_geometry(
    geometry: Union[str, "BaseGeometry"],
    source_epsg: Optional[Union[int, str]] = None,
    min_area_ha: Optional[float] = None,
    verbose: bool = False,
) -> tuple:
    """Normalise any geometry into a valid EPSG:4326 field border.

    The one entry point for turning client-supplied geometry into something the
    entity APIs accept. Four things can be wrong with an input, and each is
    fixed or reported rather than passed downstream:

    1. **Wrong CRS.** Reprojected when ``source_epsg`` says so. With no
       ``source_epsg``, coordinates are checked against WGS84 bounds — out of
       range means projected data with no CRS, which raises instead of silently
       creating a field somewhere in the Gulf of Guinea.
    2. **Not areal.** Polygon and MultiPolygon both pass through — the platform
       accepts multi-part borders, so parts are never dropped. A single-part
       MultiPolygon is unwrapped (lossless tidying). Points and lines raise.
    3. **Invalid.** Unclosed rings are closed before parsing (see
       :func:`close_wkt_rings`) and self-intersections repaired via
       ``make_valid``; a repair that yields several polygons becomes a
       MultiPolygon rather than losing the smaller parts.
    4. **Empty / unparseable.** Raises.

    Args:
        geometry: WKT string or shapely geometry.
        source_epsg: CRS of the input. None means "assume 4326 and verify".
        min_area_ha: Reject anything smaller, measured on the ellipsoid. None
            disables the check. A field under ~1 ha is usually a digitising
            artefact or a units mistake, and holds too few pixels for
            field-level analytics either way.
        verbose: Log each fix as it is applied.

    Returns:
        tuple: ``(wkt_string, notes)`` — the EPSG:4326 WKT, and a list of
        human-readable strings describing every repair applied. An empty list
        means the input was already clean.

    Raises:
        ValueError: The geometry is unusable — unparseable, empty, a
            non-areal type, out of WGS84 bounds with no ``source_epsg``,
            unrepairable, or below ``min_area_ha``.
    """
    from shapely.geometry import MultiPolygon
    from shapely.validation import make_valid

    notes: list = []

    if isinstance(geometry, str):
        try:
            geom = load_wkt(geometry)
        except Exception:
            # Most likely an unclosed ring — repairable, but only before parsing.
            try:
                geom = load_wkt(close_wkt_rings(geometry))
                notes.append("closed unclosed ring(s)")
            except Exception as exc:
                raise ValueError(f"❌ Unparseable WKT: {exc}") from exc
    elif isinstance(geometry, BaseGeometry):
        geom = geometry
    else:
        raise TypeError(f"❌ Expected WKT string or shapely geometry, got {type(geometry).__name__}")

    if geom.is_empty:
        raise ValueError("❌ Geometry is empty")

    # ── 1. CRS ────────────────────────────────────────────────────────────────
    if source_epsg is not None:
        geom = reproject_geometry(geom, source_epsg, 4326)
        if str(source_epsg) != "4326":
            notes.append(f"reprojected EPSG:{source_epsg} -> EPSG:4326")
    elif not looks_like_lonlat(geom):
        raise ValueError(
            f"❌ Coordinates are outside WGS84 bounds {geom.bounds} — this is projected data "
            f"with no CRS attached. Pass source_epsg=<code> so it can be reprojected; guessing "
            f"would place the field somewhere arbitrary."
        )

    # ── 2. Areal type ─────────────────────────────────────────────────────────
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"❌ Expected a Polygon, got {geom.geom_type}. A field border must be areal — "
            f"a point or line cannot define one."
        )

    # ── 3. Validity, before splitting a MultiPolygon ──────────────────────────
    if not geom.is_valid:
        repaired = make_valid(geom)
        if repaired.is_empty:
            raise ValueError("❌ Geometry is invalid and could not be repaired")
        notes.append("repaired invalid geometry (make_valid)")
        geom = repaired

    # ── 4. Keep every areal part ──────────────────────────────────────────────
    # Multi-part borders are supported by the platform, so parts are never
    # dropped. A repair can leave non-areal debris (a bare line where the ring
    # touched itself) inside a GeometryCollection — that is what gets discarded.
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty]
        if not parts:
            raise ValueError("❌ No polygon found in GeometryCollection")
        flat: list = []
        for part in parts:
            flat.extend(part.geoms if part.geom_type == "MultiPolygon" else [part])
        geom = flat[0] if len(flat) == 1 else MultiPolygon(flat)
        notes.append("dropped non-areal debris left by the repair")

    # Lossless tidying: a one-part MultiPolygon is just a Polygon.
    if geom.geom_type == "MultiPolygon" and len(geom.geoms) == 1:
        geom = geom.geoms[0]

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"❌ Could not reduce {geom.geom_type} to a polygonal geometry")
    if not geom.is_valid:
        raise ValueError("❌ Geometry is still invalid after repair")

    # ── 5. Minimum size ───────────────────────────────────────────────────────
    if min_area_ha is not None:
        area_ha = geodesic_area_ha(geom)
        if area_ha < min_area_ha:
            raise ValueError(
                f"❌ Field is {area_ha:.3f} ha, below the {min_area_ha} ha minimum. Slivers this "
                f"small are usually a digitising artefact or a units mistake, and carry too few "
                f"pixels for field-level analytics to say anything meaningful."
            )

    if verbose:
        for note in notes:
            logger.info(f"   🔧 {note}")

    return geom.wkt, notes


def geodesic_area_ha(geometry: "BaseGeometry") -> float:
    """Area in hectares of a **lon/lat** geometry, measured on the WGS84 ellipsoid.

    ``shapely``'s ``.area`` over degrees returns square degrees — meaningless as
    a size, since it shrinks with latitude and cannot be compared against a
    hectare threshold. Always use this for anything user-facing.
    """
    from pyproj import Geod

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geometry)[0]) / 10_000.0
