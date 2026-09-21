# utils/S3.py
"""
S3 Cloud Storage Utilities
Handles reading and writing data to/from AWS S3 buckets.
"""

import json
import os
import tempfile
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

#  Third-Party Libraries
from loguru import logger

# Internal Project Utilities
from earthdaily.agriculture.core.geometry import filter_gdf_by_spatial_relation


class S3ObjectNotFound(ValueError):
    """Raised when a requested S3 object does not exist (maps botocore ``NoSuchKey``).

    Subclasses :class:`ValueError` so existing ``except ValueError`` callers keep
    working unchanged. New callers can catch this specifically to handle the
    missing-object / first-run case without string-matching the error message::

        try:
            df = EDA_S3_cloudstore.read_csv(auth, bucket, key)
        except S3ObjectNotFound:
            df = empty_state()  # first run, nothing written yet

    The original message (``"File not found in S3: <key>"``) is preserved so any
    callers currently matching on the string continue to work.
    """

    def __init__(self, bucket_name: str, file_key: str):
        self.bucket_name = bucket_name
        self.file_key = file_key
        super().__init__(f"File not found in S3: {file_key}")


class EDA_S3_cloudstore:
    """
    Handles working with S3 cloud storage for EarthDaily Agro data.
    Provides utilities for reading CSV, Parquet, and GeoParquet files with
    optional validation and optimized spatial filtering using DuckDB, plus
    boto3-based writers (write_csv / write_parquet / write_to_s3), manifest-free
    uploaders (upload_file / upload_files), and an object_exists check for the
    keyed read-modify-write case.
    """

    def __init__(self, authenticator):
        """
        Initialize S3 cloud storage handler.

        Args:
            authenticator: EDAuthenticator instance with initialized s3_client
        """
        self.authenticator = authenticator
        self.s3_client = authenticator.ensure_s3_client()
        logger.info("✅ EDA_S3_cloudstore initialized")

    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================

    @staticmethod
    def is_geoparquet(
        authenticator, bucket_name: str, file_key: str, check_bbox: bool = False, verbose: bool = False
    ) -> Union[bool, dict]:
        """
        Check if a Parquet file is a valid GeoParquet and optionally check bbox support.

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to the Parquet file
            check_bbox: If True, also check if file supports bbox filtering
            verbose: Whether to log detailed information

        Returns:
            If check_bbox=False: bool (True if valid GeoParquet)
            If check_bbox=True: dict with keys:
                - 'is_geoparquet': bool
                - 'supports_bbox': bool
                - 'bbox_method': str (None, 'covering', or 'point')
        """
        s3_client = authenticator.ensure_s3_client()

        try:
            if verbose:
                logger.info(f"🔍 Checking if file is GeoParquet: s3://{bucket_name}/{file_key}")

            # Download file
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response["Body"].read()
            buffer = BytesIO(file_content)

            # Read parquet metadata
            parquet_file = pq.ParquetFile(buffer)
            metadata = parquet_file.schema_arrow.metadata

            # Check for 'geo' key in metadata
            if metadata and b"geo" in metadata:
                geo_metadata = metadata[b"geo"].decode("utf-8")
                geo_info = json.loads(geo_metadata)

                if verbose:
                    logger.success("✅ File is a valid GeoParquet")
                    if "version" in geo_info:
                        logger.info(f"   Version: {geo_info['version']}")
                    if "primary_column" in geo_info:
                        logger.info(f"   Primary geometry column: {geo_info['primary_column']}")

                # If not checking bbox, return simple bool
                if not check_bbox:
                    return True

                # Check bbox support
                primary_column = geo_info.get("primary_column")
                supports_bbox = False
                bbox_method = None

                if primary_column:
                    column_meta = geo_info.get("columns", {}).get(primary_column, {})

                    # Check for bbox covering
                    has_bbox_covering = "covering" in column_meta and column_meta["covering"].get("bbox") is not None

                    # Check for point encoding
                    is_point_encoding = column_meta.get("encoding") == "point"

                    if has_bbox_covering:
                        supports_bbox = True
                        bbox_method = "covering"
                    elif is_point_encoding:
                        supports_bbox = True
                        bbox_method = "point"

                if verbose:
                    if supports_bbox:
                        logger.success(f"   ✓ Bbox filtering supported (method: {bbox_method})")
                    else:
                        logger.warning("   ⚠️  Bbox filtering NOT supported")

                return {"is_geoparquet": True, "supports_bbox": supports_bbox, "bbox_method": bbox_method}
            else:
                if verbose:
                    logger.warning("⚠️  File is a regular Parquet (no GeoParquet metadata)")

                if not check_bbox:
                    return False
                else:
                    return {"is_geoparquet": False, "supports_bbox": False, "bbox_method": None}

        except Exception as e:
            if verbose:
                logger.error(f"❌ Error checking GeoParquet: {str(e)}")

            if not check_bbox:
                return False
            else:
                return {"is_geoparquet": False, "supports_bbox": False, "bbox_method": None}

    @staticmethod
    def get_parquet_info(authenticator, bucket_name: str, file_key: str, verbose: bool = True) -> Dict:
        """
        Get detailed information about a Parquet/GeoParquet file.

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to the Parquet file
            verbose: Whether to log information

        Returns:
            dict: File information including schema, metadata, and GeoParquet status
        """
        s3_client = authenticator.ensure_s3_client()

        try:
            if verbose:
                logger.info(f"📋 Inspecting Parquet file: s3://{bucket_name}/{file_key}")

            # Download file
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response["Body"].read()
            file_size = len(file_content)
            buffer = BytesIO(file_content)

            # Read parquet metadata
            parquet_file = pq.ParquetFile(buffer)
            schema = parquet_file.schema_arrow
            metadata = schema.metadata

            # Basic info
            info = {
                "file_key": file_key,
                "file_size_mb": file_size / (1024 * 1024),
                "num_rows": parquet_file.metadata.num_rows,
                "num_row_groups": parquet_file.metadata.num_row_groups,
                "columns": [field.name for field in schema],
                "column_types": {field.name: str(field.type) for field in schema},
                "is_geoparquet": False,
                "geo_metadata": None,
            }

            # Check for GeoParquet metadata
            if metadata and b"geo" in metadata:
                info["is_geoparquet"] = True
                geo_metadata = json.loads(metadata[b"geo"].decode("utf-8"))
                info["geo_metadata"] = geo_metadata

            # Log information
            if verbose:
                logger.info("📊 File Information:")
                logger.info(f"   Size: {info['file_size_mb']:.2f} MB")
                logger.info(f"   Rows: {info['num_rows']:,}")
                logger.info(f"   Row groups: {info['num_row_groups']}")
                logger.info(f"   Columns: {len(info['columns'])}")

                if info["is_geoparquet"]:
                    logger.success("   ✅ GeoParquet: Yes")
                    geo = info["geo_metadata"]
                    if "version" in geo:
                        logger.info(f"      Version: {geo['version']}")
                    if "primary_column" in geo:
                        logger.info(f"      Geometry column: {geo['primary_column']}")
                else:
                    logger.warning("   ⚠️  GeoParquet: No (regular Parquet)")

            return info

        except Exception as e:
            logger.error(f"❌ Error inspecting Parquet file: {str(e)}")
            raise

    # ========================================================================
    # BASIC READ METHODS
    # ========================================================================

    @staticmethod
    def read_from_s3(
        authenticator,
        bucket_name: str,
        file_key: str,
        file_type: str = "auto",
        safe: bool = False,
        verbose: bool = False,
        **read_kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Read a file from S3 into a pandas DataFrame or GeoPandas GeoDataFrame.

        Supports CSV, Parquet, and GeoParquet files with automatic type detection.

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to the file
            file_type: File type ('csv', 'parquet', 'geoparquet', 'auto').
                      If 'auto', infers from file extension. Default: 'auto'
            safe: If True, validates GeoParquet files before reading. Default: False
            verbose: Whether to log progress messages. Default: False
            **read_kwargs: Additional arguments to pass to pandas/geopandas read functions

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: Loaded data

        Raises:
            ClientError: If S3 operations fail
            S3ObjectNotFound: If the key does not exist (subclass of ValueError;
                catch this to handle the missing-object / first-run case)
            ValueError: If the file cannot be read or the file type is unsupported

        Examples:
            >>> df = EDA_S3_cloudstore.read_from_s3(auth, 'bucket', 'data.csv')
            >>> gdf = EDA_S3_cloudstore.read_from_s3(auth, 'bucket', 'geo.parquet', safe=True)
        """
        s3_client = authenticator.ensure_s3_client()

        # Auto-detect file type from extension
        if file_type == "auto":
            file_lower = file_key.lower()
            if file_lower.endswith(".csv"):
                file_type = "csv"
            elif file_lower.endswith(".parquet"):
                file_type = "parquet"
            else:
                raise ValueError(
                    f"Cannot auto-detect file type for: {file_key}. "
                    f"Supported extensions: .csv, .parquet. "
                    f"Or specify file_type explicitly."
                )

        # Validate GeoParquet if safe mode is enabled and it's a parquet file
        if safe and file_type in ["parquet", "geoparquet"]:
            if verbose:
                logger.info("🔒 Safe mode enabled - validating GeoParquet...")

            is_geo = EDA_S3_cloudstore.is_geoparquet(authenticator, bucket_name, file_key, verbose=verbose)

            if file_type == "geoparquet" and not is_geo:
                error_msg = f"File is not a valid GeoParquet: {file_key}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Update file_type based on validation
            if is_geo:
                file_type = "geoparquet"
                if verbose:
                    logger.info("   ✅ Validated as GeoParquet")
            else:
                file_type = "parquet"
                if verbose:
                    logger.info("   ℹ️  Regular Parquet (not GeoParquet)")

        try:
            if verbose:
                logger.info(f"📥 Reading {file_type.upper()} from s3://{bucket_name}/{file_key}")

            # Download file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response["Body"].read()
            buffer = BytesIO(file_content)

            # Read based on file type
            if file_type == "csv":
                df = pd.read_csv(buffer, **read_kwargs)
                data_type = "DataFrame"

            elif file_type == "parquet":
                # Try reading as geoparquet first, fall back to regular parquet
                try:
                    df = gpd.read_parquet(buffer, **read_kwargs)
                    # Check if it actually has geometry
                    if "geometry" in df.columns and hasattr(df, "geometry"):
                        data_type = "GeoDataFrame"
                    else:
                        # No geometry, convert to regular DataFrame
                        df = pd.DataFrame(df)
                        data_type = "DataFrame"
                except Exception:
                    # Not a geoparquet, read as regular parquet
                    buffer.seek(0)  # Reset buffer position
                    df = pd.read_parquet(buffer, **read_kwargs)
                    data_type = "DataFrame"

            elif file_type == "geoparquet":
                df = gpd.read_parquet(buffer, **read_kwargs)
                data_type = "GeoDataFrame"

            else:
                raise ValueError(
                    f"Unsupported file type: {file_type}. Supported types: 'csv', 'parquet', 'geoparquet', 'auto'"
                )

            if verbose:
                logger.success(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns as {data_type}")

            return df

        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                logger.error(f"❌ File not found: s3://{bucket_name}/{file_key}")
                raise S3ObjectNotFound(bucket_name, file_key) from e
            else:
                logger.error(f"❌ S3 client error: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"❌ Error reading file from S3: {str(e)}")
            raise

    @staticmethod
    def read_csv(
        authenticator, bucket_name: str, file_key: str, verbose: bool = False, **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Read a CSV file from S3 into a pandas DataFrame.
        Convenience wrapper around read_from_s3.
        """
        return EDA_S3_cloudstore.read_from_s3(
            authenticator=authenticator,
            bucket_name=bucket_name,
            file_key=file_key,
            file_type="csv",
            verbose=verbose,
            **pandas_kwargs,
        )

    @staticmethod
    def read_parquet(
        authenticator, bucket_name: str, file_key: str, safe: bool = False, verbose: bool = False, **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Read a Parquet file from S3 into a pandas DataFrame.
        """
        return EDA_S3_cloudstore.read_from_s3(
            authenticator=authenticator,
            bucket_name=bucket_name,
            file_key=file_key,
            file_type="parquet",
            safe=safe,
            verbose=verbose,
            **pandas_kwargs,
        )

    @staticmethod
    def read_geoparquet(
        authenticator, bucket_name: str, file_key: str, safe: bool = False, verbose: bool = False, **geopandas_kwargs
    ) -> gpd.GeoDataFrame:
        """
        Read a GeoParquet file from S3 into a GeoPandas GeoDataFrame.
        """
        return EDA_S3_cloudstore.read_from_s3(
            authenticator=authenticator,
            bucket_name=bucket_name,
            file_key=file_key,
            file_type="geoparquet",
            safe=safe,
            verbose=verbose,
            **geopandas_kwargs,
        )

    # ========================================================================
    # EXISTENCE CHECK
    # ========================================================================
    @staticmethod
    def object_exists(authenticator, bucket_name: str, file_key: str) -> bool:
        """
        Return True if an object exists at the given key, False otherwise.

        Uses ``head_object`` (a metadata-only request, no body download) and maps
        a 404 / NoSuchKey to ``False``. Any other ClientError (permissions, etc.)
        propagates. Lets callers guard the first-run case without catching
        :class:`S3ObjectNotFound`:

            if EDA_S3_cloudstore.object_exists(auth, bucket, key):
                df = EDA_S3_cloudstore.read_csv(auth, bucket, key)

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to check

        Returns:
            bool: True if the key exists, False if it does not.
        """
        s3_client = authenticator.ensure_s3_client()
        try:
            s3_client.head_object(Bucket=bucket_name, Key=file_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            logger.error(f"❌ S3 client error checking s3://{bucket_name}/{file_key}: {str(e)}")
            raise

    # ========================================================================
    # WRITE METHODS
    # ========================================================================
    @staticmethod
    def write_to_s3(
        authenticator,
        df: pd.DataFrame,
        bucket_name: str,
        file_key: str,
        file_type: str = "auto",
        verbose: bool = False,
        **write_kwargs,
    ) -> str:
        """
        Write a DataFrame to S3 as CSV or Parquet using the boto3 client.

        Symmetric counterpart to :meth:`read_from_s3` — serializes to an in-memory
        buffer and ``put_object``s it on the same ``authenticator.ensure_s3_client()``
        client, so it works without the optional ``[s3]``/s3fs extra.

        ``index=False`` is the default (matching the wheel's export convention);
        pass ``index=True`` in ``write_kwargs`` to override.

        Args:
            authenticator: EDAuthenticator instance with s3_client
            df: DataFrame (or GeoDataFrame) to write
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to write to
            file_type: 'csv', 'parquet', or 'auto' (infer from the key extension).
                       'auto' raises if the extension is neither .csv nor .parquet
                       — never a silent default. Default: 'auto'
            verbose: Whether to log progress messages. Default: False
            **write_kwargs: Extra args forwarded to ``DataFrame.to_csv`` /
                            ``DataFrame.to_parquet``

        Returns:
            str: The ``s3://bucket/key`` URI written.

        Raises:
            ClientError: If the S3 put fails
            ValueError: If file_type cannot be resolved or is unsupported
        """
        s3_client = authenticator.ensure_s3_client()

        # Resolve file type — mirror read_from_s3: infer from extension, never
        # silently default.
        if file_type == "auto":
            file_lower = file_key.lower()
            if file_lower.endswith(".csv"):
                file_type = "csv"
            elif file_lower.endswith(".parquet"):
                file_type = "parquet"
            else:
                raise ValueError(
                    f"Cannot auto-detect file type for: {file_key}. "
                    f"Supported extensions: .csv, .parquet. "
                    f"Or specify file_type explicitly."
                )

        write_kwargs.setdefault("index", False)

        if file_type == "csv":
            body = df.to_csv(**write_kwargs).encode("utf-8")
        elif file_type == "parquet":
            buffer = BytesIO()
            df.to_parquet(buffer, **write_kwargs)
            body = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: 'csv', 'parquet', 'auto'")

        if verbose:
            logger.info(f"📤 Writing {file_type.upper()} to s3://{bucket_name}/{file_key}")

        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=body)

        uri = f"s3://{bucket_name}/{file_key}"
        if verbose:
            logger.success(f"✅ Wrote {len(df)} rows, {len(df.columns)} columns to {uri}")
        return uri

    @staticmethod
    def write_csv(
        authenticator, df: pd.DataFrame, bucket_name: str, file_key: str, verbose: bool = False, **pandas_kwargs
    ) -> str:
        """
        Write a DataFrame to S3 as CSV. Convenience wrapper around write_to_s3.
        """
        return EDA_S3_cloudstore.write_to_s3(
            authenticator=authenticator,
            df=df,
            bucket_name=bucket_name,
            file_key=file_key,
            file_type="csv",
            verbose=verbose,
            **pandas_kwargs,
        )

    @staticmethod
    def write_parquet(
        authenticator, df: pd.DataFrame, bucket_name: str, file_key: str, verbose: bool = False, **pandas_kwargs
    ) -> str:
        """
        Write a DataFrame to S3 as Parquet. Convenience wrapper around write_to_s3.
        """
        return EDA_S3_cloudstore.write_to_s3(
            authenticator=authenticator,
            df=df,
            bucket_name=bucket_name,
            file_key=file_key,
            file_type="parquet",
            verbose=verbose,
            **pandas_kwargs,
        )

    @staticmethod
    def upload_file(authenticator, bucket_name: str, file_key: str, local_path: str, verbose: bool = False) -> str:
        """
        Upload an existing local file to S3 at the given key.

        Manifest-free counterpart to the publish path — for callers that just want
        to push a file (raster, report, archive) without generating the run
        manifest that ``publish_local_run_to_s3`` expects. Streams from disk via
        boto3's ``upload_file`` (no full read into memory).

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            file_key: S3 key (path) to write to
            local_path: Path to the local file to upload
            verbose: Whether to log progress messages. Default: False

        Returns:
            str: The ``s3://bucket/key`` URI written.

        Raises:
            ClientError: If the S3 upload fails
            FileNotFoundError: If local_path does not exist
        """
        s3_client = authenticator.ensure_s3_client()
        local_path = str(local_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if verbose:
            logger.info(f"📤 Uploading {local_path} → s3://{bucket_name}/{file_key}")

        s3_client.upload_file(local_path, bucket_name, file_key)

        uri = f"s3://{bucket_name}/{file_key}"
        if verbose:
            logger.success(f"✅ Uploaded to {uri}")
        return uri

    @staticmethod
    def upload_files(
        authenticator, bucket_name: str, local_paths: List[str], s3_prefix: str = "", verbose: bool = False
    ) -> List[str]:
        """
        Upload several local files under an S3 key prefix.

        Each file lands at ``<s3_prefix>/<filename>`` (filename = basename of the
        local path). Convenience loop over :meth:`upload_file`.

        Args:
            authenticator: EDAuthenticator instance with s3_client
            bucket_name: Name of the S3 bucket
            local_paths: Iterable of local file paths to upload
            s3_prefix: Key prefix to upload under (e.g. ``"runs/2026-06-12"``).
                       Empty string uploads to the bucket root. Default: ""
            verbose: Whether to log progress messages. Default: False

        Returns:
            List[str]: The ``s3://bucket/key`` URIs written, in input order.
        """
        prefix = s3_prefix.strip("/")
        uris = []
        for local_path in local_paths:
            filename = os.path.basename(str(local_path))
            file_key = f"{prefix}/{filename}" if prefix else filename
            uris.append(
                EDA_S3_cloudstore.upload_file(authenticator, bucket_name, file_key, local_path, verbose=verbose)
            )
        return uris

    # ========================================================================
    # FILTERED READ METHODS (WITH TEMP FILE)
    # ========================================================================
    @staticmethod
    def read_geoparquet_filtered(
        authenticator,
        bucket_name: str,
        file_key: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        wkt_filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        spatial_operation: str = "intersects",  # ← NEW
        buffer_distance: float = 0,  # ← NEW
        safe: bool = False,
        use_bbox_from_wkt: bool = True,
        verbose: bool = False,
        **read_kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Read GeoParquet from S3 with spatial filtering (uses temp file for bbox filtering).

        Args:
            authenticator: EDAuthenticator instance
            bucket_name: S3 bucket name
            file_key: S3 file key
            bbox: Bounding box filter (minx, miny, maxx, maxy)
            wkt_filter: WKT geometry to filter by
            columns: Columns to read
            spatial_operation: Spatial operation for WKT filter ('intersects', 'contains', 'within', 'overlaps', 'touches')
            buffer_distance: Buffer distance around WKT filter (in CRS units)
            safe: Validate GeoParquet
            use_bbox_from_wkt: If True, compute bbox from WKT for filtering (default True)
            verbose: Log progress

        Returns:
            gpd.GeoDataFrame: Filtered data
        """
        from shapely import wkt

        s3_client = authenticator.ensure_s3_client()

        # Validate if safe mode
        if safe:
            if verbose:
                logger.info("🔒 Safe mode enabled - validating GeoParquet...")
            is_geo = EDA_S3_cloudstore.is_geoparquet(authenticator, bucket_name, file_key, verbose=verbose)
            if not is_geo:
                raise ValueError(f"File is not a valid GeoParquet: {file_key}")
            if verbose:
                logger.success("   ✅ Validated as GeoParquet")

        try:
            # Compute bbox from WKT if requested and not already provided
            if wkt_filter and not bbox and use_bbox_from_wkt:
                if verbose:
                    logger.info("📐 Computing bounding box from WKT filter...")
                geom = wkt.loads(wkt_filter)
                bbox = geom.bounds
                if verbose:
                    logger.info(f"   Bbox: {bbox}")

            if verbose:
                logger.info(f"📥 Reading GeoParquet from s3://{bucket_name}/{file_key}")
                if bbox:
                    logger.info(f"   Bounding box filter: {bbox}")
                if columns:
                    logger.info(f"   Reading {len(columns)} columns only")

            # Download file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response["Body"].read()

            # For bbox filtering, we need to use a temporary file
            if bbox or columns:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name

                try:
                    read_params = dict(read_kwargs)
                    if bbox:
                        read_params["bbox"] = bbox
                    if columns:
                        read_params["columns"] = columns

                    gdf = gpd.read_parquet(tmp_path, **read_params)

                    if verbose:
                        logger.info(f"   Rows after bbox filter: {len(gdf)}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                buffer = BytesIO(file_content)
                gdf = gpd.read_parquet(buffer, **read_kwargs)

            # Apply exact geometry filter if WKT was provided
            # ← REPLACE THIS SECTION with the reusable function
            if wkt_filter:
                gdf = filter_gdf_by_spatial_relation(
                    gdf=gdf,
                    wkt_geometry=wkt_filter,
                    operation=cast(
                        Literal["intersects", "contains", "within", "overlaps", "touches"], spatial_operation
                    ),
                    buffer_distance=buffer_distance,
                    verbose=verbose,
                )

            if verbose:
                logger.success(f"✅ Loaded {len(gdf)} features, {len(gdf.columns)} columns")

            return gdf

        except Exception as e:
            logger.error(f"❌ Error reading filtered GeoParquet: {str(e)}")
            raise

    @staticmethod
    def read_geoparquet_duckdb_spatial(
        authenticator,
        bucket_name: str,
        file_key: str,
        wkt_filter: str = None,
        bbox: tuple = None,
        columns: list = None,
        spatial_predicate: str = "intersects",
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
        """Read and spatially filter GeoParquet using DuckDB's spatial extension."""
        import pyarrow.parquet as pq
        from shapely import wkt as shapely_wkt

        try:
            # Get the actual geometry column name from metadata
            # TODO: EDA_S3_cloudstore.get_s3_filesystem is undefined — broken path,
            # all three call sites need a real implementation. Tracked for cleanup.
            fs = EDA_S3_cloudstore.get_s3_filesystem(authenticator)  # type: ignore[attr-defined]
            with fs.open_input_file(f"{bucket_name}/{file_key}") as f:
                parquet_file = pq.ParquetFile(f)
                if b"geo" in parquet_file.schema_arrow.metadata:
                    import json

                    geo_metadata = json.loads(parquet_file.schema_arrow.metadata[b"geo"])
                    geom_col = geo_metadata.get("primary_column", "geometry")
                else:
                    geom_col = "geometry"

            # Get S3 credentials
            creds = authenticator.ensure_credentials()

            # Initialize DuckDB with spatial extension
            con = duckdb.connect()
            try:
                con.execute("INSTALL spatial; LOAD spatial;")
                con.execute("INSTALL httpfs; LOAD httpfs;")

                # Configure S3
                con.execute(f"""
                    SET s3_region='{creds.get("region", "us-east-1")}';
                    SET s3_access_key_id='{creds["aws_access_key_id"]}';
                    SET s3_secret_access_key='{creds["aws_secret_access_key"]}';
                """)

                if "aws_session_token" in creds:
                    con.execute(f"SET s3_session_token='{creds['aws_session_token']}';")

                s3_path = f"s3://{bucket_name}/{file_key}"

                # Build column selection
                col_select = "*" if not columns else ", ".join(columns)

                # Build spatial filter
                where_clause = ""
                if wkt_filter:
                    filter_geom = shapely_wkt.loads(wkt_filter)
                    wkt_str = filter_geom.wkt

                    where_clause = f"""
                        WHERE ST_{spatial_predicate}(
                            {geom_col},
                            ST_GeomFromText('{wkt_str}')
                        )
                    """
                elif bbox:
                    minx, miny, maxx, maxy = bbox
                    bbox_wkt = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"

                    where_clause = f"""
                        WHERE ST_{spatial_predicate}(
                            {geom_col},
                            ST_GeomFromText('{bbox_wkt}')
                        )
                    """

                query = f"""
                    SELECT {col_select}
                    FROM read_parquet('{s3_path}')
                    {where_clause}
                """

                if verbose:
                    logger.info(f"🦆 DuckDB spatial query on column '{geom_col}'")

                # Execute and convert to GeoDataFrame
                df = con.execute(query).df()

                if len(df) == 0:
                    if verbose:
                        logger.warning("⚠️  No features matched the spatial filter")
                    return gpd.GeoDataFrame()

                # Convert to GeoDataFrame with proper geometry column
                gdf = gpd.GeoDataFrame(df, geometry=geom_col)

                if verbose:
                    logger.success(f"✓ Read {len(gdf)} filtered features via DuckDB")

                return gdf

            finally:
                con.close()  # ← CRITICAL: Close connection

        except Exception as e:
            if verbose:
                logger.error(f"❌ DuckDB spatial filtering failed: {e}")
            raise

    @staticmethod
    def read_geoparquet_rowgroup_filtered(
        authenticator, bucket_name: str, file_key: str, bbox: tuple = None, columns: list = None, verbose: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Filter by row groups using statistics.
        NOTE: This only works if the GeoParquet has a bbox covering column with statistics.
        Most files won't have this.
        """
        import pyarrow.parquet as pq
        from shapely.geometry import box

        # TODO: EDA_S3_cloudstore.get_s3_filesystem is undefined — see other call sites.
        fs = EDA_S3_cloudstore.get_s3_filesystem(authenticator)  # type: ignore[attr-defined]

        try:
            with fs.open_input_file(f"{bucket_name}/{file_key}") as f:
                parquet_file = pq.ParquetFile(f)

                # Get geometry column name
                if b"geo" in parquet_file.schema_arrow.metadata:
                    import json

                    geo_metadata = json.loads(parquet_file.schema_arrow.metadata[b"geo"])
                    geom_col = geo_metadata.get("primary_column", "geometry")

                    # Check if there's a bbox covering column
                    column_meta = geo_metadata.get("columns", {}).get(geom_col, {})
                    bbox_covering = column_meta.get("covering", {}).get("bbox")

                    if not bbox_covering:
                        if verbose:
                            logger.warning("⚠️  No bbox covering - cannot use row group filtering")
                        raise ValueError("No bbox covering available")

                    # Use the bbox covering column for filtering
                    bbox_col_name = bbox_covering.get("xmin", [None])[0]  # Get column name

                    if not bbox_col_name:
                        raise ValueError("Cannot determine bbox column names")

                # For now, just read everything - row group filtering is complex
                # and requires bbox covering columns which most files don't have
                if verbose:
                    logger.warning("⚠️  Row group filtering not implemented for this file type")

                gdf = gpd.read_parquet(f, columns=columns, filesystem=fs)

                # Apply spatial filter in memory
                if bbox and len(gdf) > 0:
                    bbox_geom = box(*bbox)
                    gdf = gdf[gdf.geometry.intersects(bbox_geom)]

                    if verbose:
                        logger.info(f"✓ Filtered to {len(gdf)} features in memory")

                return gdf

        except Exception as e:
            if verbose:
                logger.warning(f"⚠️  Row group filtering failed: {e}")
            raise

    @staticmethod
    def read_geoparquet_hybrid_filtered(
        authenticator, bucket_name: str, file_key: str, bbox: tuple = None, columns: list = None, verbose: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Two-pass read: First read only geometry bounds, then full data for matches.
        Only works if file has separate bbox columns.
        """
        import pyarrow.parquet as pq  # ← ADD THIS

        # TODO: EDA_S3_cloudstore.get_s3_filesystem is undefined — see other call sites.
        fs = EDA_S3_cloudstore.get_s3_filesystem(authenticator)  # type: ignore[attr-defined]

        try:
            with fs.open_input_file(f"{bucket_name}/{file_key}") as f:
                parquet_file = pq.ParquetFile(f)
                schema = parquet_file.schema_arrow

                # Look for bbox columns
                bbox_cols = [col for col in schema.names if col in ["xmin", "ymin", "xmax", "ymax", "bbox", "bounds"]]

                if not bbox_cols:
                    if verbose:
                        logger.warning("⚠️  No bbox columns found - cannot use hybrid method")
                    raise ValueError("No bbox columns available")

                if bbox:
                    if verbose:
                        logger.info(f"📋 Pass 1: Checking bounds using columns: {bbox_cols}")

                    # Read geometry + bbox columns only
                    # Note: This reads ALL rows but fewer columns
                    table = parquet_file.read(columns=["geometry"] + bbox_cols)
                    df = table.to_pandas()

                    minx, miny, maxx, maxy = bbox

                    # Filter based on bbox columns if available
                    if all(col in df.columns for col in ["xmin", "ymin", "xmax", "ymax"]):
                        mask = (df["xmax"] >= minx) & (df["xmin"] <= maxx) & (df["ymax"] >= miny) & (df["ymin"] <= maxy)
                        filtered_df = df[mask]
                    else:
                        # Extract bounds from geometry
                        gdf_temp = gpd.GeoDataFrame(df, geometry="geometry")
                        bounds = gdf_temp.geometry.bounds
                        mask = (
                            (bounds["maxx"] >= minx)
                            & (bounds["minx"] <= maxx)
                            & (bounds["maxy"] >= miny)
                            & (bounds["miny"] <= maxy)
                        )
                        filtered_df = df[mask]

                    if verbose:
                        logger.info(f"✓ Filtered to {len(filtered_df)} features")

                    # Now read full columns for filtered rows
                    # NOTE: This requires re-reading - not truly optimized
                    if len(filtered_df) > 0:
                        # We already have the data, just select columns if needed
                        if columns:
                            # Need to re-read with specific columns
                            # This is where the method breaks down - we'd need row indices
                            # but parquet doesn't support efficient row-based filtering
                            table = parquet_file.read(columns=columns)
                            full_df = table.to_pandas().iloc[filtered_df.index]
                            gdf = gpd.GeoDataFrame(full_df, geometry="geometry")
                        else:
                            gdf = gpd.GeoDataFrame(filtered_df, geometry="geometry")
                            if columns:
                                gdf = gdf[columns]
                    else:
                        gdf = gpd.GeoDataFrame()

                    return gdf

        except Exception as e:
            if verbose:
                logger.warning(f"⚠️  Hybrid read failed: {e}")
            raise

    # ========================================================================
    # WRAPPER AROUND READING WITH FILTERING
    # ========================================================================
    @staticmethod
    def read_geoparquet_filtered_smart(
        authenticator,
        bucket_name: str,
        file_key: str,
        bbox: tuple = None,
        wkt_filter: str = None,
        columns: list = None,
        spatial_operation: str = "intersects",  # ← NEW
        buffer_distance: float = 0,  # ← NEW
        safe: bool = True,
        use_duckdb: bool = True,
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Smartly read and filter GeoParquet using the best available method.

        Args:
            spatial_operation: Spatial operation for WKT filter ('intersects', 'contains', 'within', etc.)
            buffer_distance: Buffer distance around WKT filter (in CRS units)
        """
        from shapely import wkt as shapely_wkt

        # Convert WKT to bbox if needed
        filter_geometry = None
        if wkt_filter:
            filter_geometry = shapely_wkt.loads(wkt_filter)
            if not bbox:
                bbox = filter_geometry.bounds

        # Check file capabilities. check_bbox=True branch always returns a dict;
        # narrow it explicitly so subsequent indexing typechecks.
        result = EDA_S3_cloudstore.is_geoparquet(authenticator, bucket_name, file_key, check_bbox=True, verbose=verbose)
        assert isinstance(result, dict)

        if not result["is_geoparquet"]:
            raise ValueError(f"File is not a valid GeoParquet: {file_key}")

        supports_bbox = result["supports_bbox"]

        # Strategy 1: Try DuckDB spatial if enabled and filter provided
        # Note: DuckDB only supports basic intersects, so skip if using other operations
        if use_duckdb and (wkt_filter or bbox) and spatial_operation == "intersects":
            try:
                if verbose:
                    logger.info("🦆 Strategy 1: DuckDB spatial query...")
                return EDA_S3_cloudstore.read_geoparquet_duckdb_spatial(
                    authenticator=authenticator,
                    bucket_name=bucket_name,
                    file_key=file_key,
                    wkt_filter=wkt_filter,
                    bbox=bbox,
                    columns=columns,
                    spatial_predicate="intersects",
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️  DuckDB failed: {e}, trying next method...")

        # Strategy 2: Use native bbox filtering if supported
        if supports_bbox and bbox:
            try:
                if verbose:
                    logger.info(f"📦 Strategy 2: Native bbox filtering (method: {result['bbox_method']})...")
                return EDA_S3_cloudstore.read_geoparquet_filtered(
                    authenticator=authenticator,
                    bucket_name=bucket_name,
                    file_key=file_key,
                    bbox=bbox,
                    wkt_filter=wkt_filter,
                    columns=columns,
                    spatial_operation=spatial_operation,  # ← Pass through
                    buffer_distance=buffer_distance,  # ← Pass through
                    safe=safe,
                    use_bbox_from_wkt=True,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️  Native bbox filtering failed: {e}, falling back...")

        # Strategy 3: Full read + in-memory filter (fallback)
        if verbose:
            logger.info("🔍 Strategy 3: Full read with in-memory filtering...")

        gdf = EDA_S3_cloudstore.read_geoparquet_filtered(
            authenticator=authenticator,
            bucket_name=bucket_name,
            file_key=file_key,
            bbox=None,
            wkt_filter=wkt_filter,
            columns=columns,
            spatial_operation=spatial_operation,  # ← Pass through
            buffer_distance=buffer_distance,  # ← Pass through
            safe=safe,
            use_bbox_from_wkt=False,
            verbose=verbose,
        )

        if verbose and len(gdf) > 0:
            logger.success(f"✓ Final result: {len(gdf)} features")

        return gdf
