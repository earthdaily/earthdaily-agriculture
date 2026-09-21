# functions_enhanced.py - Addition of geometry support function

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tqdm import tqdm

# Import necessary functions from api_utils

# from earthdaily.agriculture.extractors.coverage_function import CoverageExtractor
# Import necessary functions from entity_management

# BOTH FUNCTIONS TO BE REWORKED UNDER THE NEW PROJECT STRUCTURE

# def process_bulk_download_with_geometry(sfd_list, env, bearer_token, vegetation_index, start_date, clear_cover_min, partials_path, map_format, filename_sanitizer=None, use_specific_date=False):
#     """
#     Enhanced bulk download with geometry support for file-based data sources
#     """
#     from tqdm import tqdm
#     import time

#     results = {
#         'downloaded': 0,
#         'skipped': 0,
#         'errors': [],
#         'no_coverage': 0,
#         'success_list': []
#     }

#     print(f"🌍 Processing {len(sfd_list)} seasonfields with geometry-based coverage detection...")
#     if use_specific_date:
#         print(f"📅 Looking for images for this specific date: {start_date}")
#     else:
#         print(f"📅 Looking for images after: {start_date}")
#     print(f"☁️ Minimum clear coverage: {clear_cover_min}%\n")

#     start_time = time.time()

#     for i, row in tqdm(sfd_list.iterrows(), total=len(sfd_list), desc="📥 Downloading", unit="field"):
#         try:
#             sfd_id = row.get('id', f'geom_{i}')
#             sfd_name = row.get('name', f'Field_{i}')
#             grower = row.get('field.farm.grower.companyName', row.get('grower', 'Unknown_Grower'))
#             crop = row.get('crop.id', row.get('crop', 'Unknown_Crop'))

#             # Obtenir la géométrie au format WKT
#             geometry = row.get('geometry')
#             if geometry is None:
#                 results['errors'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'error_message': 'No geometry found',
#                     'image_id': 'N/A',
#                     'image_date': 'N/A'
#                 })
#                 continue

#             # Convertir la géométrie en WKT si nécessaire
#             if hasattr(geometry, 'wkt'):
#                 geometry_wkt = geometry.wkt
#             elif isinstance(geometry, str):
#                 geometry_wkt = geometry
#             else:
#                 geometry_wkt = str(geometry)

#             # Get satellite coverage using geometry
#             coverage = get_satellite_coverage_by_geometry(
#                 env,
#                 bearer_token,
#                 geometry_wkt,
#                 vegetation_index,
#                 start_date,
#                 clear_cover_min=clear_cover_min,
#                 use_specific_date=use_specific_date
#             )

#             if coverage.empty:
#                 results['no_coverage'] += 1
#                 continue

#             # Get most recent image
#             coverage = coverage.sort_values('image_date', ascending=False)
#             latest = coverage.iloc[0]

#             image_id = latest['image_id']
#             image_date = latest['image_date']

#             # Format date for filename
#             formatted_date = datetime.strptime(image_date[:10], "%Y-%m-%d").strftime("%d%m%Y")

#             # Sanitize filename components
#             if filename_sanitizer:
#                 grower_clean = filename_sanitizer(str(grower))
#                 sfd_name_clean = filename_sanitizer(str(sfd_name))
#                 image_id_upper = image_id.upper()
#                 # Include the seasonfield ID for guaranteed uniqueness
#                 tif_filename = f"{grower_clean}_{sfd_name_clean}_{sfd_id}_{image_id_upper}_{vegetation_index}_{formatted_date}"
#             else:
#                 # Fallback to basic cleaning
#                 grower_clean = str(grower).replace('|', '_').replace(' ', '_')
#                 sfd_name_clean = str(sfd_name).replace('|', '_').replace(' ', '_')
#                 image_id_upper = image_id.upper().replace('|', '_').replace(' ', '_')
#                 tif_filename = f"{grower_clean}_{sfd_name_clean}_{image_id_upper}_{vegetation_index}_{formatted_date}"

#             # Download the file using geometry-based API
#             result = download_map_with_geometry(
#                 env, bearer_token, geometry_wkt, image_id, vegetation_index,
#                 partials_path, map_format, tif_filename
#             )

#             # Generate final filename for tracking
#             import re

#             if map_format == 'tiff.zip':
#                 final_filename = re.sub(r'[<>:"/\\|?*]', '_', tif_filename) + '.tiff'
#             else:
#                 final_filename = re.sub(r'[<>:"/\\|?*]', '_', tif_filename) + '.png'

#             if result == "downloaded":
#                 results['downloaded'] += 1
#                 results['success_list'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'status': 'downloaded',
#                     'filename': final_filename,
#                     'image_date': image_date,
#                     'coverage_percent': latest.get('coveragePercent', 100)
#                 })
#             elif result == "skipped":
#                 results['skipped'] += 1
#                 results['success_list'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'status': 'skipped',
#                     'filename': final_filename,
#                     'image_date': image_date
#                 })
#             else:
#                 results['errors'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'error_message': result,
#                     'image_id': image_id,
#                     'image_date': image_date
#                 })

#         except Exception as e:
#             results['errors'].append({
#                 'sfd_id': row.get('id', 'unknown'),
#                 'sfd_name': row.get('name', 'unknown'),
#                 'grower': row.get('field.farm.grower.companyName', row.get('grower', 'unknown')),
#                 'crop': row.get('crop.id', row.get('crop', 'unknown')),
#                 'error_message': str(e),
#                 'image_id': 'N/A',
#                 'image_date': 'N/A'
#             })

#     elapsed_time = time.time() - start_time
#     print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")

#     return results

# def process_bulk_download_enhanced(sfd_list, env, bearer_token, vegetation_index, start_date, clear_cover_min, partials_path, map_format, filename_sanitizer=None, use_specific_date=False):
#     """
#     Enhanced bulk download with better filename handling (original function for ID-based data)
#     """
#     from tqdm import tqdm
#     import time

#     results = {
#         'downloaded': 0,
#         'skipped': 0,
#         'errors': [],
#         'no_coverage': 0,
#         'success_list': []
#     }

#     print(f"🌍 Processing {len(sfd_list)} seasonfields...")
#     if use_specific_date:
#         print(f"📅 Looking for images for this specific date: {start_date}")
#     else:
#         print(f"📅 Looking for images after: {start_date}")
#     print(f"☁️ Minimum clear coverage: {clear_cover_min}%\n")

#     start_time = time.time()

#     for i, row in tqdm(sfd_list.iterrows(), total=len(sfd_list), desc="📥 Downloading", unit="field"):
#         try:
#             sfd_id = row['id']
#             sfd_name = row['name']
#             grower = row.get('field.farm.grower.companyName', 'Unknown_Grower')
#             crop = row.get('crop.id', 'Unknown_Crop')

#             # Get satellite coverage
#             coverage = get_satellite_coverage(
#                 env,
#                 bearer_token,
#                 sfd_id,
#                 vegetation_index,
#                 start_date,
#                 clear_cover_min=clear_cover_min,
#                 use_specific_date=use_specific_date
#             )

#             if coverage.empty:
#                 results['no_coverage'] += 1
#                 continue

#             # Get most recent image
#             coverage = coverage.sort_values('image_date', ascending=False)
#             latest = coverage.iloc[0]

#             image_id = latest['image_id']
#             image_date = latest['image_date']

#             # Format date for filename
#             formatted_date = datetime.strptime(image_date[:10], "%Y-%m-%d").strftime("%d%m%Y")

#             # Sanitize filename components
#             if filename_sanitizer:
#                 grower_clean = filename_sanitizer(grower)
#                 sfd_name_clean = filename_sanitizer(sfd_name)
#                 image_id_upper = image_id.upper()
#                 vegetation_index = vegetation_index
#                 # Include the seasonfield ID for guaranteed uniqueness
#                 tif_filename = f"{grower_clean}_{sfd_name_clean}_{sfd_id}_{image_id_upper}_{vegetation_index}_{formatted_date}"
#             else:
#                 # Fallback to basic cleaning
#                 grower_clean = grower.replace('|', '_').replace(' ', '_')
#                 sfd_name_clean = sfd_name.replace('|', '_').replace(' ', '_')
#                 image_id_upper = image_id.upper().replace('|', '_').replace(' ', '_')
#                 tif_filename = f"{grower_clean}_{sfd_name_clean}_{image_id_upper}_{vegetation_index}_{formatted_date}"

#             # Download the file
#             result = download_map_with_skip(env, bearer_token, sfd_id, image_id, vegetation_index, partials_path, map_format, tif_filename)

#             # Generate final filename for tracking
#             import re

#             if map_format == 'tiff.zip':
#                 final_filename = re.sub(r'[<>:"/\\|?*]', '_', tif_filename) + '.tiff'
#             else:
#                 final_filename = re.sub(r'[<>:"/\\|?*]', '_', tif_filename) + '.png'

#             if result == "downloaded":
#                 results['downloaded'] += 1
#                 results['success_list'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'status': 'downloaded',
#                     'filename': final_filename,
#                     'image_date': image_date,
#                     'coverage_percent': latest.get('coveragePercent', 100)
#                 })
#             elif result == "skipped":
#                 results['skipped'] += 1
#                 results['success_list'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'status': 'skipped',
#                     'filename': final_filename,
#                     'image_date': image_date
#                 })
#             else:
#                 results['errors'].append({
#                     'sfd_id': sfd_id,
#                     'sfd_name': sfd_name,
#                     'grower': grower,
#                     'crop': crop,
#                     'error_message': result,
#                     'image_id': image_id,
#                     'image_date': image_date
#                 })

#         except Exception as e:
#             results['errors'].append({
#                 'sfd_id': row.get('id', 'unknown'),
#                 'sfd_name': row.get('name', 'unknown'),
#                 'grower': row.get('field.farm.grower.companyName', 'unknown'),
#                 'crop': row.get('crop.id', 'unknown'),
#                 'error_message': str(e),
#                 'image_id': 'N/A',
#                 'image_date': 'N/A'
#             })

#     elapsed_time = time.time() - start_time
#     print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")

#     return results


def save_error_reports(results, partials_path):
    """Save detailed error reports"""
    import os

    import pandas as pd

    today = datetime.today().strftime("%m%d%Y")

    # Save errors to CSV
    if results.get("errors"):
        error_df = pd.DataFrame(results["errors"])
        error_csv = os.path.join(partials_path, f"download_errors_{today}.csv")
        error_df.to_csv(error_csv, index=False)
        print(f"📄 Error report saved to: {error_csv}")

    # Save successful downloads to CSV
    if results.get("success_list"):
        success_df = pd.DataFrame(results["success_list"])
        success_csv = os.path.join(partials_path, f"successful_downloads_{today}.csv")
        success_df.to_csv(success_csv, index=False)
        print(f"📄 Success report saved to: {success_csv}")


def retry_failed_downloads(
    error_csv_path,
    env,
    bearer_token,
    vegetation_index,
    start_date,
    clear_cover_min,
    partials_path,
    map_format,
    filename_sanitizer=None,
    use_specific_date=False,
):
    """Retry downloads that failed previously with improved filename handling"""
    import os
    from datetime import datetime

    if not os.path.exists(error_csv_path):
        print(f"❌ Error CSV not found: {error_csv_path}")
        return None

    df_errors = pd.read_csv(error_csv_path)
    print(f"🔄 Retrying {len(df_errors)} failed downloads...")

    results = {"downloaded": 0, "errors": [], "success_list": []}

    for i, row in tqdm(df_errors.iterrows(), total=len(df_errors), desc="🔁 Retrying", unit="field"):
        try:
            sfd_id = row["sfd_id"]
            sfd_name = row["sfd_name"]
            grower = row["grower"]
            crop = row["crop"]
            image_id = row["image_id"]
            image_date = row["image_date"]
            map_format = row["map_format"]

            formatted_date = datetime.strptime(image_date[:10], "%Y-%m-%d").strftime("%d%m%Y")

            # Use the same filename sanitization logic
            if filename_sanitizer:
                grower_clean = filename_sanitizer(grower)
                sfd_name_clean = filename_sanitizer(sfd_name)
                tif_filename = f"{grower_clean}_{sfd_name_clean}_{sfd_id}_{formatted_date}"
            else:
                grower_clean = grower.replace("|", "_").replace(" ", "_")
                sfd_name_clean = sfd_name.replace("|", "_").replace(" ", "_")
                tif_filename = f"{grower_clean}_{sfd_name_clean}_{formatted_date}"

            # NOTE: download_map_with_skip was never re-exported here after a refactor;
            # `final_filename` / `filename` likewise reference values that were
            # populated by the old helper. This whole retry path is dead code
            # pending rewrite — keep ruff quiet via per-line noqa.
            result = download_map_with_skip(  # noqa: F821
                env, bearer_token, sfd_id, image_id, vegetation_index, partials_path, tif_filename
            )

            if map_format == "tiff.zip":
                final_filename = re.sub(r'[<>:"/\\|?*]', "_", tif_filename) + ".tiff"  # noqa: F841
            else:
                filename = re.sub(r'[<>:"/\\|?*]', "_", tif_filename) + ".png"  # noqa: F841

            if result == "downloaded":
                results["downloaded"] += 1
                results["success_list"].append(
                    {
                        "sfd_id": sfd_id,
                        "sfd_name": sfd_name,
                        "grower": grower,
                        "crop": crop,
                        "status": "downloaded",
                        "filename": filename,
                        "image_date": image_date,
                    }
                )
            elif "error" in result:
                results["errors"].append(
                    {
                        "sfd_id": sfd_id,
                        "sfd_name": sfd_name,
                        "grower": grower,
                        "crop": crop,
                        "error_message": result,
                        "image_id": image_id,
                        "image_date": image_date,
                    }
                )

        except Exception as e:
            results["errors"].append(
                {
                    "sfd_id": row.get("sfd_id", "unknown"),
                    "sfd_name": row.get("sfd_name", "unknown"),
                    "grower": row.get("grower", "unknown"),
                    "crop": row.get("crop", "unknown"),
                    "error_message": str(e),
                    "image_id": row.get("image_id", "N/A"),
                    "image_date": row.get("image_date", "N/A"),
                }
            )

    return results


def print_summary(results):
    """Print download summary with English messages"""
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully downloaded: {results.get('downloaded', 0)} files")
    print(f"⏭️  Already existed (skipped): {results.get('skipped', 0)} files")
    print(f"❌ Failed downloads: {len(results.get('errors', []))} files")
    print(f"❓ No coverage found: {results.get('no_coverage', 0)} fields")
    print("=" * 60)


def _resolve_project_root():
    """
    Resolve the project root directory.

    Walks up from this file's location to find the directory containing pyproject.toml.
    If not found (e.g. package installed via wheel into site-packages), walks up from
    the current working directory instead.

    Returns:
        str: Absolute path to the project root directory.
    """
    # Strategy 1: walk up from this file's location
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return str(parent)

    # Strategy 2: walk up from cwd (handles installed-package scenarios)
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return str(parent)

    # Fallback: use cwd
    logger.warning(f"Could not find pyproject.toml from __file__ or cwd. Using cwd as project root: {cwd}")
    return str(cwd)


def setup_environment(env: str = None, project_root: str = None):
    """Setup environment with logging.

    Args:
        env: Environment override ('prod' or 'preprod'). If None, reads from ENVIRONMENT env var.
        project_root: Override for workspace root (where results/, partials/, etc. are created).
            If None, resolves automatically via pyproject.toml lookup.
    """
    # Resolve the code project root (where pyproject.toml / src/.env live)
    code_root = _resolve_project_root()

    # Load .env with priority:
    #   project_root/.env > project_root/src/.env > code_root/src/.env
    #   > cwd/.env > upward search from cwd
    #
    # The last two matter for projects generated from the cookiecutter template:
    # they have no pyproject.toml, so _resolve_project_root() falls back to cwd and
    # code_root/src/.env does not exist. The previous fallback was a bare
    # load_dotenv(), which calls find_dotenv(usecwd=False) and walks up from the
    # *calling frame's* file — i.e. from inside site-packages for a wheel install —
    # so it never reached the user's project and raised "Missing required
    # environment variables" with a perfectly good .env sitting in cwd. (It happened
    # to work in notebooks only because dotenv treats IPython as interactive and
    # uses cwd; scripts and containers got the broken path.)
    env_candidates = []
    if project_root:
        workspace_root = str(Path(project_root).resolve())
        env_candidates += [
            os.path.join(workspace_root, ".env"),
            os.path.join(workspace_root, "src", ".env"),
        ]
    env_candidates += [
        os.path.join(code_root, "src", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]

    env_loaded_from = None
    for env_candidate in env_candidates:
        if os.path.isfile(env_candidate):
            load_dotenv(env_candidate)
            env_loaded_from = env_candidate
            break

    if env_loaded_from is None:
        # Nothing at a known location — search upward from cwd (not from this
        # file, which is the whole point of usecwd=True).
        discovered = find_dotenv(usecwd=True)
        if discovered:
            load_dotenv(discovered)
            env_loaded_from = discovered

    # Log the resolved path: silence here is what made the failure above hard to
    # diagnose — a missing credential looked like a bad .env rather than an
    # unread one.
    if env_loaded_from:
        logger.info(f"🔑 Loaded credentials from: {env_loaded_from}")
    else:
        logger.warning(
            f"No .env file found (searched {len(env_candidates)} known locations and upward from {os.getcwd()}). "
            "Relying on ambient environment variables."
        )

    # Use explicit env parameter if provided, otherwise fall back to ENVIRONMENT env var
    env_value = env if env else os.getenv("ENVIRONMENT", "prod")

    # Determine the env prefix for credential validation
    env_prefix = env_value.upper().replace("-", "")  # 'preprod' -> 'PREPROD'

    # Validate credentials: check env-prefixed vars first, then non-prefixed
    credential_vars = ["API_USERNAME", "API_PASSWORD", "API_CLIENT_ID", "API_CLIENT_SECRET"]
    missing_vars = []
    for var in credential_vars:
        if not os.getenv(f"{env_prefix}_{var}") and not os.getenv(var):
            missing_vars.append(f"{env_prefix}_{var} (or {var})")

    if missing_vars:
        logger.error(f"❌ Missing environment variables for {env_value}: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        raise ValueError("Missing required environment variables")

    logger.success("✅ Environment variables loaded successfully")

    # Use explicit project_root if provided, otherwise use code_root
    if project_root:
        project_root = str(Path(project_root).resolve())
    else:
        project_root = code_root

    required_folders = {
        "partials": "partial_result_dir",
        "results": "output_result_dir",
        "inputs": None,
        "logs": None,
        "cache": "cache_dir",
    }

    config = {
        "env": env_value,
        "project_root": project_root,
    }

    # Create/check folders at project root level
    for folder, alias in required_folders.items():
        folder_path = os.path.join(project_root, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"📂 Created missing folder: {folder_path}")
        else:
            logger.debug(f"✅ Folder already exists: {folder_path}")

        # Test write access
        test_file = os.path.join(folder_path, "test_file.txt")
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            logger.debug(f"📝 Write access confirmed in {folder_path}")
        except Exception as e:
            logger.error(f"❌ Cannot write to {folder_path}: {e}")
            raise

        # Assign alias if defined (store absolute path)
        if alias:
            config[alias] = folder_path

    return config


# def setup_environment():
#     """Setup environment with English messages"""
#     from dotenv import load_dotenv
#     import os

#     load_dotenv()

#     # Validate environment variables
#     required_vars = ['API_USERNAME', 'API_PASSWORD', 'ENVIRONMENT', 'API_CLIENT_ID', 'API_CLIENT_SECRET']
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
#     env_value = os.getenv('ENVIRONMENT', 'production')

#     if missing_vars:
#         print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
#         print("Please check your .env file")
#         raise ValueError("Missing required environment variables")

#     # Validate environment variables

#     print("✅ Environment variables loaded successfully")

#     required_folders = {
#         "partials": "partial_result_dir",
#         "results": "output_result_dir",
#         "inputs": None
#     }

#     config = {"env": env_value}

#     # Create/check folders
#     for folder, alias in required_folders.items():
#         if not os.path.exists(folder):
#             os.makedirs(folder, exist_ok=True)
#             print(f"📂 Created missing folder: {folder}")
#         else:
#             print(f"✅ Folder already exists: {folder}")

#         # Test write access
#         test_file = os.path.join(folder, "test_file.txt")
#         with open(test_file, "w") as f:
#             f.write("test")
#         os.remove(test_file)  # clean up
#         print(f"📝 Write access confirmed in {folder}")

#         # Assign alias if defined
#         if alias:
#             config[alias] = folder

#     return config
