<div id="top"></div>
<!-- PROJECT SHIELDS -->
<!--
*** See the bottom of this document for the declaration of the reference variables
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/earthdaily">
    <img src="https://earthdaily.com/hs-fs/hubfs/3.%20Logos%20all/EarthDaily%20Logos/EDA_logo_main.png?width=240&height=113&name=EDA_logo_main.png" alt="Logo" width="400" height="200">
  </a>

  <h1 align="center">earthdaily-agriculture</h1>

  <p align="center">
    Python client for bulk extraction of agricultural analytics — vegetation indices, weather, crop identification, disease risk, harvest detection, scoring and more — through the EarthDaily Agriculture API.
    <br />
    <a href="https://earthdaily.com/ag"><strong>Who we are</strong></a>
    <br />
    <br />
    <a href="https://github.com/earthdaily/earthdaily-agriculture">Project description</a>
    ·
    <a href="https://github.com/earthdaily/earthdaily-agriculture/issues">Report Bug</a>
    ·
    <a href="https://github.com/earthdaily/earthdaily-agriculture/issues">Request Feature</a>
  </p>
</p>


<div align="center">

[![CI][ci-shield]][ci-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Twitter][twitter-shield]][twitter-url]
[![Youtube][youtube-shield]][youtube-url]
[![PyPI][pypi-shield]][pypi-url]
[![Python][language-python-shield]][language-python-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

</div>


<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#features">Features</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#run-the-package-from-source">Run the package from source</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#support-development">Support development</a></li>
    <li><a href="#resources">Resources</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#copyrights">Copyrights</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

EarthDaily Agriculture is the agriculture focused analytic division of EarthDaily Analytics. Learn more about EarthDaily at [EarthDaily Analytics | Satellite imagery & Analytics for agriculture, insurance, surveillance](https://earthdaily.com/). EarthDaily Agriculture uses satellite imagery to provide advanced analytics that mitigate risk and increase efficiencies — leading to more sustainable outcomes for the organizations and people who feed the planet.

<p align="left">
Through the EarthDaily Agriculture services, we make geospatial analytics easy to browse and analyze, in our cloud or in your own environment. We give developers and data scientists flexibility and extensibility with analytic-ready data and digital-ag ready building blocks, allowing you to enrich your solutions with information at field, regional or continental level via our APIs and apps.
</p>

We have a team of experts around the world that understand local crops and the ag industry, as well as advanced analytics to support your business.

The `earthdaily-agriculture` Python package provides a ready-to-use library that lets any Python developer quickly experience EarthDaily Agriculture capabilities. It wraps the full analytics catalog, coverage, time series, weather, crop identification, emergence, harvest, scoring, sustainability, regional — behind a single consistent extractor API designed for bulk processing.

<p align="right">(<a href="#top">back to top</a>)</p>

## Features

* **One consistent API across every extractor.** Every extractor inherits from `BaseExtractor` and follows the same `setup_<type>_parameters()` → `process_<type>_bulk_parallel()` lifecycle. Learn one, you know them all.
* **Bulk parallel extraction.** Configurable worker count, progress bars, partial saves, retry-on-failure, and an HTML run report out of the box (`generate_report=True`).
* **Full analytics catalog.** Foundational analytics (coverage, vegetation time series, weather, field-level maps, crop ID), crop-development signals (greenness, emergence, harvest, planted area, standing crop, disease, change index, in-season monitoring), risk scoring (historical & in-season score, ZARC), sustainability (bare soil, cover crop, tillage), and regional aggregates.
* **Flexible entity sources.** Pull entities from the EarthDaily Agriculture platform or from any GeoDataFrame / CSV / parquet of your own — a column-mapping system bridges naming differences.
* **Workflow orchestration.** Chain multiple extractors via a YAML workflow file; `WorkflowManager` runs the DAG with token refresh, partials, caching, and per-run reporting.
* **Local result cache.** Per-extractor parquet cache keyed by parameter signature, skipping redundant API calls across reruns.
* **Cloud-friendly output.** Route results, partials, and logs to S3 or Azure Storage (or any S3-compatible store) via `EDAGRO_OUTPUT_PREFIX` or constructor kwargs.
* **Entity & user management.** `EntityManager` and `UserManager` cover the EarthDaily Agriculture MDM API (Farm / Field / Seasonfield, growers, agronomists, account linking).

See the [documentation](https://docs.earthdaily.com/agro/) and the notebooks under [`notebooks/`](https://github.com/earthdaily/earthdaily-agriculture/tree/main/notebooks) for working examples.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting started

### Prerequisites

Make sure you have valid credentials. If you need to get trial access, please register [here](https://earthdaily.com/ag-contact).

This package requires Python **3.10 or newer** and is tested on 3.10, 3.11, 3.12 and 3.13. Python **3.12.x** is recommended and is the primary CI target.

### Installation

#### Conda

If you are using Conda, create and activate a virtual environment first:

```
conda create --name edagro python=3.12
conda activate edagro
```

#### For Linux / Mac OS / Windows

```
pip install earthdaily-agriculture
```

The package is pure-Python and installs the same way on Linux, macOS, and Windows. For an editable / development install with notebook extras:

```
pip install -e ".[test,jupyter]"
```

### Run the package from source

1. Clone and install dependencies

```
git clone https://github.com/earthdaily/earthdaily-agriculture.git
cd earthdaily-agriculture
pip install -r requirements.txt
```

2. Create a `.env` file

You need a `.env` file (e.g. at project root) with your credentials to run the example notebooks:

```
PROD_API_CLIENT_ID=
PROD_API_CLIENT_SECRET=
PROD_API_USERNAME=
PROD_API_PASSWORD=
```

For the preprod environment, use `PREPROD_API_*` variants. If you also want to route outputs to S3, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

3. Install the Jupyter Notebook extras

```
pip install -e ".[jupyter]"
```

4. Set up the Jupyter Notebook kernel

```
python -m ipykernel install --user --name edagro
```

5. Open one of the example notebooks under `notebooks/` and run it.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Initialise the workflow manager (handles auth, workspace directories, and token refresh):

```python
from earthdaily.agriculture.services.workflow_manager import WorkflowManager

manager = WorkflowManager("prod")
```

Load entities — either from the EarthDaily Agriculture platform or from your own GeoDataFrame:

```python
manager.load_seasonfields(
    sowing_date_gte="2025-07-01",
    crop_id="WINTER_OSR",
)
```

Build and run an extractor (here: a Medium-Resolution Time Series of LAI):

```python
from earthdaily.agriculture.extractors.VTS_functions import MRTSExtractor

mrts = MRTSExtractor(
    bearer_token=manager.bearer_token,
    token_expiration=manager.token_expiration,
    config=manager.config,
    workflow_ref=manager,
)

mrts.setup_mrts_parameters(
    start_date="2025-11-01",
    end_date="2026-02-28",
    vegetation_index="LAI",
    clear_cover_min=95,
)

results = mrts.process_mrts_bulk_parallel(
    entity_list=manager.sfd_list,
    output_path=manager.output_result_dir,
    max_workers=20,
    prefix="mrts_lai",
    generate_report=True,
)

print(results["results_df"].head())
```

Use the `earthdaily.agriculture` logger:

```python
from loguru import logger

logger.enable("earthdaily.agriculture")
```

See the Jupyter notebooks under [`notebooks/`](https://github.com/earthdaily/earthdaily-agriculture/tree/main/notebooks) for end-to-end working examples per extractor.

<p align="right">(<a href="#top">back to top</a>)</p>

## Documentation

Full documentation lives at <https://docs.earthdaily.com/agro/>.

Pages you'll likely want first:

- **Quick start** — install, authenticate, first extraction.
- **Extractor parameters reference** — every `setup_*_parameters()` argument.
- **Column mapping reference** — adapt the extractor to your DataFrame's column names.
- **KPI reference** — aggregate time series into a single KPI per entity.
- **Workflow architecture** — YAML-driven multi-step pipelines.
- **Deployment patterns** — GitHub Actions cron or a Docker container on ECS / Cloud Run.
- **Cloud storage** — route outputs to S3.
- **Caching** — when and how to enable the local cache.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- RESOURCES -->
## Resources
The following links provide more information:
- [Pypi package](https://pypi.org/project/earthdaily-agriculture/)
- [Documentation](https://docs.earthdaily.com/agro/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Support development

If this project has been useful — if it helped you or your business save precious time — don't hesitate to give it a star.

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Released under the **MIT** license. See [`LICENSE`](./LICENSE) for the full text.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

For any additional information, please [email us](mailto:sales@earthdailyagro.com).

<p align="right">(<a href="#top">back to top</a>)</p>

## Copyrights

© 2026 EarthDaily Analytics | All Rights Reserved.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- List of available shields https://shields.io/category/license -->
<!-- List of available shields https://simpleicons.org/ -->
[ci-shield]: https://github.com/earthdaily/earthdaily-agriculture/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/earthdaily/earthdaily-agriculture/actions/workflows/ci.yml
[issues-shield]: https://img.shields.io/github/issues/earthdaily/earthdaily-agriculture
[issues-url]: https://github.com/earthdaily/earthdaily-agriculture/issues
[license-shield]: https://img.shields.io/github/license/earthdaily/earthdaily-agriculture
[license-url]: ./LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=social&logo=linkedin
[linkedin-url]: https://www.linkedin.com/company/earthdailyagro/
[twitter-shield]: https://img.shields.io/badge/-Twitter-black.svg?style=social&logo=x
[twitter-url]: https://twitter.com/EarthDailyAgro
[youtube-shield]: https://img.shields.io/badge/-YouTube-black.svg?style=social&logo=youtube
[youtube-url]: https://www.youtube.com/channel/UCy4X-hM2xRK3oyC_xYKSG_g
[pypi-shield]: https://img.shields.io/pypi/v/earthdaily-agriculture.svg?logo=pypi
[pypi-url]: https://pypi.org/project/earthdaily-agriculture/
[language-python-shield]: https://img.shields.io/pypi/pyversions/earthdaily-agriculture.svg?logo=python
[language-python-url]: https://pypi.org/project/earthdaily-agriculture/
