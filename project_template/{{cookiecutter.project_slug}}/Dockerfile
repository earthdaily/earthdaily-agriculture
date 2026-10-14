# {{ cookiecutter.project_name }} — container image
#
# The general-purpose image: `docker run`, scheduled jobs, CI. For large stateless runs
# use `Dockerfile.headless` (leaner, and refuses to write to the container's own disk).
# See docs/14 - Deployment_patterns.md (Pattern B) for invocation patterns.
#
# A client project is NOT a Python package — it has no pyproject.toml / src/ of its
# own. The image installs the project's requirements.txt (earthdaily-agriculture from
# PyPI, or the wheel in dist/ when scaffolded with --use-local-wheel) and runs the
# project's own app/ code.

FROM python:{{ cookiecutter.python_version }}-slim

# The rasterio / pyogrio / shapely / pyproj wheels bundle their own GDAL / GEOS / PROJ,
# so no system GDAL — its package name (libgdal32, libgdal36, ...) changes with every
# Debian release and broke this image once already. The bundled GDAL still needs
# libexpat at import; that name is stable across releases.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, for layer caching. dist/ is copied in both install modes: it
# holds the pinned wheel in `wheel` mode and only .gitkeep otherwise.
COPY requirements.txt ./
COPY dist/ ./dist/
RUN pip install --no-cache-dir -r requirements.txt

# Runtime code only; notebooks, docs and .env are excluded by .dockerignore.
COPY app/ ./app/
COPY configuration/ ./configuration/

# Non-root user, as most schedulers and restricted container runtimes require. /app must
# be writable: with no pyproject.toml here the package resolves the project root to the
# working directory and creates results/, logs/ etc. there.
RUN useradd --create-home --shell /bin/bash edagro && chown -R edagro /app
USER edagro

# Console-only logs so the scheduler captures stdout as the log stream.
ENV EDAGRO_LOG_CONSOLE_ONLY=1

# Callers pass --prefix / --shard / --num-shards / --limit;
# `docker run <image> --help` prints usage.
ENTRYPOINT ["python", "-m", "app.run_pipeline"]
CMD ["--help"]
