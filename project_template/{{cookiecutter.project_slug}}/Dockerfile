# {{ cookiecutter.project_name }} — container image
#
# Mirrors the local earthdaily-agriculture install so the same code runs in CI, on
# ECS / Cloud Run / Argo, or under `docker run`. See docs/14 -
# Deployment_patterns.md (Pattern B) for invocation patterns.
#
# Multi-stage build keeps the runtime image around 600 MB. Single-stage
# alternative (~1.2 GB) below in a comment if you need to debug a build.

# ---- Stage 1: build wheel ----
FROM python:{{ cookiecutter.python_version }}-slim AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
RUN pip install --no-cache-dir build && python -m build --wheel

# ---- Stage 2: runtime ----
FROM python:{{ cookiecutter.python_version }}-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal32 \
    && rm -rf /var/lib/apt/lists/*

# Install the wheel built in stage 1, plus the earthdaily-agriculture wheel pinned in dist/.
COPY --from=builder /build/dist/*.whl /tmp/
COPY dist/earthdaily_agriculture-{{ cookiecutter.edagro_wheel_version }}-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Non-root user — required by many orchestrators (Cloud Run, OpenShift).
RUN useradd --create-home --shell /bin/bash edagro
USER edagro
WORKDIR /home/edagro

# Default command runs the pipeline; orchestrators can override with --shard,
# --limit, etc. The image's stdout is the log stream when EDAGRO_LOG_CONSOLE_ONLY=1.
ENTRYPOINT ["python", "-m", "app.run_pipeline"]
CMD ["--help"]
