"""
Transform functions for the {{ cookiecutter.project_name }} workflow.

Transforms reshape upstream extractor outputs into downstream extractor inputs.

Each function must follow the standard signature:
    def my_transform(
        entity_list: pd.DataFrame,
        upstream_results: dict[str, dict],
        params: dict,
    ) -> pd.DataFrame
"""

import pandas as pd
from loguru import logger


def use_upstream_entities(
    entity_list: pd.DataFrame,
    upstream_results: dict[str, dict],
    params: dict,
) -> pd.DataFrame:
    """
    Replace entity_list with the results_df from an upstream step.

    Use this when a downstream extractor needs columns added by a prior
    transform-only step.

    Params (from YAML):
        depends_on (str): Upstream step whose results_df becomes the new entity_list.
    """
    source_step = params.get("depends_on")
    if not source_step or source_step not in upstream_results:
        raise ValueError(f"Upstream step '{source_step}' not found in results")

    upstream_df = upstream_results[source_step].get("results_df", pd.DataFrame())
    if upstream_df.empty:
        logger.warning(f"Upstream step '{source_step}' has empty results — returning empty entity list")
        return entity_list.iloc[0:0]

    logger.info(f"  Using {len(upstream_df)} entities from upstream step '{source_step}'")
    return upstream_df


# Add your project-specific transforms below:
# def my_transform(entity_list, upstream_results, params):
#     ...
