"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, node

from cardiomea.pipelines.features.pipeline import (
    list_rec_files_pipeline,
    # create_auto_pipeline,
    create_single_pipeline,
    extract_AP_features_pipeline,
)

from cardiomea.pipelines.visualize.pipeline import(
    create_dashboard_pipeline,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": list_rec_files_pipeline(),
        "list_rec_files": list_rec_files_pipeline(),
        # "auto_pipeline": create_auto_pipeline(),
        "single_pipeline": create_single_pipeline(),
        "intra_pipeline": extract_AP_features_pipeline(),
        "dashboard": create_dashboard_pipeline(),
    }
