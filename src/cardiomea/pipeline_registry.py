"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline as mpipeline

from cardiomea.pipelines.features.pipeline import (
    list_rec_files_pipeline,
    extract_features_pipeline,
    create_modular_pipeline,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())


    return {
        # pipelines,
        "__default__": list_rec_files_pipeline(),
        "extract_features": extract_features_pipeline(),
        "modular_pipe": create_modular_pipeline(),
    }
