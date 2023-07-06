"""
This is a boilerplate pipeline 'visualize'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from cardiomea.pipelines.visualize.nodes import (
    dashboard,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_dashboard_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=dashboard,
            inputs=[
                "cardio_db_FP",
                "cardio_db_AP",
                "params:web_port", 
                "params:raw_data.base_directory"
            ],
            outputs=None,
            tags=["dashboard"],
            name="dashboard",
        ),
    ])
