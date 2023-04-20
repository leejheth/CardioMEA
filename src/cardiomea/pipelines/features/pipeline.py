"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
# from kedro.pipeline.modular_pipeline import pipeline as mpipeline
from cardiomea.pipelines.features.nodes import (
    list_rec_files,
    write_yaml_file,
    get_R_timestamps
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

def list_rec_files_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=list_rec_files,
            inputs=[
                "data_catalog", 
                "params:raw_data.base_directory",
                "params:raw_data.file_extension"
            ],
            outputs=[
                "data_catalog_full",
                "counter"
            ],
            tags=["directory","recordings","files"],
            name="list_rec_files",
        ),
        node(
            func=write_yaml_file,
            inputs=["counter"],
            outputs=None,
            tags=["yaml","write","files"],
            name="write_yaml_file",         
        )
    ])

def extract_features_pipeline(**kwargs) -> Pipeline:    
    return pipeline([
        node(
            func=get_R_timestamps,
            inputs=[
                "data_catalog_full", 
                "key_test",
                # "params:synchronous_channels",
                # "params:test"
            ],
            outputs=[
                "R_timestamps",
                "channelIDs"
            ],
            tags=["features","R_timestamps","channelIDs"],
            name="get_R_timestamps",
        )
    ])



def create_modular_pipeline(**kwargs) -> Pipeline:   
    p_list = Pipeline([])

    for i in range(5):
        pipeline_key = f'pipeline_{i}'

        def parse_rec_files(num):
            def generate_num():
                return num*10
            return generate_num

        p_list += pipeline([
            node(
                    parse_rec_files(i),
                    inputs=None,
                    outputs=pipeline_key,
                )
        ])
        p_list += pipeline(
            pipe=extract_features_pipeline(),
            inputs={'data_catalog_full': 'data_catalog_full','key_test': pipeline_key},
            outputs={'R_timestamps': f'R_timestamps_{i}', 'channelIDs': f'channelIDs_{i}'},
            namespace=pipeline_key,
        )

    return p_list