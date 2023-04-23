"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import yaml
from kedro.pipeline import Pipeline, node, pipeline
from cardiomea.pipelines.features.nodes import (
    list_rec_files,
    write_yaml_file,
    parse_rec_file_info,
    extract_data,
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
            func=parse_rec_file_info,
            inputs=[
                "data_catalog_full", 
                "index",
            ],
            outputs=[
                "rec_info",
                "file_path_full",
            ],
            tags=["directory","files"],
            name="parse_rec_file_info",
        ),
        node(
            func=extract_data,
            inputs=[
                "file_path_full", 
                "params:signals.start_frame",
                "params:signals.length",
            ],
            outputs=[
                "signals",
                "electrode_info",
                "gain",
            ],
            tags=["extract","data","signals"],
            name="extract_data",
        ),
        node(
            func=get_R_timestamps,
            inputs=[
                "signals", 
                "params:signals.factor",
                "params:signals.min_peak_dist",
            ],
            outputs=[
                "R_timestamps",
                "channelIDs"
            ],
            tags=["features","R_timestamps","channelIDs"],
            name="get_R_timestamps",
        )
    ])



def create_auto_pipeline(**kwargs) -> Pipeline:   
    # Read the number of files to process
    with open("conf/base/file_count.yml", "r") as f:
        content = yaml.safe_load(f)
    n_files = content['n_files']

    p_list = Pipeline([])
    for i in range(n_files):
        pipeline_key = f'pipeline_{i}'

        def pass_value(num):
            def generate_num():
                return num
            return generate_num

        p_list += pipeline([
            node(
                    func=pass_value(i),
                    inputs=None,
                    outputs=pipeline_key,
                )
        ])
        p_list += pipeline(
            pipe=extract_features_pipeline(),
            inputs={'data_catalog_full': 'data_catalog_full','index': pipeline_key},
            parameters={
                'signals.start_frame': 'signals.start_frame',
                'signals.length': 'signals.length',
                'signals.factor': 'signals.factor',
                'signals.min_peak_dist': 'signals.min_peak_dist'},
            outputs={'R_timestamps': f'R_timestamps_{i}', 'channelIDs': f'channelIDs_{i}'},
            namespace=pipeline_key,
        )

    return p_list