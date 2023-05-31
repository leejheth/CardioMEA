"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import yaml
from functools import partial, update_wrapper
from kedro.pipeline import Pipeline, node, pipeline
from cardiomea.pipelines.features.nodes import (
    list_rec_files,
    write_yaml_file,
    parse_rec_file_info,
    extract_data,
    get_R_timestamps,
    get_active_area,
    get_FP_waves,
    get_FP_wave_features,
    get_HRV_features,
    get_conduction_speed,
    upload_to_sql_server,
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
            inputs=[
                "counter",
                "params:nCPUs",
            ],
            outputs=None,
            tags=["yaml","write","files"],
            name="write_yaml_file",         
        )
    ])

def extract_features_pipeline(**kwargs) -> Pipeline:    
    return pipeline([
        node(
            func=extract_data,
            inputs=[
                "file_path_full", 
                "params:signals.start_frame",
                "params:signals.length",
                "params:signals.s_freq",
            ],
            outputs=[
                "signals",
                "electrodes_info",
                "gain",
                "rec_duration",
            ],
            tags=["extract","data","signals"],
            name="extract_data",
        ),
        node(
            func=get_R_timestamps,
            inputs=[
                "signals", 
                "electrodes_info",
                "params:signals.factor",
                "params:signals.min_peak_dist",
                "params:signals.s_freq",
            ],
            outputs=[
                "R_timestamps",
                "channelIDs",
                "electrodes_info_updated"
            ],
            tags=["R_timestamps","channelIDs"],
            name="get_R_timestamps",
        ),        
        node(
            func=get_active_area,
            inputs=[
                "electrodes_info", 
                "channelIDs",
            ],
            outputs="active_area",
            tags=["activity","network"],
            name="get_active_area",
        ),
        node(
            func=get_FP_waves,
            inputs=[
                "signals", 
                "R_timestamps",
                "channelIDs",
                "params:signals.before_R",
                "params:signals.after_R",
            ],
            outputs="FP_waves",
            tags=["waveforms"],
            name="get_FP_waves",
        ),
        node(
            func=get_FP_wave_features,
            inputs=[
                "FP_waves", 
                "params:signals.before_R",
                "params:signals.T_from",
                "params:signals.T_to",
                "params:signals.s_freq",
            ],
            outputs=[
                "R_amplitudes",
                "R_widths",
                "FPDs",
            ],
            tags=["R_spikes","waveforms"],
            name="get_FP_wave_features",
        ),
        node(
            func=get_HRV_features,
            inputs=[
                "R_timestamps", 
            ],
            outputs="HRV_features",
            tags=["HRV"],
            name="get_HRV_features",
        ),
        node(
            func=get_conduction_speed,
            inputs=[
                "R_timestamps", 
                "electrodes_info_updated",
                "params:signals.s_freq",
            ],
            outputs=[
                "conduction_speed",
                "n_beats",
            ],
            tags=["conduction","propagation","R_spikes"],
            name="get_conduction_speed",
        ),
        node(
            func=upload_to_sql_server,
            inputs=[
                "rec_info",
                "file_path_full",
                "gain",
                "rec_duration",
                "electrodes_info_updated",
                "active_area",
                "R_amplitudes",
                "R_widths",
                "FPDs",
                "HRV_features",
                "conduction_speed",
                "n_beats",
                "params:tablename"
            ],
            outputs="dummy_for_pipe",
            tags=["upload","data","SQL"],
            name="upload_to_sql_server",
        ),
    ])


def create_auto_pipeline(**kwargs) -> Pipeline:   
    # Read the number of files to process
    with open("conf/base/file_count.yml", "r") as f:
        content = yaml.safe_load(f)
    n_files = content['n_files']
    nCPUs = content['n_CPUs']

    p_list = Pipeline([])
    for i in range(n_files):   
        session = int(i/nCPUs) 
        if session==0:
            pipe_input = 'first_pipe_input'
            pipe_output = f'pipe_id_{i}_{i+nCPUs}'
        else:
            pipe_input = f'pipe_id_{i-nCPUs}_{i}'
            pipe_output = f'pipe_id_{i}_{i+nCPUs}'

        parse_rec_file_info_partial = partial(parse_rec_file_info,index=i)
        update_wrapper(parse_rec_file_info_partial,parse_rec_file_info)

        pipeline_key = f'pipeline_{i}'
        rec_info_key = f'rec_info_{i}'
        file_path_full_key = f'file_path_full_{i}'
        node_key = f'parse_node_{i}'

        p_list += pipeline([
            node(
                func=parse_rec_file_info_partial,
                inputs=[
                    "data_catalog_full", 
                    pipe_input,
                ],
                outputs=[
                    rec_info_key,
                    file_path_full_key,
                ],
                tags=["directory","files"],
                name=node_key,
            ),
        ])
        p_list += pipeline(
            pipe=extract_features_pipeline(),
            inputs={'rec_info': rec_info_key, 'file_path_full': file_path_full_key},
            parameters={
                'signals.start_frame': 'signals.start_frame',
                'signals.length': 'signals.length',
                'signals.factor': 'signals.factor',
                'signals.min_peak_dist': 'signals.min_peak_dist',
                'signals.before_R': 'signals.before_R',
                'signals.after_R': 'signals.after_R',
                'signals.T_from': 'signals.T_from',
                'signals.T_to': 'signals.T_to',
                'signals.s_freq': 'signals.s_freq',
                'tablename': 'tablename',
            },
            outputs={'dummy_for_pipe': pipe_output},
            namespace=pipeline_key,
        )

    return p_list