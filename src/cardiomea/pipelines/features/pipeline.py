"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import yaml
from kedro.extras.datasets.pickle import PickleDataSet
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
    pass_value,
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
                "params:parallel.n_jobs",
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
                "params:parallel.n_jobs"
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
                "params:parallel.n_jobs",
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
                "params:parallel.n_jobs",
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
            outputs=None,
            tags=["upload","data","SQL"],
            name="upload_to_sql_server",
        ),
    ])


def create_auto_pipeline(**kwargs) -> Pipeline:   
    # Read the number of files to process
    with open("conf/base/file_count.yml", "r") as f:
        content = yaml.safe_load(f)
    n_files = content['n_files']

    p_list = Pipeline([])
    for i in range(n_files):
        ind = PickleDataSet(filepath="data/02_intermediate/loop_index.pkl", backend="pickle")
        ind.save(i)

        pipeline_key = f'pipeline_{i}'

        p_list += pipeline([
            node(
                    func=pass_value,
                    inputs="tmp_ind",
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
                'signals.min_peak_dist': 'signals.min_peak_dist',
                'parallel.n_jobs': 'parallel.n_jobs',
                'signals.before_R': 'signals.before_R',
                'signals.after_R': 'signals.after_R',
                'signals.T_from': 'signals.T_from',
                'signals.T_to': 'signals.T_to',
                'signals.s_freq': 'signals.s_freq',
                'tablename': 'tablename',
            },
            outputs=None,
            namespace=pipeline_key,
        )

    return p_list