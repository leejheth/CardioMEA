"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

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
    get_AP_waves,
    get_AP_wave_features,
    upload_AP_features_to_sql_server,
    parse_rec_file_info_FP_AP,
    merge_FP_AP_features,
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
                "params:raw_data.file_name_pattern",
                "params:raw_data.file_extension",
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
                "rec_proc_duration",
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
                "params:signals.s_freq",
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
                "rec_proc_duration",
                "electrodes_info_updated",
                "active_area",
                "R_amplitudes",
                "R_widths",
                "FPDs",
                "HRV_features",
                "conduction_speed",
                "n_beats",
                "params:tablename.FP"
            ],
            outputs=None,
            tags=["upload","data","SQL"],
            name="upload_to_sql_server",
        ),
    ])


def create_single_pipeline(**kwargs) -> Pipeline:    
    parse_pipeline = pipeline([
        node(
            func=parse_rec_file_info,
            inputs=[
                "data_catalog_full", 
                "params:file_index",
            ],
            outputs=[
                "rec_info",
                "file_path_full",
            ],
            tags=["directory","files"],
            name="parse_rec_file_info",
        )
    ])
    
    return parse_pipeline + extract_features_pipeline()

def extract_AP_features_pipeline(**kwargs) -> Pipeline:    
    return pipeline([
        node(
            func=parse_rec_file_info,
            inputs=[
                "data_catalog_full", 
                "params:file_index",
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
                "rec_proc_duration",
            ],
            tags=["extract","data","signals"],
            name="extract_data",
        ),
        node(
            func=get_AP_waves,
            inputs=[
                "signals", 
                "params:AP_wave",
                "electrodes_info",
            ],
            outputs=[
                "AP_waves",
                "electroporation_yield",
                "electrode_ids_list",
            ],
            tags=["waveforms"],
            name="get_AP_waves",
        ),
        node(
            func=get_AP_wave_features,
            inputs=[
                "AP_waves", 
                "electrode_ids_list",
                "params:AP_wave.after_upstroke",
                "params:signals.s_freq",
            ],
            outputs=[
                "AP_amplitudes",
                "depolarization_time",
                "APD50",
                "APD90",
                "electrode_ids_list_updated",
            ],
            tags=["AP","features","waveforms"],
            name="get_AP_wave_features",
        ),
        node(
            func=upload_AP_features_to_sql_server,
            inputs=[
                "rec_info",
                "file_path_full",
                "gain",
                "rec_duration",
                "rec_proc_duration",
                "electroporation_yield",
                "electrodes_info",
                "AP_amplitudes",
                "depolarization_time",
                "APD50",
                "APD90",
                "params:tablename.AP"
            ],
            outputs=None,
            tags=["upload","data","SQL"],
            name="upload_AP_features_to_sql_server",
        ),
    ])


def extract_FP_AP_features_pipeline(**kwargs) -> Pipeline:    
    return pipeline([
        node(
            func=parse_rec_file_info_FP_AP,
            inputs=[
                "data_catalog", 
                "params:raw_data.base_directory",
                "params:file_index",
            ],
            outputs=[
                "rec_info",
                "file_path_full_FP",
                "file_path_full_AP"
            ],
            tags=["directory","files"],
            name="parse_rec_file_info_FP_AP",
        ),
        node(
            func=extract_data,
            inputs=[
                "file_path_full_FP", 
                "params:signals.start_frame",
                "params:signals.length",
                "params:signals.s_freq",
            ],
            outputs=[
                "signals",
                "electrodes_info",
                "gain",
                "rec_duration",
                "rec_proc_duration",
            ],
            tags=["extract","data","signals"],
            name="extract_FP_data",
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
            func=extract_data,
            inputs=[
                "file_path_full_AP", 
                "params:signals.start_frame",
                "params:signals.length",
                "params:signals.s_freq",
            ],
            outputs=[
                "signals_AP",
                "electrodes_info_AP",
                "gain_AP",
                "rec_duration_AP",
                "rec_proc_duration_AP",
            ],
            tags=["extract","data","signals"],
            name="extract_AP_data",
        ),
        node(
            func=get_AP_waves,
            inputs=[
                "signals_AP", 
                "params:AP_wave",
                "electrodes_info_AP",
            ],
            outputs=[
                "AP_waves",
                "electroporation_yield",
                "electrode_ids_list"
            ],
            tags=["waveforms"],
            name="get_AP_waves",
        ),
        node(
            func=get_AP_wave_features,
            inputs=[
                "AP_waves", 
                "electrode_ids_list",
                "params:AP_wave.after_upstroke",
                "params:signals.s_freq",
            ],
            outputs=[
                "AP_amplitudes",
                "depolarization_time",
                "APD50",
                "APD90",
                "electrode_ids_list_updated",
            ],
            tags=["AP","features","waveforms"],
            name="get_AP_wave_features",
        ),
        node(
            func=merge_FP_AP_features,
            inputs=[
                "rec_info",
                "file_path_full_FP",
                "file_path_full_AP",
                "electrodes_info_updated",
                "R_amplitudes",
                "R_widths",
                "FPDs",
                "AP_amplitudes",
                "depolarization_time",
                "APD50",
                "APD90",
                "electrode_ids_list_updated",
                "params:tablename.FP_AP"
            ],
            outputs=None,
            tags=["FP","AP","features","electrodes"],
            name="merge_FP_AP_features",
        )
    ])