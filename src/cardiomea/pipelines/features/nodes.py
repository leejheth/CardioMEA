"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import datetime
import h5py
import glob
import numpy as np
import pandas as pd
import psycopg2
import statistics as st
import warnings
import yaml
import hrvanalysis as hrva
from collections import defaultdict
from scipy import signal
from scipy.signal import find_peaks
from lmfit import Model, Parameters
from psycopg2.extensions import register_adapter, AsIs
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import sympy as sym


def list_rec_files(data_catalog,base_directory,file_name_pattern,ext):
    """List all recording files in the directory.

    This function takes in a data catalog containing directory information (where recording files 
    are stored) along with other information (names of cell lines, compound, etc.) and returns a 
    full data catalog where path to all recording files (in the specified directory) are added.
    
    Args:
        data_catalog (pandas.DataFrame): Data catalog containing directory and other (names of cell lines, compound, etc.) information of recording files.
        base_directory (str): Base directory of the recording files (root directory of the specified file_path).
        ext (str): File extension of the recording files to be listed.

    Returns:
        data_catalog_full (pandas.DataFrame): Data catalog containing directory and other (names of cell lines, compound, etc.) information of recording files, as well as the full directory of the recording files.
                                              Missing compound entries will be filled with 'absent'.
    """
    data_catalog_full = pd.DataFrame(columns=data_catalog.columns.to_list() + ['file_path_full'])
   
    for row in range(len(data_catalog)):
        # list all files (with specified extension) in the directory
        rec_files = glob.glob(base_directory+str(data_catalog.loc[row,'file_path'])+"/"+file_name_pattern+"*"+ext)
        
        # create a new dataframe that contains both existing and new data (full file directory)
        if rec_files:
            tmp = pd.concat([data_catalog.loc[[row]]]*len(rec_files), ignore_index=True)
            tmp["file_path_full"] = rec_files
            data_catalog_full = data_catalog_full.merge(tmp, how='outer')
        else:
            print(f"No recording files are found in {base_directory+data_catalog.loc[row,'file_path']}")

    # replace NaN values in colume 'compound' to a string 'absent'
    data_catalog_full['compound'].fillna('absent', inplace=True)

    print(f"A total of {len(data_catalog_full)} recording files are found.")

    return data_catalog_full, len(data_catalog_full)


def write_yaml_file(counter, nCPUs):
    """Write a yaml file that contains the number of recording files."""
    data = dict([
        ("n_files", counter),
        ("n_CPUs", nCPUs)
    ])

    with open('conf/base/file_count.yml', 'w') as f:
        yaml.dump(data, f)


def parse_rec_file_info(data_catalog_full, index):
    """Parse the information of a recording file.

    Args:
        data_catalog_full (pandas.DataFrame): Data catalog containing directory and other (names of cell lines, compound, etc.) information of recording files, as well as the full directory of the recording files.
        index (int): Index of the recording file to be parsed.

    Returns:
        rec_info (dict): Dictionary containing the following information of the recording file:
            - cell_line (str): Name of the cell line (if provided).
            - compound (str): Name of the compound (if provided).
            - file_path (str): Directory where the recording file was found.
            - note (str): Note about the recording (if provided).
        file_path_full (str): Full directory of the recording file.
    """
    cell_line = data_catalog_full.loc[index,'cell_line']
    compound = data_catalog_full.loc[index,'compound']
    file_path = data_catalog_full.loc[index,'file_path']
    note = data_catalog_full.loc[index,'note']
    file_path_full = data_catalog_full.loc[index,'file_path_full']
    print(f"Processing recording file: {file_path_full} ...")

    rec_info = dict([
        ("cell_line", cell_line),
        ("compound", compound),
        ("file_path", file_path),
        ("note", note),
    ])

    return rec_info, file_path_full


def extract_data(file_path_full, start_frame, length, s_freq):
    """Extract data from a recording file.

    Args:
        file_path_full (str): Full directory of the recording file.
        start_frame (int): Start frame of the recording.
        length (int): Length of the recording.
        s_freq (int): sampling frequency of the recording.
        num_frames/s_freq (float): duration of recording in seconds.

    Returns:
        signals (numpy.ndarray): Extracted signals.
        electrodes_info (dict): Dictionary containing the channel information (electrode ID, X and Y locations, number of electrodes used for recording).
        gain (int): Gain of the recording.
        rec_duration (float): Duration of the recording (in seconds) in the file.
        rec_proc_duration (float): Duration of the recording data (in seconds) that will be processed.
    """
    obj = h5py.File(file_path_full, mode='r')

    if obj.get('version')[0].decode() == '20160704':
        setting_struc = 'settings'
        map_struc = 'mapping'
        data_struc = 'sig'
    elif obj.get('version')[0].decode() == '20190530':
        setting_struc = 'recordings/rec0000/well000/settings'
        map_struc = setting_struc + '/mapping'
        data_struc = 'data_store/data0000/groups/routed/raw'
    else:
        raise NotImplementedError(f"Recording file was created with version {obj.get('version')[0].decode()} and is not supported.")

    # get channel information
    mapping = obj.get(map_struc)
    channels = mapping['channel']
    electrodes = mapping['electrode']
    routed_idxs = np.where(electrodes > -1)[0] # remove unused channels
    channel_ids = channels[routed_idxs]
    electrode_ids = list(electrodes[routed_idxs])
    num_channels = len(electrode_ids)
    num_frames = obj.get(data_struc).shape[1]
    x_locs = list(mapping['x'][routed_idxs])
    y_locs = list(mapping['y'][routed_idxs])

    electrodes_info = dict([
        ("electrode_ids", electrode_ids),
        ("x_locs", x_locs),
        ("y_locs", y_locs),
        ("num_channels", num_channels)
    ])

    # get lsb value
    gain = (obj[setting_struc]['gain'][0]).astype(int)
    lsb = obj[setting_struc]['lsb'][0] * 1e6
    
    # get raw voltage traces from all recording channels
    if start_frame < (num_frames-1):
        if (start_frame+length) > num_frames:
            warnings.warn("The specified time frame exceeds the data length. Signals will be extracted until end-of-file.")
        signals = (obj.get(data_struc)[:,start_frame:start_frame+length] * lsb).astype('uint16')
    else:
        warnings.warn(f"The start frame exceeds the length of data. Signals will be extracted from the start of the recording until MIN({length}-th frame, end-of-file) instead.")
        signals = (obj.get(data_struc)[:,0:min(length,num_frames)] * lsb).astype('uint16')
    
    if obj.get('version')[0].decode() == '20160704':
        signals = signals[channel_ids,:]

    return signals, electrodes_info, gain, num_frames/s_freq, signals.shape[1]/s_freq


def get_R_timestamps(signals,electrodes_info,mult_factor,min_peak_dist,s_freq):
    """Identify R peaks in the signals.
    
    Args:
        signals (numpy.ndarray): Extracted signals.
        electrodes_info (dict): Dictionary containing the channel information (electrode ID, X and Y locations, number of electrodes used for recording).
        mult_factor (float): Multiplication factor for the threshold.
        min_peak_dist (int): Minimum distance between two R peaks.
        s_freq (int): Sampling frequency of the recording.
        
    Returns:
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        sync_channelIDs (list): List of channel IDs of R peaks that are synchronous.  
        electrodes_info_updated (dict): Updated dictionary containing the channel information (contains only channels that captured synchronous beatings).  
    """
    # build a butterworth filter of order: 3, bandwidth: 100-2000Hz, bandpass
    b, a = signal.butter(3,[100*2/s_freq,2000*2/s_freq],btype='band')

    n_channels = signals.shape[0]
    n_Rpeaks = []
    r_timestamps = []
    for ch in range(n_channels):
        # Apply the filter to the signal of the current channel
        filtered_ch = signal.filtfilt(b, a, signals[ch])

        r_locs = _R_timestamps(filtered_ch, mult_factor, min_peak_dist)
        n_Rpeaks.append(len(r_locs))
        r_timestamps.append(r_locs)
    
    # Identify synchronous beats
    sync_beats = st.mode(n_Rpeaks)
    # Indices of channels where beats (R peaks) are synchronous
    sync_channelIDs = [i for i, n_peaks in enumerate(n_Rpeaks) if n_peaks == sync_beats]

    # Extract r_locs for synchronous channels
    sync_timestamps = [r_timestamps[i] for i in sync_channelIDs]
    
    # Update electrodes info (contains only channels that captured synchronous beatings)
    electrodes_info_updated = {
        "electrode_ids": [electrodes_info['electrode_ids'][i] for i in sync_channelIDs],
        "x_locs": [electrodes_info['x_locs'][i] for i in sync_channelIDs],
        "y_locs": [electrodes_info['y_locs'][i] for i in sync_channelIDs],
        "num_channels": len(sync_channelIDs)
    }

    return sync_timestamps, sync_channelIDs, electrodes_info_updated


def _R_timestamps(signal_single,mult_factor,min_peak_dist):
    """Identify R peaks in a single channel."""
    thr = mult_factor*np.std(signal_single)
    r_locs, _ = find_peaks(signal_single, distance=min_peak_dist, prominence=thr)

    return r_locs
  

def get_active_area(electrodes_info, sync_channelIDs):
    """Calculate the percentage of electrodes which recorded synchronous beats.
    
    Args:
        electrodes_info (dict): Dictionary containing information about electrodes.
        sync_channelIDs (list): List of channel IDs of R peaks that are synchronous.

    Returns:
        active_area (float): Percentage of electrodes which recorded synchronous beats.
    """
    active_area = 100 * len(sync_channelIDs) / electrodes_info["num_channels"]
    
    return active_area


def get_FP_waves(signals,sync_timestamps,sync_channelIDs,before_R,after_R):
    """Extract FP waves from the signals.
    
    Args:
        signals (numpy.ndarray): Raw signals.
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        sync_channelIDs (list): List of channel IDs of R peaks that are synchronous.
        before_R (int): Number of frames before R peak.
        after_R (int): Number of frames after R peak.
        
    Returns:
        FP_waves (list): List of FP waves.
    """
    FP_waves=[]
    for count, ch in enumerate(sync_channelIDs):
        FP_waves.append(_FP_wave(signals[ch],sync_timestamps[count],before_R,after_R))
    
    return FP_waves


def _FP_wave(signal_single,timestamps,before_R,after_R):
    """Get an average FP wave from a single channel."""
    waves = [signal_single[peak_loc-before_R:peak_loc+after_R] for peak_loc in timestamps if (peak_loc-before_R)>0 and (peak_loc+after_R)<=len(signal_single)] 
    if not waves:
        return []
    else:
        return list(np.mean(np.vstack(waves),axis=0))


def get_FP_wave_features(FP_waves,before_R,T_from,T_to,s_freq):
    """Extract features from FP waves.
    
    Args:
        FP_waves (list): List of FP waves.
        before_R (int): Number of frames before R peak.
        T_from (int): Start of the interval to find T peak (number of frames from R peak).
        T_to (int): End of the interval to find T peak (number of frames from R peak).
        s_freq (int): Sampling frequency of the signals.
        
    Returns:
        R_amplitudes (list): List of R peak-to-peak amplitudes (in microvolts).
        R_widths (list): List of estimated R spike widths (in milliseconds).
        FPDs (list): List of field potential durations (in milliseconds).
    """
    R_amplitudes=[]
    R_widths=[]
    FPDs=[]

    for wave in FP_waves:
        R_amplitude, R_width, FPD = _FP_wave_features(wave,before_R,T_from,T_to,s_freq)
        R_amplitudes.append(R_amplitude)
        R_widths.append(R_width)
        FPDs.append(FPD)

    return R_amplitudes, R_widths, FPDs


def _FP_wave_features(wave,before_R,T_from,T_to,s_freq):
    """Extract features from a single FP wave."""
    # get R peak-to-peak amplitude
    try:
        R_amplitude = wave[before_R]-np.min(wave)
    except Exception:
        R_amplitude = np.nan
    
    # get an estimate of R spike width (time duration of R peak-to-peak)
    try:
        R_width = 1e3*(np.argmin(wave)-before_R)/s_freq # in milliseconds
        if R_width>10 or R_width<=0: # if calculate R width is >10 ms or <=0 ms, drop the data.
            R_width = np.nan
    except Exception:
        R_width = np.nan

    # get FPD
    b, a = signal.butter(3,[3*2/s_freq,100*2/s_freq],btype='band')
    try:
        T_filtered = signal.filtfilt(b, a, wave)
        Tpeak, _ = find_peaks(T_filtered[before_R+T_from:before_R+T_to], distance=len(wave), prominence=15) # find just 1 peak
        FPD = 1e3*(Tpeak[0]+T_from)/s_freq # in milliseconds
    except Exception:
        FPD = np.nan
        
    return R_amplitude, R_width, FPD


def get_HRV_features(sync_timestamps,s_freq):
    """Extract heart rate variability (HRV) features.
    
    Args:
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        s_freq (int): Sampling frequency of the signals.

    Returns:
        HRV_features (dict): Dictionary of HRV features.
    """
    res=[]
    for timestamps in sync_timestamps:
        res.append(_hrv_features(timestamps,s_freq))

    # Create a defaultdict to store mean values
    HRV_features = defaultdict(int)

    for d in res:
        for key, value in d.items():
            try:
                # averages values across channels
                HRV_features[key] += value/len(res)
            except TypeError:
                HRV_features[key] = None

    return dict(HRV_features)


def _hrv_features(timestamps,s_freq):
    """Extract HRV features from a single channel.
    Documentation that explains the features: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html
    """
    # get RR intervale in milliseconds
    RR_intervals = 1e3*np.diff(timestamps)/s_freq
    
    try:
        # get time domain features (use for >1min recordings)
        time_domain_features = hrva.get_time_domain_features(RR_intervals)
        
        # get geometrical features (use for >20min recordings)
        # geometrical_features = hrva.get_geometrical_features(RR_intervals)
        
        # get frequency domain features (use for >2min recordings)
        # frequency_domain_features = hrva.get_frequency_domain_features(RR_intervals)
        
        # get poincare plot features (use for >5min recordings)
        # poincare_plot_features = hrva.get_poincare_plot_features(RR_intervals)

        # all_hrv_features = {**time_domain_features, **geometrical_features, **frequency_domain_features, **poincare_plot_features}
        all_hrv_features = {**time_domain_features}
    
    except Exception:
        # keys = ['mean_nni','sdnn','sdsd','nni_50','pnni_50','nni_20','pnni_20','rmssd','median_nni','range_nni','cvsd','cvnni','mean_hr','max_hr','min_hr','std_hr','triangular_index','tinn','lf','hf','lf_hf_ratio','lfnu','hfnu','total_power','vlf','csi','cvi','Modified_csi','sd1','sd2','ratio_sd2_sd1']
        keys = ['mean_nni','sdnn','sdsd','nni_50','pnni_50','nni_20','pnni_20','rmssd','median_nni','range_nni','cvsd','cvnni','mean_hr','max_hr','min_hr','std_hr']
        all_hrv_features = dict([])
        for key in keys:
            all_hrv_features.update({key:None})

    return all_hrv_features


def get_conduction_speed(sync_timestamps,electrodes_info_updated,s_freq):
    """Estimate conduction speed.

    Fit a cone equation to the data and estimate conduction speed at each electrode using partial derivatives.  
    Algorithms are adapted from Bayly et al. "Estimation of Conduction Velocity Vector Fields from Epicardial Mapping Data" (1998) 
    and also from Cardio PyMEA https://github.com/csdunhamUC/cardio_pymea.git with modifications.
    
    Args:
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        electrodes_info_updated (dict): Updated electrode information.
        s_freq (int): Sampling frequency of the signals.
    
    Returns:
        speed_list (list): list of conduction speed estimated in each beat. (unit of each data point: cm/s)
        n_beats (int): Number of beats.
    """
    if not sync_timestamps: # if sync_timestamps is an empty list
        return [], 0
    else: 
        beat_clusters = np.vstack(sync_timestamps)
        n_beats = len(sync_timestamps[0])
        x_locs = electrodes_info_updated['x_locs']
        y_locs = electrodes_info_updated['y_locs']

        def cone_surface(x, y, x_p, y_p, a, b, c):
            return (a*(x-x_p)**2 + b*(y-y_p)**2)**0.5 + c

        model = Model(cone_surface, independent_vars=['x', 'y'])

        speed_list = []
        for beat in range(n_beats):
            # activation time in seconds
            t_data = (beat_clusters[:,beat]-min(beat_clusters[:,beat]))/s_freq 

            # define parameters and limits
            params = Parameters()
            params.add('x_p', value=0.2, min=-0.1, max=0.5)
            params.add('y_p', value=0.1, min=-0.2, max=0.4)
            params.add('a', value=1, min=1e-4)
            params.add('b', value=1, min=1e-4)
            params.add('c', value=1)

            # x, y locations in cm units
            x_locs_cm = np.array(x_locs)*1e-4
            y_locs_cm = np.array(y_locs)*1e-4

            # fit model to the data
            result = model.fit(t_data, params, x=x_locs_cm, y=y_locs_cm)

            # get estimated parameter values
            x_p, y_p, a, b, c = [i.value for i in result.params.values()]

            x, y = sym.symbols("x,y", real=True)
            t_xy = (a*(x-x_p)**2 + b*(y-y_p)**2)**0.5 + c

            # calculate partial derivatives
            t_partial_x_eq = sym.lambdify([x, y], t_xy.diff(x), "numpy")
            t_partial_y_eq = sym.lambdify([x, y], t_xy.diff(y), "numpy")

            T_partial_x = t_partial_x_eq(x_locs_cm,y_locs_cm) 
            T_partial_y = t_partial_y_eq(x_locs_cm,y_locs_cm)

            # calculate dx/dT and dy/dT (conduction velocity)
            velocity_x = T_partial_x / (T_partial_x**2 + T_partial_y**2)
            velocity_y = T_partial_y / (T_partial_x**2 + T_partial_y**2)

            # calculate magnitude of the vector (conduction speed)
            speed = np.sqrt(velocity_x**2 + velocity_y**2)

            speed_list.append(np.mean(speed))
        
        return speed_list, n_beats


def upload_to_sql_server(rec_info,file_path_full,gain,rec_duration,rec_proc_duration,electrodes_info_updated,active_area,R_amplitudes,R_widths,FPDs,HRV_features,conduction_speed,n_beats,tablename):
    """Upload extracted FP feature data to SQL server.
    
    Args:
        rec_info (dict): Recording information.
        file_path_full (str): Full path to the recording file.
        gain (int): Gain of the recording.
        rec_duration (float): Duration of the recording in seconds.
        rec_proc_duration (float): Duration of the fraction of recording (in seconds) that was processed.
        electrodes_info_updated (dict): Updated electrode information.
        active_area (float): Active area of the electrode (percentage).
        R_amplitudes (list): List of R peak amplitudes (in microvolts).
        R_widths (list): List of R peak widths (in milliseconds).
        FPDs (list): List of FPDs (field potential durations) (in milliseconds).
        HRV_features (dict): Dictionary of HRV features.
        conduction_speed (list): List of conduction speed estimated in each beat.
        n_beats (int): Number of beats.
        tablename (str): Name of the SQL table where the data is uploaded.
    """    
    # organize SQL column names and data types
    sql_columns = [
        ['time', 'TIMESTAMP'],
        ['cell_line', 'VARCHAR'],
        ['compound', 'VARCHAR'],
        ['note', 'VARCHAR'],
        ['file_path_full', 'VARCHAR'],
        ['gain', 'SMALLINT'],
        ['rec_duration', 'DECIMAL(4,1)'],
        ['rec_proc_duration', 'DECIMAL(4,1)'],
        ['n_electrodes_sync', 'SMALLINT'],
        ['active_area_in_percent', 'DECIMAL(4,1)'],
        ['r_amplitudes_str', 'VARCHAR'],
        ['r_amplitudes_mean', 'DECIMAL(6,1)'],
        ['r_amplitudes_std', 'DECIMAL(7,1)'],
        ['r_widths_str', 'VARCHAR'],
        ['r_widths_mean', 'DECIMAL(5,2)'],
        ['r_widths_std', 'DECIMAL(5,2)'],
        ['fpds_str', 'VARCHAR'],
        ['fpds_mean', 'DECIMAL(4,1)'],
        ['fpds_std', 'DECIMAL(5,1)'],
        ['conduction_speed_str', 'VARCHAR'],
        ['conduction_speed_mean', 'DECIMAL(3,1)'],
        ['conduction_speed_std', 'DECIMAL(4,1)'],
        ['n_beats', 'SMALLINT']
        ]

    HRV_col = [[key, 'DECIMAL'] for key, _ in HRV_features.items()]

    # add HRV features to the list of SQL columns
    sql_columns.extend(HRV_col)

    # convert to integers to prevent overflow
    R_amplitudes_int = [int(R_amplitudes[i]) if not np.isnan(R_amplitudes[i]) else R_amplitudes[i] for i in range(len(R_amplitudes))]

    # prepare data in appropriate forms
    cell_line = rec_info['cell_line']
    compound = rec_info['compound']
    note = rec_info['note']
    n_electrodes_sync = electrodes_info_updated['num_channels']
    R_amplitudes_str = ' '.join(map(str, R_amplitudes_int))
    R_amplitudes_mean = np.nanmean(R_amplitudes_int)
    R_amplitudes_std = np.std(R_amplitudes_int)
    R_widths_str = ' '.join(map(str, R_widths))
    R_widths_mean = np.nanmean(R_widths)
    R_widths_std = np.nanstd(R_widths)
    FPDs_str = ' '.join(map(str, FPDs))
    FPDs_mean = np.nanmean(FPDs)
    FPDs_std = np.nanstd(FPDs)
    conduction_speed_str = ' '.join(map(str, conduction_speed))
    conduction_speed_mean = np.mean(conduction_speed)
    conduction_speed_std = np.std(conduction_speed)
    timestamp = datetime.datetime.now()

    values = [timestamp, cell_line, compound, note, file_path_full, gain, rec_duration, rec_proc_duration, n_electrodes_sync, active_area, R_amplitudes_str, R_amplitudes_mean, R_amplitudes_std, R_widths_str, R_widths_mean, R_widths_std, FPDs_str, FPDs_mean, FPDs_std, conduction_speed_str, conduction_speed_mean, conduction_speed_std, n_beats]

    HRV_values = [value for _, value in HRV_features.items()]
    
    # add HRV features to the list of values
    values.extend(HRV_values)

    # upload data to SQL server
    _upload_to_sql_server(tablename, sql_columns, values)


def _upload_to_sql_server(tablename, sql_columns, values):
    """Make a connection to the SQL server and upload data."""
    # Get PostgreSQL database information and credentials data from text file.
    # (.txt file format: host, dbname, user, password, port in each line)
    with open('conf/local/postgresql.txt', 'r') as f:
        sql_credentials = f.read().splitlines()
    
    register_adapter(np.int64, AsIs)
    register_adapter(np.float16, AsIs)

    # connect to sql server
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host=sql_credentials[0],
            dbname=sql_credentials[1],
            user=sql_credentials[2],
            password=sql_credentials[3],
            port=sql_credentials[4]
        )
		
        # create a cursor
        cur = conn.cursor()

        # create a table if not exists
        cur.execute(f"CREATE TABLE IF NOT EXISTS {tablename} ({','.join([' '.join(c) for c in sql_columns])})")

        # insert data into the table
        query = f"INSERT INTO {tablename} ({','.join([c[0] for c in sql_columns])}) VALUES ({','.join(['%s' for i in range(len(sql_columns))])})"
        cur.execute(query, tuple(values))
        conn.commit()
                    
	    # close the communication with the PostgreSQL
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def get_AP_waves(signals, params_AP_wave, electrodes_info):
    """Extract AP waves from the signals.

    Args:
        signals (numpy.ndarray): Extracted signals.
        params_AP_wave (dict): Dictionary containing parameters for extracting AP waves.
        electrodes_info (dict): Dictionary containing the channel information (electrode ID, X and Y locations, number of electrodes used for recording).
    
    Returns:
        AP_waves (list): List of AP waves.
    """
    AP_waves=[]
    electrode_ids_list=[]
    for cnt, signal in enumerate(signals):
        # find the end point of the clipped window of the signal
        max_clipped_ind = np.flatnonzero(signal == np.amax(signal)).tolist()
        min_clipped_ind = np.flatnonzero(signal == np.amin(signal)).tolist()
        endpoint = max(max_clipped_ind + min_clipped_ind)

        if endpoint:
            endpoint = max(endpoint,2e5)    # after offset fluctuation
        else:
            endpoint = 2e5     # 10s from the onset of recording
    
        segment = signal[int(endpoint):]
        if len(segment) < params_AP_wave['before_upstroke']+params_AP_wave['after_upstroke']:
            # skip this channel if the signal was clipped almost till the end.
            continue
        
        # Find AP locations
        r_locs_tmp, _ = find_peaks(segment[params_AP_wave['before_upstroke']:-params_AP_wave['after_upstroke']], distance=params_AP_wave['after_upstroke'], prominence=params_AP_wave['min_peak_prom'], width=params_AP_wave['width_thr'])
        r_locs = r_locs_tmp + params_AP_wave['before_upstroke']

        # skip if no peak was found
        if r_locs.size==0:
            continue

        # get AP waves
        waves = []
        n_peaks = min(params_AP_wave['n_waves'],len(r_locs))
        for peaks in range(n_peaks):
            waves.append(segment[r_locs[peaks]-params_AP_wave['before_upstroke']:r_locs[peaks]+params_AP_wave['after_upstroke']]/n_peaks)
        
        # average the waves
        AP_waves.append([sum(col) / len(col) for col in zip(*waves)])
        electrode_ids_list.append(electrodes_info['electrode_ids'][cnt])
        
    # Calculate electroporation yield
    electroporation_yield = 100*len(AP_waves)/signals.shape[0]

    return AP_waves, electroporation_yield, electrode_ids_list

def get_AP_wave_features(AP_waves,electrode_ids_list,after_upstroke,s_freq):
    """Extract features from AP waves.

    Args:
        AP_waves (list): List of AP waves.
        electrode_ids_list (list): List of electrode IDs.
        after_upstroke (int): Number of frames after the upstroke.
        s_freq (int): Sampling frequency of the signals.

    Returns:
        amps (list): List of peak-to-peak amplitudes of AP waves (in millivolts).
        t_deps (list): List of depolarization times (in milliseconds).
        APD50s (list): List of action potential duration at 50% repolarization (in milliseconds).
        APD90s (list): List of action potential duration at 90% repolarization (in milliseconds).
        electrode_ids_list_updated (list): Updated list of electrode IDs.   
    """
    t_deps=[]
    amps=[]
    APD50s=[]
    APD90s=[]
    electrode_ids_list_updated=[]
    for AP_wave in AP_waves:
        # find peak (maximum of the AP wave)    
        peak_x = np.argmax(AP_wave[:after_upstroke])
        peak_y = AP_wave[peak_x]

        # skip this AP wave if FP spike is dominant
        spike_tip_width = np.flatnonzero(AP_wave > (peak_y - 80)).size
        if spike_tip_width < 100:
            continue

        # valley before the upstroke
        f_valley_x = np.argmin(AP_wave[:peak_x])
        f_valley_y = AP_wave[f_valley_x]
        # valley after the upstroke
        b_valley_x_tmp = np.argmin(AP_wave[peak_x:])
        b_valley_x = peak_x + b_valley_x_tmp
        b_valley_y = AP_wave[b_valley_x]

        # find the initiation point of the upstroke
        x = np.arange(f_valley_x,peak_x)
        y = np.array(AP_wave)[x]
        numerator = np.abs((peak_y - f_valley_y) * x - (peak_x - f_valley_x) * y + peak_x * f_valley_y - peak_y * f_valley_x)
        denominator = np.sqrt((peak_y - f_valley_y) ** 2 + (peak_x - f_valley_x) ** 2)
        knee_x = np.argmax(numerator / denominator)
        thr_x = f_valley_x + knee_x

        # get depolarization time (unit: milliseconds)
        t_dep = 1e3*(peak_x-thr_x)/s_freq
        
        # get peak-to-peak amplitude of AP wave (unit: millivolts)
        amp = peak_y - b_valley_y
        
        # get APD50
        temp = np.argmin(np.abs(AP_wave[peak_x:b_valley_x] - (peak_y - amp * 0.5)))
        APD50 = 1e3*(peak_x + temp - thr_x)/s_freq
        # get APD90
        temp = np.argmin(np.abs(AP_wave[peak_x:b_valley_x] - (peak_y - amp * 0.9)))
        APD90 = 1e3*(peak_x + temp - thr_x)/s_freq
        
        # skip this AP wave if spike tip width is greater than APD90
        if 1e3*spike_tip_width/s_freq > APD90:
            continue
        else:
            t_deps.append(t_dep) 
            amps.append(amp)
            APD50s.append(APD50)
            APD90s.append(APD90)
            electrode_ids_list_updated.append(electrode_ids_list[AP_waves.index(AP_wave)])

    return amps, t_deps, APD50s, APD90s, electrode_ids_list_updated

def upload_AP_features_to_sql_server(rec_info,file_path_full,gain,rec_duration,rec_proc_duration,electroporation_yield,electrodes_info,AP_amplitudes,depolarization_time,APD50,APD90,tablename):
    """Upload extracted AP feature data to SQL server.
    
    Args:
        rec_info (dict): Recording information.
        file_path_full (str): Full path to the recording file.
        gain (int): Gain of the recording.
        rec_duration (float): Duration of the recording in seconds.
        rec_proc_duration (float): Duration of the fraction of recording (in seconds) that was processed.
        electroporation_yield (float): Electroporation yield.
        electrodes_info (dict): electrode information.
        AP_amplitudes (list): AP amplitudes.
        depolarization_time (list): Depolarization time.
        APD50 (list): action potential duration at 50% repolarization.
        APD90 (list): action potential duration at 90% repolarization.
        tablename (str): Name of the SQL table where the data is uploaded.
    """    
    # organize SQL column names and data types
    sql_columns = [
        ['time', 'TIMESTAMP'],
        ['cell_line', 'VARCHAR'],
        ['compound', 'VARCHAR'],
        ['note', 'VARCHAR'],
        ['file_path_full', 'VARCHAR'],
        ['gain', 'SMALLINT'],
        ['rec_duration', 'DECIMAL(4,1)'],
        ['rec_proc_duration', 'DECIMAL(4,1)'],
        ['electroporation_yield', 'DECIMAL(4,1)'],
        ['n_electrodes', 'SMALLINT'],
        ['ap_amplitudes_str', 'VARCHAR'],
        ['ap_amplitudes_mean', 'DECIMAL(6,1)'],
        ['ap_amplitudes_std', 'DECIMAL(7,1)'],
        ['depolarization_time_str', 'VARCHAR'],
        ['depolarization_time_mean', 'DECIMAL(4,1)'],
        ['depolarization_time_std', 'DECIMAL(5,1)'],
        ['apd50_str', 'VARCHAR'],
        ['apd50_mean', 'DECIMAL(5,1)'],
        ['apd50_std', 'DECIMAL(6,1)'],
        ['apd90_str', 'VARCHAR'],
        ['apd90_mean', 'DECIMAL(5,1)'],
        ['apd90_std', 'DECIMAL(6,1)'],
    ]

    # convert to integers to prevent overflow
    AP_amplitudes_int = [int(AP_amplitudes[i]) if not np.isnan(AP_amplitudes[i]) else AP_amplitudes[i] for i in range(len(AP_amplitudes))]
    depolarization_time_int = [int(depolarization_time[i]) if not np.isnan(depolarization_time[i]) else depolarization_time[i] for i in range(len(depolarization_time))]
    APD50_int = [int(APD50[i]) if not np.isnan(APD50[i]) else APD50[i] for i in range(len(APD50))]
    APD90_int = [int(APD90[i]) if not np.isnan(APD90[i]) else APD90[i] for i in range(len(APD90))]

    # prepare data in appropriate forms
    cell_line = rec_info['cell_line']
    compound = rec_info['compound']
    note = rec_info['note']
    n_electrodes = electrodes_info['num_channels']
    AP_amplitudes_str = ' '.join(map(str, AP_amplitudes_int))
    AP_amplitudes_mean = np.nanmean(AP_amplitudes_int)
    AP_amplitudes_std = np.std(AP_amplitudes_int)
    depolarization_time_str = ' '.join(map(str, depolarization_time_int))
    depolarization_time_mean = np.nanmean(depolarization_time_int)
    depolarization_time_std = np.nanstd(depolarization_time_int)
    APD50_str = ' '.join(map(str, APD50_int))
    APD50_mean = np.nanmean(APD50)
    APD50_std = np.nanstd(APD50)
    APD90_str = ' '.join(map(str, APD90_int))
    APD90_mean = np.nanmean(APD90)
    APD90_std = np.nanstd(APD90)
    timestamp = datetime.datetime.now()

    values = [timestamp, cell_line, compound, note, file_path_full, gain, rec_duration, rec_proc_duration, electroporation_yield, n_electrodes, AP_amplitudes_str, AP_amplitudes_mean, AP_amplitudes_std, depolarization_time_str, depolarization_time_mean, depolarization_time_std, APD50_str, APD50_mean, APD50_std, APD90_str, APD90_mean, APD90_std]

    # upload data to SQL server
    _upload_to_sql_server(tablename, sql_columns, values)


def parse_rec_file_info_FP_AP(data_catalog, base_directory, index):    
    """Parse recording file information from the data catalog.
    
    Args:
        data_catalog (pandas.DataFrame): Data catalog containing directory and other (names of cell lines, compound, note) information of recording files.
        base_directory (str): Base directory where the recording files are stored.
        index (int): Index of the recording file in the data catalog.
    
    Returns:
        rec_info (dict): Recording information.
        file_path_full_FP (str): Full path to the FP recording file.
        file_path_full_AP (str): Full path to the AP recording file.
    """
    cell_line = data_catalog.loc[index,'cell_line']
    compound = data_catalog.loc[index,'compound']
    file_path = data_catalog.loc[index,'file_path']
    note = data_catalog.loc[index,'note']
    file_path_full_FP = base_directory+file_path+"/"+"FP_1.raw.h5"
    file_path_full_AP = base_directory+file_path+"/"+"stim_1.raw.h5"

    print(f"Processing recording file: {file_path} ...")

    rec_info = dict([
        ("cell_line", cell_line),
        ("compound", compound),
        ("file_path", file_path),
        ("note", note),
    ])

    return rec_info, file_path_full_FP, file_path_full_AP

def merge_FP_AP_features(
    rec_info, file_path_full_FP, file_path_full_AP, FP_electrodes_info, R_amplitudes, R_widths, FPDs, AP_amplitudes, depolarization_time, APD50, APD90, AP_electrodes, tablename
):
    """Merge FP and AP features and upload to SQL server.
    
    Args:
        rec_info (dict): Recording information.
        file_path_full_FP (str): Full path to the FP recording file.
        file_path_full_AP (str): Full path to the AP recording file.
        FP_electrodes_info (dict): Dictionary containing the channel information (electrode ID, X and Y locations, number of electrodes used for recording) of a FP recording file.
        R_amplitudes (list): List of R peak amplitudes (in microvolts).
        R_widths (list): List of R peak widths (in milliseconds).
        FPDs (list): List of FPDs (field potential durations) (in milliseconds).
        AP_amplitudes (list): List of AP amplitudes.
        depolarization_time (list): List of depolarization time.
        APD50 (list): List of action potential durations (APDs) at 50% repolarization.
        APD90 (list): List of action potential durations (APDs) at 90% repolarization.
        AP_electrodes (list): List of electrode IDs where APs were detected from a AP recording file.
        tablename (str): Name of the SQL table where the data is uploaded.
    """
    df_FP = pd.DataFrame(columns=['r_amplitude','r_width','fpd','fp_electrodes','file_path_full_fp'])
    df_FP['r_amplitude'] = R_amplitudes
    df_FP['r_width'] = R_widths
    df_FP['fpd'] = FPDs
    df_FP['fp_electrodes'] = FP_electrodes_info['electrode_ids']
    df_FP['file_path_full_fp'] = file_path_full_FP

    df_AP = pd.DataFrame(columns=['ap_amplitude','depolarization_time','apd50','apd90','ap_electrodes','file_path_full_ap'])
    df_AP['ap_amplitude'] = AP_amplitudes
    df_AP['depolarization_time'] = depolarization_time
    df_AP['apd50'] = APD50
    df_AP['apd90'] = APD90
    df_AP['ap_electrodes'] = AP_electrodes
    df_AP['file_path_full_ap'] = file_path_full_AP

    # merge FP and AP dataframes on electrode IDs
    df_merged = pd.merge(df_FP, df_AP, left_on='fp_electrodes', right_on='ap_electrodes', how='inner')

    df_merged['file_path'] = rec_info['file_path']
    df_merged['cell_line'] = rec_info['cell_line']
    df_merged['compound'] = rec_info['compound']
    df_merged['note'] = rec_info['note']
    df_merged['time'] = datetime.datetime.now()

    with open('conf/local/postgresql.txt', 'r') as f:
        sql_credentials = f.read().splitlines()
    
    register_adapter(np.int64, AsIs)
    register_adapter(np.float16, AsIs)

    # connect to sql server
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        engine = create_engine(f'postgresql://{sql_credentials[2]}:{sql_credentials[3]}@{sql_credentials[0]}:{sql_credentials[4]}/{sql_credentials[1]}', poolclass=NullPool)
        conn = engine.connect()
        df_merged.to_sql(tablename, engine, if_exists='append', index=False)
        conn.close()
        print('Database connection closed.')

    except Exception as error:
        print(error)
    
            