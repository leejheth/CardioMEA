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
import time
import warnings
import yaml
import hrvanalysis as hrva
from collections import defaultdict
from scipy import signal
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from lmfit import Model, Parameters
from psycopg2.extensions import register_adapter, AsIs
import sympy as sym


def list_rec_files(data_catalog,base_directory,ext):
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
    """
    data_catalog_full = pd.DataFrame(columns=data_catalog.columns.to_list() + ['file_path_full'])
    for row in range(len(data_catalog)):
        # list all files (with specified extension) in the directory
        rec_files = glob.glob(base_directory+str(data_catalog.loc[row,'file_path'])+"/*"+ext)
        
        # create a new dataframe that contains both existing and new data (full file directory)
        try:
            tmp = pd.concat([data_catalog.loc[[row]]]*len(rec_files), ignore_index=True)
            tmp["file_path_full"] = rec_files
            data_catalog_full = data_catalog_full.merge(tmp, how='outer')
        except Exception as e:
            print(e)
            print(f"No recording files are found in {base_directory+data_catalog.loc[row,'file_path']}")

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


def parse_rec_file_info(data_catalog_full, dummy, index):
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
    """
    obj = h5py.File(file_path_full, mode='r')

    # get channel information
    mapping = obj.get('mapping')
    channels = mapping['channel']
    electrodes = mapping['electrode']
    routed_idxs = np.where(electrodes > -1)[0] # remove unused channels
    channel_ids = channels[routed_idxs]
    electrode_ids = list(electrodes[routed_idxs])
    num_channels = len(electrode_ids)
    num_frames = len(obj.get('sig')[0,:])
    x_locs = list(mapping['x'][routed_idxs])
    y_locs = list(mapping['y'][routed_idxs])

    electrodes_info = dict([
        ("electrode_ids", electrode_ids),
        ("x_locs", x_locs),
        ("y_locs", y_locs),
        ("num_channels", num_channels)
    ])

    # get lsb value
    gain = (obj['settings']['gain'][0]).astype(int)
    if 'lsb' in obj['settings']:
        lsb = obj['settings']['lsb'][0] * 1e6
    else:
        lsb = 3.3 / (1024 * gain) * 1e6

    
    # get raw voltage traces from all recording channels
    if start_frame < (num_frames-1):
        if (start_frame+length) > num_frames:
            warnings.warn("The specified time frame exceeds the data length. Signals will be extracted until end-of-file.")
        signals = (obj.get('sig')[channel_ids,start_frame:start_frame+length] * lsb).astype('float16')
    else:
        warnings.warn(f"The start frame exceeds the length of data. Signals will be extracted from the start of the recording until MIN({length}-th frame, end-of-file) instead.")
        signals = (obj.get('sig')[channel_ids,0:min(length,num_frames)] * lsb).astype('float16')

    return signals, electrodes_info, gain, num_frames/s_freq


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
    # apply a digital filter
    filtered = signal.filtfilt(b, a, signals) 

    channelIDs = [ch for ch in range(len(signals))]

    n_Rpeaks=[]
    r_timestamps=[]
    for ch in channelIDs:
        n_r_locs, r_locs = _R_timestamps(filtered[ch],mult_factor,min_peak_dist)
        n_Rpeaks.append(n_r_locs)
        r_timestamps.append(r_locs)

    # identify synchronous beats
    sync_beats = st.mode(n_Rpeaks)
    # indices of channels where beats (R peaks) are synchronous
    ind_sync_channels = [ind for ind,peaks in enumerate(n_Rpeaks) if peaks==sync_beats]
    sync_timestamps = [r_timestamps[i] for i in ind_sync_channels]
    sync_channelIDs = [channelIDs[i] for i in ind_sync_channels]

    # update electrodes info (updated data contains only channels that captured synchronous beatings)
    electrodes_info_updated = dict([
        ("electrode_ids", [electrodes_info['electrode_ids'][i] for i in ind_sync_channels]),
        ("x_locs", [electrodes_info['x_locs'][i] for i in ind_sync_channels]),
        ("y_locs", [electrodes_info['y_locs'][i] for i in ind_sync_channels]),
        ("num_channels", len(ind_sync_channels))
    ])

    return sync_timestamps, sync_channelIDs, electrodes_info_updated


def _R_timestamps(signal_single,mult_factor,min_peak_dist):
    """Identify R peaks in a single channel."""
    thr = mult_factor*np.std(signal_single)
    # r_locs, _ = _peakseek(signal_single, minpeakdist=min_peak_dist, minpeakh=thr)
    r_locs, _ = find_peaks(signal_single, distance=min_peak_dist, prominence=thr)
    find_peaks

    return len(r_locs), r_locs


def _peakseek(data, minpeakdist, minpeakh):
    """Find peaks in a 1D array."""
    locs = np.where((data[1:-1] >= data[0:-2]) & (data[1:-1] >= data[2:]))[0] + 1
    
    if minpeakh:
        locs = locs[data[locs] > minpeakh]

    if minpeakdist > 1:
        while 1:
            multi = (locs[1:] - locs[0:-1]) < minpeakdist
            if not any(multi):
                break
            pks = data[locs]
            all_pks = np.array([[pks[i] for i in range(len(multi)) if multi[i]], [pks[i+1] for i in range(len(multi)) if multi[i]]])
            min_ind = np.argmin(all_pks,axis=0)

            multi_x = np.where(multi)[0]
            multi_xx = list(multi_x[min_ind==0]) + list(multi_x[min_ind==1] + 1)
            locs = np.delete(list(locs), multi_xx)
    pks = data[locs]
    
    return list(locs), pks
  

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
    
    # get an estimate of R spike width (duration of deviation from baseline)
    b, a = signal.butter(3,[100*2/s_freq,2000*2/s_freq],btype='band')
    R_filtered = signal.filtfilt(b, a, wave)
    try:
        R_width = 1e3*(np.where(abs(R_filtered)>30)[0][-1]-np.where(abs(R_filtered)>30)[0][0])/s_freq # in milliseconds
    except Exception:
        R_width = np.nan

    # get FPD
    b, a = signal.butter(3,[3*2/s_freq,100*2/s_freq],btype='band')
    T_filtered = signal.filtfilt(b, a, wave)
    # Tpeak, _ = _peakseek(T_filtered[before_R+T_from:before_R+T_to], minpeakdist=len(wave), minpeakh=30) # find just 1 peak
    Tpeak, _ = find_peaks(T_filtered[before_R+T_from:before_R+T_to], distance=len(wave), prominence=15) # find just 1 peak
    if Tpeak.size > 0:
        FPD = 1e3*(Tpeak[0]+T_from)/s_freq # in milliseconds
    else: # if no T peak is found
        FPD = np.nan
        
    return R_amplitude, R_width, FPD


def get_HRV_features(sync_timestamps):
    """Extract heart rate variability (HRV) features.
    
    Args:
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.

    Returns:
        HRV_features (dict): Dictionary of HRV features.
    """
    res=[]
    for timestamps in sync_timestamps:
        res.append(_hrv_features(timestamps))

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
    # return pd.DataFrame.from_dict(dict(HRV_features), orient='index').T


def _hrv_features(timestamps):
    """Extract HRV features from a single channel.
    Documentation that explains the features: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html
    """
    RR_intervals = np.diff(timestamps)
    
    try:
        # get time domain features
        time_domain_features = hrva.get_time_domain_features(RR_intervals)
        
        # get geometrical features
        geometrical_features = hrva.get_geometrical_features(RR_intervals)
        
        # get frequency domain features
        frequency_domain_features = hrva.get_frequency_domain_features(RR_intervals)
        
        # get csi and cvi features
        csi_cvi_features = hrva.get_csi_cvi_features(RR_intervals)
        
        # get poincare plot features
        poincare_plot_features = hrva.get_poincare_plot_features(RR_intervals)

        all_hrv_features = {**time_domain_features, **geometrical_features, **frequency_domain_features, **csi_cvi_features, **poincare_plot_features}
    
    except Exception:
        keys = ['mean_nni','sdnn','sdsd','nni_50','pnni_50','nni_20','pnni_20','rmssd','median_nni','range_nni','cvsd','cvnni','mean_hr','max_hr','min_hr','std_hr','triangular_index','tinn','lf','hf','lf_hf_ratio','lfnu','hfnu','total_power','vlf','csi','cvi','Modified_csi','sd1','sd2','ratio_sd2_sd1']
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


def upload_to_sql_server(rec_info,file_path_full,gain,rec_duration,electrodes_info_updated,active_area,R_amplitudes,R_widths,FPDs,HRV_features,conduction_speed,n_beats,tablename):
    """Upload extracted feature data to SQL server.
    
    Args:
        rec_info (dict): Recording information.
        file_path_full (str): Full path to the recording file.
        gain (int): Gain of the recording.
        rec_duration (float): Duration of the recording in seconds.
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
        ['n_electrodes_sync', 'SMALLINT'],
        ['active_area_in_percent', 'DECIMAL(4,1)'],
        ['R_amplitudes_str', 'VARCHAR'],
        ['R_amplitudes_mean', 'DECIMAL(6,1)'],
        ['R_amplitudes_std', 'DECIMAL(7,1)'],
        ['R_widths_str', 'VARCHAR'],
        ['R_widths_mean', 'DECIMAL(4,1)'],
        ['R_widths_std', 'DECIMAL(5,1)'],
        ['FPDs_str', 'VARCHAR'],
        ['FPDs_mean', 'DECIMAL(4,1)'],
        ['FPDs_std', 'DECIMAL(5,1)'],
        ['conduction_speed_str', 'VARCHAR'],
        ['conduction_speed_mean', 'DECIMAL(3,1)'],
        ['conduction_speed_std', 'DECIMAL(4,1)'],
        ['n_beats', 'SMALLINT']
        ]

    HRV_col = [[key, 'DECIMAL'] for key, _ in HRV_features.items()]

    # add HRV features to the list of SQL columns
    sql_columns.extend(HRV_col)

    # convert to integers to prevent overflow
    R_amplitudes_int = list(map(int,R_amplitudes))

    # prepare data in appropriate forms
    cell_line = rec_info['cell_line']
    compound = rec_info['compound']
    note = rec_info['note']
    n_electrodes_sync = electrodes_info_updated['num_channels']
    R_amplitudes_str = ' '.join(map(str, R_amplitudes_int))
    R_amplitudes_mean = np.mean(R_amplitudes_int)
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

    values = [timestamp, cell_line, compound, note, file_path_full, gain, rec_duration, n_electrodes_sync, active_area, R_amplitudes_str, R_amplitudes_mean, R_amplitudes_std, R_widths_str, R_widths_mean, R_widths_std, FPDs_str, FPDs_mean, FPDs_std, conduction_speed_str, conduction_speed_mean, conduction_speed_std, n_beats]

    HRV_values = [value for _, value in HRV_features.items()]
    
    # add HRV features to the list of values
    values.extend(HRV_values)

    # get PostgreSQL database information and credentials data from text file
    # .txt file format: host, database, user, password in each line
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
    
    return 1
