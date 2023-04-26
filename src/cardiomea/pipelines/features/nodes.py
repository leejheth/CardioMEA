"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import h5py
import glob
import numpy as np
import pandas as pd
import psycopg2
import statistics as st
import warnings
import yaml
from scipy import signal
from scipy.signal import find_peaks
from joblib import Parallel, delayed

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
        rec_files = glob.glob(base_directory+data_catalog.loc[row,'file_path']+"/*"+ext)
        
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


def write_yaml_file(counter):
    """Write a yaml file that contains the number of recording files."""
    data = dict(n_files = counter)

    with open('conf/base/file_count.yml', 'w') as f:
        yaml.dump(data, f)


def parse_rec_file_info(data_catalog_full, index):
    """Parse the information of a recording file.

    Args:
        data_catalog_full (pandas.DataFrame): Data catalog containing directory and other (names of cell lines, compound, etc.) information of recording files, as well as the full directory of the recording files.
        index (int): Index of the recording file to be parsed.

    Returns:
        cell_line (str): Name of the cell line (if provided).
        compound (str): Name of the compound (if provided).
        file_path (str): Directory where the recording file was found.
        note (str): Note about the recording (if provided).
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


def extract_data(file_path_full, start_frame, length):
    """Extract data from a recording file.

    Args:
        file_path_full (str): Full directory of the recording file.
        start_frame (int): Start frame of the recording.
        length (int): Length of the recording.

    Returns:
        data (dict): Dictionary containing the extracted data.
        signals (numpy.ndarray): Extracted signals.
        electrodes_info (dict): Dictionary containing the channel information (electrode ID, X and Y locations, number of electrodes used for recording).
        gain (int): Gain of the recording.
    """
    obj = h5py.File(file_path_full, mode='r')

    # get channel information
    mapping = obj.get('mapping')
    channels = np.array(mapping['channel'])
    electrodes = np.array(mapping['electrode'])
    routed_idxs = np.where(electrodes > -1)[0] # remove unused channels
    channel_ids = list(channels[routed_idxs])
    electrode_ids = list(electrodes[routed_idxs])
    num_channels = len(electrode_ids)
    num_frames = len(obj.get('sig')[0,:])
    x_locs = mapping['x']
    y_locs = mapping['y']

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
        signals = (obj.get('sig')[np.array(channel_ids),int(start_frame):int(start_frame+length)] * lsb).astype('float32')
    else:
        warnings.warn(f"The start frame exceeds the length of data. Signals will be extracted from the start of the recording until MIN({length}-th frame, end-of-file) instead.")
        signals = (obj.get('sig')[np.array(channel_ids),0:min(length,num_frames)] * lsb).astype('float32')

    return signals, electrodes_info, gain


def get_R_timestamps(signals,mult_factor,min_peak_dist,n_CPUs):
    """Identify R peaks in the signals.
    
    Args:
        signals (numpy.ndarray): Extracted signals.
        mult_factor (float): Multiplication factor for the threshold.
        min_peak_dist (int): Minimum distance between two R peaks.
        n_CPUs (int): Number of CPUs to be used for parallel processing.
        
    Returns:
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        sync_channelIDs (list): List of channel IDs of R peaks that are synchronous.    
    """
    # build a butterworth filter of order: 3, bandwidth: 100-2000Hz, bandpass
    b, a = signal.butter(3,[100*2/2e4,2000*2/2e4],btype='band')
    # apply a digital filter
    filtered = signal.filtfilt(b, a, signals) 

    channelIDs = [ch for ch in range(len(signals))]
    # Parallel processing using all available CPUs
    res = Parallel(n_jobs=n_CPUs, backend='multiprocessing')([delayed(_R_timestamps)(filtered[ch],mult_factor,min_peak_dist) for ch in channelIDs])
    n_Rpeaks, r_timestamps = map(list,zip(*res))

    # identify synchronous beats
    sync_beats = st.mode(n_Rpeaks)
    # indices of channels where beats (R peaks) are synchronous
    ind_sync_channels = [ind for ind,peaks in enumerate(n_Rpeaks) if peaks==sync_beats]
    sync_timestamps = [r_timestamps[i] for i in ind_sync_channels]
    sync_channelIDs = [channelIDs[i] for i in ind_sync_channels]

    return sync_timestamps, sync_channelIDs


def _R_timestamps(signal_single,mult_factor,min_peak_dist):
    """Identify R peaks in a single channel."""
    thr = mult_factor*np.std(signal_single)
    r_locs, _ = _peakseek(signal_single, minpeakdist=min_peak_dist, minpeakh=thr)
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
    
    return locs, pks


def get_FP_waves(signals,sync_timestamps,sync_channelIDs,before_R,after_R,n_CPUs):
    """Extract FP waves from the signals.
    
    Args:
        signals (numpy.ndarray): Raw signals.
        sync_timestamps (list): List of timestamps of R peaks that are synchronous.
        sync_channelIDs (list): List of channel IDs of R peaks that are synchronous.
        before_R (int): Number of frames before R peak.
        after_R (int): Number of frames after R peak.
        n_CPUs (int): Number of CPUs to use for parallel processing.
        
    Returns:
        FP_waves (list): List of FP waves.
    """
    # Parallel processing using all available CPUs
    FP_waves = Parallel(n_jobs=n_CPUs, backend='multiprocessing')([delayed(_FP_wave)(signals[ch],sync_timestamps[ch],before_R,after_R) for ch in sync_channelIDs])
    
    return FP_waves


def _FP_wave(signal_single,timestamps,before_R,after_R):
    """Get an average FP wave from a single channel."""
    waves = [signal_single[peak_loc-before_R:peak_loc+after_R] for peak_loc in timestamps if (peak_loc-before_R)>0 and (peak_loc+after_R)<=len(signal_single)] 
    return list(np.mean(np.vstack(waves),axis=0))


def get_FP_wave_features(FP_waves,before_R,n_CPUs):
    """Extract features from FP waves.
    
    Args:
        FP_waves (list): List of FP waves.
        n_CPUs (int): Number of CPUs to use for parallel processing.
        
    Returns:
        features (list): List of features.
    """
    # Parallel processing using all available CPUs
    res = Parallel(n_jobs=n_CPUs, backend='multiprocessing')([delayed(_FP_wave_features)(wave,before_R) for wave in FP_waves])
    FPDs, R_amplitudes, R_widths = map(list(zip(*res)))

    return FPDs, R_amplitudes, R_widths


def _FP_wave_features(wave,before_R):
    """Extract features from a single FP wave."""
    # get R peak location
    R_peak = before_R

    # get R peak-to-peak amplitude
    R_amplitude = wave[before_R]-np.min(wave)
    
    # get an estimate of R spike width (duration of deviation from baseline)
    b, a = signal.butter(3,[100*2/2e4,2000*2/2e4],btype='band')
    R_filtered = signal.filtfilt(b, a, wave)
    R_width = np.where(abs(R_filtered)>50)[0][-1]-np.where(abs(R_filtered)>50)[0][0]

    # get FPD
    b, a = signal.butter(3,[3*2/2e4,100*2/2e4],btype='band')
    T_filtered = signal.filtfilt(b, a, wave)
    Tpeak, _ = _peakseek(T_filtered[before_R+1000:after_R], minpeakdist=10000, minpeakh=30)
    Tpeak+before_R+1000
    
    
    return R_amplitude, R_width, FPD


def upload_to_sql_server(file_path):
    # get credentials data from text file
    with open('conf/local/postgresql.txt', 'r') as f:
        sql_credentials = f.read().splitlines()
    
    # connect to sql server
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host=sql_credentials[1],
            database=sql_credentials[2],
            user=sql_credentials[3],
            password=sql_credentials[4]
        )
		
        # create a cursor
        cur = conn.cursor()

        query = "INSERT INTO tablename (text_for_field1, text_for_field2, text_for_field3, text_for_field4) VALUES (%s, %s, %s, %s)"
        cur.execute(query, (field1, field2, field3, field4))
        conn.commit()
                    
	    # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
