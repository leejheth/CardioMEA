"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.18.7
"""

import h5py
import glob
import numpy as np
import pandas as pd
import psycopg2
import yaml
from scipy import signal
from scipy.signal import find_peaks
from scipy import stats as st

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
        tmp = pd.concat([data_catalog.loc[[row]]]*len(rec_files), ignore_index=True)
        tmp["file_path_full"] = rec_files
        data_catalog_full = data_catalog_full.merge(tmp, how='outer')

    print(f"A total of {len(data_catalog_full)} recording files are found.")

    return data_catalog_full, len(data_catalog_full)


def write_yaml_file(counter):
    """Write a yaml file that contains the number of recording files."""
    data = dict(n_files = counter)

    with open('conf/base/file_count.yml', 'w') as f:
        yaml.dump(data, f)



def peakseek(data, minpeakdist=4000, minpeakh=1000):
    # This is approx. 6 times faster than find_peaks function
    # data format: 1D 
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

def get_R_timestamps(rec_file,test): #,synchronous_channels):
    # import hdf5 file
    # f = h5py.File(raw_data, 'r')
    # get data information
    # gain = (f['gain'][0]).astype(int)
    # print(gain)

    print(test+3)

    return 3, 4


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
