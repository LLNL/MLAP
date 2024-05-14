import os
import sys
import os.path as path
import psutil
import glob
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from matplotlib import pyplot as plt
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer
import time
import random

# []
'''
Generate seed for initializing the random number generator
'''
def generate_seed ():
    return int (time.time())

# []
'''
Initializing the random number generator using the seed generated
using the current epoch time
'''
def init_random_generator (seed):
    random.seed(seed)
    random_state = random.getstate()
    return random_state

# []
'''
Get the master list of files containing the historical atmospheric data
'''
def get_data_file_names(data_files_dir):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_data_file_names"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nGetting the names of the data files at the dir : \n {} \n'.format(data_files_dir))
    years_list = os.listdir(data_files_dir)
    print('years_list: {} \n'.format(years_list))
    
    file_list = []
    for year in years_list:
        #print('Getting the names of the data files for the year : {}'.format(year))
        file_list_for_year = []
        for file in os.listdir(path.join(data_files_dir, year)):
            if file.startswith('wrf') and file.endswith('.nc'):
                file_list_for_year.append(file)
        #print('... Found {} files for this year'.format(len(file_list_for_year)))
        file_list.extend(file_list_for_year)

    file_list.sort()
    
    print('\nFound a total of {} files \n'.format(len(file_list)))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return file_list


# []
'''
Get the indices in the data files list for the fire time stamps
'''
def get_fire_time_indices (fire_time_stamps, data_files_list):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_fire_time_indices"')
    print('\nProcess in the module(): {}'.format(process))
    
    fire_time_indices = {}
    
    for fire_name in fire_time_stamps.keys():
        fire_time_indices_current = {}
        for time_stamp_key in fire_time_stamps[fire_name].keys():
            time_stamp = fire_time_stamps[fire_name][time_stamp_key]
            data_file_name = 'wrf_{}.nc'.format(time_stamp)
            data_file_index = data_files_list.index(data_file_name)
            fire_time_indices_current[time_stamp_key] = data_file_index
            
        fire_time_indices[fire_name] = fire_time_indices_current
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')    
    return fire_time_indices


# []
'''
Remove the data around the fires of concern
'''
def remove_data_around_fire (fire_time_indices, data_files_list):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "remove_data_around_fire"')
    print('\nProcess in the module(): {}'.format(process))
    
    fire_indices_to_delete = []
    for fire_name in fire_time_indices.keys():
        fire_start_ind = fire_time_indices[fire_name]['Start']
        fire_end_ind = fire_time_indices[fire_name]['End']
        fire_indices_to_delete.extend(range(fire_start_ind, fire_end_ind + 1))

    print('Removing {} data files around fires, out of total {}. [{} %]'.format(\
                                 len(fire_indices_to_delete), len(data_files_list),\
                                 100.0*float(len(fire_indices_to_delete))/len(data_files_list)))
    
    data_files_list = list (np.delete(np.array(data_files_list), fire_indices_to_delete))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('{} data files remaining after removing data around fires'.format(len(data_files_list)))
    print('=========================================================================')    
    return data_files_list

# []
'''
Read Fire data
'''
'''
def read_fire_data (fire_time_indices, max_history_to_consider, history_interval, \
                    data_files_list_all, data_files_location):
    
    fire_data = dict()
    
    for fire_name in fire_time_indices.keys():
        fire_time_indices[fire_name]['Hist'] = get_history_file_indices (\
            [fire_time_indices[fire_name]['Ref']], max_history_to_consider, history_interval)[0]

        fire_time_indices[fire_name]['ToRead'] = [fire_time_indices[fire_name]['Ref']]
        fire_time_indices[fire_name]['ToRead'].extend(fire_time_indices[fire_name]['Hist'])
        
    for fire_name in fire_time_indices.keys():
        fire_data_files_to_read, fire_time_stamps_to_read, fire_file_indices_data_dict = \
                    read_data_all_possible_times (fire_time_indices[fire_name]['ToRead'], \
                                                  data_files_list_all, \
                                                  data_files_location)

        fire_data[fire_name] = {
            'fire_data_files_to_read': fire_data_files_to_read,
            'fire_time_stamps_to_read': fire_time_stamps_to_read,
            'fire_file_indices_data_dict': fire_file_indices_data_dict
        }
    print('=========================================================================')    
    return fire_time_indices, fire_data
'''

# []
'''
Downsample the file indices/lists to use from the master list of files containing
the historical atmospheric data
Get the random reference time where Fuel Moisture is to be read and relative to which 
historical data is to be collected
'''
def downsample_data_files (data_files_list, percent_files_to_use, max_history_to_consider, random_state, sampling_type):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "downsample_data_files"')
    print('\nProcess in the module(): {}'.format(process))
    
    random.setstate(random_state)
    print('\nSelecting approx {} % of the data files'.format(percent_files_to_use))
    file_indices = set(range(len(data_files_list)))
    invalid_ind = set(range(max_history_to_consider))
    '''
    TODO:
    The way first "max_history_to_consider" number of indices after start are invalid, 
    "max_history_to_consider" number of indices right after each fire should also be invlalid.
    The reason is that after we remove fire data, the indices shrink. So, if the index 
    corresponding to timestamp right after fire is selected, the history data would be from
    ~ 2 weeks before it rather than until "max_history_to_consider".
    NEED to the list of invalid indices.
    '''
    valid_indices = list(file_indices - invalid_ind)
    total_files = len(valid_indices)
    downsample_files_count = round(percent_files_to_use*total_files/100.0)
    
    if (sampling_type == "random"):
        sampled_file_indices = random.sample(valid_indices, k = downsample_files_count)
    elif (sampling_type == "uniform"):
        sampled_file_indices = list(np.linspace(min(valid_indices), max(valid_indices), \
                                      downsample_files_count).astype(int))
    else:
        raise ValueError('Invalid "sampling_type": "{}". \
                        \nValid types are: "random", and "uniform"'.format(sampling_type))
        
    sampled_data_files = list(np.array(data_files_list)[sampled_file_indices])
    
    print('Selected {} data files out of {} total and {} usable considering historical data'.format(len(sampled_data_files), len(data_files_list), len(valid_indices)))
    #print('Indices of the randomly selected files: \n {}'.format(sampled_file_indices))
    #print('Names of the randomly selected files: \n {}'.format(sampled_data_files))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return sampled_file_indices, sampled_data_files





# []
'''
Get the history file indices corresponding to the sampled reference time indices
'''
def get_history_file_indices (sampled_file_indices, max_history_to_consider, history_interval):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_history_file_indices"')
    print('\nProcess in the module(): {}'.format(process))
    
    #print('\nDetermining history file indices corresponding to given file/time indices...')
    history_file_indices = []
    for fuel_moisture_time_index in sampled_file_indices:
        atm_data_time_indices = np.arange(fuel_moisture_time_index, \
                                         fuel_moisture_time_index - max_history_to_consider - 1,\
                                         - history_interval)
        atm_data_time_indices = list(np.sort(atm_data_time_indices)[:-1])
        history_file_indices.append(atm_data_time_indices)
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')    
    return history_file_indices

# []
'''
Get timestamps and datetime for the downsampled data files
'''
def get_datetime_for_data_files (sampled_data_files):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_datetime_for_data_files"')
    print('\nProcess in the module(): {}'.format(process))
    
    sampled_time_stamps = []
    sampled_datetime = []
    for filename in sampled_data_files:
        data_filename_split = filename.split('_')
        date = data_filename_split[1]
        hour = data_filename_split[2].split('.')[0]
        date_hour = date + '_' + hour
        sampled_time_stamps.append(date_hour)
        sampled_datetime.append(datetime.fromisoformat(date_hour))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return sampled_time_stamps, sampled_datetime

# []
'''
Create DataFrame using sampled file indices, filenames, timestamps, and datetime
'''
def create_df_sampled_time (sampled_file_indices, sampled_data_files, sampled_time_stamps, sampled_datetime, history_file_indices):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "create_df_sampled_time"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nCreating DataFrame using sampled file indices, filenames, timestamps, and datetime...')
    df_sampled_time = pd.DataFrame()
    df_sampled_time['ref_time_indices'] = sampled_file_indices
    df_sampled_time['sampled_data_files'] = sampled_data_files
    df_sampled_time['sampled_time_stamps'] = sampled_time_stamps
    df_sampled_time['sampled_datetime'] = sampled_datetime
    df_sampled_time['history_time_indices'] = history_file_indices
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return df_sampled_time

# []
'''
Plot the sampled datetime
'''
def plot_sampled_datetime (df_sampled_time, extracted_data_loc, xlim = None, ylim = None):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_sampled_datetime"')
    print('\nProcess in the module(): {}'.format(process))
    
    plt.figure()
    print('\nPlotting sampled datetime from the available data...')
    sampled_datetime = df_sampled_time['sampled_datetime']
    plt.scatter(range(len(sampled_datetime)), sampled_datetime)
    plt.xlabel('Indices of reference time [-]', fontsize=20)
    plt.ylabel('Datetime [-]', fontsize=20)
    if xlim:
        plt.xlim([xlim[0],xlim[1]])
    if ylim:
        plt.ylim([ylim[0],ylim[1]])
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.title('Datetime of the {} sampled reference time'.format(len(sampled_datetime)), fontsize=20)

    if xlim or ylim:
        filename = 'Sampled_Datetime_Bounded.png'
    else:
        filename = 'Sampled_Datetime_Unbounded.png'

    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    plt.figure()
    count, bins, ignored = plt.hist(sampled_datetime, 50)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.savefig(os.path.join(filedir, 'Sampled_Datetime_Bounded_Histogram'), bbox_inches='tight')
    #plt.show()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    
#[]
'''
Read a single data file
'''
def read_single_data_file (data_files_location, data_file_to_read, timestamp_to_read):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "read_single_data_file"')
    print('\nProcess in the module(): {}'.format(process))
    
    #print('\nReading data contained in the selcted file: {}...'.format(data_file_to_read))
    data_at_timestamp = {}
    
    year = data_file_to_read.split('_')[1].split('-')[0]
    dfm_file_data = xr.open_dataset(path.join(data_files_location, year, data_file_to_read))
    
    data_at_timestamp['TimeStamp' ] = timestamp_to_read
    data_at_timestamp['ny'        ] = dfm_file_data.dims['south_north']
    data_at_timestamp['nx'        ] = dfm_file_data.dims['west_east']
    data_at_timestamp['HGT'       ] = np.array(dfm_file_data['HGT'])
    data_at_timestamp['T2'        ] = np.array(dfm_file_data['T2'])
    data_at_timestamp['Q2'        ] = np.array(dfm_file_data['Q2'])
    data_at_timestamp['RH'        ] = np.array(dfm_file_data['RH'])
    data_at_timestamp['PRECIP'    ] = np.array(dfm_file_data['PRECIP'])
    data_at_timestamp['PSFC'      ] = np.array(dfm_file_data['PSFC'])
    data_at_timestamp['U10'       ] = np.array(dfm_file_data['U10'])
    data_at_timestamp['V10'       ] = np.array(dfm_file_data['V10'])
    data_at_timestamp['SWDOWN'    ] = np.array(dfm_file_data['SWDOWN'])
    data_at_timestamp['FMC_1hr'   ] = np.array(dfm_file_data['FMC_GC'])[:, :, 0]
    data_at_timestamp['FMC_10hr'  ] = np.array(dfm_file_data['FMC_GC'])[:, :, 1]
    data_at_timestamp['FMC_100hr' ] = np.array(dfm_file_data['FMC_GC'])[:, :, 2]
    data_at_timestamp['FMC_1000hr'] = np.array(dfm_file_data['FMC_GC'])[:, :, 3]
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return data_at_timestamp


#[]
'''
Read a single data file
'''
def read_single_data_file_lightweight (data_files_location, data_file_to_read, timestamp_to_read):
    #process = psutil.Process(os.getpid())
    #print('=========================================================================')
    #module_start_time = timer()
    #module_initial_memory = process.memory_info().rss
    #print('MODULE Name: "read_single_data_file"')
    #print('\nProcess in the module(): {}'.format(process))
    
    #print('\nReading data contained in the selcted file: {}...'.format(data_file_to_read))
    data_at_timestamp = {}
    
    year = data_file_to_read.split('_')[1].split('-')[0]
    dfm_file_data = xr.open_dataset(path.join(data_files_location, year, data_file_to_read))
    
    #data_at_timestamp['TimeStamp' ] = timestamp_to_read
    data_at_timestamp['ny'        ] = dfm_file_data.dims['south_north']
    data_at_timestamp['nx'        ] = dfm_file_data.dims['west_east']
    data_at_timestamp['HGT'       ] = np.array(dfm_file_data['HGT'])
    data_at_timestamp['T2'        ] = np.array(dfm_file_data['T2'])
    #data_at_timestamp['Q2'        ] = np.array(dfm_file_data['Q2'])
    data_at_timestamp['RH'        ] = np.array(dfm_file_data['RH'])
    data_at_timestamp['PRECIP'    ] = np.array(dfm_file_data['PRECIP'])
    #data_at_timestamp['PSFC'      ] = np.array(dfm_file_data['PSFC'])
    data_at_timestamp['U10'       ] = np.array(dfm_file_data['U10'])
    data_at_timestamp['V10'       ] = np.array(dfm_file_data['V10'])
    data_at_timestamp['SWDOWN'    ] = np.array(dfm_file_data['SWDOWN'])
    #data_at_timestamp['FMC_1hr'   ] = np.array(dfm_file_data['FMC_GC'])[:, :, 0]
    data_at_timestamp['FM_10hr'  ] = np.array(dfm_file_data['FMC_GC'])[:, :, 1]
    data_at_timestamp['FM_100hr' ] = np.array(dfm_file_data['FMC_GC'])[:, :, 2]
    #data_at_timestamp['FMC_1000hr'] = np.array(dfm_file_data['FMC_GC'])[:, :, 3]
    
    #module_final_memory = process.memory_info().rss
    #module_end_time = timer()
    #module_memory_consumed = module_final_memory - module_initial_memory
    #module_compute_time = module_end_time - module_start_time
    #print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
    #print('Module computing time: {:.3f} s'.format(module_compute_time))
    #print('=========================================================================')
    return data_at_timestamp


#[]
'''
Read SJSU data at desired ref and historical times
'''
def read_SJSU_data_desired_times (time_region_info, data_files_location):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "read_SJSU_data_desired_times"')
    print('\nProcess in the module(): {}'.format(process))
    
    data_read_SJSU = dict()
    for count_ref_time, item_ref_time in enumerate(time_region_info['SJSU']):
        timestamps_to_read = [item_ref_time['RefTime']]
        for hist_timestamp in item_ref_time['HistTime']:
            timestamps_to_read.append(hist_timestamp)
        for timestamp in timestamps_to_read:
            data_file_to_read = 'wrf_%s.nc'%(timestamp)
            #print(data_file_to_read)
            data_read_SJSU[timestamp] = read_single_data_file_lightweight (\
                                                               data_files_location, \
                                                               data_file_to_read, \
                                                               timestamp)
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return data_read_SJSU

#[]
'''
process elevation from data read from a single file
'''
def process_elevation_at_timestamp (data_at_timestamp):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "process_elevation_at_timestamp"')
    print('\nProcess in the module(): {}'.format(process))
    
    #print('\nProcessing elevation data into pos, neg, and zero...')
    HGT = data_at_timestamp['HGT']
    
    HGT_UPD = np.ones((data_at_timestamp['ny'], data_at_timestamp['nx']), int)
    HGT_UPD[np.where(HGT == 0)] = 0
    HGT_UPD[np.where(HGT < 0)] = -1
                     
    data_at_timestamp['HGT_UPD'] = HGT_UPD
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return data_at_timestamp


# []
'''
Get grid indices
'''
def get_grid_indices_given_data_at_timestamp (data_at_timestamp, x_clip, y_clip, \
                                              j_nevada, i_nevada, j_anchor, i_anchor, 
                                              remove_nevada = True):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_grid_indices_given_data_at_timestamp"')
    print('\nProcess in the module(): {}'.format(process))
    
    # Extract relevant info from data at timestamp
    ny, nx = data_at_timestamp['ny'], data_at_timestamp['nx']
    
    data_at_timestamp = process_elevation_at_timestamp (data_at_timestamp)
    HGT_UPD = data_at_timestamp['HGT_UPD']

    # Initialize grid indices
    grid_indices_all = np.zeros((ny, nx), int)
    grid_indices_valid = np.zeros((ny, nx), int)

    for j in range(ny):
        for i in range(nx):
           grid_indices_all[j][i] = nx*j + i
           grid_indices_valid[j][i] = nx*j + i 

    grid_indices_valid[np.where(HGT_UPD == 0)] = -1
    #grid_indices_valid[np.where(HGT_UPD == -1)] = -1000

    # Remove points for Nevada
    if remove_nevada:
        for j in range(ny):
            for i in range(nx):
                if (j - j_nevada)*(i_anchor -   (nx-1)) < (j_anchor - j_nevada)*(i -   (nx-1)) and \
                   (j - j_anchor)*(i_nevada - i_anchor) < ((ny -1)  - j_anchor)*(i - i_anchor):
                   grid_indices_valid[j][i] = -1
    
    # Remove clipped points
    if x_clip is not None:
        grid_indices_valid[:, :x_clip[0]] = -1
        grid_indices_valid[:, x_clip[1]:] = -1
    if y_clip is not None:
        grid_indices_valid[:y_clip[0], :] = -1
        grid_indices_valid[y_clip[1]:, :] = -1
    
    # Flatten the indices
    grid_indices_all_flat = grid_indices_all.flatten()
    grid_indices_valid_flat = grid_indices_valid[np.where(grid_indices_valid >= 0)].flatten()
   
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return grid_indices_all, grid_indices_valid, grid_indices_all_flat, grid_indices_valid_flat


# []
'''
Get grid indices
'''
def get_grid_indices_all (data_at_timestamp, x_clip, y_clip, j_nevada, i_nevada, j_anchor, i_anchor, remove_nevada = True):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "get_grid_indices_all"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nGetting all the grid indices from data at a chosen timestamp {}...')

    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')  
    return get_grid_indices_given_data_at_timestamp (\
                                              data_at_timestamp, x_clip, y_clip, \
                                              j_nevada, i_nevada, j_anchor, i_anchor, \
                                              remove_nevada)


# []
'''
Reconstruct valid grid indices
'''
def reconstruct_valid_grid_indices (grid_indices_valid_flat, data_at_timestamp):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "reconstruct_valid_grid_indices"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nReconstructing valid grid indices...')
    nx = data_at_timestamp['nx']
    ny = data_at_timestamp['ny']
    
    grid_indices_valid_reconst = np.ones((ny, nx), int)*(-1)
    grid_indices_valid_bool = np.zeros((ny, nx), int)
    valid_grid_ind_to_coord = {}
    
    for valid_ind in grid_indices_valid_flat:
        j_ind = valid_ind // nx
        i_ind = valid_ind - j_ind*nx

        valid_grid_ind_to_coord[valid_ind] = (j_ind, i_ind)
        grid_indices_valid_reconst[j_ind][i_ind] = valid_ind
        grid_indices_valid_bool[j_ind][i_ind] = 1
        
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')    
    return grid_indices_valid_reconst, grid_indices_valid_bool, valid_grid_ind_to_coord

#[]
'''
Plot Contours of indices at a timestamp
'''
def plot_contours_of_indices (data_at_timestamp, grid_indices_all, grid_indices_valid, grid_indices_valid_bool, grid_indices_valid_reconst, extracted_data_loc):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_contours_of_indices"')
    print('\nProcess in the module(): {}'.format(process))
    
    cmap_name = 'hot'
    cont_levels = 20
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    x_ind, y_ind = np.meshgrid(range(data_at_timestamp['nx']), range(data_at_timestamp['ny']))

    cont = ax[0][0].contourf(x_ind, y_ind, grid_indices_all, levels = cont_levels, cmap=cmap_name, extend='both')
    ax[0][0].set_title('All Indices')
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])

    cont = ax[0][1].contourf(x_ind, y_ind, grid_indices_valid, levels = cont_levels, cmap=cmap_name, extend='both')
    ax[0][1].set_title('Valid Indices')
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])

    cont = ax[1][0].contourf(x_ind, y_ind, grid_indices_valid_bool, levels = cont_levels, cmap=cmap_name, extend='both')
    ax[1][0].set_title('Reconstructed Indices (Boolean)')
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])

    cont = ax[1][1].contourf(x_ind, y_ind, grid_indices_valid_reconst, levels = cont_levels, cmap=cmap_name, extend='both')
    ax[1][1].set_title('Reconstructed Valid Indices')
    ax[1][1].set_xticks([])
    ax[1][1].set_yticks([])

    
    # Save plots
    filename = 'GridIndices.png'
 
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')


#[]
'''
Plot Contours of Data at a TimeStamp
'''
def plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc, grid_indices_valid, masked = True):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_contours_at_timestamp"')
    print('\nProcess in the module(): {}'.format(process))
    
    cmap_name = 'hot'
    masked_string = 'UnMasked' 
    cont_levels = 20
    fig, axlist = plt.subplots(3, 4, figsize=(12, 8))
    for var_ind, ax in enumerate(axlist.ravel()):
        qoi = qoi_to_plot[var_ind]
        data_to_plot = data_at_timestamp[qoi] #Unmasked
        if masked:
            mask = np.zeros_like(grid_indices_valid, dtype=bool)
            mask[np.where(grid_indices_valid < 0)] = True
            data_to_plot = np.ma.array(data_to_plot, mask=mask)
            masked_string = 'Masked'
        
        x_ind, y_ind = np.meshgrid(range(data_at_timestamp['nx']), range(data_at_timestamp['ny']))
        cont = ax.contourf(x_ind, y_ind, data_to_plot, levels = cont_levels, cmap=cmap_name, extend='both')
        #clb = plt.colorbar(cont)
        #clb.ax.tick_params(labelsize=14)
        #plt.xlabel('x-loc [Grid-Index]', fontsize=14)
        #plt.ylabel('y-loc [Grid-Index]', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.tick_params(axis='x', labelsize=14)
        #plt.tick_params(axis='y', labelsize=14)
        ax.set_title('{}'.format(qoi),fontsize=10)
        #ax.show()
        #plt.close()
    
    filename = 'Contours_{}_QoIs_{}.png'.format(masked_string, data_at_timestamp['TimeStamp'])
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
 
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')

    
#[]
'''
Plot Contours of Data at a TimeStamp
'''
def plot_contours_at_timestamp2 (data_at_timestamp, timestamp_to_read, qoi_to_plot, extracted_data_loc, grid_indices_valid, cont_levels_count, qoi_cont_range, masked = True):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_contours_at_timestamp2"')
    print('\nProcess in the module(): {}'.format(process))
    
    cmap_name = 'hot'
    masked_string = 'UnMasked' 
    if qoi_cont_range:
        cont_levels = np.linspace(qoi_cont_range[0], qoi_cont_range[1], cont_levels_count)
    else:
        cont_levels = cont_levels_count
        
    for qoi in qoi_to_plot:
        data_to_plot = data_at_timestamp[qoi] #Unmasked
        if masked:
            mask = np.zeros_like(grid_indices_valid, dtype=bool)
            mask[np.where(grid_indices_valid < 0)] = True
            data_to_plot = np.ma.array(data_to_plot, mask=mask)
            masked_string = 'Masked'
        
        plt.figure(figsize=(9, 6))
        x_ind, y_ind = np.meshgrid(range(data_at_timestamp['nx']), range(data_at_timestamp['ny']))
        cont = plt.contourf(x_ind, y_ind, data_to_plot, levels = cont_levels, cmap=cmap_name, extend='both')
        clb = plt.colorbar(cont)
        clb.ax.tick_params(labelsize=14)
        plt.xlabel('x-loc [Grid-Index]', fontsize=16)
        plt.ylabel('y-loc [Grid-Index]', fontsize=16)
        #ax.set_xticks([])
        #ax.set_yticks([])
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.title('{} at {}'.format(qoi, timestamp_to_read),fontsize=16)
        
        filename = 'Contours_{}_{}_{}.png'.format(masked_string, qoi, data_at_timestamp['TimeStamp'])
        filedir = extracted_data_loc
        os.system('mkdir -p %s'%filedir)

        plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
        #plt.show()
        #plt.close()
        
        module_final_memory = process.memory_info().rss
        module_end_time = timer()
        module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
        module_compute_time = module_end_time - module_start_time
        print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
        print('Module computing time: {:.3f} s'.format(module_compute_time))
        print('=========================================================================')
        
#[]
'''
Plot PDF of Data at a TimeStamp
'''
def plot_pdf_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_pdf_at_timestamp"')
    print('\nProcess in the module(): {}'.format(process))
    
    num_bins = 30
    fig, axlist = plt.subplots(3, 4, figsize=(12, 8))
    for var_ind, ax in enumerate(axlist.ravel()):
        qoi = qoi_to_plot[var_ind]
        hist, bin_edges = np.histogram(data_at_timestamp[qoi], bins = num_bins, density = True)
        bin_centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)
        
        #ax.bar(bin_centers, hist, color = 'lime')
        ax.plot(bin_centers, hist, color = 'b', label = qoi)
        ax.set_xticks(np.linspace(bin_centers.min(), bin_centers.max(), 4))
        ax.tick_params(axis='x', labelsize=10)
        ax.legend()
        #ax.set_title('{}'.format(qoi),fontsize=10)

    filename = 'PDFs_QoIs_{}.png'.format(data_at_timestamp['TimeStamp'])
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    
# []
'''
Sample grid indices for each ref time
'''
def sample_grid_indices (sampled_file_indices, percent_grid_points_to_use, grid_indices_valid_flat, valid_grid_ind_to_coord):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "sample_grid_indices"')
    print('\nProcess in the module(): {}'.format(process))
    
    downsample_grid_point_count = round(percent_grid_points_to_use*len(grid_indices_valid_flat)/100.0)
    print('Selecting {} grid points (approx {} % of a total of {} considerable/valid grid points)\n'.format(
            downsample_grid_point_count, percent_grid_points_to_use, len(grid_indices_valid_flat)))
    
    grid_indices_selected =np.zeros((len(sampled_file_indices), downsample_grid_point_count), int)
    j_indices_selected = np.zeros_like(grid_indices_selected, int)
    i_indices_selected = np.zeros_like(grid_indices_selected, int)
    
    for sampled_file_count in range(len(sampled_file_indices)):
        
        sampled_grid_indices = random.sample(set(grid_indices_valid_flat), \
                                             k = downsample_grid_point_count)
        
        for sampled_grid_point_count, sampled_grid_ind in enumerate(sampled_grid_indices):
            grid_indices_selected[sampled_file_count][sampled_grid_point_count] = sampled_grid_ind
            j_indices_selected[sampled_file_count][sampled_grid_point_count] = \
                                                    valid_grid_ind_to_coord[sampled_grid_ind][0]
            i_indices_selected[sampled_file_count][sampled_grid_point_count] = \
                                                    valid_grid_ind_to_coord[sampled_grid_ind][1]
    #End for
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return grid_indices_selected, j_indices_selected, i_indices_selected


# []
'''
Plot sampled grid indices for each ref time
'''
def plot_sampled_grid_points (grid_indices_selected, extracted_data_loc):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_sampled_grid_points"')
    print('\nProcess in the module(): {}'.format(process))
    
    cmap_name = 'hot'
    cont_levels = 20
    
    grid_count, time_count = grid_indices_selected.shape[1], grid_indices_selected.shape[0]
    grid_ind, time_ind = np.meshgrid(range(grid_count), range(time_count))

    plt.figure()
    cont = plt.contourf(grid_ind, time_ind, grid_indices_selected, levels = cont_levels, cmap=cmap_name, extend='both')
    clb = plt.colorbar(cont)
    clb.ax.tick_params(labelsize=14)
    plt.xlabel('Sampled Random Grid Point Count [-]', fontsize=14)
    plt.ylabel('Sampled Ref Time Count [-]', fontsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.title('Sampled grid index for each random ref time',fontsize=14)

    filename = 'Sampled_Grid_Indices.png'
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')

# []
'''
Plot sampled grid indices for each ref time in 3D
'''
def plot_sampled_grid_points_3D (j_indices_selected, i_indices_selected, extracted_data_loc, fig_size):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "plot_sampled_grid_points_3D"')
    print('\nProcess in the module(): {}'.format(process))
    
    fig = plt.figure(figsize = fig_size)
    ax = plt.axes(projection='3d')
    for ref_time_count in range(j_indices_selected.shape[0]):
        #sampled_datetime_current = sampled_datetime[ref_time_count]
        #print('Sampled DateTime: {}'.format(sampled_datetime_current))
        j_indices_at_current_time = j_indices_selected[ref_time_count]
        i_indices_at_current_time = i_indices_selected[ref_time_count]
        time_indices_at_current_time = np.ones_like(j_indices_at_current_time)*ref_time_count

        ax.scatter3D(i_indices_at_current_time,j_indices_at_current_time, 
                     time_indices_at_current_time)
   
    ax.set_xlabel('x-indices', fontsize=14)
    ax.set_ylabel('y-indices', fontsize=14)
    ax.set_zlabel('Sampled Ref Time Count [-]', fontsize=14)

    filename = 'Sampled_Grid_Indices_3D.png'
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)

    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')

# []
'''
Create a map of time indices and grid indices at that time where we need data
'''
def create_time_grid_indices_map (sampled_file_indices, history_file_indices, grid_indices_selected):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "create_time_grid_indices_map"')
    print('\nProcess in the module(): {}'.format(process))
    
    time_grid_indices_list_dict = dict()
    time_grid_indices_list_count = dict()
    time_grid_indices_set_dict = dict()
    time_grid_indices_set_count = dict()
    
    for sampled_time_count, sampled_time_ind in enumerate(sampled_file_indices):

        grid_indices_sampled_at_current_time = grid_indices_selected[sampled_time_count]

        time_grid_indices_list_dict[sampled_time_ind] = \
                np.hstack((time_grid_indices_list_dict.get(sampled_time_ind, \
                                                           np.empty((0,), dtype = int)), \
                           grid_indices_sampled_at_current_time))

        for history_time_index in history_file_indices[sampled_time_count]:
                time_grid_indices_list_dict[history_time_index] =\
                    np.hstack((time_grid_indices_list_dict.get(history_time_index,\
                                                               np.empty((0,), dtype = int)), \
                               grid_indices_sampled_at_current_time))

    # Derive other indices
    for chosen_time_index in time_grid_indices_list_dict.keys():
        time_grid_indices_list_count[chosen_time_index] = \
            len(time_grid_indices_list_dict[chosen_time_index])
        time_grid_indices_set_dict[chosen_time_index] = \
            set(time_grid_indices_list_dict[chosen_time_index])
        time_grid_indices_set_count[chosen_time_index] = \
            len(time_grid_indices_set_dict[chosen_time_index])
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return time_grid_indices_list_dict, time_grid_indices_list_count, time_grid_indices_set_dict, time_grid_indices_set_count

# []
'''
Read data at a desired time index and grid index
'''
def read_data_at_time_grid_as_dict (labels_to_read, labels_ind_in_nc_file, \
                                    features_to_read, \
                                    time_ind_to_read, grid_ind_to_read,\
                                    valid_grid_ind_to_coord, dfm_file_data):
    #print('=========================================================================')
    #print('MODULE Name: ""')
    
    data_at_time_and_grid = dict()
    
    j_ind_to_read, i_ind_to_read = valid_grid_ind_to_coord[grid_ind_to_read]
    
    for feature in features_to_read:      
        if (feature == 'UMag10'):
            data_at_time_and_grid[feature] = \
                    (np.array(dfm_file_data['U10'])[j_ind_to_read, i_ind_to_read]**2 +
                     np.array(dfm_file_data['V10'])[j_ind_to_read, i_ind_to_read]**2)**(0.5)
        else:
            data_at_time_and_grid[feature] = \
                np.array(dfm_file_data[feature])[j_ind_to_read, i_ind_to_read]
    
    for label, label_ind in zip(labels_to_read, labels_ind_in_nc_file):
        data_at_time_and_grid[label] = \
                np.array(dfm_file_data['FMC_GC'])[j_ind_to_read, i_ind_to_read][label_ind]
    
    #print('=========================================================================')
    return data_at_time_and_grid

# []
'''
Read data at a desired time index and grid index
'''
def read_data_at_time_grid_as_array (labels_to_read, labels_ind_in_nc_file, \
                                    features_to_read, \
                                    time_ind_to_read, grid_ind_to_read,\
                                    valid_grid_ind_to_coord, dfm_file_data):
    #print('=========================================================================')
    #print('MODULE Name: ""')
    data_at_time_and_grid = np.empty((len(features_to_read) + len(labels_to_read), ), \
                                         dtype = np.float16)
    
    j_ind_to_read, i_ind_to_read = valid_grid_ind_to_coord[grid_ind_to_read]
    
    for feature_count, feature in enumerate(features_to_read):
        if (feature == 'UMag10'):
            data_at_time_and_grid[feature_count] = \
                    (np.array(dfm_file_data['U10'])[j_ind_to_read, i_ind_to_read]**2 +
                     np.array(dfm_file_data['V10'])[j_ind_to_read, i_ind_to_read]**2)**(0.5)
        else:
            data_at_time_and_grid[feature_count] = \
                    np.array(dfm_file_data[feature])[j_ind_to_read, i_ind_to_read]
    
    for label_count , (label, label_ind) in enumerate(zip(labels_to_read, labels_ind_in_nc_file)):
        data_at_time_and_grid[len(features_to_read) + label_count] = \
                np.array(dfm_file_data['FMC_GC'])[j_ind_to_read, i_ind_to_read][label_ind]
    
    #print('=========================================================================')
    return data_at_time_and_grid

# []
'''
Read data at sampled time and grid indices
'''
def read_data_at_sampled_times_and_grids (labels_to_read, labels_ind_in_nc_file, \
                                          features_to_read, valid_grid_ind_to_coord, \
                                          time_grid_indices_set_dict, \
                                          data_files_location, data_files_list, \
                                          data_at_time_grid_type = 'dict'):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "read_data_at_sampled_times_and_grids"')
    print('\nProcess in the module(): {}'.format(process))
    
    data_at_sampled_times_and_grids = dict()
    
    for time_ind_to_read in time_grid_indices_set_dict.keys():
        data_file_to_read = data_files_list[time_ind_to_read]
        year = data_file_to_read.split('_')[1].split('-')[0]
        '''
        print('\ntime_ind_to_read: {}, data_file_to_read: {}'.format(\
                                                         time_ind_to_read, data_file_to_read))
        '''
        # Read the data file at the current time index
        file_read_start_time = timer()
        file_read_initial_memory = process.memory_info().rss
        
        dfm_file_data = xr.open_dataset(path.join(data_files_location, year, data_file_to_read))
        
        file_read_final_memory = process.memory_info().rss
        file_read_memory_consumed = \
                            file_read_final_memory - file_read_initial_memory
        file_read_end_time = timer()
        '''
        print('"xr.open_dataset ()"memory consumed: {:.3f} KB'.format(\
                                                        file_read_memory_consumed/(1024)))
        print('"xr.open_dataset ()" computing time: {:.3f} s'.format(\
                                                         file_read_end_time - file_read_start_time))
        '''
        # Extract sampled grid data at the current time index
        grid_indices_to_read_at_current_time = time_grid_indices_set_dict[time_ind_to_read]
        data_at_sampled_grids_at_current_time = dict()
        
        extract_grid_data_start_time = timer()
        extract_grid_data_initial_memory = process.memory_info().rss
        
        for grid_ind_to_read_at_current_time in grid_indices_to_read_at_current_time:
            if data_at_time_grid_type == 'dict':
                data_at_sampled_grids_at_current_time[grid_ind_to_read_at_current_time] = \
                    read_data_at_time_grid_as_dict(labels_to_read, labels_ind_in_nc_file, \
                                                   features_to_read, \
                                                   time_ind_to_read, grid_ind_to_read_at_current_time,\
                                                   valid_grid_ind_to_coord, dfm_file_data)
            elif data_at_time_grid_type == 'array':
                data_at_sampled_grids_at_current_time[grid_ind_to_read_at_current_time] = \
                    read_data_at_time_grid_as_array(labels_to_read, labels_ind_in_nc_file, \
                                                    features_to_read, \
                                                    time_ind_to_read, grid_ind_to_read_at_current_time,\
                                                    valid_grid_ind_to_coord, dfm_file_data)
            else:
                print("Unrecognized data type")
            

        extract_grid_data_final_memory = process.memory_info().rss
        extract_grid_data_memory_consumed = \
                            extract_grid_data_final_memory - extract_grid_data_initial_memory
        extract_grid_data_end_time = timer()
        '''
        print('"Extract grid data at current time" memory consumed: {:.3f} KB'.format(\
                                                        extract_grid_data_memory_consumed/(1024)))
        print('"Extract grid data at current time" computing time: {:.3f} s'.format(\
                                           extract_grid_data_end_time - extract_grid_data_start_time))
        '''
        data_at_sampled_times_and_grids[time_ind_to_read] = \
                                                data_at_sampled_grids_at_current_time
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return data_at_sampled_times_and_grids, module_memory_consumed, module_compute_time

# []
'''
Create DataFrame of FM and Historical Data. Also Extract Time and Grid Info.
'''
def create_dataframe_FM_atm_data (data_at_sampled_times_and_grids, data_at_timestamp, \
                                  sampled_file_indices, history_file_indices, \
                                  sampled_time_stamps, history_interval, \
                                  grid_indices_selected, \
                                  j_indices_selected, i_indices_selected,\
                                  labels_to_read, features_to_read):
    
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "create_dataframe_FM_atm_data"')
    print('\nProcess in the module(): {}'.format(process))
    
    ## Define the Sizes to Contain Sampled Data
    num_sampled_times  = grid_indices_selected.shape[0]
    num_sampled_points = grid_indices_selected.shape[1]
    num_data_points    = num_sampled_times*num_sampled_points
    num_hist_indices   = len(history_file_indices[0])
    
    FM_time_ind  = np.zeros((num_data_points, 1), np.int32)
    FM_ts        = np.zeros((num_data_points, 1), '<U13') #list()
    his_time_ind = np.zeros((num_data_points, num_hist_indices), np.int32)

    grid_ind     = np.zeros((num_data_points, 1), np.int32)
    j_ind        = np.zeros((num_data_points, 1), np.int32)
    i_ind        = np.zeros((num_data_points, 1), np.int32)

    HGT          = np.zeros((num_data_points, 1), np.float16)

    label_data_at_ref_time = dict()
    for label in labels_to_read:
        label_data_at_ref_time[label] = np.zeros((num_data_points, 1), np.float16)

    atm_data_at_hist_times = dict()
    for feature in features_to_read:
        atm_data_at_hist_times[feature] = np.zeros((num_data_points, num_hist_indices), np.float16)
        
        
    ## Fill in the Data Arrays
    for sampled_time_count, sampled_time_ind in enumerate(sampled_file_indices):
        hist_indices = np.array(history_file_indices[sampled_time_count])

        for sampled_grid_point_count in range(num_sampled_points):
            data_point_count = sampled_time_count*num_sampled_points + sampled_grid_point_count

            # Time Indices
            FM_time_ind [ data_point_count] = sampled_time_ind
            FM_ts [       data_point_count] = sampled_time_stamps [sampled_time_count]
            his_time_ind [data_point_count] = hist_indices

            # Grid Identifier
            grid_index = grid_indices_selected[sampled_time_count][sampled_grid_point_count]
            j_loc      = j_indices_selected[   sampled_time_count][sampled_grid_point_count]
            i_loc      = i_indices_selected[   sampled_time_count][sampled_grid_point_count]
            #print(sampled_time_count, sampled_grid_point_count, data_point_count)

            # Grid Indices
            grid_ind [    data_point_count] = grid_index
            j_ind [       data_point_count] = j_loc
            i_ind [       data_point_count] = i_loc

            # Height for Features
            HGT [data_point_count] = data_at_timestamp['HGT'][j_loc][i_loc]

            # FM at Ref Time (Labels)
            for label_count, label in enumerate(labels_to_read):
                label_data_at_ref_time[label][data_point_count] = \
                        data_at_sampled_times_and_grids[sampled_time_ind][grid_index][\
                                                  len(features_to_read) + label_count]

            # Historical Atmospheric Data (Features)
            for feature_count, feature in enumerate(features_to_read):
                for hist_ind_count, hist_ind in enumerate(hist_indices):
                    atm_data_at_hist_times[feature][data_point_count][hist_ind_count] = \
                            data_at_sampled_times_and_grids[hist_ind][grid_index][\
                                                      feature_count] 
    
    ## Create DataFrame
    df = pd.DataFrame()
    df['FM_time_ind'] = FM_time_ind.flatten()
    df['FM_ts'] = FM_ts
    df['his_time_ind'] = list(his_time_ind)

    df['grid_ind'] = grid_ind
    df['j_ind'] = j_ind
    df['i_ind'] = i_ind

    df['HGT'] = HGT
    
    for label_count, label in enumerate(labels_to_read):
        df[label] = label_data_at_ref_time[label]
        
    for hist_ind_count, hist_ind in enumerate(hist_indices):
        for feature_count, feature in enumerate(features_to_read):
            hist_hr = (num_hist_indices-hist_ind_count)*history_interval
            feature_header = '{}[-{}hr]'.format(feature, hist_hr)
            df[feature_header] = atm_data_at_hist_times[feature].T[hist_ind_count]
            
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    
    return df


# []
'''
Create DataFrame of FM and Historical Data For a Desired Time Stamp. Also Extract Time and Grid Info.
'''
# Account for if (feature == 'UMag10') and change accordingly in Step 5
def create_dataframe_FM_atm_at_timestamp (sampled_time_stamps, hist_stamps, data_read_SJSU, \
                                       history_interval, \
                                       grid_indices_valid_flat, valid_grid_ind_to_coord):
    
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "create_dataframe_FM_atm_at_timestamp"')
    print('\nProcess in the module(): {}'.format(process))
    
    ## Define the Sizes to Contain Sampled Data
    num_sampled_times  = 1
    num_sampled_points = len(grid_indices_valid_flat)
    num_data_points    = num_sampled_times*num_sampled_points
    num_hist_indices   = len(hist_stamps)

    FM_time_ind  = np.zeros((num_data_points, 1), int)
    FM_ts        = np.zeros((num_data_points, 1), '<U13') #list()
    his_time_ind = np.zeros((num_data_points, num_hist_indices), int)

    grid_ind     = np.zeros((num_data_points, 1), int)
    j_ind        = np.zeros((num_data_points, 1), int)
    i_ind        = np.zeros((num_data_points, 1), int)

    FM_10hr      = np.zeros((num_data_points, 1), float)
    FM_100hr     = np.zeros((num_data_points, 1), float)

    HGT          = np.zeros((num_data_points, 1), float)

    U10_hist     = np.zeros((num_data_points, num_hist_indices), float)
    V10_hist     = np.zeros((num_data_points, num_hist_indices), float)
    T2_hist      = np.zeros((num_data_points, num_hist_indices), float)
    RH_hist      = np.zeros((num_data_points, num_hist_indices), float)
    PRECIP_hist  = np.zeros((num_data_points, num_hist_indices), float)
    SWDOWN_hist  = np.zeros((num_data_points, num_hist_indices), float)

    ## Fill in The Data Arrays
    for sampled_time_count, sampled_time_stamp in enumerate(sampled_time_stamps):
        #print('sampled_time_stamp: ', sampled_time_stamp)

        for sampled_grid_point_count in range(num_sampled_points):
            data_point_count = sampled_time_count*num_sampled_points + sampled_grid_point_count

            # Grid Identifier
            grid_index = grid_indices_valid_flat[data_point_count]
            j_loc      = valid_grid_ind_to_coord[grid_index][0]
            i_loc      = valid_grid_ind_to_coord[grid_index][1]
            #print(sampled_time_count, sampled_grid_point_count, data_point_count)

            # Time Indices
            #FM_time_ind [ data_point_count] = sampled_time_ind
            FM_ts [       data_point_count] = sampled_time_stamps [sampled_time_count]
            #his_time_ind [data_point_count] = hist_indices

            # Grid Indices
            grid_ind [    data_point_count] = grid_index
            j_ind [       data_point_count] = j_loc
            i_ind [       data_point_count] = i_loc

            # FM for Labels
            FM_10hr [ data_point_count] = data_read_SJSU[sampled_time_stamp]['FM_10hr' ][j_loc][i_loc]
            FM_100hr [data_point_count] = data_read_SJSU[sampled_time_stamp]['FM_100hr'][j_loc][i_loc]

            # Height for Features
            HGT [data_point_count] = data_read_SJSU[sampled_time_stamp]['HGT'][j_loc][i_loc]


            # History Data for Features
            for hist_ind_count, hist_stamp in enumerate(hist_stamps):
                U10_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['U10'][j_loc][i_loc]
                V10_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['V10'][j_loc][i_loc]
                T2_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['T2'][j_loc][i_loc]
                RH_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['RH'][j_loc][i_loc]
                PRECIP_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['PRECIP'][j_loc][i_loc]
                SWDOWN_hist[data_point_count][hist_ind_count] = data_read_SJSU[hist_stamp]['SWDOWN'][j_loc][i_loc]


    ## Create DataFrame
    df = pd.DataFrame()
    #df['FM_time_ind'] = FM_time_ind.flatten()
    df['FM_ts'] = FM_ts.flatten()
    #df['his_time_ind'] = list(his_time_ind)

    df['grid_ind'] = grid_ind
    df['j_ind'] = j_ind
    df['i_ind'] = i_ind

    df['FM_10hr'] = FM_10hr
    df['FM_100hr'] = FM_100hr

    df['HGT'] = HGT


    num_hist_indices = len(hist_stamps)
    for hist_ind_count in range(num_hist_indices):
        df['U10[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)] =\
                                        U10_hist[:,hist_ind_count]
        df['V10[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)]=\
                                        V10_hist[:,hist_ind_count]
        df['T2[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)] =\
                                        T2_hist[:,hist_ind_count]
        df['RH[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)] =\
                                        RH_hist[:,hist_ind_count]
        df['PREC[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)] =\
                                        PRECIP_hist[:,hist_ind_count]
        df['SWDOWN[-{}hr]'.format((num_hist_indices-hist_ind_count)*history_interval)] =\
                                        SWDOWN_hist[:,hist_ind_count]
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    
    return df
