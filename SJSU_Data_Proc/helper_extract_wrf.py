import os
import sys
import os.path as path
import glob
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from datetime import date, datetime, timedelta
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
    print('\nGetting the names of the data files at the dir : \n {} \n'.format(data_files_dir))
    years_list = os.listdir(data_files_dir)
    print('years_list: {} \n'.format(years_list))
    
    file_list = []
    for year in years_list:
        print('Getting the names of the data files for the year : {}'.format(year))
        file_list_for_year = []
        for file in os.listdir(path.join(data_files_dir, year)):
            if file.startswith('wrf') and file.endswith('.nc'):
                file_list_for_year.append(file)
        print('... Found {} files for this year'.format(len(file_list_for_year)))
        file_list.extend(file_list_for_year)

    file_list.sort()
    print('\nFound a total of {} files \n'.format(len(file_list)))
    print('=========================================================================')
    return file_list

# []
'''
Downsample the file indices/lists to use from the master list of files containing
the historical atmospheric data
Get the random reference time where Fuel Moisture is to be read and relative to which 
historical data is to be collected
'''
def downsample_data_files (data_files_list, percent_files_to_use, max_history_to_consider, random_state):
    random.setstate(random_state)
    print('\nRandomly selecting approx {} % of the data files'.format(percent_files_to_use))
    file_indices = set(range(len(data_files_list)))
    invalid_ind = set(range(max_history_to_consider))
    valid_indices = list(file_indices - invalid_ind)
    total_files = len(valid_indices)
    downsample_files_count = round(percent_files_to_use*total_files/100.0)
    sampled_file_indices = random.sample(valid_indices, k = downsample_files_count)
    sampled_data_files = list(np.array(data_files_list)[sampled_file_indices])
    print('Selected {} data files out of {} total and {} usable considering historical data'.format(len(sampled_data_files), len(data_files_list), len(valid_indices)))
    #print('Indices of the randomly selected files: \n {}'.format(sampled_file_indices))
    #print('Names of the randomly selected files: \n {}'.format(sampled_data_files))
    print('=========================================================================')
    return sampled_file_indices, sampled_data_files

# []
'''
Get the history file indices corresponding to the sampled reference time indices
'''
def get_history_file_indices (sampled_file_indices, max_history_to_consider, history_interval):
    history_file_indices = []
    for fuel_moisture_time_index in sampled_file_indices:
        atm_data_time_indices = np.arange(fuel_moisture_time_index, \
                                         fuel_moisture_time_index - max_history_to_consider - 1,\
                                         - history_interval)
        atm_data_time_indices = list(np.sort(atm_data_time_indices)[:-1])
        history_file_indices.append(atm_data_time_indices)
        
    return history_file_indices

# []
'''
Get timestamps and datetime for the downsampled data files
'''
def get_datetime_for_data_files (sampled_data_files):
    print('\nGetting time stamps and datetime of the downsampled data files...')
    sampled_time_stamps = []
    sampled_datetime = []
    for filename in sampled_data_files:
        data_filename_split = filename.split('_')
        date = data_filename_split[1]
        hour = data_filename_split[2].split('.')[0]
        date_hour = date + '_' + hour
        sampled_time_stamps.append(date_hour)
        sampled_datetime.append(datetime.fromisoformat(date_hour))
    print('=========================================================================')
    return sampled_time_stamps, sampled_datetime

# []
'''
Create DataFrame using sampled file indices, filenames, timestamps, and datetime
'''
def create_df_sampled_time (sampled_file_indices, sampled_data_files, sampled_time_stamps, sampled_datetime, history_file_indices):
    print('\nCreating DataFrame using sampled file indices, filenames, timestamps, and datetime...')
    df_sampled_time = pd.DataFrame()
    df_sampled_time['ref_time_indices'] = sampled_file_indices
    df_sampled_time['sampled_data_files'] = sampled_data_files
    df_sampled_time['sampled_time_stamps'] = sampled_time_stamps
    df_sampled_time['sampled_datetime'] = sampled_datetime
    df_sampled_time['history_time_indices'] = history_file_indices
    print('=========================================================================')
    return df_sampled_time

# []
'''
Plot the sampled datetime
'''
def plot_sampled_datetime (df_sampled_time, extracted_data_loc, xlim = None, ylim = None):       
    plt.figure()
    
    sampled_datetime = df_sampled_time['sampled_datetime']
    plt.scatter(range(len(sampled_datetime)), sampled_datetime)
    plt.xlabel('Indices of refernce time [-]', fontsize=20)
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
    plt.show()
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.close()

#[]
'''
Get grid indices
'''
def get_grid_indices_all (data_files_location, sampled_file_indices, sampled_data_files):
    random_ind_of_downsampled_files = random.choice(range(len(sampled_file_indices)))
    
    file_ind_to_read = sampled_file_indices[random_ind_of_downsampled_files]
    data_file_to_read = sampled_data_files[random_ind_of_downsampled_files]
    year = data_file_to_read.split('_')[1].split('-')[0]
    dfm_file_data = xr.open_dataset(path.join(data_files_location, year, data_file_to_read))
    
    ny, nx = dfm_file_data.dims['south_north'], dfm_file_data.dims['west_east']
    grid_indices = np.zeros((ny, nx), int)
    for j in range(ny):
        for i in range(nx):
           grid_indices[j][i] = nx*j + i
    grid_indices_flattened = grid_indices.flatten()
    
    return data_file_to_read, grid_indices, grid_indices_flattened
    
#[]
'''
Read a single data file
'''
def read_single_data_file ():
    data_at_time = {}
    
    return data_at_time
    
# []
'''
Downsample the grid indices to use from all the grid points where data are available
'''
def downsample_grid_indices (data_file_name, dfm_file_data, percent_grid_points_to_use, max_history_to_consider, history_interval, frames_in_file):
    df_for_single_file = pd.DataFrame()

    ny, nx = dfm_file_data.dims['south_north'], dfm_file_data.dims['west_east']
    n_grid_points = nx*ny
    #print('Dimensions: {} X {}, Number of grid points: {}'.format(nx, ny, n_grid_points))
    grid_indices = list(range(n_grid_points))
    downsample_grid_point_count = round(percent_grid_points_to_use*n_grid_points/100.0)
    print('Selecting {} grid points (approx {} % of a total of {} grid points)\n'.format(
        downsample_grid_point_count, percent_grid_points_to_use, n_grid_points))
    #print('-----------------------------------------------------------------------')
    
    grid_indices_selected = []
    while (len(grid_indices_selected) < downsample_grid_point_count):
        grid_ind_sampled = random.choice(grid_indices)
        if grid_ind_sampled in grid_indices_selected:
            #print('Grid index {} sampled is already in the grid indices selected: \n{}'.format(
            #grid_ind_sampled, grid_indices_selected))
            pass
        j_ind = grid_ind_sampled // nx
        i_ind = grid_ind_sampled - j_ind*nx
        #print('Grid point # {} : grid_ind = {}, i = {}, j = {}'.format(
            #len(grid_indices_selected), grid_ind_sampled, i_ind, j_ind))

        FM_time_index, AtmData_time_indices, df_at_gp = create_df_at_gp (data_file_name, dfm_file_data, i_ind, j_ind, max_history_to_consider, history_interval, frames_in_file)
        #print('DataFrame at grid point: \n {}'.format(df_at_gp))
        if (not df_at_gp.isna().values.any()):
            df_for_single_file = df_for_single_file.append(df_at_gp).reset_index(drop = True)
            grid_indices_selected.append(grid_ind_sampled)
        #else:
            #print('NaN found at the grid point...ignoring this grid point')
        #print('-----------------------------------------------------------------------')

    #print('=========================================================================')
    return df_for_single_file

# []
'''
Create DataFrame at a grid point
'''
def create_df_at_gp (data_file_name, dfm_file_data, i_ind, j_ind, max_history_to_consider, history_interval, frames_in_file):
    df_at_gp = pd.DataFrame()
    FM_time_index_range = list(range(max_history_to_consider, frames_in_file ))
    FM_time_index = random.choice(FM_time_index_range)
    AtmData_time_indices = np.arange(FM_time_index, FM_time_index-max_history_to_consider, -history_interval)
    AtmData_time_indices = list(np.sort(AtmData_time_indices)[:-1])
    
    #print('FM_time_index: {}'.format(FM_time_index))
    #print('AtmData_time_indices: \n{}'.format(AtmData_time_indices))
    
    data_at_gp = dfm_file_data.isel(south_north = j_ind).isel(west_east = i_ind)
    
    # Identity Fields
    df_at_gp['WRF_Nelson_File'] = [data_file_name]
    df_at_gp['lat'] = np.array(data_at_gp['latitude']).flatten()
    df_at_gp['lon'] = np.array(data_at_gp['longitude']).flatten()
    '''
    df_at_gp['YYYY'] = np.array(data_at_gp['YYYY'].isel(time=FM_time_index)).flatten()
    df_at_gp['MM'] = np.array(data_at_gp['MM'].isel(time=FM_time_index)).flatten()
    df_at_gp['DD'] = np.array(data_at_gp['DD'].isel(time=FM_time_index)).flatten()
    df_at_gp['HH'] = np.array(data_at_gp['HH'].isel(time=FM_time_index)).flatten()
    '''
    FM_ref_time = '%d:%02d:%02d:%02d'%(np.array(data_at_gp['YYYY'].isel(time=FM_time_index)).flatten()[0],
                                       np.array(data_at_gp['MM'].isel(time=FM_time_index)).flatten()[0],
                                       np.array(data_at_gp['DD'].isel(time=FM_time_index)).flatten()[0],
                                       np.array(data_at_gp['HH'].isel(time=FM_time_index)).flatten()[0])
    df_at_gp['Ref_Time_for_FM'] = [FM_ref_time]
    
    # Label Fields
    df_at_gp['FM_1hr']  = np.array(data_at_gp['mean_wtd_moisture_1hr'].isel(time=FM_time_index)).flatten()
    df_at_gp['FM_10hr'] = np.array(data_at_gp['mean_wtd_moisture_10hr'].isel(time=FM_time_index)).flatten()
    df_at_gp['FM_100hr'] = np.array(data_at_gp['mean_wtd_moisture_100hr'].isel(time=FM_time_index)).flatten()
    df_at_gp['FM_1000hr'] = np.array(data_at_gp['mean_wtd_moisture_1000hr'].isel(time=FM_time_index)).flatten()
    
    # Historical Atm Data
    for hist_data_ind in AtmData_time_indices:
        #hist_data_ind = AtmData_indices[0]
        U10_data = np.array(data_at_gp['eastward_10m_wind'].isel(
                            time=hist_data_ind)).flatten()
        V10_data = np.array(data_at_gp['northward_10m_wind'].isel(
                            time=hist_data_ind)).flatten()
        T2_data  = np.array(data_at_gp['air_temperature_2m'].isel(
                            time=hist_data_ind)).flatten()
        Precip_data = np.array(data_at_gp['accumulated_precipitation_amount'].isel(
                            time=hist_data_ind)).flatten()
        RH2_data = np.array(data_at_gp['air_relative_humidity_2m'].isel(
                            time=hist_data_ind)).flatten()
        SDSF_data = np.array(data_at_gp['surface_downwelling_shortwave_flux'].isel(
                            time=hist_data_ind)).flatten()

        df_at_gp['U10[%s]'%str(hist_data_ind - FM_time_index)] = U10_data
        df_at_gp['V10[%s]'%str(hist_data_ind - FM_time_index)] = V10_data
        df_at_gp['T2[%s]'%str(hist_data_ind - FM_time_index)] = T2_data
        df_at_gp['Precip[%s]'%str(hist_data_ind - FM_time_index)] = Precip_data
        df_at_gp['RH2[%s]'%str(hist_data_ind - FM_time_index)] = RH2_data
        df_at_gp['SDSF[%s]'%str(hist_data_ind - FM_time_index)] = SDSF_data

    return FM_time_index, AtmData_time_indices, df_at_gp