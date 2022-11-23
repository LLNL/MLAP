import os
import sys
import os.path as path
import glob
import numpy as np
import pandas as pd
import xarray as xr
import pickle
#from matplotlib import pyplot as plt
#plt.style.use('seaborn-white')
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
    print('\nGetting the names of data files at the dir : \n {}'.format(data_files_dir))
    file_list = []
    for file in os.listdir(data_files_dir):
        if file.startswith('wrfout') and file.endswith('.nc'):
            file_list.append(file)
    file_list.sort()
    print('Found {} files'.format(len(file_list)))
    print('=========================================================================')
    return file_list

# []
'''
Downsample the file indices/lists to use from the master list of files containing
the historical atmospheric data
'''
def downsample_data_files (data_files_list, percent_files_to_use):
    #random.setstate(random_state)
    print('\nRandomly selecting approx {} % of the data files'.format(percent_files_to_use))
    file_indices = list(range(len(data_files_list)))
    total_files = len(file_indices)
    downsample_files_count = round(percent_files_to_use*total_files/100.0)
    sampled_file_indices = random.sample(file_indices, k = downsample_files_count)
    sampled_data_files = list(np.array(data_files_list)[sampled_file_indices])
    print('Selected {} data files'.format(len(sampled_data_files)))
    print('Indices of the randomly selected files \n {}'.format(sampled_file_indices))
    print('Names of the randomly selected files \n {}'.format(sampled_data_files))
    print('=========================================================================')
    return sampled_file_indices, sampled_data_files

# []
'''
Downsample the grid indices to use from all the grid points where data are available
'''
def downsample_grid_indices (dfm_file_data, percent_grid_points_to_use):
    #random.setstate(random_state)
    print('\nRandomly selecting {} % of the grid points'.format(percent_grid_points_to_use))
    ny, nx = dfm_file_data.dims['south_north'], dfm_file_data.dims['west_east']
    n_grid_points = nx*ny
    print('Dimensions: {} X {}, Num of grid points: {}'.format(nx, ny, n_grid_points))
    grid_indices = list(range(n_grid_points))
    downsample_grid_point_count = round(percent_grid_points_to_use*n_grid_points/100.0)
    print('Selected {} grid points'.format(downsample_grid_point_count))
    sampled_grid_indices = random.sample(grid_indices, k = downsample_grid_point_count)
    j_indices = np.array(sampled_grid_indices) //  nx
    i_indices = sampled_grid_indices - nx*j_indices
    print('Sampled grid indices: \n{}'.format(sampled_grid_indices))
    print('Sampled i-indices: \n{}'.format(i_indices))
    print('Sampled j-indices: \n{}'.format(j_indices))
    
    return i_indices, j_indices, sampled_grid_indices

# []
'''
Create DataFrame at a grid point
'''
def create_df_at_gp (dfm_file_data, i_ind, j_ind, max_history_to_consider, history_interval, frames_in_file):
    df_at_gp = pd.DataFrame()
    FM_time_index_range = list(range(max_history_to_consider, frames_in_file ))
    FM_time_index = random.choice(FM_time_index_range)
    AtmData_time_indices = np.arange(FM_time_index, FM_time_index-max_history_to_consider, -history_interval)
    AtmData_time_indices = list(np.sort(AtmData_time_indices)[:-1])
    
    data_at_gp = dfm_file_data.isel(south_north = j_ind).isel(west_east = i_ind)
    
    df_at_gp['lat'] = np.array(data_at_gp['latitude']).flatten()
    df_at_gp['lon'] = np.array(data_at_gp['longitude']).flatten()
    
    df_at_gp['FM_10hr'] = np.array(data_at_gp['mean_wtd_moisture_10hr'].isel(time=FM_time_index)).flatten()
    df_at_gp['FM_1hr']  = np.array(data_at_gp['mean_wtd_moisture_1hr'].isel(time=FM_time_index)).flatten()
    
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