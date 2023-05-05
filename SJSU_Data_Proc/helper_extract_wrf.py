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
    print('\nDetermining history file indices corresponding to sampled file/time indices...')
    history_file_indices = []
    for fuel_moisture_time_index in sampled_file_indices:
        atm_data_time_indices = np.arange(fuel_moisture_time_index, \
                                         fuel_moisture_time_index - max_history_to_consider - 1,\
                                         - history_interval)
        atm_data_time_indices = list(np.sort(atm_data_time_indices)[:-1])
        history_file_indices.append(atm_data_time_indices)
    print('=========================================================================')    
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
    plt.show()
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.close()
    print('=========================================================================')
    
#[]
'''
Read a single data file
'''
def read_single_data_file (data_files_location, data_file_to_read, timestamp_to_read):
    print('\nReading data contained in the randomly selcted file: {}...'.format(data_file_to_read))
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
    print('=========================================================================')
    return data_at_timestamp
 
#[]
'''
process elevation from data read from a single file
'''
def process_elevation_at_timestamp (data_at_timestamp):
    print('\nProcessing elevation data into pos, neg, and zero...')
    HGT = data_at_timestamp['HGT']
    
    HGT_UPD = np.ones((data_at_timestamp['ny'], data_at_timestamp['nx']), int)
    HGT_UPD[np.where(HGT == 0)] = 0
    HGT_UPD[np.where(HGT < 0)] = -1
                     
    data_at_timestamp['HGT_UPD'] = HGT_UPD
    print('=========================================================================')
    return data_at_timestamp
    
#[]
'''
Plot Contours of Data at a TimeStamp
'''
def plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc):
    cmap_name = 'rainbow'
    cont_levels = 20
    fig, axlist = plt.subplots(3, 4, figsize=(12, 8))
    for var_ind, ax in enumerate(axlist.ravel()):
        qoi = qoi_to_plot[var_ind]
        x_ind, y_ind = np.meshgrid(range(data_at_timestamp['nx']), range(data_at_timestamp['ny']))
        cont = ax.contourf(x_ind, y_ind, data_at_timestamp[qoi], levels = cont_levels, cmap=cmap_name, extend='both')
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
    
    filename = 'contours_{}.png'.format(data_at_timestamp['TimeStamp'])
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    plt.show()
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.close()
    print('=========================================================================')

#[]
'''
Plot PDF of Data at a TimeStamp
'''
def plot_pdf_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc):
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

    filename = 'pdfs_{}.png'.format(data_at_timestamp['TimeStamp'])
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    plt.show()
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.close()
    print('=========================================================================')

# []
'''
Get grid indices
'''
def get_grid_indices_all (data_files_location, sampled_file_indices, sampled_data_files, sampled_time_stamps):
    print('\nGetting all the grid indices from a randomly selcted file...')
    random_ind_of_downsampled_files = random.choice(range(len(sampled_file_indices)))

    #file_ind_to_read = sampled_file_indices[random_ind_of_downsampled_files]
    data_file_to_read = sampled_data_files[random_ind_of_downsampled_files]
    timestamp_to_read = sampled_time_stamps[random_ind_of_downsampled_files]
    print('The selected file is: {}'.format(data_file_to_read))

    # Read the data at timestamp and process elevation
    data_at_timestamp = read_single_data_file (data_files_location, data_file_to_read, timestamp_to_read)
    data_at_timestamp = process_elevation_at_timestamp (data_at_timestamp)

    # Extract relevant info from data at timestamp
    ny, nx = data_at_timestamp['ny'], data_at_timestamp['nx']
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

    grid_indices_all_flat = grid_indices_all.flatten()
    grid_indices_valid_flat = grid_indices_valid[np.where(grid_indices_valid >= 0)].flatten()
    print('=========================================================================')
    return data_file_to_read, grid_indices_all, grid_indices_valid, grid_indices_all_flat, grid_indices_valid_flat


# []
'''
Reconstruct valid grid indices
'''
def reconstruct_valid_grid_indices (grid_indices_valid_flat, data_at_timestamp):
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
        
    print('=========================================================================')    
    return grid_indices_valid_reconst, grid_indices_valid_bool, valid_grid_ind_to_coord

#[]
'''
Plot Contours of indices at a timestamp
'''
def plot_contours_of_indices (data_at_timestamp, grid_indices_all, grid_indices_valid, grid_indices_valid_bool, grid_indices_valid_reconst):
    cmap_name = 'rainbow'
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

    print('=========================================================================')


# []
'''
Sample grid indices for each ref time
'''
def sample_grid_indices (sampled_file_indices, percent_grid_points_to_use, grid_indices_valid_flat, valid_grid_ind_to_coord):
    downsample_grid_point_count = round(percent_grid_points_to_use*len(grid_indices_valid_flat)/100.0)
    print('Selecting {} grid points (approx {} % of a total of {} considerable/valid grid points)\n'.format(
            downsample_grid_point_count, percent_grid_points_to_use, len(grid_indices_valid_flat)))
    
    grid_indices_selected = []
    j_indices_selected = []
    i_indices_selected = []
    
    for sampled_file_count in range(len(sampled_file_indices)):
        sampled_grid_indices = random.sample(set(grid_indices_valid_flat), k = downsample_grid_point_count)
        
        j_indices_current_time = []
        i_indices_current_time = []
        
        for sampled_grid_ind in sampled_grid_indices:
            j_indices_current_time.append(valid_grid_ind_to_coord[sampled_grid_ind][0])
            i_indices_current_time.append(valid_grid_ind_to_coord[sampled_grid_ind][1])
        
        grid_indices_selected.append(sampled_grid_indices)
        j_indices_selected.append(j_indices_current_time)
        i_indices_selected.append(i_indices_current_time)
        
    print('=========================================================================')
    return np.array(grid_indices_selected), np.array(j_indices_selected), np.array(i_indices_selected)

# []
'''
Plot sampled grid indices for each ref time
'''
def plot_sampled_grid_points (grid_indices_selected, extracted_data_loc):
    cmap_name = 'rainbow'
    cont_levels = 20
    grid_count, time_count = grid_indices_selected.shape[1], grid_indices_selected.shape[0]
    
    grid_ind, time_ind = np.meshgrid(range(grid_count), range(time_count))

    plt.figure()
    cont = plt.contourf(grid_ind, time_ind, grid_indices_selected, levels = cont_levels, cmap=cmap_name, extend='both')
    clb = plt.colorbar(cont)
    clb.ax.tick_params(labelsize=14)
    plt.xlabel('Grid-Index [-]', fontsize=14)
    plt.ylabel('Sampled Ref Time Count [-]', fontsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.title('Sampled grid index for each random ref time',fontsize=14)

    filename = 'Sampled_Grid_Indices.png'
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    plt.show()
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.close()
    print('=========================================================================')

# []
'''
Plot sampled grid indices for each ref time in 3D
'''
def plot_sampled_grid_points_3D (j_indices_selected, i_indices_selected, extracted_data_loc):
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    for ref_time_count in range(j_indices_selected.shape[0]):
        #sampled_datetime_current = sampled_datetime[ref_time_count]
        #print('Sampled DateTime: {}'.format(sampled_datetime_current))
        j_indices_at_current_time = j_indices_selected[ref_time_count]
        i_indices_at_current_time = i_indices_selected[ref_time_count]
        time_indices_at_current_time = np.ones_like(j_indices_at_current_time)*ref_time_count

        ax.scatter3D(i_indices_at_current_time,j_indices_at_current_time, 
                     time_indices_at_current_time)
        ax.set_xlabel('x-indices')
        ax.set_ylabel('y-indices')
        ax.set_zlabel('Sampled Ref Time Count [-]')
        #ax.set_title('Sampled grid points at each of the sampled datetimes')
        '''
        print(time_indices_at_current_time)
        print(j_indices_at_current_time)
        print(i_indices_at_current_time)
        '''

# []
'''
Create a maps of time indices and grid indices at that time where we need data
'''
def create_time_grid_indices_map (sampled_file_indices, history_file_indices, grid_indices_selected):
    time_grid_indices_list_dict = {}
    time_grid_indices_list_count = {}
    time_grid_indices_set_dict = {}
    time_grid_indices_set_count = {}
    
    for sampled_time_count in range(len(sampled_file_indices)):
        grid_indices_sampled_at_current_time = grid_indices_selected[sampled_time_count]

        if sampled_file_indices[sampled_time_count] not in time_grid_indices_list_dict.keys():
            time_grid_indices_list_dict[sampled_file_indices[sampled_time_count]] = \
                grid_indices_sampled_at_current_time
        else:
            time_grid_indices_list_dict[sampled_file_indices[sampled_time_count]]= \
                np.hstack((time_grid_indices_list_dict[sampled_file_indices[sampled_time_count]], grid_indices_sampled_at_current_time))

        for history_time_index in history_file_indices[sampled_time_count]:
            if history_time_index not in time_grid_indices_list_dict.keys():
                time_grid_indices_list_dict[history_time_index] = \
                    grid_indices_sampled_at_current_time
            else:
                time_grid_indices_list_dict[history_time_index] =\
                    np.hstack((time_grid_indices_list_dict[history_time_index], grid_indices_sampled_at_current_time))
        
    
    # Derive other indices
    for sampled_time_index in time_grid_indices_list_dict.keys():
        time_grid_indices_list_count[sampled_time_index] = \
            len(time_grid_indices_list_dict[sampled_time_index])
        time_grid_indices_set_dict[sampled_time_index] = \
            set(time_grid_indices_list_dict[sampled_time_index])
        time_grid_indices_set_count[sampled_time_index] = \
            len(time_grid_indices_set_dict[sampled_time_index])

    return time_grid_indices_list_dict, time_grid_indices_list_count, time_grid_indices_set_dict, time_grid_indices_set_count

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