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
        #print('Getting the names of the data files for the year : {}'.format(year))
        file_list_for_year = []
        for file in os.listdir(path.join(data_files_dir, year)):
            if file.startswith('wrf') and file.endswith('.nc'):
                file_list_for_year.append(file)
        #print('... Found {} files for this year'.format(len(file_list_for_year)))
        file_list.extend(file_list_for_year)

    file_list.sort()
    print('\nFound a total of {} files \n'.format(len(file_list)))
    print('=========================================================================')
    return file_list


# []
'''
Get the indices in the data files list for the fire time stamps
'''
def get_fire_time_indices (fire_time_stamps, data_files_list):
    fire_time_indices = {}
    
    for fire_name in fire_time_stamps.keys():
        fire_time_indices_current = {}
        for time_stamp_key in fire_time_stamps[fire_name].keys():
            time_stamp = fire_time_stamps[fire_name][time_stamp_key]
            data_file_name = 'wrf_{}.nc'.format(time_stamp)
            data_file_index = data_files_list.index(data_file_name)
            fire_time_indices_current[time_stamp_key] = data_file_index
            
        fire_time_indices[fire_name] = fire_time_indices_current
        
    print('=========================================================================')    
    return fire_time_indices


# []
'''
Remove the data around the fires of concern
'''
def remove_data_around_fire (fire_time_indices, data_files_list):    
    fire_indices_to_delete = []
    for fire_name in fire_time_indices.keys():
        fire_start_ind = fire_time_indices[fire_name]['Start']
        fire_end_ind = fire_time_indices[fire_name]['End']
        fire_indices_to_delete.extend(range(fire_start_ind, fire_end_ind + 1))

    print('Removing {} data files around fires, out of total {}. [{} %]'.format(\
                                 len(fire_indices_to_delete), len(data_files_list),\
                                 100.0*float(len(fire_indices_to_delete))/len(data_files_list)))
    
    data_files_list = list (np.delete(np.array(data_files_list), fire_indices_to_delete))
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
def downsample_data_files (data_files_list, percent_files_to_use, max_history_to_consider, random_state):
    random.setstate(random_state)
    print('\nRandomly selecting approx {} % of the data files'.format(percent_files_to_use))
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
    #print('\nDetermining history file indices corresponding to given file/time indices...')
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
    #print('\nGetting time stamps and datetime of the downsampled data files...')
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
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    print('=========================================================================')
    
#[]
'''
Read a single data file
'''
def read_single_data_file (data_files_location, data_file_to_read, timestamp_to_read):
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
    #print('=========================================================================')
    return data_at_timestamp
 
#[]
'''
process elevation from data read from a single file
'''
def process_elevation_at_timestamp (data_at_timestamp):
    #print('\nProcessing elevation data into pos, neg, and zero...')
    HGT = data_at_timestamp['HGT']
    
    HGT_UPD = np.ones((data_at_timestamp['ny'], data_at_timestamp['nx']), int)
    HGT_UPD[np.where(HGT == 0)] = 0
    HGT_UPD[np.where(HGT < 0)] = -1
                     
    data_at_timestamp['HGT_UPD'] = HGT_UPD
    #print('=========================================================================')
    return data_at_timestamp
    
# []
'''
Get grid indices
'''
def get_grid_indices_all (data_files_location, sampled_file_indices, sampled_data_files, sampled_time_stamps, x_clip, y_clip, j_nevada, i_nevada, j_anchor, i_anchor, remove_nevada = True):
    
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
    print('=========================================================================')
    return grid_indices_all, grid_indices_valid, grid_indices_all_flat, grid_indices_valid_flat


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
def plot_contours_of_indices (data_at_timestamp, grid_indices_all, grid_indices_valid, grid_indices_valid_bool, grid_indices_valid_reconst, extracted_data_loc):
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
    print('=========================================================================')


#[]
'''
Plot Contours of Data at a TimeStamp
'''
def plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc, grid_indices_valid, masked = True):
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
    print('=========================================================================')

    
#[]
'''
Plot Contours of Data at a TimeStamp
'''
def plot_contours_at_timestamp2 (data_at_timestamp, timestamp_to_read, qoi_to_plot, extracted_data_loc, grid_indices_valid, cont_levels_count, masked = True, qoi_cont_range = None):
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

    filename = 'PDFs_QoIs_{}.png'.format(data_at_timestamp['TimeStamp'])
    filedir = extracted_data_loc
    os.system('mkdir -p %s'%filedir)
    
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    #plt.show()
    #plt.close()
    print('=========================================================================')
    
# []
'''
Sample grid indices for each ref time
'''
def sample_grid_indices (sampled_file_indices, percent_grid_points_to_use, grid_indices_valid_flat, valid_grid_ind_to_coord):
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
    
    print('=========================================================================')
    return grid_indices_selected, j_indices_selected, i_indices_selected


# []
'''
Plot sampled grid indices for each ref time
'''
def plot_sampled_grid_points (grid_indices_selected, extracted_data_loc):
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
    print('=========================================================================')

# []
'''
Plot sampled grid indices for each ref time in 3D
'''
def plot_sampled_grid_points_3D (j_indices_selected, i_indices_selected, extracted_data_loc, fig_size):
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
    print('=========================================================================')

# []
'''
Create a maps of time indices and grid indices at that time where we need data
'''
def create_time_grid_indices_map (sampled_file_indices, history_file_indices, grid_indices_selected):
    time_grid_indices_list_dict = {}
    time_grid_indices_list_count = {}
    time_grid_indices_set_dict = {}
    time_grid_indices_set_count = {}
    
    for sampled_time_count, sampled_time_ind in enumerate(sampled_file_indices):
        
        grid_indices_sampled_at_current_time = grid_indices_selected[sampled_time_count]

        if sampled_time_ind not in time_grid_indices_list_dict.keys():
            time_grid_indices_list_dict[sampled_time_ind] = \
                grid_indices_sampled_at_current_time
        else:
            time_grid_indices_list_dict[sampled_time_ind]= \
                np.hstack((time_grid_indices_list_dict[sampled_time_ind], \
                           grid_indices_sampled_at_current_time))

        for history_time_index in history_file_indices[sampled_time_count]:
            if history_time_index not in time_grid_indices_list_dict.keys():
                time_grid_indices_list_dict[history_time_index] = \
                    grid_indices_sampled_at_current_time
            else:
                time_grid_indices_list_dict[history_time_index] =\
                    np.hstack((time_grid_indices_list_dict[history_time_index], \
                               grid_indices_sampled_at_current_time))
        
    
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
Read data at all the possible time indices
'''
'''
def read_data_all_possible_times (file_indices_to_read, data_files_list, \
                                  data_files_location):
    file_indices_data_dict = dict()
    
    #file_indices_to_read = list(time_grid_indices_list_dict.keys())
    data_files_to_read = list(np.array(data_files_list)[file_indices_to_read])
    time_stamps_to_read, datetime_to_read = get_datetime_for_data_files (data_files_to_read)
    print('Read a total of {} files ( ref time + history)'.format(len(file_indices_to_read)))
          
    for file_index_to_read_count in range(len(file_indices_to_read)):
        file_index_to_read = file_indices_to_read [file_index_to_read_count]
        data_file_to_read = data_files_to_read [file_index_to_read_count]
        timestamp_to_read = time_stamps_to_read [file_index_to_read_count]
        #print('data_file_to_read: {}, timestamp_to_read: {}'.format(data_file_to_read, timestamp_to_read))
        data_at_timestamp = read_single_data_file (data_files_location, data_file_to_read, \
                                                   timestamp_to_read)
        data_at_timestamp = process_elevation_at_timestamp (data_at_timestamp)
        file_indices_data_dict[file_index_to_read] = data_at_timestamp
    
    print('=========================================================================')
    return data_files_to_read, time_stamps_to_read, file_indices_data_dict
'''

# []
'''
Save data read at all the possible time indices
'''
'''
def save_data_read_at_all_possible_times (file_indices_to_read, data_files_to_read, time_stamps_to_read,\
                                          file_indices_data_dict, \
                                          extracted_data_loc, collection_of_read_data_files):
    
    collection_of_read_data_filename = os.path.join(extracted_data_loc, collection_of_read_data_files)
    if os.path.exists(collection_of_read_data_filename):
        print('The picke file "{}" already exists at "{}"'.format(collection_of_read_data_files, \
                                                                extracted_data_loc))                                                   
    else:
        data_to_save = {'file_indices_to_read': file_indices_to_read,
                     'data_files_to_read': data_files_to_read,
                     'time_stamps_to_read': time_stamps_to_read,
                     'file_indices_data_dict': file_indices_data_dict}

        collection_of_read_data_files_handle = open(collection_of_read_data_filename, 'wb')
        pickle.dump(data_to_save, collection_of_read_data_files_handle)
        collection_of_read_data_files_handle.close()

        print('Wrote all the read data files in "{}" at "{}"'.format(collection_of_read_data_files, \
                                                                    extracted_data_loc))
    print('=========================================================================')
'''

# []
'''
Read data at all the possible time indices saved in a pickle file
'''
'''
def read_data_from_pickle_all_possible_times (extracted_data_loc, collection_of_read_data_files):
    collection_of_read_data = {}
    collection_of_read_data_filename = os.path.join(extracted_data_loc, collection_of_read_data_files)
    if os.path.exists(collection_of_read_data_filename):
        print('Reading the picke file "{}" at "{}"'.format(collection_of_read_data_files, \
                                                          extracted_data_loc))
        collection_of_read_data_files_handle = open(collection_of_read_data_filename, 'rb')
        collection_of_read_data = pickle.load(collection_of_read_data_files_handle)
        collection_of_read_data_files_handle.close()
    else:
        exception_message = 'Pickle file "{}" NOT found at "{}"'.format(collection_of_read_data_files, \
                                                                        extracted_data_loc)
        raise Exception(exception_message)
        
    print('=========================================================================')
    return collection_of_read_data
''' 

# []
'''
Create DataFrame of FM and Historical Data. Also Extract Time and Grid Info.
'''
'''
def create_dataframe_FM_atm_data (data_at_times, \
                                  sampled_file_indices, history_file_indices, \
                                  sampled_time_stamps, history_interval, \
                                  grid_indices_selected, \
                                  j_indices_selected, i_indices_selected):
    
    ## Define the Sizes to Contain Sampled Data
    num_sampled_times  = grid_indices_selected.shape[0]
    num_sampled_points = grid_indices_selected.shape[1]
    num_data_points    = num_sampled_times*num_sampled_points
    num_hist_indices   = len(history_file_indices[0])

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
    for sampled_time_count, sampled_time_ind in enumerate(sampled_file_indices):
        hist_indices = np.array(history_file_indices[sampled_time_count])

        for sampled_grid_point_count in range(num_sampled_points):
            data_point_count = sampled_time_count*num_sampled_points + sampled_grid_point_count

            # Grid Identifier
            grid_index = grid_indices_selected[sampled_time_count][sampled_grid_point_count]
            j_loc      = j_indices_selected[   sampled_time_count][sampled_grid_point_count]
            i_loc      = i_indices_selected[   sampled_time_count][sampled_grid_point_count]
            #print(sampled_time_count, sampled_grid_point_count, data_point_count)

            # Time Indices
            FM_time_ind [ data_point_count] = sampled_time_ind
            FM_ts [       data_point_count] = sampled_time_stamps [sampled_time_count]
            his_time_ind [data_point_count] = hist_indices

            # Grid Indices
            grid_ind [    data_point_count] = grid_index
            j_ind [       data_point_count] = j_loc
            i_ind [       data_point_count] = i_loc

            # FM for Labels
            FM_10hr [ data_point_count] = data_at_times[sampled_time_ind]['FMC_10hr' ][j_loc][i_loc]
            FM_100hr [data_point_count] = data_at_times[sampled_time_ind]['FMC_100hr'][j_loc][i_loc]

            # Height for Features
            HGT [data_point_count] = data_at_times[sampled_time_ind]['HGT'][j_loc][i_loc]

            # History Data for Features
            for hist_ind_count, hist_ind in enumerate(hist_indices):
                U10_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['U10'][j_loc][i_loc]
                V10_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['V10'][j_loc][i_loc]
                T2_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['T2'][j_loc][i_loc]
                RH_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['RH'][j_loc][i_loc]
                PRECIP_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['PRECIP'][j_loc][i_loc]
                SWDOWN_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['SWDOWN'][j_loc][i_loc]
    
    ## Create DataFrame
    df = pd.DataFrame()
    df['FM_time_ind'] = FM_time_ind.flatten()
    df['FM_ts'] = FM_ts
    df['his_time_ind'] = list(his_time_ind)

    df['grid_ind'] = grid_ind
    df['j_ind'] = j_ind
    df['i_ind'] = i_ind

    df['FM_10hr'] = FM_10hr
    df['FM_100hr'] = FM_100hr

    df['HGT'] = HGT

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
    
    print('=========================================================================')
    return df
'''


# []
'''
Create DataFrame of FM and Historical Data For a Fire of Interest. Also Extract Time and Grid Info.
'''
'''
def create_dataframe_FM_atm_data_fire (fire_name, fire_time_indices, data_at_times, \
                                       history_interval, \
                                       grid_indices_valid_flat, valid_grid_ind_to_coord):
   
    ## Extract Some Info
    fire_time_ind_ref  = fire_time_indices[fire_name]['Ref']
    sampled_file_indices = [fire_time_ind_ref] # Make list to align with extracted data format
    history_file_indices = [fire_time_indices[fire_name]['Hist']] # Make list to align with extracted data format
    sampled_time_stamps  = [data_at_times[fire_time_ind_ref]['TimeStamp']]

    ## Define the Sizes to Contain Sampled Data
    num_sampled_times  = 1
    num_sampled_points = len(grid_indices_valid_flat)
    num_data_points    = num_sampled_times*num_sampled_points
    num_hist_indices   = len(history_file_indices[0])

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
    for sampled_time_count, sampled_time_ind in enumerate(sampled_file_indices):
        hist_indices = np.array(history_file_indices[sampled_time_count])

        for sampled_grid_point_count in range(num_sampled_points):
            data_point_count = sampled_time_count*num_sampled_points + sampled_grid_point_count

            # Grid Identifier
            grid_index = grid_indices_valid_flat[data_point_count]
            j_loc      = valid_grid_ind_to_coord[grid_index][0]
            i_loc      = valid_grid_ind_to_coord[grid_index][1]
            #print(sampled_time_count, sampled_grid_point_count, data_point_count)

            # Time Indices
            FM_time_ind [ data_point_count] = sampled_time_ind
            FM_ts [       data_point_count] = sampled_time_stamps [sampled_time_count]
            his_time_ind [data_point_count] = hist_indices

            # Grid Indices
            grid_ind [    data_point_count] = grid_index
            j_ind [       data_point_count] = j_loc
            i_ind [       data_point_count] = i_loc

            # FM for Labels
            FM_10hr [ data_point_count] = data_at_times[sampled_time_ind]['FMC_10hr' ][j_loc][i_loc]
            FM_100hr [data_point_count] = data_at_times[sampled_time_ind]['FMC_100hr'][j_loc][i_loc]

            # Height for Features
            HGT [data_point_count] = data_at_times[sampled_time_ind]['HGT'][j_loc][i_loc]

            # History Data for Features
            for hist_ind_count, hist_ind in enumerate(hist_indices):
                U10_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['U10'][j_loc][i_loc]
                V10_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['V10'][j_loc][i_loc]
                T2_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['T2'][j_loc][i_loc]
                RH_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['RH'][j_loc][i_loc]
                PRECIP_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['PRECIP'][j_loc][i_loc]
                SWDOWN_hist[data_point_count][hist_ind_count] = data_at_times[hist_ind]['SWDOWN'][j_loc][i_loc]
    
    ## Create DataFrame
    df = pd.DataFrame()
    df['FM_time_ind'] = FM_time_ind.flatten()
    df['FM_ts'] = FM_ts
    df['his_time_ind'] = list(his_time_ind)

    df['grid_ind'] = grid_ind
    df['j_ind'] = j_ind
    df['i_ind'] = i_ind

    df['FM_10hr'] = FM_10hr
    df['FM_100hr'] = FM_100hr

    df['HGT'] = HGT

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
    
    print('=========================================================================')
    return df
'''

# []
'''
Downsample the grid indices to use from all the grid points where data are available
'''
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
'''

# []
'''
Create DataFrame at a grid point
'''
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
'''