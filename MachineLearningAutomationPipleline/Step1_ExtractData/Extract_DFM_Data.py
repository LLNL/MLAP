#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python Extract_DFM_Data.ipynb

# # Import Modules

# ## Standard Packages

# In[ ]:


import os
import sys
import os.path as path
import psutil
import glob
import random
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import json
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer


# ## User-Defined Functions

# In[ ]:


from Extract_DFM_Data_Helper import *


# # Global Start Time and Memory

# In[ ]:


global_start_time = timer()
process = psutil.Process(os.getpid())
global_initial_memory = process.memory_info().rss
print('\nProcess in Main(): {}'.format(process))


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Extract/json_extract_data_000.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]


# ### Load the JSON file for extracting data

# In[ ]:


print('Loading input from JSON file: \n {}'.format(json_file_extract_data))


# In[ ]:


with open(json_file_extract_data) as json_file_handle:
    json_content_extract_data = json.load(json_file_handle)


# In[ ]:


#json_content_extract_data


# # Variables to be Used for Extracting WRF Data

# ## DataSet Defintion

# In[ ]:


# The current data set params
data_set_defn = json_content_extract_data['data_set_defn']

data_set_count = data_set_defn['data_set_count']
percent_files_to_use = data_set_defn['percent_files_to_use']  # f1 = what percent of available files to use
percent_grid_points_to_use = data_set_defn['percent_grid_points_to_use']  # f2 = what percent of grid points to use
max_history_to_consider = data_set_defn['max_history_to_consider'] # n_history in hours
history_interval        = data_set_defn['history_interval']


# In[ ]:


sample_first = json_content_extract_data['sampling_type']['sample_first']
sampling_type_time = json_content_extract_data['sampling_type']['time']
sampling_type_space = json_content_extract_data['sampling_type']['space']


# In[ ]:


features_labels = json_content_extract_data['features_labels']
qois_to_read = features_labels['qois_to_read']
labels_to_read = features_labels['labels_to_read']
labels_ind_in_nc_file = features_labels['labels_ind_in_nc_file']


# ## Flags, Other Params etc.

# ### Nevada Data

# In[ ]:


nevada_data = json_content_extract_data['nevada_data']
remove_nevada = nevada_data['remove_nevada']
j_nevada, i_nevada = nevada_data['j_nevada'], nevada_data['i_nevada']
j_anchor, i_anchor = nevada_data['j_anchor'], nevada_data['i_anchor']


# ### Remove/Extract Fire Data

# In[ ]:


fire_flags = json_content_extract_data['fire_flags']
remove_fire_data_from_train_test = fire_flags['remove_fire_data_from_train_test']
extract_fire_data = fire_flags['extract_fire_data']


# ### Clip Data for Train/Test

# In[ ]:


clip_data_train_test = json_content_extract_data['clip_data_train_test']
x_clip_train_test = clip_data_train_test['x_clip']
y_clip_train_test = clip_data_train_test['y_clip']


# ## Paths and File Names

# #### Global

# In[ ]:


# WRF data set location and the extracted data set location
data_files_location = json_content_extract_data['paths']['data_files_location']
extracted_data_base_loc = json_content_extract_data['paths']['extracted_data_base_loc']


# #### DataSet Specific (Train and Test Data Extracted from WRF)

# In[ ]:


data_set_name = 'data_train_test_extracted_%03d'%(data_set_count)
extracted_data_loc = os.path.join(extracted_data_base_loc, data_set_name)
os.system('mkdir -p %s'%extracted_data_loc)

#collection_of_read_data_files = '{}_files_read.pkl'.format(data_set_name)
extracted_data_file_name = '{}_df.pkl'.format(data_set_name)

tab_data_file_name = '{}_tab_data.csv'.format(data_set_name)


# ## Relevant Fire TimeStamps

# In[ ]:


fire_time_stamps = json_content_extract_data['fire_time_stamps']


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # File Names

# In[ ]:


data_files_list_all = get_data_file_names(data_files_location)
data_files_list = data_files_list_all


# # Remove Files Corresponding To Fire Data

# ## Get Indices for Fire Time Stamps

# In[ ]:


if remove_fire_data_from_train_test or extract_fire_data:
    fire_time_indices = get_fire_time_indices (fire_time_stamps, data_files_list_all)


# ## Remove the Files with Indices for Fire Time Stamps

# In[ ]:


if remove_fire_data_from_train_test:
    data_files_list = remove_data_around_fire (fire_time_indices, data_files_list)


# In[ ]:


#len(data_files_list)


# # Deal with just first few files to check for correctness of script. Be sure to undo this

# In[ ]:


#data_files_list = data_files_list[0:18]


# # Downsample Files

# In[ ]:


if (sample_first == 'time'):
    sampled_file_indices, sampled_data_files = downsample_data_files (                                            data_files_list, percent_files_to_use,                                             max_history_to_consider, random_state,                                             sampling_type_time)


# # Get History File Indices

# In[ ]:


if (sample_first == 'time'):
    history_file_indices = get_history_file_indices (sampled_file_indices,                                                      max_history_to_consider, history_interval)


# # Create timestamps and datetime of downsampled data files

# In[ ]:


if (sample_first == 'time'):
    sampled_time_stamps, sampled_datetime = get_datetime_for_data_files (sampled_data_files)


# # Create DataFrame using sampled file indices, filenames, timestamps, and datetime

# In[ ]:


if (sample_first == 'time'):
    df_sampled_time = create_df_sampled_time (sampled_file_indices, sampled_data_files,                                               sampled_time_stamps, sampled_datetime,                                               history_file_indices)


# In[ ]:


#df_sampled_time[df_sampled_time['ref_time_indices'] < max_history_to_consider+10]


# In[ ]:


#df_sampled_time.head(30)


# # Plot Sampled Datetime

# In[ ]:


if (sample_first == 'time') and json_content_extract_data['plot_options']['plot_sampled_datetime']:
    plot_sampled_datetime (df_sampled_time, extracted_data_loc)


# # Read Quantities in a Selected Data File

# ## Read the Data in a Specified or Randomly Selected File

# In[ ]:


data_in_a_file = json_content_extract_data['data_in_a_file']
prescribe_file = data_in_a_file['prescribe_file_flag']
if prescribe_file:
    data_file_to_read = data_in_a_file['data_file_to_read']
    timestamp_to_read = data_file_to_read.split('_')[1] + '_' +                         data_file_to_read.split('_')[2].split('.')[0]
elif (sample_first == 'time'):
    random_ind_of_downsampled_files = random.choice(range(len(sampled_file_indices)))
    file_ind_to_read = sampled_file_indices[random_ind_of_downsampled_files]
    data_file_to_read = sampled_data_files[random_ind_of_downsampled_files]
    timestamp_to_read = sampled_time_stamps[random_ind_of_downsampled_files]


# In[ ]:


data_file_to_read, timestamp_to_read


# In[ ]:


data_at_timestamp = read_single_data_file (data_files_location, data_file_to_read,                                            timestamp_to_read)


# ## Processing Elevation Data into Pos, Neg, and Zero

# In[ ]:


data_at_timestamp = process_elevation_at_timestamp (data_at_timestamp)


# # Get and Plot Grid Indices (All and Considerable)

# ## Get Grid Indices

# In[ ]:


grid_indices_all, grid_indices_valid, grid_indices_all_flat, grid_indices_valid_flat =                         get_grid_indices_all (data_at_timestamp,                                               x_clip_train_test, y_clip_train_test,                                               j_nevada, i_nevada, j_anchor, i_anchor,                                               remove_nevada)


# ## Reconstruct Grid Indices

# In[ ]:


grid_indices_valid_reconst, grid_indices_valid_bool, valid_grid_ind_to_coord =                 reconstruct_valid_grid_indices (grid_indices_valid_flat, data_at_timestamp)


# ## Plot Grid Indices

# In[ ]:


if json_content_extract_data['plot_options']['plot_contours_of_indices']:
    plot_contours_of_indices (data_at_timestamp, grid_indices_all, grid_indices_valid,                               grid_indices_valid_bool, grid_indices_valid_reconst,                               extracted_data_loc)


# In[ ]:


#len(grid_indices_valid_flat)


# # Plot Quantities in the Selected Data File

# ## Plot the Contours of QoIs for the Data Just Read Above

# ### Unmasked Data

# In[ ]:


if json_content_extract_data['plot_options']['plot_contours_of_qoi']:
    qoi_to_plot = json_content_extract_data['qoi_to_plot']['contours']
    plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc,                                 grid_indices_valid, masked = False)


# ### Masked Data

# In[ ]:


if json_content_extract_data['plot_options']['plot_contours_of_qoi']:
    qoi_to_plot = json_content_extract_data['qoi_to_plot']['contours']
    plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc,                                 grid_indices_valid, masked = True)


# ## Plot the PDFs of QoIs for the Data Just Read Above

# In[ ]:


if json_content_extract_data['plot_options']['plot_pdfs_of_qoi']:
    qoi_to_plot = json_content_extract_data['qoi_to_plot']['pdfs']
    plot_pdf_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc)


# ## Plot the Contours of QoIs With Colorbars

# In[ ]:


if json_content_extract_data['plot_options']['plot_fm_contours_with_cb']:
    qoi_to_plot = json_content_extract_data['qoi_to_plot']['contours_with_cb']
    cont_levels_count = json_content_extract_data['qoi_to_plot']['cont_levels_count']
    qoi_cont_range = json_content_extract_data['qoi_to_plot']['qoi_cont_range']


# In[ ]:


if json_content_extract_data['plot_options']['plot_fm_contours_with_cb']:
    plot_contours_at_timestamp2 (data_at_timestamp, timestamp_to_read, qoi_to_plot,                                  extracted_data_loc, grid_indices_valid,                                  cont_levels_count, qoi_cont_range, masked = True)


# # Sample and Plot Grid Indices for Each Sampled Ref Time

# ## Sample Grid Indices

# In[ ]:


if (sample_first == 'time'):
    grid_indices_selected, j_indices_selected, i_indices_selected =         sample_grid_indices (sampled_file_indices, percent_grid_points_to_use,                              grid_indices_valid_flat, valid_grid_ind_to_coord)


# In[ ]:


#grid_indices_selected


# ## Plot Sampled Grid Indices

# In[ ]:


if (sample_first == 'time') and json_content_extract_data['plot_options']['plot_sampled_grid_indices_2d']:
    plot_sampled_grid_points (grid_indices_selected, extracted_data_loc)


# ## Plot Sampled Grid Indices in 3D

# In[ ]:


if (sample_first == 'time') and json_content_extract_data['plot_options']['plot_sampled_grid_indices_3d']:
    plot_sampled_grid_points_3D (j_indices_selected, i_indices_selected,                                  extracted_data_loc, (6, 6)) #fig_size hard-coded


# # Create a Dict of Time Indices and Grid Indices

# In[ ]:


if (sample_first == 'time'):
    time_grid_indices_list_dict, time_grid_indices_list_count,     time_grid_indices_set_dict, time_grid_indices_set_count =         create_time_grid_indices_map (sampled_file_indices, history_file_indices,                                       grid_indices_selected)


# In[ ]:


#len(time_grid_indices_list_dict.keys())
#len(time_grid_indices_set_dict.keys())


# In[ ]:


#sampled_file_indices
#grid_indices_selected.shape


# In[ ]:


#time_grid_indices_list_dict
#time_grid_indices_list_count
#time_grid_indices_set_dict
#time_grid_indices_set_count


# # Read Data at Sampled Time and Grid Indices

# In[ ]:


#features_to_read, labels_to_read, labels_ind_in_nc_file


# In[ ]:


if (sample_first == 'time'):
    data_at_sampled_times_and_grids, read_data_memory, read_data_time =         read_data_at_sampled_times_and_grids(labels_to_read, labels_ind_in_nc_file,                                              qois_to_read, valid_grid_ind_to_coord,                                              time_grid_indices_set_dict,                                              data_files_location, data_files_list,                                              'array')


# In[ ]:


#np.set_printoptions (suppress=True)
#data_at_sampled_times_and_grids.keys()


# # Create DataFrame of Data at Sampled Time and Grid Indices

# In[ ]:


if (sample_first == 'time'):
    df = create_dataframe_FM_atm_data (data_at_sampled_times_and_grids, data_at_timestamp,                                      sampled_file_indices, history_file_indices,                                       sampled_time_stamps, history_interval,                                       grid_indices_selected,                                       j_indices_selected, i_indices_selected,                                      labels_to_read, qois_to_read)


# ## Save The Data Extracted  at Sampled Time and Grid Points

# In[ ]:


if (sample_first == 'time'):
    df.to_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))


# ## Load and Test The Extracted Data Saved in Pickle File

# In[ ]:


#df_from_pickle = pd.read_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))
#df_from_pickle.head(10)


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = (global_final_memory - global_initial_memory)/(1024*1024)
global_compute_time = global_end_time - global_start_time
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed))
print('Total computing time: {:.3f} s'.format(global_compute_time))
print('=========================================================================')


# # Save Other Relevant Info in a CSV File

# In[ ]:


print('Saving relevant info in a CSV file')
print('=========================================================================')


# In[ ]:


if (sample_first == 'time'):
    if ('UMag10' in qois_to_read and         'U10' not in qois_to_read and         'V10' not in qois_to_read):
        num_features = len(history_file_indices[0]) * (len(qois_to_read)    )
    else:
        num_features = len(history_file_indices[0]) * (len(qois_to_read) - 1)

    data_for_csv = { 'max_hist':                   [max_history_to_consider],
                     'hist_interval':              [history_interval],
                     'num_hist':                   [len(history_file_indices[0])],
                     'num_qois':                   [len(qois_to_read) - 1],
                     'num_total_files':            [len(data_files_list)],
                     'percent_files_to_use':       [percent_files_to_use],
                     'num_sampled_times':          [grid_indices_selected.shape[0]],
                     'num_data_files_to_read':     [grid_indices_selected.shape[0] * \
                                                    (len(history_file_indices[0]) + 1)],
                     'num_grid_points_sn':         [grid_indices_all.shape[0]],
                     'num_grid_points_we':         [grid_indices_all.shape[1]],
                     'num_total_grid_points':      [len(grid_indices_all_flat)],
                     'num_valid_grid_points':      [len(grid_indices_valid_flat)],
                     'percent_grid_points_to_use': [percent_grid_points_to_use],
                     'num_sampled_grid_points':    [grid_indices_selected.shape[1]],
                     'num_time_grid_to_read':      [grid_indices_selected.shape[0] * \
                                                    (len(history_file_indices[0]) + 1) * \
                                                    grid_indices_selected.shape[1]],
                     'rows_feature_mat':           [grid_indices_selected.shape[0] * \
                                                    grid_indices_selected.shape[1]],
                     'cols_feature_mat':           [num_features],
                     'read_data_memory':           [read_data_memory], 
                     'read_data_time':             [read_data_time],
                     'global_memory':              [global_memory_consumed], 
                     'global_time':                [global_compute_time]
    }
    tabulated_data = pd.DataFrame(data_for_csv)
    tabulated_data.to_csv(os.path.join(extracted_data_loc, tab_data_file_name), index = False)
#tabulated_data


# In[ ]:


print("SUCCESS: Done Extraction of Data")
print('=========================================================================')

