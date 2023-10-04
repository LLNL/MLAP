#!/usr/bin/env python
# coding: utf-8

# ## Conver this notebook to executable python script using:

# In[1]:


#jupyter nbconvert --to python Extract_DFM_Reanalysis_Data.ipynb


# # Import Modules

# ## Standard Packages

# In[2]:


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
plt.style.use('seaborn-white')
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer


# ## User-Defined Functions

# In[3]:


from Extract_DFM_Reanalysis_Data_Helper import *


# # Global Start Time and Memory

# In[4]:


global_start_time = timer()
process = psutil.Process(os.getpid())
global_initial_memory = process.memory_info().rss


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[5]:


input_json_file = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/01_WRF_Nelson_Data_Extracted/InputJsonFiles/input_json_extract_data_000.json'


# ### Input file name when using python script on command line

# In[6]:


#input_json_file = sys.argv[1]


# ### Load the Input JSON File

# In[7]:


print('Loading input from JSON file: \n {}'.format(input_json_file))


# In[8]:


with open(input_json_file) as input_json_file_handle:
    input_json_data = json.load(input_json_file_handle)


# In[9]:


#input_json_data


# # Variables to be Used for Extracting WRF Data

# ## DataSet Defintion

# In[10]:


# The current data set params
data_set_defn = input_json_data['data_set_defn']

data_set_count = data_set_defn['data_set_count']
percent_files_to_use = data_set_defn['percent_files_to_use']  # f1 = what percent of available files to use
percent_grid_points_to_use = data_set_defn['percent_grid_points_to_use']  # f2 = what percent of grid points to use
max_history_to_consider = data_set_defn['max_history_to_consider'] # n_history in hours
history_interval        = data_set_defn['history_interval']


# ## Flags, Other Params etc.

# ### Nevada Data

# In[11]:


nevada_data = input_json_data['nevada_data']
remove_nevada = nevada_data['remove_nevada']
j_nevada, i_nevada = nevada_data['j_nevada'], nevada_data['i_nevada']
j_anchor, i_anchor = nevada_data['j_anchor'], nevada_data['i_anchor']


# ### Remove/Extract Fire Data

# In[12]:


fire_flags = input_json_data['fire_flags']
remove_fire_data_from_train_test = fire_flags['remove_fire_data_from_train_test']
extract_fire_data = fire_flags['extract_fire_data']


# ### Clip Data for Train/Test

# In[13]:


clip_data_train_test = input_json_data['clip_data_train_test']
x_clip_train_test = clip_data_train_test['x_clip']
y_clip_train_test = clip_data_train_test['y_clip']


# ## Paths and File Names

# In[14]:


paths = input_json_data['paths']


# #### Global

# In[15]:


# WRF data set location and the extracted data set location
data_files_location = paths['data_files_location']
extracted_data_base_loc = paths['extracted_data_base_loc']


# #### DataSet Specific (Train and Test)

# In[16]:


data_set_name = 'data_train_test_extracted_%03d'%(data_set_count)

extracted_data_loc = os.path.join(extracted_data_base_loc, data_set_name)
os.system('mkdir -p %s'%extracted_data_loc)

collection_of_read_data_files = '{}_files_read.pkl'.format(data_set_name)
extracted_data_file_name = '{}_df.pkl'.format(data_set_name)

tab_data_file_name = '{}_tab_data.csv'.format(data_set_name)


# #### DataSet Specific (Fire)

# In[17]:


fire_data_set_name = 'data_fire_extracted_%03d'%(data_set_count)

fire_data_loc = os.path.join(extracted_data_base_loc, fire_data_set_name)
os.system('mkdir -p %s'%fire_data_loc)

fire_data_file_name = '{}.pkl'.format(fire_data_set_name)


# ## Relevant Fire TimeStamps

# In[18]:


fire_time_stamps = input_json_data['fire_time_stamps']


# # Generate seed for the random number generator

# In[19]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # File Names

# In[20]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
data_files_list_all = get_data_file_names(data_files_location)
data_files_list = data_files_list_all
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "get_data_file_names" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Remove Files Corresponding To Fire Data

# ## Get Indices for Fire Time Stamps

# In[21]:


if remove_fire_data_from_train_test or extract_fire_data:
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    fire_time_indices = get_fire_time_indices (fire_time_stamps, data_files_list_all)
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = module_final_memory - module_initial_memory
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
    print('Module "get_fire_time_indices" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Remove the Files with Indices for Fire Time Stamps

# In[22]:


if remove_fire_data_from_train_test:
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    data_files_list = remove_data_around_fire (fire_time_indices, data_files_list)
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = module_final_memory - module_initial_memory
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
    print('Module "remove_data_around_fire" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# In[23]:


#len(data_files_list)


# # Deal with just first few files to check for correctness of script. Be sure to undo this

# In[24]:


#data_files_list = data_files_list[0:30]


# # Downsample Files

# In[25]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
sampled_file_indices, sampled_data_files = downsample_data_files (                                        data_files_list, percent_files_to_use,                                         max_history_to_consider, random_state)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "downsample_data_files" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Get History File Indices

# In[26]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
history_file_indices = get_history_file_indices (sampled_file_indices,                                                  max_history_to_consider, history_interval)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "get_history_file_indices" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Create timestamps and datetime of downsampled data files

# In[27]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
sampled_time_stamps, sampled_datetime = get_datetime_for_data_files (sampled_data_files)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "get_datetime_for_data_files" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Create DataFrame using sampled file indices, filenames, timestamps, and datetime

# In[28]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
df_sampled_time = create_df_sampled_time (sampled_file_indices, sampled_data_files,                                           sampled_time_stamps, sampled_datetime,                                           history_file_indices)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "create_df_sampled_time" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# In[29]:


#df_sampled_time[df_sampled_time['ref_time_indices'] < max_history_to_consider+10]


# In[30]:


df_sampled_time.head(30)


# # Plot Sampled Datetime

# In[31]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_sampled_datetime (df_sampled_time, extracted_data_loc)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_sampled_datetime" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Read Quantities in a Selected Data File

# ## Read the Data in a Specified or Randomly Selected File

# In[32]:


data_in_a_file = input_json_data['data_in_a_file']
prescribe_file = data_in_a_file['prescribe_file_flag']
if prescribe_file:
    data_file_to_read = data_in_a_file['data_file_to_read']
    timestamp_to_read = data_file_to_read.split('_')[1] + '_' +                         data_file_to_read.split('_')[2].split('.')[0]
else:
    random_ind_of_downsampled_files = random.choice(range(len(sampled_file_indices)))
    file_ind_to_read = sampled_file_indices[random_ind_of_downsampled_files]
    data_file_to_read = sampled_data_files[random_ind_of_downsampled_files]
    timestamp_to_read = sampled_time_stamps[random_ind_of_downsampled_files]


# In[33]:


data_file_to_read, timestamp_to_read


# In[34]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
data_at_timestamp = read_single_data_file (data_files_location, data_file_to_read,                                            timestamp_to_read)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "read_single_data_file" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Processing Elevation Data into Pos, Neg, and Zero

# In[35]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
data_at_timestamp = process_elevation_at_timestamp (data_at_timestamp)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "process_elevation_at_timestamp" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Get and Plot Grid Indices (All and Considerable)

# ## Get Grid Indices

# In[36]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
grid_indices_all, grid_indices_valid, grid_indices_all_flat, grid_indices_valid_flat =                         get_grid_indices_all (data_files_location, sampled_file_indices,                                               sampled_data_files, sampled_time_stamps,                                               x_clip_train_test, y_clip_train_test,                                               j_nevada, i_nevada, j_anchor, i_anchor,                                               remove_nevada)

module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "get_grid_indices_all" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Reconstruct Grid Indices

# In[37]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
grid_indices_valid_reconst, grid_indices_valid_bool, valid_grid_ind_to_coord =                 reconstruct_valid_grid_indices (grid_indices_valid_flat, data_at_timestamp)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "reconstruct_valid_grid_indices" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Plot Grid Indices

# In[38]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_contours_of_indices (data_at_timestamp, grid_indices_all, grid_indices_valid,                           grid_indices_valid_bool, grid_indices_valid_reconst,                           extracted_data_loc)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_contours_of_indices" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# In[39]:


#len(grid_indices_valid_flat)


# # Plot Quantities in the Selected Data File

# ## Plot the Contours of QoIs for the Data Just Read Above

# In[40]:


qoi_to_plot = input_json_data['qoi_to_plot']['contours']


# ### Unmasked Data

# In[41]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc,                             grid_indices_valid, masked = False)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_contours_at_timestamp" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ### Masked Data

# In[42]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_contours_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc,                             grid_indices_valid, masked = True)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_contours_at_timestamp" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Plot the PDFs of QoIs for the Data Just Read Above

# In[43]:


qoi_to_plot = input_json_data['qoi_to_plot']['pdfs']


# In[44]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_pdf_at_timestamp (data_at_timestamp, qoi_to_plot, extracted_data_loc)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_pdf_at_timestamp" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Plot the Contours of QoIs With Colorbars

# In[45]:


qoi_to_plot = input_json_data['qoi_to_plot']['contours_with_cb']
cont_levels_count = input_json_data['qoi_to_plot']['cont_levels_count']
qoi_cont_range = input_json_data['qoi_to_plot']['qoi_cont_range']


# In[46]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_contours_at_timestamp2 (data_at_timestamp, timestamp_to_read, qoi_to_plot,                              extracted_data_loc, grid_indices_valid,                              cont_levels_count, qoi_cont_range, masked = True)
#plot_contours_at_timestamp2 (data_at_timestamp, timestamp_to_read, qoi_to_plot, extracted_data_loc, grid_indices_valid, cont_levels_count, masked = True)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_contours_at_timestamp2" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Sample and Plot Grid Indices for Each Sampled Ref Time

# ## Sample Grid Indices

# In[47]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
grid_indices_selected, j_indices_selected, i_indices_selected =     sample_grid_indices (sampled_file_indices, percent_grid_points_to_use,                          grid_indices_valid_flat, valid_grid_ind_to_coord)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "sample_grid_indices" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Plot Sampled Grid Indices

# In[48]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_sampled_grid_points (grid_indices_selected, extracted_data_loc)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_sampled_grid_points" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# ## Plot Sampled Grid Indices in 3D

# In[49]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
plot_sampled_grid_points_3D (j_indices_selected, i_indices_selected,                              extracted_data_loc, (6, 6)) #fig_size hard-coded
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "plot_sampled_grid_points_3D" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# # Create a Dict of Time Indices and Grid Indices

# In[50]:


module_start_time = timer()
module_initial_memory = process.memory_info().rss
time_grid_indices_list_dict, time_grid_indices_list_count, time_grid_indices_set_dict, time_grid_indices_set_count =     create_time_grid_indices_map (sampled_file_indices, history_file_indices,                                   grid_indices_selected)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "create_time_grid_indices_map" computing time: {:.3f} s'.format(module_end_time - module_start_time))


# In[51]:


#len(time_grid_indices_list_dict.keys())
#len(time_grid_indices_set_dict.keys())


# In[52]:


#sampled_file_indices
#grid_indices_selected


# In[53]:


#time_grid_indices_list_dict
#time_grid_indices_list_count
#time_grid_indices_set_dict
#time_grid_indices_set_count


# # Read Data at Sampled Time and Grid

# # Read Files At All Possible Time Indices (Ref + History)

# ## Read Data at All Times

# In[54]:


'''
module_start_time = timer()
module_initial_memory = process.memory_info().rss

file_indices_to_read = list(time_grid_indices_list_dict.keys())
data_files_to_read, time_stamps_to_read, file_indices_data_dict = \
                read_data_all_possible_times (file_indices_to_read, data_files_list, \
                                             data_files_location)

module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "read_data_all_possible_times" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# ## Save Data In a Pickle File

# In[55]:


'''
module_start_time = timer()
module_initial_memory = process.memory_info().rss
save_data_read_at_all_possible_times (file_indices_to_read, data_files_to_read, \
                                      time_stamps_to_read, file_indices_data_dict, \
                                      extracted_data_loc, collection_of_read_data_files)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "save_data_read_at_all_possible_times" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# ## Delete Data No Longer Needed To Free Memory

# In[56]:


'''
global_final_memory = process.memory_info().rss
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed so far: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))

print('Deleting Some Variables')
del file_indices_data_dict

global_final_memory = process.memory_info().rss
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed so far: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
'''


# ## Read Data at All Possible Times Saved in a Pickle File

# In[57]:


'''
module_start_time = timer()
module_initial_memory = process.memory_info().rss
collection_of_read_data = read_data_from_pickle_all_possible_times (extracted_data_loc, \
                                                              collection_of_read_data_files)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "read_data_from_pickle_all_possible_times" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# In[58]:


#collection_of_read_data['file_indices_data_dict'].keys()


# # Extract and Save Data at The Sampled Time and Grid Points

# ## Extract Data at The Sampled Time and Grid Points

# In[59]:


'''
data_at_times = collection_of_read_data['file_indices_data_dict']
#data_at_times.keys()
'''


# In[60]:


'''
module_start_time = timer()
module_initial_memory = process.memory_info().rss
df = create_dataframe_FM_atm_data (data_at_times, \
                                   sampled_file_indices, history_file_indices, \
                                   sampled_time_stamps, history_interval, \
                                   grid_indices_selected, \
                                   j_indices_selected, i_indices_selected)
module_final_memory = process.memory_info().rss
module_end_time = timer()
module_memory_consumed = module_final_memory - module_initial_memory
print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
print('Module "create_dataframe_FM_atm_data" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# In[61]:


#df.head(5)


# ## Save The Data Extracted  at Sampled Time and Grid Points

# In[62]:


'''
df.to_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))
'''


# ## Load and Test The Extracted Data Saved in Pickle File

# In[63]:


'''
df_from_pickle = pd.read_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))
df_from_pickle.head(5)
'''


# # Save Other Relevant Info in A CSV File

# In[64]:


'''
data_for_csv = { 'max_history_to_consider':    [max_history_to_consider],
                 'history_interval':           [history_interval],
                 'num_hist_indices':           [len(history_file_indices[0])],
                 'num_total_files':            [len(data_files_list)],
                 'percent_files_to_use':       [percent_files_to_use],
                 'num_sampled_times':          [grid_indices_selected.shape[0]],
                 'num_data_files_to_read':     [len(data_files_to_read)],
                 'num_grid_points_sn':         [grid_indices_all.shape[0]],
                 'num_grid_points_we':         [grid_indices_all.shape[1]],
                 'num_total_grid_points':      [len(grid_indices_all_flat)],
                 'num_valid_grid_points':      [len(grid_indices_valid_flat)],
                 'percent_grid_points_to_use': [percent_grid_points_to_use],
                 'num_sampled_grid_points':    [grid_indices_selected.shape[1]],
                 'num_data_points':            [len(df)]               
}
tabulated_data = pd.DataFrame(data_for_csv)
tabulated_data.to_csv(os.path.join(extracted_data_loc, tab_data_file_name), index = False)
'''


# In[65]:


'''
tabulated_data
'''


# # Extract Fire Data

# ## Read Fire Data

# In[66]:


'''
if extract_fire_data:
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss

    fire_time_indices, fire_data = read_fire_data (\
                                fire_time_indices, max_history_to_consider, history_interval, \
                                data_files_list_all, data_files_location)

    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = module_final_memory - module_initial_memory
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
    print('Module "read_fire_data" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# ## Create DataFrame for Fire Data

# In[67]:


'''
if extract_fire_data:
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    fire_data_extracted = dict()

    for fire_name in fire_time_indices.keys():
        data_at_times = fire_data[fire_name]['fire_file_indices_data_dict']
        df_fire = create_dataframe_FM_atm_data_fire (fire_name, fire_time_indices, data_at_times, \
                                                     history_interval, \
                                                     grid_indices_valid_flat, valid_grid_ind_to_coord)
        fire_data_extracted[fire_name] = df_fire

    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = module_final_memory - module_initial_memory
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed/(1024*1024)))
    print('Module "create_dataframe_FM_atm_data_fire" computing time: {:.3f} s'.format(module_end_time - module_start_time))
'''


# ## Save DataFrame for Fire Data

# In[68]:


'''
if extract_fire_data:
    fire_data_file_handle = open(os.path.join(fire_data_loc, fire_data_file_name), 'wb')
    pickle.dump(fire_data_extracted, fire_data_file_handle)
    fire_data_file_handle.close()
    print('Wrote fire data in "{}" at "{}"'.format(fire_data_file_name, fire_data_loc))
'''


# ## Load and Test The Extracted Fire Data Saved in Pickle File

# In[69]:


'''
if extract_fire_data:
    fire_data_file_handle = open(os.path.join(fire_data_loc, fire_data_file_name), 'rb')
    fire_data_pickled = pickle.load(fire_data_file_handle)
    fire_data_file_handle.close()
    print('Read fire data from "{}" at "{}"'.format(fire_data_file_name, fire_data_loc))
'''


# In[70]:


'''
if extract_fire_data:
    fire_data_pickled['Woosley'].head(5)
'''


# ## Delete Fire Data No Longer Needed

# In[71]:


'''
if extract_fire_data:
    del fire_data, data_at_times, fire_data_extracted
'''


# # Global End Time and Memory

# In[72]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Extraction of Data")

