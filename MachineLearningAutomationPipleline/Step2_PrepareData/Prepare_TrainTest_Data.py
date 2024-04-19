#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python Prepare_TrainTest_Data.ipynb

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


current_running_file_dir = sys.path[0]
current_running_file_par = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, os.path.join(current_running_file_par, 'Step1_ExtractData'))


# In[ ]:


#print('current_running_file_dir:', current_running_file_dir)
#print('current_running_file_par:', current_running_file_par)
#print('PATH: ', sys.path)


# In[ ]:


from Extract_DFM_Data_Helper import *
from Prepare_TrainTest_Data_Helper import *


# # Global Start Time and Memory

# In[ ]:


global_start_time = timer()
process = psutil.Process(os.getpid())
global_initial_memory = process.memory_info().rss


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Extract/json_extract_data_000.json'
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Prep/json_prep_data_label_007.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]
#json_file_prep_data = sys.argv[2]


# ### Load the JSON file for extracting data

# In[ ]:


print('Loading the JSON file for extracting data: \n {}'.format(json_file_extract_data))


# In[ ]:


with open(json_file_extract_data) as json_file_handle:
    json_content_extract_data = json.load(json_file_handle)


# In[ ]:


#json_content_extract_data


# ### Load the JSON file for preparing data

# In[ ]:


print('Loading the JSON file for preparing data: \n {}'.format(json_file_prep_data))


# In[ ]:


with open(json_file_prep_data) as json_file_handle:
    json_content_prep_data = json.load(json_file_handle)


# In[ ]:


#json_content_prep_data


# # Variables to be Used for Preparing Train and Test Data

# ## DataSet Defintion

# In[ ]:


# The current data set params
data_set_count = json_content_extract_data['data_set_defn']['data_set_count']


# In[ ]:


features_labels = json_content_extract_data['features_labels']
qois_to_read = features_labels['qois_to_read']


# ## Define Label, FM Threshold etc.

# In[ ]:


label_count = json_content_prep_data['label_defn']['label_count']


# In[ ]:


FM_labels = json_content_prep_data['FM_labels']


# In[ ]:


FM_label_type = FM_labels['label_type']

if (FM_label_type == 'Binary'):
    FM_binary_threshold = FM_labels['FM_binary_threshold']
if (FM_label_type == 'MultiClass'):
    FM_MC_levels = FM_labels['FM_MC_levels']


# In[ ]:


qois_to_use = json_content_prep_data['features']['qois_to_use']
qois_derived = json_content_prep_data['features']['qois_derived']


# In[ ]:


prune_data = json_content_prep_data['prune_data']


# ## Paths and File Names

# #### Global

# In[ ]:


# WRF data set location and the extracted data set location
extracted_data_base_loc = json_content_extract_data['paths']['extracted_data_base_loc']
prepared_data_base_loc  = json_content_prep_data['paths']['prepared_data_base_loc']


# #### DataSet Specific (Train and Test Data Extracted from WRF)

# In[ ]:


data_set_name = 'data_train_test_extracted_%03d'%(data_set_count)
extracted_data_loc = os.path.join(extracted_data_base_loc, data_set_name)
extracted_data_file_name = '{}_df.pkl'.format(data_set_name)


# #### DataSet and Label Specific (Train and Test Data Prepared)

# In[ ]:


prepared_data_set_name = 'dataset_%03d_label_%03d_%s'%(data_set_count, label_count, FM_label_type)

prepared_data_loc = os.path.join(prepared_data_base_loc, prepared_data_set_name)
os.system('mkdir -p %s'%prepared_data_loc)

prepared_data_file_name = '{}.pkl'.format(prepared_data_set_name)
sampled_time_ind_file_name = '{}_time_ind.csv'.format(prepared_data_set_name)
sampled_grid_ind_file_name = '{}_grid_ind.csv'.format(prepared_data_set_name)


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # Load The Pickled Extracted Data (Train, Test) from WRF 

# ## Load The Train/Test Data Saved in Pickle File

# In[ ]:


df_tt_prep = load_pickled_data (extracted_data_loc, extracted_data_file_name)


# In[ ]:


#df_tt_prep.keys()


# In[ ]:


#df_tt_prep['PRECIP[-8hr]']


# In[ ]:


#len(df_tt_prep)


# ## Prune Extracted Data

# In[ ]:


df_tt_prep = prune_desired_data (df_tt_prep, prune_data)


# ## Reduce The Size of Extracted Train/Test Data

# In[ ]:


df_tt_prep = reduce_data_size (df_tt_prep)


# In[ ]:


#df_tt_prep.dtypes


# ## Assess The sampled Time  and Grid Indices

# In[ ]:


#df_tt_prep[['FM_time_ind', 'FM_ts', 'his_time_ind','j_ind', 'i_ind']]
#df_tt_prep['FM_time_ind']
#sorted_time_indices = df_tt_prep['FM_time_ind'].sort_values().unique()


# In[ ]:


df_time_indices = assess_sampled_indices (df_tt_prep, 'FM_time_ind',                                             prepared_data_loc, sampled_time_ind_file_name)


# In[ ]:


df_grid_indices = assess_sampled_indices (df_tt_prep, 'grid_ind',                                             prepared_data_loc, sampled_grid_ind_file_name)


# In[ ]:


#df_time_indices[0:30]


# In[ ]:


#len(df_time_indices)


# In[ ]:


#df_grid_indices[3000:3010]


# # Get Column Names in the Train and Test Data

# In[ ]:


keys_identity, keys_FM, keys_U10, keys_V10, keys_UMag10, keys_T2, keys_RH, keys_PREC, keys_SW,                             keys_HGT = get_keys_from_extracted_data (df_tt_prep)


# In[ ]:


#keys_U10, keys_V10, keys_UMag10


# In[ ]:


keys_FM_Binary, keys_FM_MC = define_binary_and_MC_FM_labels (keys_FM)


# ### Define Groups of Keys

# In[ ]:


keys_labels = define_labels(FM_label_type, keys_FM, keys_FM_Binary, keys_FM_MC)


# In[ ]:


#keys_FM_MC


# In[ ]:


keys_features  = define_features(keys_HGT, keys_UMag10, keys_T2, keys_RH, keys_PREC, keys_SW,                    qois_to_use)


# In[ ]:


#keys_features


# 
# # Compute New Columns or Remove Some

# ## Compute Wind Magnitude 

# In[ ]:


#features_to_read


# In[ ]:


if ('UMag10' not in qois_to_read):
    df_tt_prep = compute_wind_mag (df_tt_prep, keys_U10, keys_V10, keys_UMag10)


# In[ ]:


#df_tt_prep[keys_U10 + keys_V10 + keys_UMag10]
#df_tt_prep


# ## Drop Wind Components

# In[ ]:


if ('UMag10' not in qois_to_read):
    df_tt_prep = drop_wind_components (df_tt_prep, keys_U10, keys_V10)


# In[ ]:


#df_tt_prep[keys_UMag10]


# ## Compute VPD

# In[ ]:


if 'VPD' in qois_derived:
    df_tt_prep, keys_VPD = compute_VPD (df_tt_prep, keys_T2, keys_RH)
    keys_features += keys_VPD


# In[ ]:


#keys_T2_Cel, keys_VP_s, keys_VP, keys_VPD


# In[ ]:


#df_tt_prep[keys_T2 + keys_T2_Cel + keys_RH + keys_VP_s + keys_VP + keys_VPD]
#df_tt_prep[keys_T2 +               keys_RH +                       keys_VPD]


# In[ ]:


#plt.hist(df_tt_prep['VPD[-6hr]'])


# ## Compute Binary FM Labels

# In[ ]:


if FM_label_type == 'Binary':
    df_tt_prep = compute_binary_FM_labels(df_tt_prep,                                           keys_FM, keys_FM_Binary, FM_binary_threshold)


# In[ ]:


#len(df_tt_prep.keys())
#df_tt_prep[keys_FM + keys_FM_Binary].dtypes


# In[ ]:


#df_tt_prep[keys_FM + keys_FM_Binary][985:995]


# ## Compute MC FM Labels

# In[ ]:


if FM_label_type == 'MultiClass':
    df_tt_prep = compute_MC_FM_labels(df_tt_prep,                                       keys_FM, keys_FM_MC, FM_MC_levels)


# In[ ]:


#df_tt_prep[keys_FM + keys_FM_MC].dtypes
#df_tt_prep[keys_FM + keys_FM_MC]


# # Plot FM Labels

# In[ ]:


FM_hr = json_content_prep_data['qoi_to_plot']['FM_hr']


# In[ ]:


plot_FM_labels (df_tt_prep, FM_label_type, FM_hr,                 prepared_data_set_name, prepared_data_loc)


# # Split Data into Identity, Features, and Labels

# In[ ]:


data_tt_prep = split_data_into_groups (df_tt_prep,                                        keys_identity, keys_labels, keys_features)


# # Save The Prepared Data

# In[ ]:


prepared_data = data_tt_prep
with open(os.path.join(prepared_data_loc, prepared_data_file_name), 'wb') as file_handle:
    pickle.dump(prepared_data, file_handle)
print('Wrote prepared data in "{}" at "{}"'.format(prepared_data_file_name, prepared_data_loc))


# # Load and Test The Prepared Data Saved in Pickle File

# In[ ]:


prepared_data_read = load_pickled_data (prepared_data_loc, prepared_data_file_name)


# In[ ]:


#with open(os.path.join(prepared_data_loc, prepared_data_file_name), 'rb') as file_handle:
 #   prepared_data_read = pickle.load(file_handle)
print('Read prepared data from "{}" at "{}"'.format(prepared_data_file_name, prepared_data_loc))


# In[ ]:


#prepared_data_read['identity'].


# In[ ]:


#prepared_data_read['labels'].head(5)


# In[ ]:


#prepared_data_read['labels'][prepared_data_read['labels']['FM_10hr_bin'] == 1]


# In[ ]:


#prepared_data_read['features'].dtypes
#prepared_data_read['features'].head(5)


# In[ ]:


#prepared_data_read['features']['UMag10[-8hr]']


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Preparation of Data")

