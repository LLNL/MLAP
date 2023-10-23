#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# In[ ]:


#jupyter nbconvert --to python Prepare_TrainTest_Data.ipynb


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


sys.path.insert(0, os.path.join(os.path.pardir, 'Step1_ExtractData'))


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


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/01_WRF_Nelson_Data_Extracted/InputJsonFiles/json_extract_data_000.json'
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/02_TrainTest_Data_Prepared/InputJsonFiles/json_prep_data_label_000.json'


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


# # Variables to be Used for Preparing Train, Test, and Fire Data

# ## DataSet Defintion

# In[ ]:


# The current data set params
data_set_count = json_content_extract_data['data_set_defn']['data_set_count']


# ## Define Label, FM Threshold etc.

# In[ ]:


label_count = json_content_prep_data['label_defn']['label_count']


# In[ ]:


FM_labels = json_content_prep_data['FM_labels']


# In[ ]:


FM_label_type = FM_labels['label_type']
FM_binary_threshold = FM_labels['FM_binary_threshold']
FM_MC_levels = FM_labels['FM_MC_levels']


# ## Paths and File Names

# #### Global

# In[ ]:


# WRF data set location and the extracted data set location
extracted_data_base_loc = json_content_extract_data['paths']['extracted_data_base_loc']
prepared_data_base_loc  = json_content_prep_data['paths']['prepared_data_base_loc']


# #### DataSet Specific (Train, Test, Fire Data Extracted from WRF)

# In[ ]:


data_set_name = 'data_train_test_extracted_%02d'%(data_set_count)
extracted_data_loc = os.path.join(extracted_data_base_loc, data_set_name)
extracted_data_file_name = '{}_df.pkl'.format(data_set_name)

fire_data_set_name = 'data_fire_extracted_%02d'%(data_set_count)
fire_data_loc = os.path.join(extracted_data_base_loc, fire_data_set_name)
fire_data_file_name = '{}.pkl'.format(fire_data_set_name)


# #### DataSet Specific (Train, Test, Fire Prepared)

# In[ ]:


prepared_data_set_name = 'dataset_%03d_label_%03d'%(data_set_count, label_count)

prepared_data_loc = os.path.join(prepared_data_base_loc, prepared_data_set_name)
os.system('mkdir -p %s'%prepared_data_loc)

prepared_data_file_name = '{}.pkl'.format(prepared_data_set_name)


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # Load The Pickled Extracted Data (Train, Test, Fire) from WRF 

# ## Load The Train/Test Data Saved in Pickle File

# In[ ]:


df_tt_extracted = pd.read_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))
#df_tt_extracted[998:1002]


# ## Load The Fire Data Saved in Pickle File

# In[ ]:


fire_data_file_handle = open(os.path.join(fire_data_loc, fire_data_file_name), 'rb')
fire_data_extracted = pickle.load(fire_data_file_handle)
fire_data_file_handle.close()
print('Read fire data from "{}" at "{}"'.format(fire_data_file_name, fire_data_loc))


# In[ ]:


#fire_data_extracted['Woosley'].head(5)


# ## Ensure The Train/test and Fire Data Have the Same Keys

# In[ ]:


#df_tt_extracted.keys() == fire_data_extracted['Woosley'].keys()


# # Get Column Names in the Train, Test, and Fire Data

# In[ ]:


keys_identity, keys_FM, keys_U10, keys_V10, keys_UMag10, keys_T2, keys_RH, keys_PREC, keys_SW,                             keys_HGT = get_keys_from_extracted_data (df_tt_extracted)


# ### Define Binary and MC FM labels

# In[ ]:


keys_FM_Binary, keys_FM_MC = define_binary_and_MC_FM_labels (keys_FM)


# ### Define Groups of Keys

# In[ ]:


keys_labels = keys_FM + keys_FM_Binary + keys_FM_MC
keys_features = keys_UMag10 + keys_T2 + keys_RH + keys_PREC + keys_SW #+ keys_HGT


# # Compute New Columns or Remove Some

# ## Compute Wind Magnitude 

# In[ ]:


df_tt_prep = compute_wind_mag (df_tt_extracted, keys_U10, keys_V10, keys_UMag10)

fire_data_prep = dict()
for fire_name in fire_data_extracted.keys():
    fire_data_prep[fire_name] = compute_wind_mag (fire_data_extracted[fire_name], 
                                                  keys_U10, keys_V10, keys_UMag10)


# In[ ]:


#df_tt_prep[keys_U10 + keys_V10 + keys_UMag10]
#fire_data_prep['Woosley'][keys_U10 + keys_V10 + keys_UMag10]


# ## Drop Wind Components

# In[ ]:


df_tt_prep = drop_wind_components (df_tt_prep, keys_U10, keys_V10)

for fire_name in fire_data_prep.keys():
    fire_data_prep[fire_name] = drop_wind_components (                                        fire_data_prep[fire_name], keys_U10, keys_V10)


# In[ ]:


#df_tt_prep[keys_UMag10]
#fire_data_prep['Woosley'][keys_UMag10]


# ## Compute Binary FM Labels

# In[ ]:


df_tt_prep = compute_binary_FM_labels(df_tt_prep,                                       keys_FM, keys_FM_Binary, FM_binary_threshold)

for fire_name in fire_data_prep.keys():
    fire_data_prep[fire_name] = compute_binary_FM_labels (fire_data_prep[fire_name],                                       keys_FM, keys_FM_Binary, FM_binary_threshold)


# In[ ]:


#len(df_tt_prep.keys())
#len(fire_data_prep['Woosley'].keys())


# In[ ]:


#df_tt_prep[keys_FM + keys_FM_Binary][985:995]
#fire_data_prep['Woosley'][keys_FM +keys_FM_Binary]


# ## Compute MC FM Labels

# In[ ]:


df_tt_prep = compute_MC_FM_labels(df_tt_prep,                                   keys_FM, keys_FM_MC, FM_MC_levels)

for fire_name in fire_data_prep.keys():
    fire_data_prep[fire_name] = compute_MC_FM_labels (fire_data_prep[fire_name],                                       keys_FM, keys_FM_MC, FM_MC_levels)


# In[ ]:


#df_tt_prep[keys_FM + keys_FM_MC]
#fire_data_prep['Woosley'][keys_FM + keys_FM_MC]


# ## Plot Binary FM Labels

# In[ ]:


columns_to_plot = json_content_prep_data['qoi_to_plot']['FM_binary_columns_to_plot']
plot_binary_FM_labels (df_tt_prep, columns_to_plot, prepared_data_set_name, prepared_data_loc)


# ## Plot MC FM Labels

# In[ ]:


col_to_plot = json_content_prep_data['qoi_to_plot']['FM_MC_column_to_plot']
plot_MC_FM_labels (df_tt_prep, col_to_plot, prepared_data_set_name, prepared_data_loc)


# # Split Data into Identity, Features, and Labels

# In[ ]:


data_tt_prep, data_fire_prep = split_data_into_groups (                                            df_tt_prep, fire_data_prep,                                             keys_identity, keys_labels, keys_features)


# # Save The Prepared Data

# In[ ]:


prepared_data = {'tt': data_tt_prep,
                 'fire': data_fire_prep}
prepared_data_file_handle = open(os.path.join(prepared_data_loc, prepared_data_file_name), 'wb')
pickle.dump(prepared_data, prepared_data_file_handle)
prepared_data_file_handle.close()
print('Wrote prepared data in "{}" at "{}"'.format(prepared_data_file_name, prepared_data_loc))


# # Load and Test The Prepared Data Saved in Pickle File

# In[ ]:


prepared_data_file_handle = open(os.path.join(prepared_data_loc, prepared_data_file_name), 'rb')
prepared_data_read = pickle.load(prepared_data_file_handle)
prepared_data_file_handle.close()
print('Read prepared data from "{}" at "{}"'.format(prepared_data_file_name, prepared_data_loc))


# In[ ]:


#prepared_data_read['tt']['all'].head(5)
#prepared_data_read['fire']['Woosley']['identity'].head(5)


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Preparation of Data")

