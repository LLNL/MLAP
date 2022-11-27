#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


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


# In[2]:


from helper_functions import generate_seed, init_random_generator
from helper_functions import get_data_file_names, downsample_data_files
from helper_functions import downsample_grid_indices
from helper_functions import create_df_at_gp


# # Variables to be used for preparing training data

# In[3]:


# WRF data set location and the extracted data set location
data_files_location = '/p/lustre1/mirocha2/SJSU_DATA/akochanski/PGnE_climo/dfm'
extracted_data_loc = '/p/lustre2/jha3/Wildfire/Wildfire_SJSU/data_extracted'

# The current data set params
data_set_count = 0
percent_files_to_use = 5.0         # f1 = what percent of available files to use
percent_grid_points_to_use = 0.005  # f2 = what percent of grid points to use
max_history_to_consider = 5 # n_history in hours
history_interval        = 2

# Some fixed stuff
frames_in_file          = 153
label_fields = ['mean_wtd_moisture_1hr', 'mean_wtd_moisture_10hr']
identity_fields = ['latitude', 'longitude']
feature_fields = ['eastward_10m_wind', 'northward_10m_wind',                  'air_temperature_2m',                   'accumulated_precipitation_amount',                   'air_relative_humidity_2m',                   'surface_downwelling_shortwave_flux'] 


# # Generate seed for the random number generator

# In[4]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # Paths, File Names, Downsample Files

# In[5]:


data_files_list = get_data_file_names(data_files_location)
sampled_file_indices, sampled_data_files = downsample_data_files (data_files_list, percent_files_to_use)


# # Grid Dimensions, Downsample Grid Points

# In[6]:


df_for_all_files = pd.DataFrame()
for file_count, data_file_name in enumerate(sampled_data_files):
    print ('\nReading data from file # {}, with name :- {}'.format(file_count, data_file_name))
    print('-----------------------------------------------------------------------')
    dfm_file_data = xr.open_dataset(path.join(data_files_location, data_file_name))
    
    df_for_single_file = downsample_grid_indices (dfm_file_data, percent_grid_points_to_use, 
                                                  max_history_to_consider, history_interval, frames_in_file)
    
    df_for_all_files = df_for_all_files.append(df_for_single_file).reset_index(drop = True)


# In[7]:


df_for_all_files.head(10)


# # Save the extracted data

# In[8]:


data_set_name = 'extracted_data_%02d'%(data_set_count)
extracted_data_file_name = '{}_files_{}pc_grid_points_{}pc_max_history_{}_hist_interval_{}.pkl'.format(
                            data_set_name, # name of data set
                            percent_files_to_use, # f1 = what percent of available files to use
                            percent_grid_points_to_use, # f2 = what percent of grid points to use
                            max_history_to_consider, # n_history in hours
                            history_interval)
extracted_data = {'percent_files_to_use': percent_files_to_use,
                 'percent_grid_points_to_use': percent_grid_points_to_use,
                 'max_history_to_consider': max_history_to_consider,
                 'history_interval': history_interval,
                 'df_for_all_files': df_for_all_files}
extracted_data_file_handle = open(os.path.join(
    extracted_data_loc, extracted_data_file_name), 'wb')
pickle.dump(extracted_data, extracted_data_file_handle)
extracted_data_file_handle.close()


# # Load extracted data from pickle file

# In[9]:


loaded_data = pickle.load(open(os.path.join(
    extracted_data_loc, extracted_data_file_name), 'rb'))


# In[10]:


loaded_data['df_for_all_files'].head(10)


# In[ ]:




