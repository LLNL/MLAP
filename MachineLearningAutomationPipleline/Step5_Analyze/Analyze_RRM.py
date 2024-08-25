#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python Analyze_RRM.ipynb

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
import time


# ## Scikit-Learn

# In[ ]:


#from sklearn.svm import SVC, SVR
#from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#from sklearn.neural_network import MLPClassifier, MLPRegressor

#from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
#from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report


# ## User-Defined Functions

# In[ ]:


current_running_file_dir = sys.path[0]
current_running_file_par = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, os.path.join(current_running_file_par, 'Step1_ExtractData'))
sys.path.insert(0, os.path.join(current_running_file_par, 'Step2_PrepareData'))
sys.path.insert(0, os.path.join(current_running_file_par, 'Step3_TrainModel'))


# In[ ]:


from Extract_DFM_Data_Helper import *
from Prepare_TrainTest_Data_Helper import *
from TrainModel_Helper import *
from Analyze_Helper import *


# # Global Start Time and Memory

# In[ ]:


global_start_time = timer()
process = psutil.Process(os.getpid())
global_initial_memory = process.memory_info().rss


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Extract/json_extract_data_039.json'
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Prep/json_prep_data_label_006.json'
json_file_train_model  = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Train/json_train_model_007.json'
json_file_trends      = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Trends/json_trends_002.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]
#json_file_prep_data = sys.argv[2]
#json_file_train_model = sys.argv[3]
#json_file_trends  = sys.argv[4]


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


# ### Load the JSON file for training model

# In[ ]:


print('Loading the JSON file for training model: \n {}'.format(json_file_train_model))


# In[ ]:


with open(json_file_train_model) as json_file_handle:
    json_content_train_model = json.load(json_file_handle)


# In[ ]:


#json_content_train_model


# ### Load the JSON file for analysis of trends

# In[ ]:


print('Loading the JSON file for analysis: \n {}'.format(json_file_trends))


# In[ ]:


with open(json_file_trends) as json_file_handle:
    json_content_trends = json.load(json_file_handle)


# In[ ]:


#json_content_trends


# # Variables to be Used for Analysis

# ## DataSet Defintion

# In[ ]:


# The current data set params
data_set_count = json_content_extract_data['data_set_defn']['data_set_count']
max_history_to_consider = json_content_extract_data['data_set_defn']['max_history_to_consider']
history_interval = json_content_extract_data['data_set_defn']['history_interval']


# In[ ]:


features_labels = json_content_extract_data['features_labels']
qois_to_read = features_labels['qois_to_read']


# ## Nevada Data

# In[ ]:


nevada_data = json_content_extract_data['nevada_data']
remove_nevada = nevada_data['remove_nevada']
j_nevada, i_nevada = nevada_data['j_nevada'], nevada_data['i_nevada']
j_anchor, i_anchor = nevada_data['j_anchor'], nevada_data['i_anchor']


# ## Clip Data for Train/Test

# In[ ]:


clip_data_train_test = json_content_extract_data['clip_data_train_test']
x_clip_train_test = clip_data_train_test['x_clip']
y_clip_train_test = clip_data_train_test['y_clip']


# ## Define Label, FM Threshold etc.

# In[ ]:


label_count = json_content_prep_data['label_defn']['label_count']


# In[ ]:


FM_labels = json_content_prep_data['FM_labels']


# In[ ]:


FM_label_type = FM_labels['label_type']

class_labels = None
if (FM_label_type == 'Binary'):
    FM_binary_threshold = FM_labels['FM_binary_threshold']
    class_labels = range(2)
if (FM_label_type == 'MultiClass'):
    FM_MC_levels = FM_labels['FM_MC_levels']
    class_labels = range(len(FM_MC_levels) -1)


# In[ ]:


FM_hr = json_content_prep_data['qoi_to_plot']['FM_hr']


# In[ ]:


qois_to_use = json_content_prep_data['features']['qois_to_use']
qois_derived = json_content_prep_data['features']['qois_derived']


# In[ ]:


prune_data = json_content_prep_data['prune_data']


# ## Define ML Model and Params etc.

# ### Model Definition 

# In[ ]:


model_count = json_content_train_model['models']['model_count']
scaler_type = json_content_train_model['models']['scaler_type']
model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
#model_params = json_content_train_model['models']['params']


# ## Define Trends Analysis Inputs

# In[ ]:


trends_count = json_content_trends['trends_count']


# In[ ]:


base_data_loc = json_content_trends["base_data_loc"]


# In[ ]:


year_range = json_content_trends["year_range"]
year = json_content_trends["year"]
start_time_stamp = json_content_trends["start_time_stamp"]
start_time_for_avg = json_content_trends["start_time_for_avg"]
end_time_for_avg = json_content_trends["end_time_for_avg"]


# In[ ]:


base_file_name = json_content_trends["base_file_name"]
CA_mask_file_name = json_content_trends["CA_mask_file_name"]
CA_mask_file = os.path.join(base_data_loc, CA_mask_file_name)


# In[ ]:


prediction_interval = json_content_trends["prediction_interval"]


# ### Paths

# In[ ]:


trends_data_paths = json_content_trends['paths']
trends_data_base_loc = trends_data_paths['trends_data_base_loc']
os.system('mkdir -p %s'%trends_data_base_loc)


# # Paths and File Names

# #### Global

# In[ ]:


data_files_location = json_content_extract_data['paths']['data_files_location']
trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
trends_data_base_loc = json_content_trends['paths']['trends_data_base_loc']


# In[ ]:


#raw_data_paths = json_content_analyze['paths']['raw_data']


# #### DataSet, Label, and Model Specific (Trained Model)

# In[ ]:


trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count,                                                         label_count, FM_label_type,                                                         model_count, model_name)

trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)

trained_model_file_name = '{}_model.pkl'.format(trained_model_name)


# In[ ]:


os.system('mkdir -p %s'%filedir)


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # ML Model

# ## Load the Model

# In[ ]:


trained_model_file = os.path.join(trained_model_loc, trained_model_file_name)
model = pickle.load(open(trained_model_file, 'rb'))
print ('\nLoaded the ML model file at: {}\n'.format(trained_model_file))
print ('The model loaded is: {} \n'.format(model))
print ('Model params: \n {}'.format(model.get_params()))


# # Read all the RRM data

# In[ ]:


U_var = 'WINDSPD_10M'
T_var = 'TREFHT'
SW_var = 'FSDS'
RH_var = 'RHREFHT'


# In[ ]:


data_U = read_single_RRM_file (base_data_loc, year_range, base_file_name, U_var)


# In[ ]:


data_T = read_single_RRM_file (base_data_loc, year_range, base_file_name, T_var)


# In[ ]:


data_SW = read_single_RRM_file (base_data_loc, year_range, base_file_name, SW_var)


# In[ ]:


data_RH = read_single_RRM_file (base_data_loc, year_range, base_file_name, RH_var)


# ## Play around with data read

# In[ ]:


#data_U
#np.array(data_U['WINDSPD_10M'][0]).shape
#data_T
#np.array(data_T['TREFHT'][0]).shape
#data_RH
#np.array(data_RH['RHREFHT'][0]).shape
#data_SW
#np.array(data_SW['FSDS'][0]).shape


# ## Geometry Info

# In[ ]:


lat_lon_file = os.path.join(base_data_loc, 'CAne32x32v1pg2.latlon.nc')
CA_mask_file = os.path.join(base_data_loc, 'CA_shp_mask_CAx32v1pg2.nc')


# In[ ]:


lat_lon_data = xr.open_dataset(lat_lon_file)
CA_mask_data = xr.open_dataset(CA_mask_file)


# In[ ]:


ca_mask_info = np.array(CA_mask_data['CA_shp_mask_CAx32v1pg2'])
masked_ind = np.where(ca_mask_info == 1)[0]
lat_info = np.array(CA_mask_data['lat'])
lon_info = np.array(CA_mask_data['lon'])
area_info = np.array(CA_mask_data['area'])


# # Create Dataframes at Time Stamps of Interest

# In[ ]:


start_time_stamp_for_avg = '{}-{}'.format(year, start_time_for_avg)
end_time_stamp_for_avg = '{}-{}'.format(year, end_time_for_avg)
fuel_moisture_time_index_start = get_time_diff_hours(start_time_stamp, start_time_stamp_for_avg)
fuel_moisture_time_index_end   = get_time_diff_hours(start_time_stamp,   end_time_stamp_for_avg)


# In[ ]:


fuel_moisture_time_indices_for_averagring = np.arange(fuel_moisture_time_index_start,                                                       fuel_moisture_time_index_end + 1,                                                      prediction_interval)


# In[ ]:


#fuel_moisture_time_indices_for_averagring


# In[ ]:


pred_sum = np.zeros_like(labels_pred, np.float64)


# In[ ]:


#pred_sum, pred_avg


# In[ ]:


#pred_avg.min(), pred_avg.max()


# In[ ]:


for fuel_moisture_time_index in fuel_moisture_time_indices_for_averagring:
    current_time_stamp = np.array(data_U['time'])[fuel_moisture_time_index].strftime('%Y-%m-%d_%H')
    df_features = create_df_at_timestamp (fuel_moisture_time_index, max_history_to_consider, history_interval,                                       data_U, data_T, data_RH, data_SW)
    df_features_masked = df_features.iloc[masked_ind]
    
    ### Scale Features
    print ('Data scaler type: {}'.format(scaler_type))
    scaler = define_scaler (scaler_type)
    X_gt = df_features_masked
    scaler.fit(X_gt)
    X_gt_scaled = scaler.transform(X_gt)
    
    ### Prediction and Evaluation with Trained Model
    labels_pred = predict(model, X_gt_scaled, "Data at TimeStamp")
    pred_sum += labels_pred
    
    ### Plots
    '''
    fig, ax = plt.subplots()
    cont_levels = np.linspace(0, 0.3, 11)
    cont = ax.scatter(lon_info[masked_ind], lat_info[masked_ind], c = labels_pred, marker = 's', s= 5, cmap = 'hot', vmin=0, vmax=0.3)
    cbar = fig.colorbar(cont, ticks=cont_levels, extend = 'max')
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title(current_time_stamp)
    '''

pred_avg = pred_sum/len(fuel_moisture_time_indices_for_averagring)


# In[ ]:


### Plots
fig, ax = plt.subplots()
cont_levels = np.linspace(0, 0.3, 11)
cont = ax.scatter(lon_info[masked_ind], lat_info[masked_ind], c = pred_avg -0.05, marker = 's', s= 3, cmap = 'hot', vmin=0, vmax=0.3)
cbar = fig.colorbar(cont, ticks=cont_levels, extend = 'max')
cbar.ax.tick_params(labelsize=12)
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('{} through {} every {} hrs'.format(start_time_stamp_for_avg, end_time_stamp_for_avg, prediction_interval))


fig_file_name = 'FM_{}_{}_every_{}_hrs'.format(year_range, year, prediction_interval)
plt.savefig(os.path.join(trends_data_base_loc, fig_file_name), bbox_inches='tight')


# In[ ]:


yearly_avg_file_name = 'FM_{}_{}_every_{}_hrs.pkl'.format(year_range, year, prediction_interval)
pickle.dump({'pred_avg': pred_avg}, open(os.path.join(trends_data_base_loc,yearly_avg_file_name), "wb"))


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Training and Testing of Model")

