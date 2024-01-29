#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python Analyze.ipynb

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


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Extract/json_extract_data_015.json'
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Prep/json_prep_data_label_001.json'
json_file_train_model  = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Train/json_train_model_003.json'
json_file_analyze      = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Analyze/json_analyze_001.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]
#json_file_prep_data = sys.argv[2]
#json_file_train_model = sys.argv[3]
#json_file_analyze  = sys.argv[4]


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


# ### Load the JSON file for analysis

# In[ ]:


print('Loading the JSON file for analysis: \n {}'.format(json_file_analyze))


# In[ ]:


with open(json_file_analyze) as json_file_handle:
    json_content_analyze = json.load(json_file_handle)


# In[ ]:


#json_content_analyze


# # Variables to be Used for Analysis

# ## DataSet Defintion

# In[ ]:


# The current data set params
data_set_count = json_content_extract_data['data_set_defn']['data_set_count']
max_history_to_consider = json_content_extract_data['data_set_defn']['max_history_to_consider']
history_interval = json_content_extract_data['data_set_defn']['history_interval']


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


FM_hr = json_content_prep_data['qoi_to_plot']['FM_hr']


# ## Define ML Model and Params etc.

# ### Model Definition 

# In[ ]:


model_count = json_content_train_model['models']['model_count']
scaler_type = json_content_train_model['models']['scaler_type']
model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
model_params = json_content_train_model['models']['params']


# ## Define Analysis Inputs

# In[ ]:


analysis_count = json_content_analyze['analysis_count']


# ### Paths

# In[ ]:


analysis_data_paths = json_content_analyze['paths']
analysis_data_desired = json_content_analyze['analysis_data_desired']


# ### Data Types of Interest

# In[ ]:


analysis_data_defined = [analysis_data_elem                          for analysis_data_elem in analysis_data_desired                          if analysis_data_elem in json_content_analyze]


# In[ ]:


print ('Analysis desired to be performed on the following data sets:\n {}'.format(                                                            analysis_data_desired))

print ('Time and Region Info available for these data sets out of those desired:\n {}'                                                    .format(analysis_data_defined))


# ### Analysis Plots Prefernces

# In[ ]:


analysis = json_content_train_model['evaluation']
fig_size_x = analysis['fig_size_x']
fig_size_y = analysis['fig_size_y']
font_size  = analysis['font_size']

if (FM_label_type == 'Regression'):
    max_data_size_scatter = analysis['max_data_size_scatter']
    x_lim      = analysis['x_lim']
else:
    normalize_cm = analysis['normalize_cm']


# # Paths and File Names

# #### Global

# In[ ]:


trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
analysis_data_base_loc = json_content_analyze['paths']['analysis_data_base_loc']


# In[ ]:


raw_data_paths = json_content_analyze['paths']['raw_data']


# #### DataSet, Label, and Model Specific (Trained Model)

# In[ ]:


trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count,                                                         label_count, FM_label_type,                                                         model_count, model_name)

trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)

trained_model_file_name = '{}.pkl'.format(trained_model_name)


# #### DataSet, Label, Model, and TimeStamp Specific (Analysis Data)

# In[ ]:


analysis_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s_analysis_%03d'%(                                                        data_set_count,                                                         label_count, FM_label_type,                                                         model_count, model_name,                                                        analysis_count)

analysis_loc = os.path.join(analysis_data_base_loc, analysis_name)
os.system('mkdir -p %s'%analysis_loc)


# In[ ]:


time_region_info = get_time_region_info (analysis_data_defined, json_content_analyze)


# In[ ]:


#time_region_info


# In[ ]:


analysis_data_locations_all_types = dict()
for analysis_data_type in time_region_info.keys():
    #print('\nTime Info for {}:'.format(analysis_data_type))
    
    analysis_data_locations = []
    for count_ref_time, item_ref_time in enumerate(time_region_info[analysis_data_type]):
        #print ('... Reference Time {}: {}'.format(count_ref_time + 1, item_ref_time['RefTime']))
        analysis_data_loc = os.path.join(analysis_loc,                                          analysis_data_type,
                                         item_ref_time['RefTime'])
        analysis_data_locations.append(analysis_data_loc)
        os.system('mkdir -p %s'%analysis_data_loc)
    
    analysis_data_locations_all_types[analysis_data_type] = analysis_data_locations


# In[ ]:


#analysis_data_locations_all_types


# #### Features for Fire Data

# In[ ]:


'''
scaler = MinMaxScaler()
scaler.fit(X_fire)
X_fire_scaled = scaler.transform(X_fire)
'''


# # ML Model

# In[ ]:


print ('FM label type: {}'.format(FM_label_type))
print ('ML model considered: {}'.format(model_name))


# ## Load the Model

# In[ ]:


trained_model_file = os.path.join(trained_model_loc, trained_model_file_name)
model = pickle.load(open(trained_model_file, 'rb'))
print ('\nLoaded the ML model file at: {}\n'.format(trained_model_file))


# In[ ]:


print ('The model loaded is: {} \n'.format(model))
print ('Model params: \n {}'.format(model.get_params()))


# In[ ]:


time_region_info = get_time_region_info (analysis_data_defined, json_content_analyze)


# In[ ]:


time_region_info = get_history_time_stamps_all_data_types (                            time_region_info, max_history_to_consider, history_interval)


# In[ ]:


time_region_info


# ## HRRR Data

# In[ ]:


analysis_data_type = 'HRRR'


# In[ ]:


raw_data_path = raw_data_paths[analysis_data_type]
raw_data_path


# In[ ]:


qoi = 'ugrd10m'


# In[ ]:


desired_time = '2020-09-04_00'


# In[ ]:


year, month, day, hour = split_timestamp (desired_time)


# In[ ]:


#year, month, day, hour


# In[ ]:





# # Prediction with Trained Model

# ## Prediction on Fire Data

# In[ ]:


t1 = time.time()
labels_pred = model.predict(X_fire_scaled)
print ("Prediction Time:", round(time.time()-t1, 3), "s")


# In[ ]:


if (label_type == 'bin' or 'label_type' == 'MC'):
    accuracy = accuracy_score(labels_pred, y_fire)
else:
    accuracy = model.score(X_fire_scaled, y_fire)
conf_mat = None


# In[ ]:


if (label_type == 'bin'):
    conf_mat = confusion_matrix(y_fire, labels_pred, labels = [0, 1])
    print('Classification Report: \n')
    print(classification_report(y_fire, labels_pred, labels=[0, 1]))
    average_precision = average_precision_score(y_fire, labels_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
elif (label_type == 'MC'):
    conf_mat = confusion_matrix(y_fire, labels_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print('Confusion Mat is not suitable for label_type: {}'.format(label_type))

print('Accuracy Score: {}'.format(accuracy))
print('Confusion Matrix: \n{}'.format(conf_mat))


# ## Plot Ground Truth and Prediction of Fire Data

# In[ ]:


ny, nx = (480, 396)
j_indices = idy_fire['j_ind']
i_indices = idy_fire['i_ind']
ground_truth = y_fire


# In[ ]:


if label_type == 'Regr':
    ground_truth_mat = np.full((ny,nx), np.nan)
    pred_mat = np.full_like(ground_truth_mat, np.nan )
    error_mat = np.full_like(ground_truth_mat, np.nan)
else:
    ground_truth_mat = np.ones((ny,nx), int)*(-1)
    pred_mat = np.ones_like(ground_truth_mat, int)*(-1)
    error_mat = np.ones_like(ground_truth_mat, int)*(-1)


# In[ ]:


for j_loc, i_loc, gt_val, pred_val in zip (j_indices, i_indices, ground_truth, labels_pred):
    ground_truth_mat[j_loc][i_loc] = gt_val
    pred_mat[        j_loc][i_loc] = pred_val
    if (label_type == 'bin' or 'label_type' == 'MC'):
        error_mat[       j_loc][i_loc] = (gt_val == pred_val)
    else:
        error_mat[       j_loc][i_loc] = 100.0*(pred_val/gt_val - 1.0)


# In[ ]:


#error_mat


# In[ ]:


cmap_name = input_json_data['plot_options']['cmap_name']
cont_levels = input_json_data['plot_options']['cont_levels']
cont_levels = np.linspace(0, 0.28, 21)
cont_levels_err = np.linspace(-75.0, 75.0, 21)
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

x_ind, y_ind = np.meshgrid(range(nx), range(ny))

cont = ax[0].contourf(x_ind, y_ind, ground_truth_mat, levels = cont_levels, cmap=cmap_name, extend='both')
plt.colorbar(cont)
ax[0].set_title('Ground Truth')
ax[0].set_xticks([])
ax[0].set_yticks([])

cont = ax[1].contourf(x_ind, y_ind, pred_mat, levels = cont_levels, cmap=cmap_name, extend='both')
plt.colorbar(cont)
ax[1].set_title('Prediction')
ax[1].set_xticks([])
ax[1].set_yticks([])

cont = ax[2].contourf(x_ind, y_ind, error_mat, levels = cont_levels_err, cmap=cmap_name, extend='both')
plt.colorbar(cont)
ax[2].set_title('Correct Match')
ax[2].set_xticks([])
ax[2].set_yticks([])

filename = trained_model_name.split('.')[0] + '_{}_Fire.png'.format(fire_name)
filedir = analysis_loc
os.system('mkdir -p %s'%filedir)

plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Training and Testing of Model")

