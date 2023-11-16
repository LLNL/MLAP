#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# In[ ]:


#jupyter nbconvert --to python TrainModel.ipynb


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


from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report


# ## User-Defined Functions

# In[ ]:


current_running_file_dir = sys.path[0]
current_running_file_par = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, os.path.join(current_running_file_par, 'Step1_ExtractData'))


# In[ ]:


from Extract_DFM_Data_Helper import *


# # Global Start Time and Memory

# In[ ]:


global_start_time = timer()
process = psutil.Process(os.getpid())
global_initial_memory = process.memory_info().rss


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_extract_data = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/01_WRF_Nelson_Data_Extracted/InputJsonFiles/json_extract_data_005.json'
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/02_TrainTest_Data_Prepared/InputJsonFiles/json_prep_data_label_000.json'
json_file_train_model  = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/03_Trained_Models/InputJsonFiles/json_train_model_000.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]
#json_file_prep_data = sys.argv[2]
#json_file_train_model = sys.argv[3]


# ### Load the JSON file for extracting data

# In[ ]:


print('\nLoading the JSON file for extracting data: \n {}'.format(json_file_extract_data))


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


# # Variables to be Used for Training Model

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

if (FM_label_type == 'Binary'):
    FM_binary_threshold = FM_labels['FM_binary_threshold']
if (FM_label_type == 'MultiClass'):
    FM_MC_levels = FM_labels['FM_MC_levels']


# In[ ]:


FM_hr = json_content_prep_data['qoi_to_plot']['FM_hr']


# ## Define ML Model and Params etc.

# ### Model Defintion

# In[ ]:


model_count = json_content_train_model['models']['model_count']
scaler_type = json_content_train_model['models']['scaler_type']
model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
model_params = json_content_train_model['models']['params']


# # Paths and File Names

# #### Global

# In[ ]:


prepared_data_base_loc = json_content_prep_data[ 'paths']['prepared_data_base_loc']
trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']


# #### DataSet and Label Specific (Train and Test Data Prepared)

# In[ ]:


prepared_data_set_name = 'dataset_%03d_label_%03d_%s'%(data_set_count,                                                        label_count, FM_label_type)

prepared_data_loc = os.path.join(prepared_data_base_loc, prepared_data_set_name)
#os.system('mkdir -p %s'%prepared_data_loc)

prepared_data_file_name = '{}.pkl'.format(prepared_data_set_name)


# #### DataSet, Label, and Model Specific (Trained Model)

# In[ ]:


trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count,                                                         label_count, FM_label_type,                                                         model_count, model_name)

trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
os.system('mkdir -p %s'%trained_model_loc)

trained_model_file_name = '{}.pkl'.format(trained_model_name)


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # Load The Prepared Data Saved in Pickle File

# In[ ]:


with open(os.path.join(prepared_data_loc, prepared_data_file_name), 'rb') as file_handle:
    prepared_data = pickle.load(file_handle)
print('\nRead prepared data from "{}" at "{}"\n'.format(prepared_data_file_name, prepared_data_loc))


# # Get Features and Labels to Use

# ## Features

# In[ ]:


features_to_use = prepared_data['features'].keys()


# In[ ]:


#features_to_use


# ## Labels

# In[ ]:


#prepared_data['labels'].keys()


# In[ ]:


if (FM_label_type == 'Regression'):
    labels_to_use = ['FM_{}hr'.format(FM_hr)]
elif (FM_label_type == 'Binary'):
    labels_to_use = ['FM_{}hr_bin'.format(FM_hr)]
elif (FM_label_type == 'MultiClass'):
    labels_to_use = ['FM_{}hr_MC'.format(FM_hr)]
else:
    raise ValueError('Invalid "label_type": {} in "FM_labels".                     \nValid types are: "Regression", "MultiClass", and "Binary"'.format(                                                                            FM_label_type))


# In[ ]:


#labels_to_use


# ## Extract Features and Labels from Prepared Train/Test Data

# In[ ]:


X_tt     = prepared_data['features'][features_to_use]
y_tt     = prepared_data['labels'][labels_to_use]
idy_tt   = prepared_data['identity']
#all_tt = prepared_data['all']


# In[ ]:


#X_tt, y_tt, all_tt


# ## Scale Features

# In[ ]:


print ('Data scaler type: {}'.format(scaler_type))


# In[ ]:


if (scaler_type == 'Standard'):
    scaler = StandardScaler()
elif (scaler_type == 'MinMax'):
    scaler = MinMaxScaler()
elif (scaler_type == 'MaxAbs'):
    scaler = MaxAbsScaler()
elif (scaler_type == 'Robust'):
    scaler = RobustScaler()
else:
    raise ValueError('Invalid "scaler_type": "{}" in "models".                     \nValid types are:                     "Standard", "MinMax", "MaxAbs", and "Robust"'.format(                                                             scaler_type))


# In[ ]:


scaler.fit(X_tt)
X_tt_scaled = scaler.transform(X_tt)


# In[ ]:


#X_tt_scaled.shape


# #### Clarify if train/test split should be performed after or before scaling

# ## Train /Test Split

# In[ ]:


test_data_frac = json_content_train_model['models']['test_data_frac']


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(                             X_tt_scaled, y_tt.to_numpy(), test_size = test_data_frac)


# In[ ]:


#type(labels_test)


# # ML Model

# ## Define the Model

# In[ ]:


print ('FM label type: {}'.format(FM_label_type))
print ('ML model considered: {}'.format(model_name))


# In[ ]:


if (FM_label_type == 'Regression'):
    match model_name:
        case 'SVM':
            model = SVR()
        case 'RF':
            model = RandomForestRegressor()
        case 'MLP':
            model = MLPRegressor()
            
elif (FM_label_type == 'Binary' or FM_label_type == 'MultiClass'):
    match model_name:
        case 'SVM':
            model = SVC()
        case 'RF':
            model = RandomForestClassifier()
        case 'MLP':
            model = MLPClassifier()


# In[ ]:


print ('The model chosen is: {} \n'.format(model))
print ('Deafult model params: \n {}'.format(model.get_params()))


# In[ ]:


print ('Updating the model params with the dict: \n {}'.format(model_params))


# In[ ]:


model.set_params(**model_params)


# In[ ]:


print ('Updated model params: \n {}'.format(model.get_params()))


# ## Train the Model

# In[ ]:


t0 = time.time()
model.fit(features_train, labels_train.ravel())
print ("Training Time:", round(time.time()-t0, 3), "s")


# ## Save the Model

# In[ ]:


trained_model_file = os.path.join(trained_model_loc, trained_model_file_name)
pickle.dump(model, open(trained_model_file, 'wb'))
print ('\nSaved the ML model file at: {}\n'.format(trained_model_file))


# # Global End Time and Memory

# In[ ]:


global_final_memory = process.memory_info().rss
global_end_time = timer()
global_memory_consumed = global_final_memory - global_initial_memory
print('Total memory consumed: {:.3f} MB'.format(global_memory_consumed/(1024*1024)))
print('Total computing time: {:.3f} s'.format(global_end_time - global_start_time))
print('=========================================================================')
print("SUCCESS: Done Training of ML Model")

