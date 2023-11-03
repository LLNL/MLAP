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


from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
json_file_prep_data    = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/02_TrainTest_Data_Prepared/InputJsonFiles/json_prep_data_label_002.json'
json_file_train_model  = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/03_Trained_Models/InputJsonFiles/json_train_model_000.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_extract_data = sys.argv[1]
#json_file_prep_data = sys.argv[2]
#json_file_train_model = sys.argv[3]


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

# ### Models 

# In[ ]:


model_count = json_content_train_model['models']['model_count']
model_considered = json_content_train_model['models']['model_considered'] # ['RF', SVM', 'MLP']


# ### Fire of Interest

# In[ ]:


fire_name = json_content_train_model['fire_data']['fire_name']


# ## Paths and File Names

# #### Global

# In[ ]:


prepared_data_base_loc = json_content_prep_data[ 'paths']['prepared_data_base_loc']
trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
analysis_data_base_loc = json_content_train_model['paths']['analysis_data_base_loc']


# #### DataSet and Label Specific (Train and Test Data Prepared)

# In[ ]:


prepared_data_set_name = 'dataset_%03d_label_%03d_%s'%(data_set_count,                                                        label_count, FM_label_type)

prepared_data_loc = os.path.join(prepared_data_base_loc, prepared_data_set_name)
os.system('mkdir -p %s'%prepared_data_loc)

prepared_data_file_name = '{}.pkl'.format(prepared_data_set_name)


# #### DataSet, Label, and Model Specific (Trained Model)

# In[ ]:


trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count,                                                         label_count, FM_label_type,                                                         model_count, model_considered)

trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
os.system('mkdir -p %s'%trained_model_loc)

trained_model_file_name = '{}.pkl'.format(trained_model_name)


# #### DataSet Specific (Analysis Data)

# In[ ]:


analysis_set_name = 'analysis_%03d'%(data_set_count)

analysis_loc = os.path.join(analysis_data_base_loc, analysis_set_name, label_type, model_considered)
os.system('mkdir -p %s'%analysis_loc)


# In[ ]:


#analysis_loc


# # Generate seed for the random number generator

# In[ ]:


seed = generate_seed()
random_state = init_random_generator(seed)


# # Load The Prepared Data Saved in Pickle File

# In[ ]:


with open(os.path.join(prepared_data_loc, prepared_data_file_name), 'rb') as file_handle:
    prepared_data = pickle.load(file_handle)
print('Read prepared data from "{}" at "{}"'.format(prepared_data_file_name, prepared_data_loc))


# # Get Features and Labels to Use

# In[ ]:


prepared_data['all']
#prepared_data['fire']['Woosley']['identity'].head(5)


# ## Get the Headers for Features and Labels

# In[ ]:


features_to_use = prepared_data['tt']['features'].keys()
if (label_type == 'Regr'):
    label_to_use = 'FM_{}hr'.format(FM_hr)
else:
    label_to_use = 'FM_{}hr_{}'.format(FM_hr, label_type)


# In[ ]:


label_to_use


# ## Extract Features and Labels

# In[ ]:


X_tt     = prepared_data['tt']['features'][features_to_use]
y_tt     = prepared_data['tt']['labels'][label_to_use]
idy_tt   = prepared_data['tt']['identity']

X_fire   = prepared_data['fire'][fire_name]['features'][features_to_use]
y_fire   = prepared_data['fire'][fire_name]['labels'][label_to_use]
idy_fire = prepared_data['fire'][fire_name]['identity']

#all = prepared_data['tt']['all']


# In[ ]:


y_tt


# ## Scale Features

# #### Features for Train/Test

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X_tt)
X_tt_scaled = scaler.transform(X_tt)


# In[ ]:


#X_tt_scaled


# #### Features for Fire Data

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X_fire)
X_fire_scaled = scaler.transform(X_fire)


# In[ ]:


#X_fire_scaled.max()


# ## Train /Test Split

# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(X_tt_scaled, y_tt, test_size=0.2)


# # Model

# ## Define the Model

# In[ ]:


print ('label_type: {}'.format(label_type))
print ('Model: {}'.format(model_considered))

if (label_type == 'Regr'):
    match model_considered:
        case 'SVM':
            model = SVR(kernel='rbf')
        case 'RF':
            model = RandomForestRegressor(max_depth=2, random_state=0)
        case 'MLP':
            model = MLPRegressor(random_state=1, max_iter=500)
else: # 'bin' or 'MC'
    match model_considered:
        case 'SVM':
            model = SVC(kernel="linear", class_weight = "balanced")
        case 'RF':
            model = RandomForestClassifier(max_depth=2, random_state=0)
        case 'MLP':
            model = MLPClassifier(solver = 'sgd', activation = 'relu', max_iter= 10000, 
                    random_state = 0, hidden_layer_sizes = [15,15,15])


# In[ ]:


model


# ## Train the Model

# In[ ]:


t0 = time.time()
model.fit(features_train, labels_train)
print ("Training Time:", round(time.time()-t0, 3), "s")


# ## Save the Model

# In[ ]:


pickle.dump(model, open(trained_model_file, 'wb'))


# ## Load the Model

# In[ ]:


trained_model_file


# In[ ]:


model = pickle.load(open(trained_model_file, 'rb'))


# In[ ]:


model


# # Prediction with Trained Model

# ## Prediction on Train Data

# In[ ]:


t1 = time.time()
labels_pred = model.predict(features_train)
print ("Prediction Time:", round(time.time()-t1, 3), "s")


# In[ ]:


if (label_type == 'bin' or 'label_type' == 'MC'):
    accuracy = accuracy_score(labels_pred, labels_train)
else:
    accuracy = model.score(features_train, labels_train)
conf_mat = None


# In[ ]:


if (label_type == 'bin'):
    conf_mat = confusion_matrix(labels_train, labels_pred, labels = [0, 1])
    print('Classification Report: \n')
    print(classification_report(labels_train, labels_pred, labels=[0, 1]))
    average_precision = average_precision_score(labels_train, labels_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
elif (label_type == 'MC'):
    conf_mat = confusion_matrix(labels_train, labels_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print('Confusion Mat is not suitable for label_type: {}'.format(label_type))

print('Accuracy Score: {}'.format(accuracy))
print('Confusion Matrix: \n{}'.format(conf_mat))


# ## Prediction on Test Data

# In[ ]:


t1 = time.time()
labels_pred = model.predict(features_test)
print ("Prediction Time:", round(time.time()-t1, 3), "s")


# In[ ]:


if (label_type == 'bin' or 'label_type' == 'MC'):
    accuracy = accuracy_score(labels_pred, labels_test)
else:
    accuracy = model.score(features_test, labels_test)
conf_mat = None


# In[ ]:


if (label_type == 'bin'):
    conf_mat = confusion_matrix(labels_test, labels_pred, labels = [0, 1])
    print('Classification Report: \n')
    print(classification_report(labels_test, labels_pred, labels=[0, 1]))
    average_precision = average_precision_score(labels_test, labels_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
elif (label_type == 'MC'):
    conf_mat = confusion_matrix(labels_test, labels_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print('Confusion Mat is not suitable for label_type: {}'.format(label_type))

print('Accuracy Score: {}'.format(accuracy))
print('Confusion Matrix: \n{}'.format(conf_mat))


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

