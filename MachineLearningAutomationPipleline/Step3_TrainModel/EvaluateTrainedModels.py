#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python EvaluateTrainedModels.ipynb

# # Import Packages

# In[ ]:


import os
import sys
import os.path as path
import numpy as np
import pandas as pd
import csv
import pickle
import json
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer
import time


# In[ ]:


from TrainModel_Helper import *


# # Read the Input JSON File

# ### Input file name when using jupyter notebook

# In[ ]:


json_file_eval_models = '/p/lustre2/jha3/Wildfire/Wildfire_LDRD_SI/InputJson/Eval/json_eval_000.json'


# ### Input file name when using python script on command line

# In[ ]:


#json_file_eval_models = sys.argv[1]


# ### Load the JSON file for evaluating trained models

# In[ ]:


print('Loading the JSON file for evaluating trained models: \n {}'.format(json_file_eval_models))


# In[ ]:


with open(json_file_eval_models) as json_file_handle:
    json_content_eval_models = json.load(json_file_handle)


# In[ ]:


#json_content_eval_models


# # Simulation Directory

# In[ ]:


sim_dir = json_content_eval_models['paths']['sim_dir']


# # `json` Input Files

# In[ ]:


json_extract_base = json_content_eval_models['paths']['json_extract_base']
json_prep_base = json_content_eval_models['paths']['json_prep_base']
json_train_base = json_content_eval_models['paths']['json_train_base']


# In[ ]:


json_extract_base = os.path.join(sim_dir, json_extract_base)
json_prep_base = os.path.join(sim_dir, json_prep_base)
json_train_base = os.path.join(sim_dir, json_train_base)


# # Collect Metrics of Desired Trained Models

# In[ ]:


collection_options = json_content_eval_models['collection_options']


# In[ ]:


json_extract_counts = collection_options['json_extract_counts']
json_prep_train_maps = collection_options['json_prep_train_maps']
FM_label_type = collection_options['FM_label_type']
metric_names = collection_options['metric_names']
metric_on_sets = collection_options['metric_on_sets']


# ## Create label and train pair

# In[ ]:


label_train_pair, col_names = create_label_train_pair (json_prep_train_maps)


# In[ ]:


#json_extract_counts
#label_train_pair
#col_names
#metric_names
#metric_on_sets


# ## Create data definition

# In[ ]:


data_defn = create_data_definition (json_extract_base, json_extract_counts)


# In[ ]:


#data_defn


# ## Collect evaluation metrics

# In[ ]:


metric_name = metric_names[0]
metric_on_set = metric_on_sets[2]
eval_metric_col = '{}_{}'.format(metric_name, metric_on_set)


# In[ ]:


#eval_metric_col


# In[ ]:


df_metrics = gather_metrics_for_all_label_train_pairs (label_train_pair, col_names,                                                   json_train_base, json_extract_counts,                                                   FM_label_type, eval_metric_col)


# In[ ]:


#df_metrics


# In[ ]:


'''
trained_models_metrics, data_defn = create_trained_models_metrics (\
                                      json_prep_base, json_prep_counts, \
                                      json_train_base, json_train_counts, \
                                      json_extract_base, json_extract_counts)
'''


# In[ ]:


'''
df_train_combined = pd.DataFrame()
df_test_combined = pd.DataFrame()
for metric_name in metric_names:
    df_train, df_test = plot_trained_models_metrics (FM_label_type, json_extract_counts, \
                                                     trained_models_metrics, metric_name)
    for col in df_train.columns:
        df_train = df_train.rename(columns = {col: col + '-' + metric_name})
        df_train_combined = pd.concat([df_train_combined, df_train], axis=1)  
        
        df_test = df_test.rename(columns = {col: col + '-' + metric_name})
        df_test_combined = pd.concat([df_test_combined, df_test], axis=1)
'''


# In[ ]:


#df_test


# In[ ]:


#data_defn


# In[ ]:


#df_train_combined


# In[ ]:


#df_test_combined


# ## Effect of Max History

# ## Effect of History Interval

# ## Effect of Temporal Data Size

# ## Effect of Spatial Data Size

# ## Scatter Plots

# In[ ]:


'''
json_extract_counts = [15, 16, 17] #[2, 3, 4, 5, 6, 7, 8]
json_prep_counts = [1]
json_train_counts = [1, 3]
'''


# In[ ]:


'''
max_data_size_scatter = 800
fig_size_x = 8
fig_size_y = 8
font_size  = 10
x_lim      = [0, 0.7]
'''


# In[ ]:


#data_identifier = "Train"


# In[ ]:


'''
label_count = 1 # Regression
json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
#print(json_prep)
with open(json_prep) as json_file_handle:
    json_content_prep_data = json.load(json_file_handle)
label_count = json_content_prep_data['label_defn']['label_count']
FM_label_type = json_content_prep_data['FM_labels']['label_type']
#print('label_count: {}, FM_label_type: {}\n'.format(label_count, FM_label_type))


fig, ax = plt.subplots(len(json_extract_counts), len(json_train_counts), figsize=(12, 8))

for train_count_ind, train_count in enumerate(json_train_counts):
    json_train   = '%s_%03d.json'%(json_train_base, train_count)
    #print(json_train)
    with open(json_train) as json_file_handle:
        json_content_train_model = json.load(json_file_handle)
    model_count = json_content_train_model['models']['model_count']
    model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
    #print('Model Count: {}, Model Name: {}\n'.format(model_count, model_name))

    for data_count_ind, data_count in enumerate(json_extract_counts):
        json_extract = '%s_%03d.json'%(json_extract_base, data_count)
        #print(json_extract)
        with open(json_extract) as json_file_handle:
            json_content_extract_data = json.load(json_file_handle)
        data_set_count = json_content_extract_data['data_set_defn']['data_set_count']
        #print('Data Set Count: {}'.format(data_set_count))

        # Names of trained model and related files
        trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
        trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count, \
                                                    label_count, FM_label_type, \
                                                    model_count, model_name)

        trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
        trained_model_file_name = '{}_model.pkl'.format(trained_model_name)
        
        train_data_features_file_name  = '{}_features_train.pkl'.format(trained_model_name)
        train_data_labels_file_name    = '{}_labels_train.pkl'.format(trained_model_name)

        test_data_features_file_name   = '{}_features_test.pkl'.format(trained_model_name)
        test_data_labels_file_name     = '{}_labels_test.pkl'.format(trained_model_name)
        
        print('trained_model_file_name: {}'.format(trained_model_file_name))
        #print('train_data_features_file_name: {}'.format(train_data_features_file_name))
        #print('train_data_labels_file_name: {}'.format(train_data_labels_file_name))
        #print('test_data_features_file_name: {}'.format(test_data_features_file_name))
        #print('test_data_labels_file_name: {}'.format(test_data_labels_file_name))
        
        trained_model_file = os.path.join(trained_model_loc, trained_model_file_name)
        model = pickle.load(open(trained_model_file, 'rb'))
        #print ('\nLoaded the ML model file at: {}\n'.format(trained_model_file))
        #print ('The model loaded is: {} \n'.format(model))
        
        if (data_identifier == "Train"):
            #print('Loading the saved features and labels used in training')
            features_train = pickle.load(open(os.path.join(\
                                    trained_model_loc, train_data_features_file_name), 'rb'))
            labels_train   =  pickle.load(open(os.path.join(\
                                    trained_model_loc, train_data_labels_file_name), 'rb'))
            labels_pred_train = predict(model, features_train, "Train Data")
            accuracy_train = get_accuracy_score(model, FM_label_type, \
                                   features_train, labels_train, labels_pred_train,\
                                   "Train Data")
            labels_gt = labels_train
            labels_pred = labels_pred_train
            accuracy = accuracy_train
        else:
            #print('Loading the saved features and labels meant for testing')
            features_test = pickle.load(open(os.path.join(\
                            trained_model_loc, test_data_features_file_name), 'rb'))
            labels_test   =  pickle.load(open(os.path.join(\
                                    trained_model_loc, test_data_labels_file_name), 'rb'))
            labels_pred_test = predict(model, features_test, "Test Data")
            accuracy_test = get_accuracy_score(model, FM_label_type, \
                                   features_test, labels_test, labels_pred_test,\
                                   "Test Data")
            
            labels_gt = labels_test
            labels_pred = labels_pred_test
            accuracy = accuracy_test
        
        labels_gt_range = [labels_gt.min(), labels_gt.max()]
        data_indices = range(len(labels_gt))
        if (max_data_size_scatter < 1):
            data_ind_subset = data_indices
        else:
            data_ind_subset = random.sample(data_indices, k = max_data_size_scatter)
            
        ax[data_count_ind, train_count_ind].scatter(labels_gt[data_ind_subset], labels_pred[data_ind_subset])
        ax[data_count_ind, train_count_ind].plot(labels_gt_range, labels_gt_range, '--r')
        ax[data_count_ind, train_count_ind].set_xlabel('Ground Truth', fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_ylabel('Prediction', fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_title('Model: {}, Accuracy: {:.3f}'.format(model_name, accuracy), fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_xlim(x_lim)
        ax[data_count_ind, train_count_ind].set_ylim(x_lim)
        #ax[data_count_ind, train_count_ind].set_yticks(fontsize = font_size, rotation = 0)
        #ax[data_count_ind, train_count_ind].set_xticks(fontsize = font_size, rotation = 0)
        #print('\n')

#print('\n')
'''


# In[ ]:


#data_identifier = "Test"


# In[ ]:


'''
label_count = 1 # Regression
json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
#print(json_prep)
with open(json_prep) as json_file_handle:
    json_content_prep_data = json.load(json_file_handle)
label_count = json_content_prep_data['label_defn']['label_count']
FM_label_type = json_content_prep_data['FM_labels']['label_type']
#print('label_count: {}, FM_label_type: {}\n'.format(label_count, FM_label_type))


fig, ax = plt.subplots(len(json_extract_counts), len(json_train_counts), figsize=(12, 8))

for train_count_ind, train_count in enumerate(json_train_counts):
    json_train   = '%s_%03d.json'%(json_train_base, train_count)
    #print(json_train)
    with open(json_train) as json_file_handle:
        json_content_train_model = json.load(json_file_handle)
    model_count = json_content_train_model['models']['model_count']
    model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
    #print('Model Count: {}, Model Name: {}\n'.format(model_count, model_name))

    for data_count_ind, data_count in enumerate(json_extract_counts):
        json_extract = '%s_%03d.json'%(json_extract_base, data_count)
        #print(json_extract)
        with open(json_extract) as json_file_handle:
            json_content_extract_data = json.load(json_file_handle)
        data_set_count = json_content_extract_data['data_set_defn']['data_set_count']
        #print('Data Set Count: {}'.format(data_set_count))

        # Names of trained model and related files
        trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
        trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count, \
                                                    label_count, FM_label_type, \
                                                    model_count, model_name)

        trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
        trained_model_file_name = '{}_model.pkl'.format(trained_model_name)
        
        train_data_features_file_name  = '{}_features_train.pkl'.format(trained_model_name)
        train_data_labels_file_name    = '{}_labels_train.pkl'.format(trained_model_name)

        test_data_features_file_name   = '{}_features_test.pkl'.format(trained_model_name)
        test_data_labels_file_name     = '{}_labels_test.pkl'.format(trained_model_name)
        
        #print('trained_model_file_name: {}'.format(trained_model_file_name))
        #print('train_data_features_file_name: {}'.format(train_data_features_file_name))
        #print('train_data_labels_file_name: {}'.format(train_data_labels_file_name))
        #print('test_data_features_file_name: {}'.format(test_data_features_file_name))
        #print('test_data_labels_file_name: {}'.format(test_data_labels_file_name))
        
        trained_model_file = os.path.join(trained_model_loc, trained_model_file_name)
        model = pickle.load(open(trained_model_file, 'rb'))
        #print ('\nLoaded the ML model file at: {}\n'.format(trained_model_file))
        #print ('The model loaded is: {} \n'.format(model))
        
        if (data_identifier == "Train"):
            #print('Loading the saved features and labels used in training')
            features_train = pickle.load(open(os.path.join(\
                                    trained_model_loc, train_data_features_file_name), 'rb'))
            labels_train   =  pickle.load(open(os.path.join(\
                                    trained_model_loc, train_data_labels_file_name), 'rb'))
            labels_pred_train = predict(model, features_train, "Train Data")
            accuracy_train = get_accuracy_score(model, FM_label_type, \
                                   features_train, labels_train, labels_pred_train,\
                                   "Train Data")
            labels_gt = labels_train
            labels_pred = labels_pred_train
            accuracy = accuracy_train
        else:
            #print('Loading the saved features and labels meant for testing')
            features_test = pickle.load(open(os.path.join(\
                            trained_model_loc, test_data_features_file_name), 'rb'))
            labels_test   =  pickle.load(open(os.path.join(\
                                    trained_model_loc, test_data_labels_file_name), 'rb'))
            labels_pred_test = predict(model, features_test, "Test Data")
            accuracy_test = get_accuracy_score(model, FM_label_type, \
                                   features_test, labels_test, labels_pred_test,\
                                   "Test Data")
            
            labels_gt = labels_test
            labels_pred = labels_pred_test
            accuracy = accuracy_test
        
        labels_gt_range = [labels_gt.min(), labels_gt.max()]
        data_indices = range(len(labels_gt))
        if (max_data_size_scatter < 1):
            data_ind_subset = data_indices
        else:
            data_ind_subset = random.sample(data_indices, k = max_data_size_scatter)
            
        ax[data_count_ind, train_count_ind].scatter(labels_gt[data_ind_subset], labels_pred[data_ind_subset])
        ax[data_count_ind, train_count_ind].plot(labels_gt_range, labels_gt_range, '--r')
        ax[data_count_ind, train_count_ind].set_xlabel('Ground Truth', fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_ylabel('Prediction', fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_title('Model: {}, Accuracy: {:.3f}'.format(model_name, accuracy), fontsize = font_size)
        ax[data_count_ind, train_count_ind].set_xlim(x_lim)
        ax[data_count_ind, train_count_ind].set_ylim(x_lim)
        #ax[data_count_ind, train_count_ind].set_yticks(fontsize = font_size, rotation = 0)
        #ax[data_count_ind, train_count_ind].set_xticks(fontsize = font_size, rotation = 0)
        #print('\n')

#print('\n')
'''

