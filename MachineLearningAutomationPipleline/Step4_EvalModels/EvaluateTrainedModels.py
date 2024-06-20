#!/usr/bin/env python
# coding: utf-8

# ## Convert this notebook to executable python script using:

# - jupyter nbconvert --to python EvaluateTrainedModels.ipynb

# # Import Modules

# ## Standard Packages

# In[ ]:


import os
import sys
import os.path as path
import numpy as np
import pandas as pd
import csv
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits import mplot3d
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer
import time


# ## User-Defined Functions

# In[ ]:


current_running_file_dir = sys.path[0]
current_running_file_par = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, os.path.join(current_running_file_par, 'Step3_TrainModel'))


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


# # Evaluation Identifier

# In[ ]:


eval_count = json_content_eval_models['evaluation']['count']
identifier_text = json_content_eval_models['evaluation']['identifier_text']


# # Simulation Directory

# In[ ]:


sim_dir = json_content_eval_models['paths']['sim_dir']


# # Paths and File Names

# In[ ]:


eval_model_base_loc = json_content_eval_models['paths']['eval_model_base_loc']
eval_model_name = 'eval_%03d_%s'%(eval_count, identifier_text)
eval_model_loc = os.path.join(eval_model_base_loc, eval_model_name)
os.system('mkdir -p %s'%eval_model_loc)


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


# In[ ]:


data_defn.to_csv(os.path.join(eval_model_loc, eval_model_name+'_data_defn.csv'),                                    index=False, float_format = '%.4f')


# ## Collect evaluation metrics and plot them

# In[ ]:


for metric_name in metric_names:
    for metric_on_set in metric_on_sets:
        df_metrics = gather_metrics_for_all_label_train_pairs (                                                  label_train_pair, col_names,                                                   json_train_base, json_extract_counts,                                                   FM_label_type, metric_name, metric_on_set,                                                   eval_model_loc, eval_model_name)
        #print (df_metrics)       

        create_bar_plots (df_metrics, FM_label_type, metric_name, metric_on_set,                                            eval_model_loc, eval_model_name)
        create_heatmap (df_metrics, FM_label_type, metric_name, metric_on_set,                                            eval_model_loc, eval_model_name)

