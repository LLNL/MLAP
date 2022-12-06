#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[ ]:

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

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

# # Import helper functions
# In[ ]:
from helper_training import *


# # Variables to be used for preparing training data
# In[ ]:
extracted_data_loc = '/Users/jha3/LLNL/LLNL_Research/03_Wildfire/Wildfire_LDRD_SI/Wildfire_SJSU/data_extracted'
labeled_data_loc = '/Users/jha3/LLNL/LLNL_Research/03_Wildfire/Wildfire_LDRD_SI/Wildfire_SJSU/data_labeled'
# threshold_FM_fire_risk
threshold_FM_fire_risk = 0.1 # FM below which there is fire risk
# FM levels
FM_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# FM_hr (1 or 10)
FM_hr = 10
label_type = 'MC'# 'binary' # 'MC'

# # Load the extracted WRF data
# In[ ]:
extracted_data_files = os.listdir(extracted_data_loc)
extracted_data_files.sort()
extracted_data_file_name = extracted_data_files[3]
data_file_identity = extracted_data_file_name[10:-4]
loaded_data = pickle.load(open(os.path.join(extracted_data_loc, extracted_data_file_name), 'rb'))
df_for_all_files = loaded_data['df_for_all_files']
print('Num of data points in the extracted file: {}'.format(len(df_for_all_files)))

# # Prepare the training data

# In[]:
# # # Compure U_mag
df_for_all_files = compute_u_mag(df_for_all_files)

# In[]:
# # # 
df_labeled = re_label_binary(df_for_all_files, threshold_FM_fire_risk, labeled_data_loc, data_file_identity)

# In[]:
df_labeled = re_label_multi_class(df_for_all_files, FM_levels, labeled_data_loc, data_file_identity)

# # Define the features and labels to use in the training
# In[ ]:
features_to_use, label_to_use = get_features_labels_headers(df_for_all_files, FM_hr, label_type)

# # SVM Stuff

# ### Scikit-Learn Imports

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report
from time import time


# ### Work with Multiple Features of Interest

# In[ ]:
X, y = get_features_labels(df_labeled, features_to_use, label_to_use)

# In[]:
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# In[]:
features_train, features_test, labels_train, labels_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#sheets_train, sheets_test, labels_train, labels_test = split_labels_sheets(labels_train, labels_test)
len(features_train), len(features_test), len(labels_train), len(labels_test)


# #### Create and Train the Model

# In[ ]:


########################## SVM #################################
# Training Time is significantly more with large values of C
if (label_type == 'binary'):
    clf = SVC(kernel="linear", class_weight = "balanced") #class_weight
elif (label_type == 'MC'):
    clf = MLPClassifier(solver = 'sgd', activation = 'relu', max_iter= 10000, 
                        random_state = 0, hidden_layer_sizes = [15,15,15])
else:
    print('label_type: {} is unrecognized'.format(label_type))
    
# Try RBF etc., gamma term, epsilon, C
# Grid search to optimize parameters
# Semantic segmentation of pixels
# Try Random Forest
# Google earth Engine for satellite images in addition to QGIS

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print ("Training Time:", round(time()-t0, 3), "s")


# #### Save the Model to a File

# In[ ]:


# save the model to disk
'svm_N1205_T080_C1 --> svm_N(number of total data)_T(percentage for training)_C(model parameter in model constructor, default 1)'
svm_model_file = 'models_trained/svm_N1205_T080_C1'
pickle.dump(clf, open(svm_model_file, 'wb'))


# #### Load the Model from a File (A template...use specific model for specific pupose)

# In[ ]:


# load the model from disk


# ## Predict on the Train Data itself using SVM

# In[ ]:
clf = pickle.load(open('models_trained/svm_N1205_T080_C1', 'rb'))
t1 = time()
labels_pred = clf.predict(features_train)
print ("Prediction Time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_pred, labels_train)

if (label_type == 'binary'):
    conf_mat = confusion_matrix(labels_train, labels_pred, labels = [0, 1])
    print('Classification Report: \n')
    print(classification_report(labels_train, labels_pred, labels=[0, 1]))
    average_precision = average_precision_score(labels_train, labels_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
elif (label_type == 'MC'):
    conf_mat = confusion_matrix(labels_train, labels_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print('label_type: {} is unrecognized'.format(label_type))

print('Accuracy Score: {}'.format(accuracy))
print('Confusion Matrix: \n{}'.format(conf_mat))


# ## Predict on the Test Data using SVM

# In[ ]:


### use the trained classifier to predict labels for the test features using SVM predict
clf = pickle.load(open('models_trained/svm_N1205_T080_C1', 'rb'))
t1 = time()
labels_pred = clf.predict(features_test)
print ("Prediction Time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_pred, labels_test)

if (label_type == 'binary'):
    conf_mat = confusion_matrix(labels_test, labels_pred, labels = [0, 1])
    print('Classification Report: \n')
    print(classification_report(labels_test, labels_pred, labels=[0, 1]))
    average_precision = average_precision_score(labels_test, labels_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
elif (label_type == 'MC'):
    conf_mat = confusion_matrix(labels_test, labels_pred, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
else:
    print('label_type: {} is unrecognized'.format(label_type))

print('Accuracy Score: {}'.format(accuracy))
print('Confusion Matrix: \n{}'.format(conf_mat))

dummy = 0

"""
precision, recall, thresholds = precision_recall_curve(labels_test, pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
"""


# ## Work with Feature Pairs

# In[ ]:

'''
features = features_to_use
for i in range(len(features)):
    for j in range(i+1, len(features)):
        #print('i: {}, j: {}'.format(i, j))
        v1 = features[i]
        v2 = features[j]
        print('---- v1: {}, v2: {} ------'.format(v1, v2))
        X, y = get_features_labels(df_combined, [v1, v2])
        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.1, random_state=42)
        sheets_train, sheets_test, labels_train, labels_test = split_labels_sheets(labels_train, labels_test)
        
        #len(features_train), len(features_test)
        clf = SVC(kernel="rbf", class_weight = "balanced") #class_weight
        
        ### fit the classifier on the training features and labels
        t0 = time()
        clf.fit(features_train, labels_train)
        print ("Training Time:", round(time()-t0, 3), "s")

        ### use the trained classifier to predict labels for the test features
        t1 = time()
        pred = clf.predict(features_test)
        #print ("Prediction Time:", round(time()-t1, 3), "s")
        #print ('pred: {}'.format(pred))

        accuracy = accuracy_score(pred, labels_test)
        #print('Accuracy Score: {}'.format(accuracy))
        
        plot_scatter_svm (clf, features_test, labels_test, v1, v2, accuracy, plot_base_dir = 'plots', scatter_dir = 'SVM')
        plot_scatter_arr (features_test, labels_test, v1, v2, accuracy, plot_base_dir = 'plots', scatter_dir = 'scatter/test_data')


# ## Plot Statistics/ Histogram

# In[ ]:


col_for_hist = features_to_use


# In[ ]:


plot_hist_dfmap_combined(sheet_to_df_map, ['Fuel01', 'Fuel02', 'Fuel03','Fuel04'],
                         col_for_hist, 'histogram/Burnable', plot_base_dir = 'plots')


# In[ ]:


plot_hist_dfmap_combined(sheet_to_df_map, ['Fuel05'],
                         col_for_hist, 'histogram/Unburnable', plot_base_dir = 'plots')


# ## Create Scatter Plots

# In[ ]:


burnable_sheets = ['Fuel01', 'Fuel02', 'Fuel03','Fuel04']
unburnable_sheets = ['Fuel05']
v_list = features_to_use
for i in range(len(v_list)):
    for j in range(i+1, len(v_list)):
        #print('i: {}, j: {}'.format(i, j))
        plot_scatter_df(sheet_to_df_map, burnable_sheets, unburnable_sheets, v_list[i], v_list[j], 'plots', 'scatter/all_data')

'''