#!/usr/bin/env python
# coding: utf-8



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

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report


# []
'''
Define scaler for data (train/test as well as analysis [SJSU, HRRR, RRM])
'''
def define_scaler (scaler_type):
    if (scaler_type == 'Standard'):
        scaler = StandardScaler()
    elif (scaler_type == 'MinMax'):
        scaler = MinMaxScaler()
    elif (scaler_type == 'MaxAbs'):
        scaler = MaxAbsScaler()
    elif (scaler_type == 'Robust'):
        scaler = RobustScaler()
    else:
        raise ValueError('Invalid "scaler_type": "{}" in "models". \
                        \nValid types are: \
                        "Standard", "MinMax", "MaxAbs", and "Robust"'.format(\
                                                                 scaler_type))
    #'========================================================================='    
    return scaler


# []
'''
Define ML Model
'''
def define_model (FM_label_type, model_name):
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
                
    #'========================================================================='    
    return model


# []
'''
Predict labels on data/features using a trained model
'''
def predict(model, features_gt, data_identifier):
    t1 = time.time()
    labels_pred = model.predict(features_gt)
    print ('Prediction Time for {} is {} s'.format(\
                        data_identifier, round(time.time()-t1, 3)))
    
    return labels_pred


# []
'''
Get Accuracy Score
'''
def get_accuracy_score (model, FM_label_type, features_gt, labels_gt, labels_pred, \
                        data_identifier):
    
    if (FM_label_type == 'Binary' or FM_label_type == 'MultiClass'):
        accuracy = accuracy_score(labels_pred, labels_gt)
    else:
        accuracy = model.score(features_gt, labels_gt)
    
    print ('Accuracy for {} is: {}'.format(data_identifier, accuracy))
    return accuracy


# []
'''
Get Confusion Matrix
'''
def get_confusion_matrix (FM_label_type, labels_gt, labels_pred, \
                          data_identifier, class_labels):
    
    conf_mat = confusion_matrix(labels_gt, labels_pred, labels = class_labels)
    print('\nConfusion Matrix for {} is: \n{}'.format(data_identifier, conf_mat))
    
    return conf_mat


# []
'''
Get Classification Report
'''
def get_classification_report (FM_label_type, labels_gt, labels_pred, \
                          data_identifier, class_labels):
    
    print('\nClassification Report for {}: \n'.format(data_identifier))
    print(classification_report(labels_gt, labels_pred, labels=class_labels))
    

# []
'''
Plot Scatter for Rgeression
'''
def plot_scatter_regression (labels_gt, labels_pred, accuracy, model_name, \
                             plot_loc, fig_name, \
                             max_data_size_scatter, fig_size_x, fig_size_y, \
                             font_size, x_lim):
    
    labels_gt_range = [labels_gt.min(), labels_gt.max()]
    data_indices = range(len(labels_gt))
    if (max_data_size_scatter < 1):
        data_ind_subset = data_indices
    else:
        data_ind_subset = random.sample(data_indices, k = max_data_size_scatter)
    
    if not os.path.exists(plot_loc):
        os.system('mkdir -p %s' %(plot_loc))
    scatter_plot_path = os.path.join(plot_loc, fig_name)

    plt.figure(figsize = (fig_size_x, fig_size_y))

    plt.scatter(labels_gt[data_ind_subset], labels_pred[data_ind_subset])
    plt.plot(labels_gt_range, labels_gt_range, '--r')
    plt.xlabel('Ground Truth', fontsize = font_size)
    plt.ylabel('Prediction', fontsize = font_size)
    plt.title('Model: {}, Accuracy: {:.3f}'.format(model_name, accuracy), fontsize = font_size)
    plt.xlim(x_lim)
    plt.ylim(x_lim)
    plt.yticks(fontsize = font_size, rotation = 0)
    plt.xticks(fontsize = font_size, rotation = 0)

    plt.savefig(scatter_plot_path, bbox_inches='tight')