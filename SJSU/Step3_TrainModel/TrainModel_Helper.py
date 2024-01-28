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

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

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
            case 'Linear':
                model = LinearRegression()
            case 'SVM':
                model = SVR()
            case 'RF':
                model = RandomForestRegressor()
            case 'MLP':
                model = MLPRegressor()
            case 'GB':
                model = GradientBoostingRegressor()
            
            
    elif (FM_label_type == 'Binary' or FM_label_type == 'MultiClass'):
        match model_name:
            case 'SVM':
                model = SVC()
            case 'RF':
                model = RandomForestClassifier()
            case 'MLP':
                model = MLPClassifier()
            case 'GB':
                model = GradientBoostingClassifier
                
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
    

# []
'''
Plot Confusion Matrix for Classification
'''
def plot_confusion_matrix (conf_mat, accuracy, model_name, \
                           plot_loc, fig_name, \
                           fig_size_x, fig_size_y, \
                           font_size,\
                           normalize_cm):
    
    if normalize_cm:
        num_format = '{:.3f}'
        conf_mat_row_sum = conf_mat.sum(axis = 1)[:,np.newaxis]
        conf_mat = conf_mat/conf_mat_row_sum
    else:
        num_format = '{:.0f}'
        
    if not os.path.exists(plot_loc):
        os.system('mkdir -p %s' %(plot_loc))
    cm_plot_path = os.path.join(plot_loc, fig_name)
    
    fig, ax = plt.subplots(figsize = (fig_size_x, fig_size_y))
    
    im = plt.imshow(conf_mat, cmap = 'viridis')
    plt.ylabel('Ground Truth', fontsize = font_size)
    plt.xlabel('Prediction', fontsize = font_size)
    plt.title('Model: {}, Accuracy: {:.3f}'.format(model_name, accuracy), fontsize = font_size)

    plt.tick_params(axis='both', which='major', labelsize=font_size, labelbottom = False, bottom=False, top = False, labeltop=True)
    for (i, j), z in np.ndenumerate(conf_mat):
        ax.text(j, i, num_format.format(z), fontsize = font_size, ha='center', va='center')
    cbar = plt.colorbar(shrink = 0.8)
    cbar.ax.tick_params(labelsize=font_size)
    
    plt.savefig(cm_plot_path, bbox_inches='tight')
    
    
# []
'''
Create a dict of metrics from trained models
'''
def create_trained_models_metrics (json_prep_base, json_prep_counts, \
                                   json_train_base, json_train_counts, \
                                   json_extract_base, json_extract_counts):
    
    trained_models_metrics = dict()
    
    for label_count in json_prep_counts:
        metric_for_label = dict()
        json_prep    = '%s_%03d.json'%(json_prep_base, label_count)
        #print(json_prep)
        with open(json_prep) as json_file_handle:
            json_content_prep_data = json.load(json_file_handle)
        label_count = json_content_prep_data['label_defn']['label_count']
        FM_label_type = json_content_prep_data['FM_labels']['label_type']
        #print('label_count: {}, FM_label_type: {}'.format(label_count, FM_label_type))

        for train_count in json_train_counts:
            metric_for_model = dict()
            json_train   = '%s_%03d.json'%(json_train_base, train_count)
            #print(json_train)
            with open(json_train) as json_file_handle:
                json_content_train_model = json.load(json_file_handle)
            model_count = json_content_train_model['models']['model_count']
            model_name = json_content_train_model['models']['model_name'] # ['RF', SVM', 'MLP']
            #print('Model Count: {}, Model Name: {}'.format(model_count, model_name))

            accuracy_train = []
            accuracy_test = []
            
            data_nomenclature = []
            temporal_data_percent = []
            spatial_data_percent = []
            max_history = []
            hist_interval = []
            
            data_defn_dict = dict()
            
            for data_count in json_extract_counts:
                data_nomenclature.append(data_count)

                json_extract = '%s_%03d.json'%(json_extract_base, data_count)
                #print(json_extract)
                with open(json_extract) as json_file_handle:
                    json_content_extract_data = json.load(json_file_handle)
                
                data_set_defn = json_content_extract_data['data_set_defn']
                data_set_count = data_set_defn['data_set_count']
                percent_files_to_use = data_set_defn['percent_files_to_use']
                percent_grid_points_to_use = data_set_defn['percent_grid_points_to_use']
                max_history_to_consider = data_set_defn['max_history_to_consider']
                history_interval        = data_set_defn['history_interval']
                #print('Data Set Count: {}'.format(data_set_count))
                
                temporal_data_percent.append(percent_files_to_use)
                spatial_data_percent.append(percent_grid_points_to_use)
                max_history.append(max_history_to_consider)
                hist_interval.append(history_interval)

                # Names of trained model and related files
                trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
                trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count, \
                                                            label_count, FM_label_type, \
                                                            model_count, model_name)

                trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
                model_eval_file = '{}_eval'.format(trained_model_name)
                model_metric_file = os.path.join(trained_model_loc, model_eval_file+'.csv')
                #trained_model_file_name = '{}.pkl'.format(trained_model_name)
                #print('Trained Model Location: {}'.format(trained_model_loc))
                #print('Trained Model Metrric File: {}'.format(model_metric_file))

                eval_metric = pd.read_csv(model_metric_file).to_dict(orient='records')[0]
                #print('Eval Metrics: {}'.format(eval_metric))
                accuracy_train.append(eval_metric['accuracy_train'])
                accuracy_test.append(eval_metric['accuracy_test'])
                #print('\n')

            #print('label_count: {}, FM_label_type: {}, Model Count: {}, Model Name: {}'.format(
             #   label_count, FM_label_type, model_count, model_name))
            #print('data_nomenclature: {}'.format(data_nomenclature))
            #print('accuracy_train: {}'.format(accuracy_train))
            #print('accuracy_test: {}'.format(accuracy_test))

            metric_for_model['data_nomenclature'] = data_nomenclature
            metric_for_model['accuracy_train'] = accuracy_train
            metric_for_model['accuracy_test'] = accuracy_test
            
            data_defn_dict['temporal_data [%]'] = temporal_data_percent
            data_defn_dict['spatial_data [%]'] = spatial_data_percent
            data_defn_dict['max_history [hrs]'] = max_history
            data_defn_dict['hist_interval [hrs]'] = hist_interval
            
            data_defn = pd.DataFrame(data_defn_dict, index = data_nomenclature)

            #print('metric_for_model: \n{}'.format(metric_for_model))
            #print('\n')
            metric_for_label[model_name] = metric_for_model

        #print('metric_for_label: \n{}'.format(metric_for_label))
        #print('\n')

        trained_models_metrics[FM_label_type] =  metric_for_label
    
    #print('trained_models_metrics: \n{}'.format(trained_models_metrics))
    #print('\n')   
    #'=========================================================================' 
    return trained_models_metrics, data_defn



# []
'''
Plot metrics from trained models
'''
def plot_trained_models_metrics (FM_label_type, json_extract_counts, trained_models_metrics):
    
    label = FM_label_type
    ds_name = json_extract_counts
    models = list(trained_models_metrics[FM_label_type].keys())
    
    if (label == 'Regression'):
        ylabel_text = '$R^2$'
    else:
        ylabel_text = 'Accuracy'
    
    train_accuracy_all_models = dict()
    test_accuracy_all_models = dict()
    for model in models:
        train_accuracy_all_models[model] = trained_models_metrics[label][model]['accuracy_train']
        test_accuracy_all_models[model] = trained_models_metrics[label][model]['accuracy_test']

        df_train = pd.DataFrame(train_accuracy_all_models, index = ds_name)
        df_test  = pd.DataFrame(test_accuracy_all_models, index = ds_name)

    ax1 = df_train.plot.bar(rot = 0)
    ax1.set_xlabel('Data Set Name')
    ax1.set_ylabel(ylabel_text)
    ax1.set_title('Train')

    ax2 = df_test.plot.bar(rot = 0)
    ax2.set_xlabel('Data Set Name')
    ax2.set_ylabel(ylabel_text)
    ax2.set_title('Test')
    
    #'========================================================================='
    return df_train, df_test
    