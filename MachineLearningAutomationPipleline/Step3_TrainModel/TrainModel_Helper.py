#!/usr/bin/env python
# coding: utf-8


### ==== === Standard === === === === === === === ===
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

### ==== === Models === === === === === === === ===
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

### ==== === Preprocessors === === === === === === === ===
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

### ==== === Metrics === === === === === === === ===
### ==== === Regression === === === === === === === ===
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error
### ==== === Classification === === === === === === === ===
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report


# []
'''
Define features to use
'''
def define_features_to_use (features_in_prep_data, qois_for_training):
    features_to_use = []
    for qoi in qois_for_training:
        keys_qoi = [key for key in features_in_prep_data if qoi in key]
        #print(keys_qoi)
        features_to_use += keys_qoi
    
    return features_to_use
    

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
    
    return np.reshape(labels_pred, (len(labels_pred), 1))


# []
'''
Compute error and percent error
'''
def compute_errors (labels_gt, labels_pred):
    print ('Computing errors with ground truth and predicted labels')
    labels_error = labels_pred - labels_gt
    labels_error_abs = abs (labels_error)
    labels_pc_err = (labels_pred/labels_gt - 1.0)*100
    labels_pc_err_abs = abs (labels_pc_err)
    
    return labels_error, labels_error_abs, labels_pc_err, labels_pc_err_abs


# []
'''
Compute 90th and 95th percentile of error and percent error
Get the best 90 and 95 percent of ground truth and error
'''
def compute_best_90_95_labels (labels_gt, labels_pred, labels_error):
    print ('Computing 90-th and 95-th percentiles of error')
    labels_error_p90 = np.percentile(labels_error, 90, axis=0)[0]
    labels_error_p95 = np.percentile(labels_error, 95, axis=0)[0]
    
    labels_gt_best90 = labels_gt[np.where(labels_error < labels_error_p90)]
    labels_pred_best90 = labels_pred[np.where(labels_error < labels_error_p90)]
    labels_gt_best95 = labels_gt[np.where(labels_error < labels_error_p95)]
    labels_pred_best95 = labels_pred[np.where(labels_error < labels_error_p95)]
    
    print('P90: {:.4f}, P95: {:.4f}'.format(labels_error_p90, labels_error_p95))
    print('Data SIZE:- Orig: {}, Best 90%: {}, Best 95%: {}'.format(len(labels_gt),\
                                                                    len(labels_gt_best90),\
                                                                    len(labels_gt_best95)))

    return labels_error_p90, labels_error_p95, \
           labels_gt_best90, labels_pred_best90, \
           labels_gt_best95, labels_pred_best95



# []
'''
Get Metrics
'''
def get_metrics_regression (labels_gt, labels_pred, data_identifier):
    
    r2_scr = r2_score (labels_gt, labels_pred)
    ev_scr = explained_variance_score (labels_gt, labels_pred)
    mse = mean_squared_error (labels_gt, labels_pred)
    rmse = np.sqrt (mse)
    max_err = max_error (labels_gt, labels_pred)
    mae = mean_absolute_error (labels_gt, labels_pred)
    medae = median_absolute_error (labels_gt, labels_pred)
    
    print ('Metrics for {}:\n \
            ... R2: {:.4f}, EV: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, \n \
            ... Max_Err : {:.4f}, MAE: {:.4f}, MedAE: {:.4f}'.format(data_identifier, \
                                                         r2_scr, ev_scr, mse, rmse, \
                                                         max_err, mae, medae))
    reg_metrics = dict()
    reg_metrics['r2_score'] = r2_scr
    reg_metrics['ev_score'] = ev_scr
    reg_metrics['mse'] = mse
    reg_metrics['rmse'] = rmse
    reg_metrics['max_err'] = max_err
    reg_metrics['mae'] = mae
    reg_metrics['medae'] = medae

    return reg_metrics

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
def plot_scatter_regression (labels_gt, labels_pred, \
                             reg_metrics, \
                             model_name, \
                             plot_loc, fig_name, \
                             max_data_size_scatter, fig_size_x, fig_size_y, \
                             font_size, x_lim, label_log):
    
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

    if label_log:
        plt.scatter(np.exp(labels_gt[data_ind_subset]), np.exp(labels_pred[data_ind_subset]))
        plt.plot(np.exp(labels_gt_range), np.exp(labels_gt_range), '--r')
    else:
        plt.scatter(labels_gt[data_ind_subset], labels_pred[data_ind_subset])
        plt.plot(labels_gt_range, labels_gt_range, '--r')
    plt.xlabel('Ground Truth', fontsize = font_size)
    plt.ylabel('Prediction', fontsize = font_size)
    plt.title('Model: {}, R2: {:.3f}, RMSE: {:.3f}, MAE: {:.4f}'.format(model_name, \
                                                          reg_metrics['r2_score'], \
                                                          reg_metrics['rmse'], \
                                                          reg_metrics['mae']), fontsize = font_size)
    plt.xlim(x_lim)
    plt.ylim(x_lim)
    plt.yticks(fontsize = font_size, rotation = 0)
    plt.xticks(fontsize = font_size, rotation = 0)

    plt.savefig(scatter_plot_path, bbox_inches='tight')
    

# []
'''
Get metrics and plot scatter for Rgeression
'''
def get_metrics_plot_scatter_regression (labels_gt, labels_pred, data_identifier, \
                             model_name, plot_loc, fig_name, \
                             max_data_size_scatter, fig_size_x, fig_size_y, \
                             font_size, x_lim, label_log):
    
    reg_metrics = get_metrics_regression (labels_gt, labels_pred, data_identifier)
    
    plot_scatter_regression (labels_gt, labels_pred, reg_metrics, model_name, \
                            plot_loc, fig_name, \
                            max_data_size_scatter, fig_size_x, fig_size_y, \
                            font_size, x_lim, label_log)
    
    return reg_metrics

    
# []
'''
Plot Confusion Matrix for Classification
'''
def plot_confusion_matrix (conf_mat, accuracy, model_name, \
                           plot_loc, fig_name, \
                           fig_size_x, fig_size_y, \
                           font_size,\
                           normalize_cm, class_labels):
    
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

    plt.xticks(class_labels)
    plt.yticks(class_labels)
    plt.tick_params(axis='both', which='major', labelsize=font_size, labelbottom = False, bottom=False, top = False, labeltop=True)
    for (i, j), z in np.ndenumerate(conf_mat):
        ax.text(j, i, num_format.format(z), fontsize = font_size, ha='center', va='center')
    cbar = plt.colorbar(shrink = 0.8)
    cbar.ax.tick_params(labelsize=font_size)
    
    plt.savefig(cm_plot_path, bbox_inches='tight')
    

# []
'''
Create train test metrics
'''
def create_train_test_metrics (reg_metrics_train, reg_metrics_test, \
                               reg_metrics_test_p90, reg_metrics_test_p95):
    data_for_csv = {'r2_score_train':    [reg_metrics_train['r2_score']],
                    'ev_score_train':    [reg_metrics_train['ev_score']] ,
                    'mse_train':         [reg_metrics_train['mse']],
                    'rmse_train':        [reg_metrics_train['rmse']],
                    'max_err_train':     [reg_metrics_train['max_err']],
                    'mae_train':         [reg_metrics_train['mae']],
                    'medae_train':       [reg_metrics_train['medae']],
                    'r2_score_test':     [reg_metrics_test['r2_score']],
                    'ev_score_test':     [reg_metrics_test['ev_score']] ,
                    'mse_test':          [reg_metrics_test['mse']],
                    'rmse_test':         [reg_metrics_test['rmse']],
                    'max_err_test':      [reg_metrics_test['max_err']],
                    'mae_test':          [reg_metrics_test['mae']],
                    'medae_test':        [reg_metrics_test['medae']],
                    'r2_score_test_p90': [reg_metrics_test_p90['r2_score']],
                    'ev_score_test_p90': [reg_metrics_test_p90['ev_score']] ,
                    'mse_test_p90':      [reg_metrics_test_p90['mse']],
                    'rmse_test_p90':     [reg_metrics_test_p90['rmse']],
                    'max_err_test_p90':  [reg_metrics_test_p90['max_err']],
                    'mae_test_p90':      [reg_metrics_test_p90['mae']],
                    'medae_test_p90':    [reg_metrics_test_p90['medae']],
                    'r2_score_test_p95': [reg_metrics_test_p95['r2_score']],
                    'ev_score_test_p95': [reg_metrics_test_p95['ev_score']] ,
                    'mse_test_p95':      [reg_metrics_test_p95['mse']],
                    'rmse_test_p95':     [reg_metrics_test_p95['rmse']],
                    'max_err_test_p95':  [reg_metrics_test_p95['max_err']],
                    'mae_test_p95':      [reg_metrics_test_p95['mae']],
                    'medae_test_p95':    [reg_metrics_test_p95['medae']],
                   }
    
    return data_for_csv
    
    
# [] 
def plot_fm (y_gt, labels_pred, j_indices, i_indices, FM_label_type, analysis_data_loc, analysis_fuel_map_file_name, class_labels = None):
    ny, nx = (480, 396)
    ground_truth = y_gt.to_numpy()
    
    if FM_label_type == 'Regression':
        ground_truth_mat = np.full((ny,nx), np.nan)
        pred_mat = np.full_like(ground_truth_mat, np.nan )
        error_mat = np.full_like(ground_truth_mat, np.nan)
    else:
        ground_truth_mat = np.ones((ny,nx), int)*(-1)
        pred_mat = np.ones_like(ground_truth_mat, int)*(-np.nan)
        error_mat = np.ones_like(ground_truth_mat, int)*(-np.nan)
        
        
    for j_loc, i_loc, gt_val, pred_val in zip (j_indices, i_indices, ground_truth, labels_pred):
        ground_truth_mat[j_loc][i_loc] = gt_val
        pred_mat[        j_loc][i_loc] = pred_val
        if (FM_label_type == 'Binary' or 'FM_label_type' == 'MultiClass'):
            error_mat[       j_loc][i_loc] = (gt_val == pred_val)
        else:
            error_mat[       j_loc][i_loc] = 100.0*(pred_val/gt_val - 1.0)
            
    
    cmap_name = 'viridis'

    if (FM_label_type == 'Binary' or 'FM_label_type' == 'MultiClass'):
        cont_levels = class_labels
        cont_levels_err = [0, 1]
    else:
        cont_levels = np.linspace(0, 0.1, 21)
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
    if (FM_label_type == 'Binary' or 'FM_label_type' == 'MultiClass'):
        ax[2].set_title('Correct Match')
    else:
        ax[2].set_title('Percentage Error')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    filename = os.path.join(analysis_data_loc, analysis_fuel_map_file_name)
    plt.savefig(filename, bbox_inches='tight')


# []
'''
Create label and train pair for evaluation
'''
def create_label_train_pair (json_prep_train_maps):
    label_train_pair = []
    col_names = []
    for item in json_prep_train_maps:
        json_label = item ['json_label']
        set_info = item ['set_info']
        json_train = item ['json_train']
        subset_info = item ['subset_info']

        for label, set_text in zip(json_label, set_info):
            #print (label, set_text)
            for train, subset_text in zip (json_train, subset_info):
                #print (train, subset_text)
                label_train_pair.append((label, train))
                col_names.append('{}, {}'.format(set_text, subset_text))
                
    return label_train_pair, col_names


# []
'''
Create data definition
'''
def create_data_definition (json_extract_base, json_extract_counts):
    data_name = []
    max_hist = []
    hist_int = []
    num_hist = []
    num_sampled_times = []
    num_sampled_gp = []
    num_time_grid_to_read = []
    rows_feature = []
    cols_feature = []

    for data_set_count in json_extract_counts:

        json_extract = '%s_%03d.json'%(json_extract_base, data_set_count)
        #print(json_extract)
        with open(json_extract) as json_file_handle:
            json_content_extract_data = json.load(json_file_handle)

        extracted_data_base_loc = json_content_extract_data['paths']['extracted_data_base_loc']    
        data_set_name = 'data_train_test_extracted_%03d'%(data_set_count)
        extracted_data_loc = os.path.join(extracted_data_base_loc, data_set_name)
        tab_data_file_name = '{}_tab_data.csv'.format(data_set_name)
        #print (tab_data_file_name)
        tabulated_data = pd.read_csv(os.path.join(extracted_data_loc, tab_data_file_name))
        #print (tabulated_data)

        data_name.append(data_set_count)
        max_hist.append(tabulated_data['max_hist'][0])
        hist_int.append(tabulated_data['hist_interval'][0])
        num_hist.append(tabulated_data['num_hist'][0])
        num_sampled_times.append(tabulated_data['num_sampled_times'][0])
        num_sampled_gp.append(tabulated_data['num_sampled_grid_points'][0])
        num_time_grid_to_read.append(tabulated_data['num_time_grid_to_read'][0])
        rows_feature.append(tabulated_data['rows_feature_mat'][0])
        cols_feature.append(tabulated_data['cols_feature_mat'][0])

    data_defn = pd.DataFrame()
    data_defn['data_name'] = data_name
    data_defn['max_hist'] = max_hist
    data_defn['hist_int'] = hist_int
    data_defn['num_hist'] = num_hist
    data_defn['num_sampled_times'] = num_sampled_times
    data_defn['num_sampled_gp'] = num_sampled_gp
    data_defn['num_time_grid_to_read'] = num_time_grid_to_read
    data_defn['rows_feature'] = rows_feature
    data_defn['cols_feature'] = cols_feature
    
    #-----------------------------------------------
    return data_defn
    

# []
'''
Gather metrics for a (label, train) pair
'''
def gather_metrics_for_label_train_pair (json_extract_counts, json_content_train_model, \
                                         label_count, FM_label_type, train_count, \
                                         eval_metric_col):
    
    trained_model_base_loc = json_content_train_model['paths']['trained_model_base_loc']
    model_name = json_content_train_model['models']['model_name']
    
    metrics = []
    for data_set_count in json_extract_counts:
        trained_model_name = 'dataset_%03d_label_%03d_%s_model_%03d_%s'%(data_set_count, \
                                                    label_count, FM_label_type, \
                                                    train_count, model_name)

        trained_model_loc = os.path.join(trained_model_base_loc, trained_model_name)
        model_eval_file = '{}_eval'.format(trained_model_name)
        model_metric_file = os.path.join(trained_model_loc, model_eval_file+'.csv')
        #print('Trained Model Location: {}'.format(trained_model_loc))
        #print('Trained Model Metrric File: {}'.format(model_metric_file))
        
        eval_metric = pd.read_csv(model_metric_file).to_dict(orient='records')[0]
        metrics.append(eval_metric[eval_metric_col])
    
    #-----------------------------------------------
    return metrics
 
    
# []
'''
Gather metrics for all (label, train) pairs
'''
def gather_metrics_for_all_label_train_pairs (label_train_pair, col_names, \
                                              json_train_base, json_extract_counts, \
                                              FM_label_type, eval_metric_col):
    
    df_metrics = pd.DataFrame(index = json_extract_counts)
    for (label_count, train_count), col_name in zip(label_train_pair, col_names):
        #print(label_count, train_count, col_name)
        json_train   = '%s_%03d.json'%(json_train_base, train_count)
        #print(json_train)
        with open(json_train) as json_file_handle:
            json_content_train_model = json.load(json_file_handle)


        metrics = gather_metrics_for_label_train_pair (json_extract_counts, json_content_train_model, \
                                                       label_count, FM_label_type, train_count, \
                                                       eval_metric_col)
        df_metrics[col_name] = metrics
    
    #-----------------------------------------------
    return df_metrics


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

            r2_score_train = []
            r2_score_test = []
            rmse_train = []
            rmse_test = []
            medae_train = []
            medae_test = []
            
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
                r2_score_train.append(eval_metric['r2_score_train'])
                r2_score_test.append(eval_metric['r2_score_test'])
                rmse_train.append(eval_metric['rmse_train'])
                rmse_test.append(eval_metric['rmse_test'])
                medae_train.append(eval_metric['medae_train'])
                medae_test.append(eval_metric['medae_test'])
                #print('\n')

            #print('label_count: {}, FM_label_type: {}, Model Count: {}, Model Name: {}'.format(
             #   label_count, FM_label_type, model_count, model_name))
            #print('data_nomenclature: {}'.format(data_nomenclature))
            #print('accuracy_train: {}'.format(accuracy_train))
            #print('accuracy_test: {}'.format(accuracy_test))

            metric_for_model['data_nomenclature'] = data_nomenclature
            metric_for_model['r2_score_train'] = r2_score_train
            metric_for_model['r2_score_test'] = r2_score_test
            metric_for_model['rmse_train'] = rmse_train
            metric_for_model['rmse_test'] = rmse_test
            metric_for_model['medae_train'] = medae_train
            metric_for_model['medae_test'] = medae_test
            
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
def plot_trained_models_metrics (FM_label_type, json_extract_counts, trained_models_metrics, metric_name):
    
    label = FM_label_type
    ds_name = json_extract_counts
    models = list(trained_models_metrics[FM_label_type].keys())
    
    if (label == 'Regression'):
        if (metric_name == 'r2_score'):
            ylabel_text = '$R^2$'
        elif (metric_name == 'rmse'):
            ylabel_text = 'RMSE'
        elif (metric_name == 'medae'):
            ylabel_text = 'Median Abs Error'
        else:
            raise ValueError('Invalid "metric_name": {}. \
                            \nValid types are: "r2_score", "RMSE", and "medae"'.format(\
                                                                                    metric_name))
    else: # Classification
        ylabel_text = 'Accuracy'
    
    train_metric_all_models = dict()
    test_metric_all_models = dict()
    for model in models:
        train_metric_all_models[model] = trained_models_metrics[label][model][metric_name+'_train']
        test_metric_all_models[model] = trained_models_metrics[label][model][metric_name+'_test']

        df_train = pd.DataFrame(train_metric_all_models, index = ds_name)
        df_test  = pd.DataFrame(test_metric_all_models, index = ds_name)

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
    