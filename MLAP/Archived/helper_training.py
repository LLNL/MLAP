#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def compute_u_mag(df_extracted):
    keys_all = df_extracted.keys()
    keys_U10 = [key for key in keys_all if 'U10' in key]
    keys_V10 = [key for key in keys_all if 'V10' in key]
    
    for U10_key, V10_key in zip(keys_U10, keys_V10):
        assert U10_key[3:] == V10_key[3:]
        UMag10_key = 'UMag10' + U10_key[3:]
        df_extracted[UMag10_key] = (df_extracted[U10_key]**2 + df_extracted[V10_key]**2)**(0.5)
    
    df_extracted = df_extracted.drop(keys_U10 + keys_V10, axis = 'columns')
    return df_extracted

def re_label_binary(df_extracted, threshold_FM_fire_risk, labeled_data_loc, data_file_identity):
    keys_FM = [key for key in df_extracted.keys() if 'FM' in key and key.endswith('hr')]
    for FM_key in keys_FM:
        FM_key_binary = FM_key + '_binary'
        # If FM > threshold_FM_fire_risk, then no risk, i.e., 0. Otherwise fire risk or 1
        df_extracted[FM_key_binary] = np.where(df_extracted[FM_key] > threshold_FM_fire_risk , 0, 1)
        
        FM_plot_file_name = os.path.join(labeled_data_loc, 'histogram_binary_{}_{}.png'.format(FM_key, data_file_identity))
        if not os.path.exists(labeled_data_loc):
            os.system('mkdir {}'.format(labeled_data_loc))
        ax = df_extracted[[FM_key, FM_key_binary]].plot.hist(bins = 24, alpha = 0.5)
        ax.figure.savefig(FM_plot_file_name)
        
    return df_extracted

def re_label_multi_class(df_extracted, FM_levels, labeled_data_loc, data_file_identity):
    keys_FM = [key for key in df_extracted.keys() if 'FM' in key and key.endswith('hr')]
    for FM_key in keys_FM:
        FM_key_MC = FM_key + '_MC'
        
        conditions = []
        labels = []
        for num_levels in range(len(FM_levels)-1):
            conditions.append((df_extracted[FM_key] > FM_levels[num_levels]) & (df_extracted[FM_key] <= FM_levels[num_levels+1]))
            labels.append(num_levels)
            #labels.append('FM%02d'%(num_levels))
            
        df_extracted[FM_key_MC] = np.select(conditions, labels, default=labels[0])
        
        FM_plot_file_name = os.path.join(labeled_data_loc, 'histogram_MC_{}_{}.png'.format(FM_key, data_file_identity))
        if not os.path.exists(labeled_data_loc):
            os.system('mkdir {}'.format(labeled_data_loc))
        ax = df_extracted[[FM_key_MC]].plot.hist(bins = 24, alpha = 0.5)
        ax.figure.savefig(FM_plot_file_name)
        
    return df_extracted

def get_features_labels_headers(df_labeled, FM_hr, label_type):
    keys_all = df_labeled.keys()
    keys_FM = [key for key in df_labeled.keys() if 'FM' in key]
    keys_coord = ['lat', 'lon']
    features_to_use = [key for key in keys_all if key not in keys_FM+keys_coord]
    label_to_use = 'FM_{}hr_{}'.format(FM_hr, label_type)
    
    return features_to_use, label_to_use

def get_features_labels(df_labeled, features_to_use, label_to_use):
    X = df_labeled[features_to_use]
    y = df_labeled[label_to_use]
    
    return X, y

   
def plot_hist_df_var(df, var, hist_dir, plot_base_dir = 'plots'):
    plt.figure(figsize=(40,20))
    plot_dir = os.path.join(plot_base_dir, hist_dir)
    if not os.path.exists(plot_dir):
        os.system('mkdir -p %s' %(plot_dir))
    plot_path = os.path.join(plot_dir, '%s.png'%(var))
    
    df = df[(df[var] != 'nan')]
    x = df[var].astype(float)
    #print ('x:', x, type(x))
    #x = df[var]astype(float).dropna() # Type coversion not needed if all data are float
    #print('Variable: ', var)
    if 'time_ms' in var: 
        print ('Length before pruning:', len(x))
        x = x[(x >= 0)]
        print ('Length after pruning:', len(x))
        
    colors = ['#E69F00', '#56B4E9', '#009E73']
    plt.hist(x, bins = 50, density=False,
             color = colors[0])
    
    plt.legend(prop={'size': 30})
    #plt.title(var, fontsize = 30)
    plt.xlabel(var, fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    
    plt.savefig(plot_path)
    plt.close()
    
def plot_hist_dfmap_combined(sheet_to_df_map, sheets, cols, hist_dir, plot_base_dir = 'plots'):
    df_combined = combine_df_sheets(sheet_to_df_map, sheets, cols)
    for var in cols:
        plot_hist_df_var(df_combined, var, hist_dir, plot_base_dir)
 
    
#######################################################################################################   
def plot_scatter_svm(clf, X_test, y_test, v1, v2, accuracy, plot_base_dir = 'plots', scatter_dir = 'SVM'):    
    plt.figure(figsize=(40,20))
    scatter_plot_dir = os.path.join(plot_base_dir, scatter_dir)
    if not os.path.exists(scatter_plot_dir):
        os.system('mkdir -p %s' %(scatter_plot_dir))
    scatter_plot_path = os.path.join(scatter_plot_dir, 'V1_%s__V2_%s.png'%(v1, v2))
    
    v1_list = [item[0] for item in X_test]
    v2_list = [item[1] for item in X_test]
    x_min = min(v1_list); x_max = max(v1_list)
    y_min = min(v2_list); y_max = max(v2_list)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    import pylab as pl
    hx = .005*(x_max - x_min)  # step size in the mesh- v1 dir
    hy = .005*(y_max - y_min)  # step size in the mesh- v2 dir
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)
    
    # Plot also the test points
    unburnable_v1 = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    unburnable_v2 = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    burnable_v1    = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    burnable_v2    = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    
    plt.scatter(unburnable_v1, unburnable_v2, color = "b", marker = 'o', s=50, label="Unburnable")
    plt.scatter(burnable_v1,    burnable_v2, color = "r", marker = 's', s=50, label="Burnable")
    plt.xlabel(v1, fontsize=30)
    plt.ylabel(v2, fontsize=30)
    plt.title('accuracy: {}'.format(accuracy), fontsize=30)
    plt.yticks(fontsize=30, rotation= 0)
    plt.xticks(fontsize=30, rotation= 0)
    plt.legend(fontsize=30)
    
    plt.savefig(scatter_plot_path)
    plt.close()
    
#######################################################################################################
def plot_scatter_arr(X_test, y_test, v1, v2, accuracy, plot_base_dir = 'plots', scatter_dir = 'scatter'):    
    plt.figure(figsize=(40,20))
    scatter_plot_dir = os.path.join(plot_base_dir, scatter_dir)
    if not os.path.exists(scatter_plot_dir):
        os.system('mkdir -p %s' %(scatter_plot_dir))
    scatter_plot_path = os.path.join(scatter_plot_dir, 'V1_%s__V2_%s.png'%(v1, v2))
    
    v1_list = [item[0] for item in X_test]
    v2_list = [item[1] for item in X_test]
    x_min = min(v1_list); x_max = max(v1_list)
    y_min = min(v2_list); y_max = max(v2_list)
    
    # Plot the test points
    unburnable_v1 = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    unburnable_v2 = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    burnable_v1    = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    burnable_v2    = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    
    plt.scatter(unburnable_v1, unburnable_v2, color = "b", marker = 'o', s=50, label="Unburnable")
    plt.scatter(burnable_v1,    burnable_v2, color = "r", marker = 's', s=50, label="Burnable")
    plt.xlabel(v1, fontsize=30)
    plt.ylabel(v2, fontsize=30)
    plt.title('accuracy: {}'.format(accuracy), fontsize=30)
    plt.yticks(fontsize=30, rotation= 0)
    plt.xticks(fontsize=30, rotation= 0)
    plt.legend(fontsize=30)
    
    plt.savefig(scatter_plot_path)
    plt.close()
        
#######################################################################################################
def plot_scatter_df(sheet_to_df_map, burnable_sheets, unburnable_sheets, v1, v2, plot_base_dir = 'plots', scatter_dir = 'scatter_test'):
    plt.figure(figsize=(40,20))
    scatter_plot_dir = os.path.join(plot_base_dir, scatter_dir)
    if not os.path.exists(scatter_plot_dir):
        os.system('mkdir -p %s' %(scatter_plot_dir))
    scatter_plot_path = os.path.join(scatter_plot_dir, 'V1_%s__V2_%s.png'%(v1, v2))
    
    for sheet in burnable_sheets:
        x = [float(item) for item in sheet_to_df_map[sheet][v1]]
        y = [float(item) for item in sheet_to_df_map[sheet][v2]]
        plt.scatter(x, y, marker = 's', s=50, label = sheet) 
        
    for sheet in unburnable_sheets:
        x = [float(item) for item in sheet_to_df_map[sheet][v1]]
        y = [float(item) for item in sheet_to_df_map[sheet][v2]]
        plt.scatter(x, y, marker = 'o', s=50, label = sheet)

    plt.xlabel(v1, fontsize=30)
    plt.ylabel(v2, fontsize=30)
    plt.yticks(fontsize=30, rotation= 0)
    plt.xticks(fontsize=30, rotation= 0)
    plt.legend(fontsize=30, bbox_to_anchor=(0, 1.10, 1, 0), loc=2, ncol=5, borderaxespad=0)

    plt.savefig(scatter_plot_path)
    plt.close()