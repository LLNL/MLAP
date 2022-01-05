#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import sys
import os

def re_label_data(sheet_to_df_map):
    for sheet in sheet_to_df_map.keys():
        df_inp = sheet_to_df_map[sheet] 
        df_out = df_inp.copy()
        df_out['NewLabel'] = [1 if '_Burnable' in df_inp['Label'][idx] else 0 for idx in range(len(df_inp))]
        sheet_to_df_map[sheet] = df_out
    return sheet_to_df_map

def combine_df_sheets(sheet_to_df_map, sheets, cols):
    df_combined = pd.DataFrame() #create a new dataframe that's empty
    for sheet in sheets:
        df_combined = df_combined.append(sheet_to_df_map[sheet][cols])
    return df_combined

def get_features_labels(df_combined, features_to_use):
    X = np.array(df_combined[features_to_use].values)
    y = np.array(df_combined[['NewLabel', 'SheetName']].values)
    #y = np.array(df_combined['NewLabel'].values)
    idx_nan = [idx for idx in range(len(X)) if np.isnan(X[idx]).any()]
    X_clean = [X[i] for i in range(len(X)) if i not in idx_nan]
    y_clean = [y[i] for i in range(len(y)) if i not in idx_nan]
    #return X_clean, y_clean
    return np.array(X_clean), np.array(y_clean)

def split_labels_sheets(labels_train, labels_test):
    sheets_train = labels_train[:, 1].astype(str)
    sheets_test  = labels_test[:, 1].astype(str)
    labels_train = labels_train[:, 0].astype(np.int64)
    labels_test  = labels_test[:, 0].astype(np.int64)
    
    return sheets_train, sheets_test, labels_train, labels_test


   
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