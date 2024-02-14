#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# []
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
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
#plt.style.use('seaborn-white')
from datetime import date, datetime, timedelta, time
from timeit import default_timer as timer


# []
'''
Load the extracted data
'''
def load_extracted_data (extracted_data_loc, extracted_data_file_name):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "load_extracted_data"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nLoading exracted data from file:\n ... {} \n ... at: {}\n'.format(\
                                                   extracted_data_file_name, extracted_data_loc))
    df_tt_extracted = pd.read_pickle(os.path.join(extracted_data_loc, extracted_data_file_name))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return df_tt_extracted

# []
'''
Get keys from the extracted data 
'''
def get_keys_from_extracted_data (df_extracted, train_test = True):
    keys_all = df_extracted.keys()
    
    # Keys for identity
    if train_test:
        keys_identity = ['FM_time_ind', 'FM_ts', 'his_time_ind', 'grid_ind', 'j_ind', 'i_ind']
    else:
        keys_identity = ['FM_ts', 'grid_ind', 'j_ind', 'i_ind']

    # Keys for potential labels
    keys_FM = ['FM_10hr', 'FM_100hr']

    # Keys for features
    keys_U10 = [key for key in keys_all if 'U10' in key]
    keys_V10 = [key for key in keys_all if 'V10' in key]
    keys_T2 = [key for key in keys_all if 'T2' in key]
    keys_RH = [key for key in keys_all if 'RH' in key]
    keys_PREC = [key for key in keys_all if 'PREC' in key]
    keys_SW = [key for key in keys_all if 'SW' in key]
    keys_HGT = [key for key in keys_all if 'HGT' in key]
    
    keys_UMag10 = []
    for U10_key, V10_key in zip(keys_U10, keys_V10):
        assert U10_key[3:] == V10_key[3:]
        UMag10_key = 'UMag10' + U10_key[3:]
        keys_UMag10.append(UMag10_key)
                     
    #print('=========================================================================')
    return keys_identity, keys_FM, keys_U10, keys_V10, keys_UMag10, \
                            keys_T2, keys_RH, keys_PREC, keys_SW, keys_HGT


# []
'''
Define Binary and MC FM labels
'''
def define_binary_and_MC_FM_labels (keys_FM):
    keys_FM_Binary = []
    keys_FM_MC     = []
    for FM_key in keys_FM:
        FM_key_binary = FM_key + '_bin'
        FM_key_MC     = FM_key + '_MC'
        keys_FM_Binary.append(FM_key_binary)
        keys_FM_MC.append(FM_key_MC)
    
    return keys_FM_Binary, keys_FM_MC


# []
'''
Define groups of Keys (Labels)
'''
def define_labels(FM_label_type, keys_FM, keys_FM_Binary, keys_FM_MC):
    if (FM_label_type == 'Regression'):
        keys_labels = keys_FM
    elif (FM_label_type == 'Binary'):
        keys_labels = keys_FM + keys_FM_Binary
    elif (FM_label_type == 'MultiClass'):
        keys_labels = keys_FM + keys_FM_MC
    else:
        raise ValueError('Invalid "label_type": {} in "FM_labels". \
                        \nValid types are: "Regression", "MultiClass", and "Binary"'.format(\
                                                                                FM_label_type))
    return keys_labels


# []
'''
Define groups of Keys (Features)
'''
def define_features(keys_UMag10, keys_T2, keys_RH, keys_PREC, keys_SW, \
                   features_to_use):

    keys_features = []
    if 'UMag10' in features_to_use:
        keys_features += keys_UMag10 
    if 'T2' in features_to_use:
        keys_features += keys_T2
    if 'RH' in features_to_use:
        keys_features += keys_RH 
    if 'PREC' in features_to_use:
        keys_features += keys_PREC
    if 'SW' in features_to_use:
        keys_features += keys_SW
        
    return keys_features


# []
'''
Compute UMag from U and V
'''
def compute_wind_mag (df_extracted, keys_U10, keys_V10, keys_UMag10):
    for U10_key, V10_key, UMag_key in zip(keys_U10, keys_V10, keys_UMag10):
        assert U10_key[3:] == V10_key[3:]
        df_extracted[UMag_key] = (df_extracted[U10_key]**2 + df_extracted[V10_key]**2)**(0.5)

    return df_extracted
    
# []
'''
Drop wind components U, and v
'''
def drop_wind_components (df_extracted, keys_U10, keys_V10):
    df_extracted = df_extracted.drop(keys_U10 + keys_V10, axis = 'columns')
    return df_extracted


# []
'''
Compute binary FM labels
'''
def compute_binary_FM_labels(df, keys_FM, keys_FM_Binary, FM_binary_threshold):
    for FM_key, FM_key_binary in zip(keys_FM, keys_FM_Binary):
        # If FM > threshold_FM_fire_risk, then no risk, i.e., 0. Otherwise fire risk or 1
        df[FM_key_binary] = np.where(df[FM_key] > FM_binary_threshold , 0, 1)
    
    return df

# []
'''
Compute MC FM labels
'''
def compute_MC_FM_labels(df, keys_FM, keys_FM_MC, FM_levels):
    for FM_key, FM_key_MC in zip(keys_FM, keys_FM_MC):
        conditions = []
        labels = []
        for num_levels in range(len(FM_levels)-1):
            conditions.append((df[FM_key] > FM_levels[num_levels]) &\
                              (df[FM_key] <= FM_levels[num_levels+1]))
            labels.append(num_levels)
            #labels.append('FM%02d'%(num_levels))

        df[FM_key_MC] = np.select(conditions, labels, default=labels[0])

    return df

# []
'''
Plot FM labels
'''
def plot_FM_labels (df, FM_label_type, FM_hr, \
                    prepared_data_set_name, prepared_data_loc):
    
    if (FM_label_type == 'Regression'):
        columns_to_plot = ['FM_{}hr'.format(FM_hr)]
    elif (FM_label_type == 'Binary'):
        columns_to_plot = ['FM_{}hr_bin'.format(FM_hr)]
    elif (FM_label_type == 'MultiClass'):
        columns_to_plot = ['FM_{}hr_MC'.format(FM_hr)]
    else:
        raise ValueError('Invalid "label_type": {} in "FM_labels". \
                        \nValid types are: "Regression", "MultiClass", and "Binary"'.format(\
                                                                                FM_label_type))
        
    plt.figure()
    for col_label in columns_to_plot:
        plt.hist(df[col_label], bins = 19, density=False, label = col_label)
    plt.legend()
    plt.xlabel('Fuel Moisture ({})'.format(FM_label_type), fontsize = 14)
    plt.ylabel('Frequency', fontsize = 14)

    filename = '{}_FM_{}hr.png'.format(prepared_data_set_name, FM_hr)
    filedir = prepared_data_loc
    os.system('mkdir -p %s'%filedir)

    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight')
    plt.show()
    
    
# []
'''
Split data into groups of keys
'''
def split_data_into_groups (df, keys_identity, keys_labels, keys_features):
    data_tt_prep = dict()
    data_tt_prep['all'] = df
    data_tt_prep['identity'] = df[keys_identity]
    data_tt_prep['labels'] = df[keys_labels]
    data_tt_prep['features'] = df[keys_features]
    
    '''
    data_fire_prep = dict()
    for fire_name in fire_data_prep.keys():
        data_this_fire = dict()
        data_this_fire['all'] = fire_data_prep[fire_name]
        data_this_fire['identity'] = fire_data_prep[fire_name][keys_identity]
        data_this_fire['labels'] = fire_data_prep[fire_name][keys_labels]
        data_this_fire['features'] = fire_data_prep[fire_name][keys_features]

        data_fire_prep[fire_name] = data_this_fire
    '''
    return data_tt_prep #, data_fire_prep
