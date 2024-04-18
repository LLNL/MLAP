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
def load_pickled_data (data_loc, data_file_name):
    process = psutil.Process(os.getpid())
    print('=========================================================================')
    module_start_time = timer()
    module_initial_memory = process.memory_info().rss
    print('MODULE Name: "load_pickled_data"')
    print('\nProcess in the module(): {}'.format(process))
    
    print('\nLoading data from file:\n ... {} \n ... at: {}\n'.format(\
                                                   data_file_name, data_loc))
    data_loaded = pd.read_pickle(os.path.join(data_loc, data_file_name))
    
    module_final_memory = process.memory_info().rss
    module_end_time = timer()
    module_memory_consumed = (module_final_memory - module_initial_memory)/(1024*1024)
    module_compute_time = module_end_time - module_start_time
    print('Module memory consumed: {:.3f} MB'.format(module_memory_consumed))
    print('Module computing time: {:.3f} s'.format(module_compute_time))
    print('=========================================================================')
    return data_loaded


'''
Prune the desired data
'''
def prune_desired_data (df, prune_data):
    for qoi in prune_data:
        #print(prune_data[qoi])
        df = df[(df[qoi] >= prune_data[qoi][0]) & (df[qoi] <= prune_data[qoi][1])]
        #print(len(df))

    return df


# []
'''
Reduce the size of extracted train/test data
'''
def reduce_data_size (df):
    print ('Reducing data size (float64 to float16, int64 to int32)')
    for key in df.keys():
        data_type = df[key].dtypes
        #print('key: {}, data_type: {}'.format(key, data_type))
        if (data_type == 'int64'):
            df = df.astype({key: 'int32'})
            #print('... Changing data type to int32')

        if (data_type == 'float64'):
            df = df.astype({key: 'float16'})
            #print('... Changing data type to float16')
            
    return df


# []
'''
Assess the sampled time and grid indices
'''
def assess_sampled_indices (df_tt_prep, index_col_name, prepared_data_loc, sampled_ind_file_name):
    
    df_indices = pd.DataFrame()
    
    df_indices['sampled_ind'] = df_tt_prep[index_col_name].unique()
    df_indices['sampled_ind_sorted'] = df_tt_prep[index_col_name].sort_values().unique()

    ind_diff = np.diff(df_indices['sampled_ind_sorted'])
    ind_diff.resize((len(ind_diff)+1,))
    ind_diff[len(ind_diff)-1] = -1
    df_indices['sampled_ind_diff'] = ind_diff
    df_indices['sampled_ind_diff_sorted'] = np.sort(ind_diff)
    
    df_indices.to_csv(os.path.join(prepared_data_loc, sampled_ind_file_name), index=True)
    
    return df_indices

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
    keys_UMag10 = [key for key in keys_all if 'UMag10' in key]
    keys_T2 = [key for key in keys_all if 'T2' in key]
    keys_RH = [key for key in keys_all if 'RH' in key]
    keys_PREC = [key for key in keys_all if 'PREC' in key]
    keys_SW = [key for key in keys_all if 'SW' in key]
    keys_HGT = [key for key in keys_all if 'HGT' in key]
                     
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
def define_features(keys_HGT, keys_UMag10, keys_T2, keys_RH, keys_PREC, keys_SW, \
                   qois_to_use):

    keys_features = []
    if 'HGT' in qois_to_use:
        keys_features += keys_HGT
    if 'UMag10' in qois_to_use:
        keys_features += keys_UMag10 
    if 'T2' in qois_to_use:
        keys_features += keys_T2
    if 'RH' in qois_to_use:
        keys_features += keys_RH 
    if 'PREC' in qois_to_use:
        keys_features += keys_PREC
    if 'SW' in qois_to_use:
        keys_features += keys_SW
        
    return keys_features


# []
'''
Compute UMag from U and V
'''
def compute_wind_mag (df_extracted, keys_U10, keys_V10, keys_UMag10):
    print ('Computing wind magnitude from wind components')
    for U10_key, V10_key, UMag_key in zip(keys_U10, keys_V10, keys_UMag10):
        assert U10_key[3:] == V10_key[3:]
        df_extracted[UMag_key] = (df_extracted[U10_key]**2 + df_extracted[V10_key]**2)**(0.5)

    return df_extracted
    
# []
'''
Drop wind components U, and v
'''
def drop_wind_components (df_extracted, keys_U10, keys_V10):
    print ('Dropping wind components')
    df_extracted = df_extracted.drop(keys_U10 + keys_V10, axis = 'columns')
    return df_extracted


# []
'''
Compute VPD
'''
def compute_VPD (df_tt_prep, keys_T2, keys_RH):
    print ('Computing Vapor Pressure Deficit (VPD) from T2 and RH')

    a0 = 6.105851
    a1 = 0.4440316
    a2 = 0.1430341e-1
    a3 = 0.2641412e-3
    a4 = 0.2995057e-5
    a5 = 0.2031998e-7
    a6 = 0.6936113e-10
    a7 = 0.2564861e-13
    a8 = -0.3704404e-15

    keys_T2_Cel = []
    keys_VP_s = []
    keys_VP = []
    keys_VPD = []
    for T2_key, RH_key in zip(keys_T2, keys_RH):
        assert T2_key[2:] == RH_key[2:]

        T2_Cel_key = 'T2_Cel{}'.format(T2_key[2:])
        VP_s_key = 'VP_s{}'.format(T2_key[2:])
        VP_key = 'VP{}'.format(T2_key[2:])
        VPD_key = 'VPD{}'.format(T2_key[2:])

        keys_T2_Cel.append(T2_Cel_key)
        keys_VP_s.append(VP_s_key)
        keys_VP.append(VP_key)
        keys_VPD.append(VPD_key)

        dtt = df_tt_prep[T2_key] - 273.16
        e_s = a0 + dtt*(a1+dtt*(a2+dtt*(a3+dtt*(a4+dtt*(a5+dtt*(a6+dtt*(a7+a8*dtt)))))))
        e   = e_s*df_tt_prep[RH_key]/100.00
        VPD = e_s - e

        #df_tt_prep[T2_Cel_key] =  dtt
        #df_tt_prep[VP_s_key] = e_s
        #df_tt_prep[VP_key] = e
        df_tt_prep[VPD_key] = VPD
    
    return df_tt_prep, keys_VPD

# []
'''
Compute binary FM labels
'''
def compute_binary_FM_labels(df, keys_FM, keys_FM_Binary, FM_binary_threshold):
    print ('Computing Binary FM labels')
    for FM_key, FM_key_binary in zip(keys_FM, keys_FM_Binary):
        # If FM > threshold_FM_fire_risk, then no risk, i.e., 0. Otherwise fire risk or 1
        df[FM_key_binary] = np.where(df[FM_key] > FM_binary_threshold , 0, 1)
        df = df.astype({FM_key_binary: 'int16'})
    
    return df

# []
'''
Compute MC FM labels
'''
def compute_MC_FM_labels(df, keys_FM, keys_FM_MC, FM_levels):
    print ('Computing MultiClass FM labels')
    for FM_key, FM_key_MC in zip(keys_FM, keys_FM_MC):
        conditions = []
        labels = []
        for num_levels in range(len(FM_levels)-1):
            conditions.append((df[FM_key] > FM_levels[num_levels]) &\
                              (df[FM_key] <= FM_levels[num_levels+1]))
            labels.append(num_levels)
            #labels.append('FM%02d'%(num_levels))

        df[FM_key_MC] = np.select(conditions, labels, default=labels[0])
        df = df.astype({FM_key_MC: 'int16'})

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
    #plt.show()
    
    
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
