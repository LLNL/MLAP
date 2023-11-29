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

#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier, MLPRegressor

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, classification_report

# []
'''
Get year, month, day, hour from timestamp
'''
def split_timestamp (timestamp):
    hour = timestamp.split('_')[-1]
    yr_mo_day = timestamp.split('_')[0].split('-')
    year, month, day = yr_mo_day[0], yr_mo_day[1], yr_mo_day[2]
    
    #'========================================================================='
    return year, month, day, hour


# []
'''
Get Time and Region Info on which analysis is to be performed
'''
def get_time_region_info (analysis_data_defined, json_content_analyze):
    time_region_info = dict()
    for analysis_data_type in analysis_data_defined:
        print('Extracting Time and Region Info for {}'.format(analysis_data_type))
        if (analysis_data_type == 'SJSU' or \
            analysis_data_type == 'HRRR' or \
            analysis_data_type == 'RRM'):
            time_region_info[analysis_data_type] = json_content_analyze[analysis_data_type]
        else:
            raise ValueError('Invalid analysis data type: {}. \
                        \nValid types are: "SJSU", "HRRR", and "RRM"'.format(analysis_data_type))
            
    for analysis_data_type in time_region_info.keys():
        print('\nTime and Region Info for {}:'.format(analysis_data_type))
        for count_ref_time, item_ref_time in enumerate(time_region_info[analysis_data_type]):
            print ('... Reference Time {}: {}'.format(count_ref_time, item_ref_time['RefTime']))

            for count_regions, (x_clip, y_clip) in enumerate(\
                zip (item_ref_time['regions_x_indices'], item_ref_time['regions_y_indices'])):
                print ('... ... Region {}:, x_clip: {}, y_clip: {}'.format(\
                                count_regions, x_clip, y_clip))
            
    #'========================================================================='    
    return time_region_info

# []
'''
Get history time stamps corresponding to a reference time
'''
def get_history_time_stamps (ref_timestamp, max_history_to_consider, history_interval):
    hist_timestamps = []
    
    hist = history_interval
    while hist <= max_history_to_consider:
        ref_time = datetime.fromisoformat(ref_timestamp)
        hist_time = ref_time - timedelta(hours = hist)
        hist_timestamp = hist_time.isoformat('_', 'hours')
        #print(hist, hist_time, hist_timestamp)
        hist_timestamps.append(hist_timestamp)
        hist += history_interval
        
    #'========================================================================='    
    return hist_timestamps


# []
'''
Get history time stamps for all the reference times of interest
'''
def get_history_time_stamps_all_data_types (time_region_info, max_history_to_consider, history_interval):
    for analysis_data_type in time_region_info.keys():
        #print('\nGetting history timestamps for {}:'.format(analysis_data_type))
        for count_ref_time, item_ref_time in enumerate(time_region_info[analysis_data_type]): 
            time_region_info[analysis_data_type][count_ref_time]['HistTime'] = \
                            get_history_time_stamps (item_ref_time['RefTime'], \
                                                     max_history_to_consider, history_interval)
    
    #'========================================================================='    
    return time_region_info
    
