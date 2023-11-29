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
Get Time and Region Info on which analysis is to be performed
'''
def get_time_region_info (analysis_data_defined, json_content_analyze):
    time_region_info = dict()
    for analysis_data_elem in analysis_data_defined:
        print('Extracting Time and Region Info for {}'.format(analysis_data_elem))
        if (analysis_data_elem == 'SJSU' or \
            analysis_data_elem == 'HRRR' or \
            analysis_data_elem == 'RRM'):
            time_region_info[analysis_data_elem] = json_content_analyze[analysis_data_elem]
        else:
            raise ValueError('Invalid analysis data type: {}. \
                        \nValid types are: "SJSU", "HRRR", and "RRM"'.format(analysis_data_elem))
            
    for analysis_data_elem in time_region_info.keys():
        print('\nTime and Region Info for {}:'.format(analysis_data_elem))
        for count_ref_time, item_ref_time in enumerate(time_region_info[analysis_data_elem]):
            print ('... Reference Time {}: {}'.format(count_ref_time, item_ref_time['RefTime']))

            for count_regions, (x_clip, y_clip) in enumerate(\
                zip (item_ref_time['regions_x_indices'], item_ref_time['regions_y_indices'])):
                print ('... ... Region {}:, x_clip: {}, y_clip: {}'.format(\
                                count_regions, x_clip, y_clip))
            
    #'========================================================================='    
    return time_region_info
