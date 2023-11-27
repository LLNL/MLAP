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