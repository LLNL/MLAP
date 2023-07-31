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
Get keys from the extracted data 
'''
def get_keys_from_extracted_data (df_extracted):
    keys_all = df_extracted.keys()
    
    # Keys for identity
    keys_identity = ['FM_time_ind', 'FM_ts', 'his_time_ind', 'grid_ind', 'j_ind', 'i_ind']

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
                     
    print('=========================================================================')
    return keys_identity, keys_FM, keys_U10, keys_V10, keys_UMag10, \
                            keys_T2, keys_RH, keys_PREC, keys_SW, keys_HGT


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

