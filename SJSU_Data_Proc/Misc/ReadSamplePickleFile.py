#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:49:08 2023

@author: jha3
"""
# In[]
import os
import sys
#import wrf
import numpy as np
import math
import xarray as xr
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import gridspec
import json
import pickle
from datetime import date, datetime, timedelta, time

from timeit import default_timer as timer

plt.style.use('seaborn-white')

# In[]
sim_start_time = timer()

# In[]
# Open pickled data
pickle_file_dir = '/Users/jha3/LLNL/LLNL_Research/03_Wildfire/Wildfire_LDRD_SI/SJSU/Sample_Data_2023_03_14'
pickle_file_for_1hr = 'wrf_results_2020-06-25_01.pkl'
pickle_file_name = os.path.join(pickle_file_dir, pickle_file_for_1hr)
with open(pickle_file_name, 'rb') as pickle_file_handle:
    pickled_data_read = pickle.load(pickle_file_handle)
    
# In[]
sim_end_time = timer()
print('Total computing time: {} s'.format(sim_end_time - sim_start_time))