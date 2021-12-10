#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from datetime import datetime
import sys

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
