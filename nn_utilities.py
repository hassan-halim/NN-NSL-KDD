# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:10:15 2019

@author: hhalim
"""

import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing

#encode text values to  dummy variables
#example [1,0,0], [0,1,0], [0,0,1] encode the data for red, green blue
# we can simply use OneHotEncoder object to do this
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for d in dummies.columns:
        dummy_name = f"{name}-{d}"
        df[dummy_name] = dummies[d]
    df.drop(name, axis=1,inplace = True)

def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(v) else 0 for x in l]
        name2 = f"{name}-{tv}"
        df[name2] = l

#encode text values too indexes, [1], [2], [3] for red, reen, blue
        
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

#encode a numeric column as zscore (standardization)
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
        
    if sd is None:
        sd = df[name].std()
    
    df[name] = (df[name] - mean) / sd

#convert all missing values to the median
def mmissing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

#convert all missing value in a column to default value
def missing_default(df, name, default_val):
    df[name].fillna(default_val)

#convert dataframe to the input tensorflow needs (X,Y)
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)
# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)

# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
        * (normalized_high - normalized_low) + normalized_low