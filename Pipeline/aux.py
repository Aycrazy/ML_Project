
import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from upload_and_vizualize import camel_to_snake


def get_desired_features(dict, desired_features):
    features = [key for key,value in dict.items() if value in desired_features]
    return features

def check_na(df):
    df_lng = pd.melt(df)
    null_vars = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_vars)


def check_diff(df1,df2):
    diff = set(df1.index)-set(df2.index)
    return(len(diff))

def describe_extremes(df,column):
    column_str = str(column)
    very_high = df[column_str] > 1.000000
    print(df[column_str].debt_ratio.describe())
    print(len(df[column_str])/len(df))

def add_categoricals(dict, new_categories):
    count=1
    for cat in new_categories:
        dict[len(dict)+count] = cat

def update_keys(dict):
    return {key-1:value for key,value in dict.items()}


def add_dummies(df,need_dummies):
    new_cols = pd.DataFrame()
    for col in need_dummies:
        #print(np.array(df[col]))
        new_cols = pd.concat([new_cols.reset_index(drop=True),pd.get_dummies(df[col])],axis=1)

    df_w_dummies = pd.concat([df,new_cols],axis=1).reset_index(drop=True)
    return df_w_dummies