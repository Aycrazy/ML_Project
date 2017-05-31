import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
import sklearn as sk 
import json 
import re
from multiprocessing import Pool
from functools import partial
from time import time
from sklearn import svm, ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import *
from sklearn.feature_selection import RFE
from sklearn.grid_search import ParameterGrid
from sklearn.svm import LinearSVC
from sklearn.metrics import *
import csv
from errno import EEXIST
from os import makedirs,path
from datetime import datetime as dr
from datetime import date
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from upload_and_vizualize import camel_to_snake
from datetime import datetime as dt
from datetime import date


### SHOULD BE WITHIN CLEANING.PY ###

## FROM PREPROCESSING ##

#interest_var = ['PGM_SYS_ID','ACTIVITY_ID','AGENCY_TYPE_DESC','STATE_CODE','AIR_LCON_CODE','COMP_DETERMINATION_UID','ENF_RESPONSE_POLICY_CODE','PROGRAM_CODES']
def replace_with_value(data_file, variables, values):
    '''
    '''
    for variable in variables:
        value = values[variables.index(variable)]
        data_file[variable] = data_file[variable].fillna(value)

def convert_to_datetime(series_row, date_format):
    if str(series_row) == 'nan':
        return float('nan')
    return dt.strptime(series_row, date_format)

def convert_to_year(series_row):
    if str(series_row) == 'NaT' or str(series_row)== 'nan':
        return float('nan')
    else:
        return str(series_row.year)

def to_date_time(df, date_format, date_col):
    #add datetime column
    df[date_col] = df[date_col].apply(convert_to_datetime, date_format=date_format)
    df[date_col+'_year'] = df[date_col].apply(convert_to_year)

    return df

def convert_to_month(series_row):
    if str(series_row) == 'NaT' or str(series_row)== 'nan':
        return float('nan')
    else:
        return str(series_row.month)

def get_month_year_col(df, date_column, date_format):
    df[date_column+'_datetime'] = df[date_column].apply(convert_to_datetime, date_format=date_format)
    df[date_column+'_month'] = df[date_column+'_datetime'].apply(convert_to_month)
    df[date_column+'_year'] = df[date_column+'_datetime'].apply(convert_to_year)
    return df

def filter_date(df, date_format, date_col, start=None, end=None):
    df = to_date_time(df, date_format, date_col)
    
    
    if start:
        timestart = dt.strptime(start,"%Y/%m/%d")
        #print(start)
        df = df[df[date_col] >= timestart ]
        #print(df.head())
    if end:
        timeend = dt.strptime(end,"%Y/%m/%d")
        #print(end)
        df = df[df[date_col] <= timeend ]
        #print(df.head())
    
    return df

def filter_col(df, fac_id, features, date_col):
    #filter needed
    df = df[[fac_id] + [date_col] + [date_col+'_year'] + features]
    return df




## FROM PREPROCESSING, MODIFIED ##
def add_dummy(df, variable_list, sep_char = None, drop_one=False, drop_original=False):
    '''
    Input: 
        - df: pandas dataframe
        - variable_list: a list of variables to dummitize
        - drop_one: whether to drop first dummy
        - drop_original: whether to drop original categorical variable
    Output: dataframe with tht dummy variables added
    '''
    for variable in variable_list:
        if sep_char:
            df_dummy = df[variable].str.get_dummies(sep=sep_char)
            df_dummy.columns = [variable+ '_' +str(col) for col in df_dummy.columns]

        else:
            df_dummy = pd.get_dummies(df[variable], drop_first=drop_one, prefix = variable)
        
        df = pd.concat([df, df_dummy], axis=1)
        if drop_original:
            df = df.drop(variable, 1)
    return (df, df_dummy.columns)


def aggr_dummy_cols(df, final_df, colnames, mode = None):
    for col in colnames:
        
        cross = pd.crosstab(df['id_+_date'], columns=df[col])
        
        if mode == 'cat':
            cross.columns = [cross.columns.name+ '_' +str(col) for col in cross.columns]
        
        elif mode == 'dum':
            cross = cross.drop(0, axis = 1)
            cross.columns = [cross.columns.name for col in cross.columns]
        
        else:
            cross.columns = [cross.columns.name for col in cross.columns]
            
        cross.columns.name = None
        cross.reset_index(inplace=True)
        
        
        if final_df.empty:
            final_df = final_df.append(cross)
        else:
            final_df = pd.merge(final_df, cross, how = 'left', on = 'id_+_date')
            
    return final_df

## FROM ULAB PIPELINE ##

def generate_continous_variable(data_file, variable_list):
    '''
    function that can take a categorical variable and create 
    binary variables from it
    '''
    for variable in variable_list:
        list_values = list(data_file.groupby(variable).groups.keys())
        for i,value in enumerate(list_values):
            data_file[variable] = data_file[variable].replace(value,i)

    return data_file