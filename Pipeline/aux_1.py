
import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from upload_and_vizualize import camel_to_snake
from datetime import datetime as dt
from datetime import date 

def get_desired_features(dict, desired_features):
    '''
    This function creates a list of desired column numbers that are representative of columns
    with desired features
    Inputs:
        dict = dictionary
        desired_features = list of desired features
    Output:
        list with column numbers
    '''

    features = [key for key,value in dict.items() if value in desired_features]
    return features

def check_na(df):
    '''
    Given a pandas dataframe this function will create a crosstab table for all columns and indicate which
    have null values
    '''
    df_lng = pd.melt(df)
    null_vars = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_vars)


def check_diff(df1,df2):
    '''
    Given two pandas dataframes, this function will give a count of the values with value differences
    '''

    diff = set(df1.index)-set(df2.index)
    return(len(diff))

def describe_extremes(df,column, greater_than):
    '''
    Given a pandas dataframe and a column of choice, this function will return information about
    extreme valuse above a certain normal value.
    Inputs:
        df = pandas dataframe
        column = column of choice
        greater_than = value that signifies the lower bound of an extereme value
    Outputs:
        prints distribution of column values greater than given lower bound and its percentage of
        the whole set
    '''

    column_str = str(column)
    very_high = df[column_str] > greater_than
    print(df[column_str].debt_ratio.describe())
    print(len(df[column_str])/len(df))

def add_categoricals(dict, new_categories):
    '''
    Given a dictionary of column features.  This will add new column features to that dictionary
    '''

    count=1
    for cat in new_categories:
        dict[len(dict)+count] = cat

def update_keys(dict):
    '''
    If an index is taken from a dataframe.  This will update the dictionary of columns accordingly
    '''
    return {key-1:value for key,value in dict.items()}


def add_dummies(df,need_dummies):
    '''
    Given a dataframe and columns that need to have dummy variables created from them. This will add
    dummy columns based on those values.
    Inputs:
        df = dataframe
        need_dummies = column names that need dummies
    Outputs:
        dataframe with new dummy variables
    '''
    new_cols = pd.DataFrame()
    for col in need_dummies:
        #print(np.array(df[col]))
        new_cols = pd.concat([new_cols.reset_index(drop=True),pd.get_dummies(df[col])],axis=1)

    df_w_dummies = pd.concat([df,new_cols],axis=1).reset_index(drop=True)
    return df_w_dummies

def convert_to_datetime(series_row, date_format):
    if str(series_row) == 'nan':
        return float('nan')
    return dt.strptime(series_row, date_format)

def convert_to_weekday(series_row, output):
    if str(series_row) == 'NaT' or str(series_row)== 'nan':
        return float('nan')
    if output == 'day_num':
        return date.weekday(series_row)
    output_dict = {"weekday":'%A', "month_name":'%B',"month_num":'%m',"year":"%Y"}
    return date.strftime(series_row,output_dict[output])

def convert_to_bool(df, column, conversion):
    return df[column].replace(conversion)

def add_date_cols(df, date_columns, date_format, date_types):
    '''
    date_types = list of datetime output indicators
    '''
    if 'month' in date_types:
        print(date_types)
        date_types = [v for v in date_types if v != 'month']
        date_types.extend(['month_name','month_num'])
    for date_column in date_columns: 
        df[date_column+'_datetime'] = df[date_column].apply(convert_to_datetime, date_format=date_format)
        for new_col in date_types:
            df[date_column+'_'+new_col] = df[date_column+'_datetime'].apply(convert_to_weekday, output=new_col)
            
def get_occupied_frame(df, date_columns, date_format, date_types, bool_param = None, occ_column = None, conversion= None):
    add_date_cols(df,date_columns, date_format, date_types)
    if bool_param != None:
        df[occ_column] = convert_to_bool(df,occ_column,conversion)
        return df[df[occ_column] == bool_param]