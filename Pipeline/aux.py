
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