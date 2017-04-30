import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def read(file_name):
    '''
    This was to experiment with uploading a csv using python csv library.
    Needless to say it is pretty slow.
    '''
    df = pd.DataFrame()

    pattern = r'[(?!=.)]([a-z]*)'
    file_type = re.findall(pattern, file_name)[0]

    assert file_type == 'csv'
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for index,row in enumerate(reader):
            df = pd.concat([df,pd.DataFrame(row).transpose()])
   
    return df.reset_index(drop=True)

def read_file(file_name):
    '''
    Given a '.csv' or '.xls' file, this function will return a pandas dataframe
    '''

    pattern = r'[(?!=.)]([a-z]*)'
    file_type = re.findall(pattern, file_name)[0]
    
    if file_type == 'csv':
        df = pd.read_csv(file_name)
      
    elif file_type == 'xls':
        df = pd.read_excel(file_name)
    
    return df

def list_describe(df,optional_string=None):
    '''
    Given a pandas dataframe this function creates individual describe tables 
    for every column in a pandas dataframe.
    Inputs:
        df - pandas dataframe
        optional_string - optional string used to collect columns with similar names
        to try and cap possible redundancies in columns
    '''

    opt_columns = []
    all_cols = {}
    all_cols_caps = {}
    for index,column in enumerate(df.columns):
        if optional_string:
            if column.startswith(optional_string):
                opt_columns.append(column)

        print(df[str(column)].describe().to_frame(),'\n')
        if not index:
            continue
        all_cols[index] = camel_to_snake(column)
        all_cols_caps[index] = column

    return all_cols, all_cols_caps, opt_columns


def des_num_dep (df,column_name):
    '''
    Given a pandas data frame with cumulative sum and percentage of 
    dependents by category
    Input: df = pandas dataframe object
           column_name = name of column of interest
    Returns: new df with descriptive stats
    '''
    df = df[column_name].value_counts().to_frame()
    df['cumsum'] = df[column_name].cumsum()
    total = df[column_name].sum()
    df['percentage'] = (df['cumsum'] / total)*100 

    return df


def bin_feature(df, column_name, given_range):
    '''
    credit - borrows key idea from Hector Salvador: https://github.com/hectorsalvador/ML_for_Public_Policy/blob/master/Homework%202/pipeline/features.py 
    Given a pandas dataframe, a column_name, and a bins range.
    Categorize column entries into discrete bins.
    Input: df = pandas dataframe object
           column_name = name of column of interest
           given_range = a premade range of bins
    '''

    bins = pd.DataFrame()
    discrete_feature = 'discrete_'+column_name
    bins[discrete_feature] = pd.cut(df[column_name], bins=given_range)

    df = pd.concat([df, bins], axis = 1)

    return df,discrete_feature

def create_plots(df,column_name,choice):
    '''
    create plots of a given table column (or feature), and color choice (by number)
    Input: df = pandas dataframe object
           column_name = name of column of interest
    '''

    possible_colors = ['r','b','g','orange','purple']
    choice = choice%len(possible_colors)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    df[column_name].plot(kind = "hist", alpha = 0.2, bins = 20, color = possible_colors[choice], ax = ax1); 
    ax1.set_title(column_name);
    ax1.grid()
    df.boxplot(column=column_name, ax = ax2); ax2.set_title('Boxplot of '+column_name)

    plt.tight_layout()

def create_hist_box(df,cols_dict,ignore):
    '''
    Given a pandas dataframe this function utilizes create_plots
    to creat plots for all columns except 'ignore' columns
    Input: df = pandas dataframe object
           cols_dict = dictionary of columns and column numbers
    '''
    
    for index,col in enumerate(cols_dict.values()):
        if col in ignore:
            continue
        create_plots(df,col,index)

def create_area_graphs(df,feature, bins, outcome_feature, my_color):
    '''
    Given a pandas dataframe, a column, and an outcome feature the function creates
    a matplotlib line graph
    Input: df = pandas dataframe object
           feature = name of column of interest
    '''
    if bins:
        df,discrete_feature = bin_feature(df, feature, bins)
        df[[discrete_feature, outcome_feature]].groupby(discrete_feature).mean().plot(kind='area',rot=70, color = my_color)

    else:
        df[[feature, outcome_feature]].groupby(feature).mean().plot(kind='area',rot=70, color = my_color)


def special_plot(df, column_name,point_of_interest1, point_of_interest2):
    '''
    CREDIT: Heavily Inspired by Juan Arroyo-Miranda: https://github.com/ju-arroyom/Machine-Learning-Homeworks/blob/master/hw2/ML-PS2.ipynb
    
    Given a dataframe, column_name, and points of interest, we can take more granular look at
    our data.
    Inputs:
        df = pandas dataframe
        column_name = column of interest
        point_of_interest1 = data value that acts as an upper-bound for normal values
        point_of_interest2 = data value that acts as a lower-bound for extraordinary values
    Output:
        A set of income distribution histograms
    '''

    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2, figsize=(10,10))
    # Overall Income Distribution
    df.boxplot(column=column_name, ax = ax1); ax1.set_title('Overall '+column_name)
    # Up to 80th percentile
    df[column_name][(df[column_name]<=point_of_interest1)].plot(kind = "hist", alpha = 0.2, color ='blue', bins= 20, ax = ax2); 
    ax2.set_title(column_name+' Dist. up to 80th percentile');
    ax2.grid()

    # Monthly income above $point_of_interest1 and below $point_of_interest2
    df[column_name][(df[column_name]> point_of_interest1) & (df[column_name]<=1000000)].plot(kind = "hist", alpha = 0.2, color ='r', bins= 20, ax = ax3); 
    ax3.set_title(column_name+' Dist. > 80th percentile < $1,000,000');
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
    ax3.set_xlabel('Income in Sci Notation')
    ax3.grid()
    # Income distribution above $point_of_interest2
    df[column_name][(df[column_name]>point_of_interest2)].plot(kind = "hist", alpha = 0.2, color ='black', bins= 20, ax = ax4); 
    ax4.set_title(column_name+' Dist. above $1,000,000');
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
    ax4.set_xlabel('Income in Sci Notation')
    ax4.grid()
    plt.tight_layout()
   
