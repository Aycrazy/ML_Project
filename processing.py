from datetime import datetime as dt
import pandas as pd
import numpy as np


def convert_to_datetime(series_row, date_format):
    if str(series_row) == 'nan':
        return float('nan')
    return dt.strptime(series_row, date_format)
    

def convert_to_month(series_row):
    if str(series_row) == 'NaT' or str(series_row)== 'nan':
        return float('nan')
    else:
        return str(series_row.month)

def convert_to_year(series_row):
    if str(series_row) == 'NaT' or str(series_row)== 'nan':
        return float('nan')
    else:
        return str(series_row.year)
    
    
def get_month_year_col(df, date_column, date_format):
    df[date_column+'_datetime'] = df[date_column].apply(convert_to_datetime, date_format=date_format)
    df[date_column+'_month'] = df[date_column+'_datetime'].apply(convert_to_month)
    df[date_column+'_year'] = df[date_column+'_datetime'].apply(convert_to_year)
    return df


def date_and_filter(df, date_format, date_col, interest_var, fac_id):
    #add datetime column
    df = get_month_year_col(df, date_col, date_format)
    
    #filter year
    year = dt.strptime('2006/12/31',"%Y/%m/%d")
    df = df[df['SETTLEMENT_ENTERED_DATE_datetime'] > year ]
    
    #filter needed
    date_time = date_col + '_datetime'
    date_year = date_col + '_year'
    df = df[interest_var + [fac_id] + [date_time] + [date_year]]
    return df


def combine_col(df, year_col, col_list, file):
    for column in col_list:
        df[year_col + '_' + column] = df[year_col] + '_' + file + '_' + df[column]
    return df


def add_dummy(df, variable_list, drop_one=False, drop_original=False):
    '''
    Input: 
        - df: pandas dataframe
        - variable_list: a list of variables to dummitize
        - drop_one: whether to drop first dummy
        - drop_original: whether to drop original categorical variable
    Output: dataframe with tht dummy variables added
    '''
    for variable in variable_list:
            
        df_dummy = pd.get_dummies(df[variable], drop_first=drop_one, prefix = variable)
        df = pd.concat([df, df_dummy], axis=1)
        if drop_original:
            df = df.drop(variable, 1)
    return df


def add_count_cols(df, fac_id, year_col, list_var, file_name):
    # In order to have a dataframe ready for analysis, drop unnecessary columns BEFORE using this function
    
    # Add columns that combines year, file name, and variable values
    pre_dum = combine_col(df, year_col, list_var, file_name)
    cols = pre_dum.columns
    variable_list = list(cols[-(len(list_var)): ])

    # Dummify the variables into columns
    dummied = add_dummy(df, variable_list, drop_original=True)
    
    # Since some of the activities were done numerous times per year, sum by year (make into continuous variable)
    df = dummied.groupby(fac_id).sum()
    
    return df



def sum_by_year_in_cols(df, fac_id, year_list, continuous_var, year_var, file):
    for year in year_list:
        df[year + '_' + file + '_' + year_var] = [0] * len(df)
    
    for i, row in df.iterrows():
        for column in df.columns:
            if str(row[year_var]) in column:
                df.loc[i,column] = row[continuous_var]
    
    df = df.groupby([fac_id, year_var]).sum().reset_index()
    df = df.drop([year_var, continuous_var], 1)
    
    return df
        