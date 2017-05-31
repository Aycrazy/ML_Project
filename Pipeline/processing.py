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
from cleaning import *
from aux import *
from upload_and_vizualize import *
import aux_1

## CONFIG DATA ##
START_DATE= '2007/01/01'
END_DATE = '2016/12/31'
fac_id = 'PGM_SYS_ID'

VIOLATION = 'violation'
INSPECTION = 'inspection'
STACKTEST = 'stacktest'
TITLEV = 'titlev'
FORMALACT = 'formalact'
INFORMALACT = 'informalact'


### DON'T FORGET TO CHANGE THE DATA_FILE!!! ###
df_dict ={'violation': {'data_file': 'ICIS-AIR_downloads/ICIS-AIR_VIOLATION_HISTORY.csv',
                        'interest_var': ['AGENCY_TYPE_DESC','AIR_LCON_CODE','ENF_RESPONSE_POLICY_CODE','POLLUTANT_CODES','PROGRAM_CODES','HPV_RESOLVED_DATE'],
                         'date_col': 'HPV_DAYZERO_DATE',
                       'date_format':'%m-%d-%Y'},
          
           'inspection': {'data_file': 'ICIS-AIR_downloads/ICIS-AIR_FCES_PCES.csv',
                          'interest_var': ['STATE_EPA_FLAG','COMP_MONITOR_TYPE_CODE','PROGRAM_CODES'],
                          'date_col': 'ACTUAL_END_DATE',
                          'date_format':'%m-%d-%Y'},
          
           'stacktest': {'data_file':'ICIS-AIR_downloads/ICIS-AIR_STACK_TESTS.csv',
                         'interest_var':['COMP_MONITOR_TYPE_CODE','POLLUTANT_CODES','AIR_STACK_TEST_STATUS_CODE'],
                        'date_col': 'ACTUAL_END_DATE',
                        'date_format':'%m/%d/%Y'},
          
           'titlev':{'data_file': 'ICIS-AIR_downloads/ICIS-AIR_TITLEV_CERTS.csv',
                     'interest_var':['COMP_MONITOR_TYPE_CODE','FACILITY_RPT_DEVIATION_FLAG'],
                        'date_col': 'ACTUAL_END_DATE',
                    'date_format':'%m/%d/%Y'},
          
           'formalact':{'data_file': 'ICIS-AIR_downloads/ICIS-AIR_FORMAL_ACTIONS.csv',
                       'interest_var':['ENF_TYPE_CODE','PENALTY_AMOUNT'],
                        'date_col': 'SETTLEMENT_ENTERED_DATE',
                       'date_format':'%m/%d/%Y'},
          
           'informalact':{'data_file': 'ICIS-AIR_downloads/ICIS-AIR_INFORMAL_ACTIONS.csv',
                          'interest_var':['ENF_TYPE_CODE'],
                        'date_col': 'ACHIEVED_DATE',
                         'date_format':'%m/%d/%Y'}}

def general_read_file(df_dict, table_name, start_date, end_date):
    df = pd.DataFrame()
    
    table = df_dict[table_name]
    data_file = table['data_file']
    date_col = table['date_col']
    DATE_FORMAT = table['date_format']
    features = table['interest_var']
    
    df = read_file(data_file)
    
    if table == 'violation':
        df = violation[violation['ENF_RESPONSE_POLICY_CODE'] != 'FRV']
            
    df = filter_date(df, DATE_FORMAT, date_col, start=start_date, end=end_date)
    df = filter_col(df, fac_id, features, date_col)
        
    return df
    

def process_violation(violation_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    outcome = ['ENF_RESPONSE_POLICY_CODE']
    cat_var = ['AGENCY_TYPE_DESC', 'AIR_LCON_CODE']
    dum_var = ['PROGRAM_CODES', 'POLLUTANT_CODES']
    
    df = violation_df[violation_df['HPV_DAYZERO_DATE_year'] >= start_year]
    df = df[df['HPV_DAYZERO_DATE_year'] <= end_year]
    
    ## Replace NaN with 'None' (string) --> Making 'None' it's own category
    nan = df.columns[df.isnull().any()].tolist()
    values = ['None']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize 
    df, colnames_out = add_dummy(df, outcome, drop_original = True)
    df, colnames_dum = add_dummy(df, dum_var, sep_char = ' ')
    
    #return df
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.HPV_DAYZERO_DATE_year
            
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    final_df = aggr_dummy_cols(df, final_df, colnames_dum, 'dum')
    final_df = aggr_dummy_cols(df, final_df, colnames_out)
    
    '''re_separate = r'(.[^_]*)_(.*)'
    sep = lambda x: pd.Series([i for i in re.split(re_separate,x)])
    final_id_year = final_df['id_+_date'].apply(sep)
    final_df =pd.concat([final_id_year.rename(columns={1:'PGM_SYS_ID',2:'HPV_DAYZERO_DATE_year'}), final_df], axis=1)
    final_df.drop([0, 3, 'id_+_date'], axis = 1, inplace = True)
'''
    
    return final_df


def process_inspection(inspection_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    cat_var = ['STATE_EPA_FLAG','COMP_MONITOR_TYPE_CODE']
    dum_var = ['PROGRAM_CODES']
    
    df = inspection_df[inspection_df['ACTUAL_END_DATE_year'] >= start_year]
    df = df[df['ACTUAL_END_DATE_year'] <= end_year]
    
    ## Replace NaN with 'None' (string) --> Making 'None' it's own category
    nan = df.columns[df.isnull().any()].tolist()
    values = ['None']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize 
    (df, colnames_dum) = add_dummy(df, dum_var, sep_char = ',') 
    
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.ACTUAL_END_DATE_year
            
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    final_df = aggr_dummy_cols(df, final_df, colnames_dum, 'dum')
    
    return final_df


def process_titlev(titlev_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    cat_var = ['COMP_MONITOR_TYPE_CODE']
    bim_var = ['FACILITY_RPT_DEVIATION_FLAG']
    
    df = titlev_df[titlev_df['ACTUAL_END_DATE_year'] >= start_year]
    df = df[df['ACTUAL_END_DATE_year'] <= end_year]
    
    ## Replace NaN with 0 (string) --> THIS ONLY APPLIES TO THE BIM_VAR!!
    nan = df.columns[df.isnull().any()].tolist()
    values = ['N']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize 
    df = generate_continous_variable(df, bim_var)
    
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.ACTUAL_END_DATE_year
    
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    final_df = aggr_dummy_cols(df, final_df, bim_var, 'dum')
    
    return final_df


def process_stacktest(stacktest_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    #I HAVEN'T PUT IN POLLUTANT_CODES
    cat_var = ['AIR_STACK_TEST_STATUS_CODE', 'COMP_MONITOR_TYPE_CODE']
    
    df = stacktest_df[stacktest_df['ACTUAL_END_DATE_year'] >= start_year]
    df = df[df['ACTUAL_END_DATE_year'] <= end_year]
    
    ## Replace NaN with 0 (string) --> THIS ONLY APPLIES TO THE BIM_VAR!!
    nan = df.columns[df.isnull().any()].tolist()
    values = ['None']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize --> Not really needed here 
    
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.ACTUAL_END_DATE_year
    
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    
    return final_df

def process_formalact(formalact_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    cat_var = ['ENF_TYPE_CODE']
    cont_var = ['PENALTY_AMOUNT']
    
    df = formalact_df[formalact_df['SETTLEMENT_ENTERED_DATE_year'] >= start_year]
    df = df[df['SETTLEMENT_ENTERED_DATE_year'] <= end_year]
    
    ## Replace NaN with 0 (string) --> No need for this but ill leave it here
    nan = df.columns[df.isnull().any()].tolist()
    values = ['None']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize --> Not really needed here 
    
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.SETTLEMENT_ENTERED_DATE_year
    
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    
    sum_df = df.groupby('id_+_date')[cont_var[0]].sum().to_frame().reset_index()
    final_df = pd.merge(final_df, sum_df, on = 'id_+_date')
    
    return final_df


def process_informalact(informalact_df, start_year, end_year):
    final_df = pd.DataFrame()
    
    cat_var = ['ENF_TYPE_CODE']
    
    df = informalact_df[informalact_df['ACHIEVED_DATE_year'] >= start_year]
    df = df[df['ACHIEVED_DATE_year'] <= end_year]
    
    ## Replace NaN with 0 (string) --> No need for this but ill leave it here
    nan = df.columns[df.isnull().any()].tolist()
    values = ['None']*len(nan)
    replace_with_value(df, nan, values)
    
    ## Dummitize --> Not really needed here 
    
    df['id_+_date'] = df.PGM_SYS_ID +'_'+ df.ACHIEVED_DATE_year
    
    final_df = aggr_dummy_cols(df, final_df, cat_var, 'cat')
    
    return final_df

def process_noninspectHPV(violhist, fce, start_year, end_year):
    #removing FRVs
    violhist = violhist[violhist.ENF_RESPONSE_POLICY_CODE != 'FRV']
    violhist = violhist[violhist['HPV_DAYZERO_DATE_year'] >= start_year]
    violhist = violhist[violhist['HPV_DAYZERO_DATE_year'] <= end_year]
    
    #for fce
    fce = fce[fce['ACTUAL_END_DATE_year'] >= start_year]
    fce = fce[fce['ACTUAL_END_DATE_year'] <= end_year]
    
    #Steps before merge
    violhist['year'] = violhist['HPV_DAYZERO_DATE_year']
    merged_hpv_fce = pd.merge(violhist, fce, how='left', left_on=['PGM_SYS_ID', 'HPV_DAYZERO_DATE'], right_on=['PGM_SYS_ID','ACTUAL_END_DATE'])
    # Find violations that resulted from something other than an inspection 
    viol_by_other = merged_hpv_fce
    viol_by_other.COMP_MONITOR_TYPE_CODE.fillna(0, inplace=True)   #this will be NaN because it was not inspected
    viol_by_other = viol_by_other[viol_by_other['COMP_MONITOR_TYPE_CODE'] == 0]  #violations not resulting from inspections
    # Get the columns needed
    viol_by_other = viol_by_other[['PGM_SYS_ID','year']]
    viol_other_year = viol_by_other.groupby(['PGM_SYS_ID','year']).size().reset_index() # to get count of HPV by year
    violhist2 = violhist[['PGM_SYS_ID','year']]
    # Outer merge
    merged_viols = pd.merge(violhist2,viol_other_year, how = 'outer', on = ['PGM_SYS_ID','year'])
    merged_viols.rename(columns={'year': 'Year', 0:'NonInspection_HPV_Count'}, inplace=True)
    merged_viols.NonInspection_HPV_Count.fillna(0, inplace=True)
    merged_viols = merged_viols.dropna(axis=0)
    merged_viols['id_+_date'] = merged_viols.PGM_SYS_ID +'_'+ merged_viols.Year
    merged_viols.drop(['PGM_SYS_ID', 'Year'], axis = 1, inplace = True)
    
    return merged_viols

def change_to_zero(series_row):
    if type(series_row) != str :
        return 0
    else:
        return 1

def change_to_zero_float(series_row):
    if series_row >= 1:
        return 1
    else:
        return 0
    
def generate_label(start_year, end_year):
    
    violhist = read_file('ICIS-AIR_downloads/ICIS-AIR_VIOLATION_HISTORY.csv')
    fce = read_file('ICIS-AIR_downloads/ICIS-AIR_FCES_PCES.csv')

    date_types = ['year']
    date_format = df_dict['violation']['date_format']
    date_col = [df_dict['violation']['date_col']]
    #print(date_format,date_col)
    get_occupied_frame(violhist,date_col,date_format,date_types)

    date_col = [df_dict['inspection']['date_col']]
    get_occupied_frame(fce,date_col,date_format,date_types)

    fce = fce[['PGM_SYS_ID','STATE_EPA_FLAG','COMP_MONITOR_TYPE_CODE','PROGRAM_CODES','ACTUAL_END_DATE','ACTUAL_END_DATE_year']]
    violhist = violhist[['PGM_SYS_ID','AGENCY_TYPE_DESC','AIR_LCON_CODE','ENF_RESPONSE_POLICY_CODE','POLLUTANT_CODES','PROGRAM_CODES','HPV_DAYZERO_DATE','HPV_DAYZERO_DATE_year']]
    violhist = violhist[violhist.ENF_RESPONSE_POLICY_CODE == 'HPV']
    #removing FRVs
    #violhist = violhist[violhist.ENF_RESPONSE_POLICY_CODE != 'FRV']
    violhist = violhist[violhist['HPV_DAYZERO_DATE_year'] >= start_year]
    violhist = violhist[violhist['HPV_DAYZERO_DATE_year'] <= end_year]
    
    #for fce
    fce = fce[fce['ACTUAL_END_DATE_year'] >= start_year]
    fce = fce[fce['ACTUAL_END_DATE_year'] <= end_year]
    
    merged_hpv_fce = pd.merge(violhist, fce, how='right', left_on=['PGM_SYS_ID', 'HPV_DAYZERO_DATE'], right_on=['PGM_SYS_ID','ACTUAL_END_DATE'])
    
    #print(merged_hpv_fce.ENF_RESPONSE_POLICY_CODE.iloc[1].apply(change_to_zero))
    
    merged_hpv_fce['Outcome'] = merged_hpv_fce.ENF_RESPONSE_POLICY_CODE.apply(change_to_zero)
    #finding 0's
    
    #print(merged_hpv_fce.Outcome.value_counts())
    
    output = merged_hpv_fce.groupby(['PGM_SYS_ID', 'ACTUAL_END_DATE_year']).sum().reset_index()
    
    #print(output.Outcome.value_counts())
    
    output.Outcome = output.Outcome.apply(change_to_zero_float)
    
    return output

def generate_features(start_date, end_date):

    violation = general_read_file(df_dict, VIOLATION, START_DATE, END_DATE)
    inspection = general_read_file(df_dict, INSPECTION, START_DATE, END_DATE)
    titlev = general_read_file(df_dict, TITLEV, START_DATE, END_DATE)
    stacktest = general_read_file(df_dict, STACKTEST, START_DATE, END_DATE)
    formalact = general_read_file(df_dict, FORMALACT, START_DATE, END_DATE)
    informalact = general_read_file(df_dict, INFORMALACT, START_DATE, END_DATE)    
        
    violation_df = process_violation(violation, start_date, end_date)
    inspection_df = process_titlev(titlev, start_date, end_date)
    stacktest_df = process_stacktest(stacktest, start_date, end_date)
    formalact_df = process_formalact(formalact, start_date, end_date)
    informalact_df = process_informalact(informalact, start_date, end_date)
    noninspectHPV_df = process_noninspectHPV(violation, inspection, start_date, end_date)
    
    final_inspect_viol_df = pd.merge(inspection_df, violation_df, how = 'left', right_on = ["id_+_date"], left_on = ["id_+_date"])
    final_w_iv_stacktest_df = pd.merge(final_inspect_viol_df.dropna(), stacktest_df, how = 'left', right_on = ["id_+_date"], left_on = ["id_+_date"])
    final_w_ivs_formal_df = pd.merge(final_w_iv_stacktest_df, formalact_df, how = 'left', right_on = ["id_+_date"], left_on = ["id_+_date"])
    final_w_ivsf_informal_df = pd.merge(final_w_ivs_formal_df, informalact_df, how = 'left', right_on = ["id_+_date"], left_on = ["id_+_date"])
    final_w_ivsfi_noninspect_df = pd.merge(final_w_ivsf_informal_df, noninspectHPV_df, how = 'left', right_on = ["id_+_date"], left_on = ["id_+_date"])

    re_separate = r'(.[^_]*)_(.*)'
    sep = lambda x: pd.Series([i for i in re.split(re_separate,x)])
    final_id_year = final_w_ivsfi_noninspect_df['id_+_date'].apply(sep)
    final_df =pd.concat([final_id_year.rename(columns={1:'PGM_SYS_ID',2:'HPV_DAYZERO_DATE_year'}), final_w_ivsfi_noninspect_df], axis=1)
    final_df.drop([0, 3, 'id_+_date'], axis = 1, inplace = True)

    return final_df