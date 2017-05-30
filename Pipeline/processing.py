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

def general_read_file(df_dict, start_date, end_date):
    #df = pd.DataFrame()
    
    violation = []
    inspection = []
    stacktest = []
    titlev = []
    formalact = []
    informalact = []
    
    for table, var in df_dict.items():
        date_col = var['date_col']
        features = var['interest_var']
        DATE_FORMAT = var['date_format']
        
        if table == 'violation':
            violation = read_file(VIOLATION)
            violation = violation[violation['ENF_RESPONSE_POLICY_CODE'] != 'FRV']
            
            violation = filter_date(violation, DATE_FORMAT, date_col, start=start_date, end=end_date)
            violation = filter_col(violation, fac_id, features, date_col)
        
        elif table == 'inspection':
            inspection = read_file(INSPECTION)
            
            inspection = filter_date(inspection, DATE_FORMAT, date_col, start=start_date, end=end_date)
            inspection = filter_col(inspection, fac_id, features, date_col)
        
        elif table == 'titlev':
            titlev = read_file(TITLEV)
            
            titlev = filter_date(titlev, DATE_FORMAT, date_col, start=start_date, end=end_date)
            titlev = filter_col(titlev, fac_id, features, date_col)
        
        elif table == 'stacktest':
            stacktest = read_file(STACKTEST)
            
            stacktest = filter_date(stacktest, DATE_FORMAT, date_col, start=start_date, end=end_date)
            stacktest = filter_col(stacktest, fac_id, features, date_col)
        
        elif table == 'formalact':
            formalact = read_file(FORMALACT)
            
            formalact = filter_date(formalact, DATE_FORMAT, date_col, start=start_date, end=end_date)
            formalact = filter_col(formalact, fac_id, features, date_col)
        
        elif table == 'informalact':
            informalact = read_file(INFORMALACT)
            
            informalact = filter_date(informalact, DATE_FORMAT, date_col, start=start_date, end=end_date)
            informalact = filter_col(informalact, fac_id, features, date_col)
        
    return (violation, inspection, titlev, stacktest, formalact, informalact)
    



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
    final_df = aggr_dummy_cols(df, final_df, bim_var, 'bim')
    
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
    #NEED TO ADD THE SUM of the amount!
    
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