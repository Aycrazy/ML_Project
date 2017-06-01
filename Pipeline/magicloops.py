
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from imblearn.under_sampling import RandomUnderSampler
from processing import *
import random
from scipy import optimize
import time
from upload_and_vizualize import *


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


'''
The following functions were modified from the code of rayidghani (Github ID), https://github.com/rayidghani/magicloops/blob/master/magicloops.py:
define_clfs_params, generate_binary_at_k, evaluate_at_k, plot_precision_recall_n, clf_loop
'''

## CONFIG DATA ##
START_DATE= '2000/01/01'
END_DATE = '2016/12/31'
fac_id = 'PGM_SYS_ID'

VIOLATION = 'violation'
INSPECTION = 'inspection'
STACKTEST = 'stacktest'
TITLEV = 'titlev'
FORMALACT = 'formalact'
INFORMALACT = 'informalact'


def define_clfs_params(grid_size):
    '''
    Choose the grid size
    Input: desired grid size
    Output: dictionary of base classifiers and dictionary of different classifier specifications
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
  

    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    
    tiny_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1]},
    'SGD': { 'loss': ['hinge','log'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.001,0.1,1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,25,50],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']}
            }
    
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'tiny'):
        return clfs, tiny_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0



def generate_binary_at_k(y_scores, cutoff_value, k):
    '''
    Calculate whether predictions were accurate at threshold 100-k
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_value else 0 for x in y_scores[:cutoff_index]]
    return test_predictions_binary, cutoff_index



def evaluate_at_k(y_true, y_scores, cutoff_value, k):
    '''
    Return precision, recall, f1 scores 
    '''
    preds_at_k, cutoff_index = generate_binary_at_k(y_scores, cutoff_value, k)
    y_true = y_true[:cutoff_index]
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f_score = f1_score(y_true, preds_at_k)
    return (precision, recall, f_score)



def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Make a precision recall graph
    Input:
        - y_true: true y values
        - y_prob: predicted y probabilities
        - model_name: model/classifier
    Output: prints a precision recall graph
    ''' 
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()



NOTEBOOK = 1

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, train_test_year, specification):
    '''
    Run different models with different classifiers
    Input:
    - models_to_run: a list of models to run
    - clfs: base classifiers
    - X_train, X_test, y_train, y_test: divided data
    - specification: whether data were standardized, imbalanced sampled
    Output: a dataframe of evaluating the different classifiers
    '''
    results_df =  pd.DataFrame(columns=('Model','Classifier', 'Parameters', 'AUC-ROC', 'Accuracy', 'Prec@5', 'Prec@10', 'Prec@20',
                                       'Rec@5', 'Rec@10','Rec@20', 'F@5', 'F@10', 'F@20'))

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print (models_to_run[index] + specification)
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)

                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, np.array(y_test)), reverse=True))
                
                accuracy = clf.score(X_test, y_test)
                roc = roc_auc_score(y_test, y_pred_probs)
                p5, r5, f5 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, .5, 5.0)
                p10, r10, f10 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted,.5, 10.0)
                p20, r20, f20 = evaluate_at_k(y_test_sorted,y_pred_probs_sorted, .5, 20.0)
                
                results_df.loc[len(results_df)] = [train_test_year + models_to_run[index] + specification, clf, p, roc, accuracy,                                               
                                                   p5, p10, p20, r5, r10, r20, f5, f10, f20]

                if NOTEBOOK == 1:
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
            except IndexError as e:
                print ('Error:',e)
                continue
    return results_df

def train_test_dates(year_lst):
    dictionaries = []
    for i in range(1, len(year_lst)):
        subdict ={}
        subdict['test'] = str(year_lst[i])
        sublist = []
        for j in range(0,i):      
            theyears = year_lst[j:i]
            if len(theyears) == 1:
                sub = (str(theyears[0]),str(theyears[0]))
                sublist.append(sub)
            else:
                year_start = theyears[0]
                year_end = theyears[-1]
                sub = (str(year_start), str(year_end))
                sublist.append(sub)
            subdict['train'] = sublist
        dictionaries.append(subdict)
    return dictionaries

def find_best_classifier_by_model(result_df, eval_method):
    '''
    Find the best classifier by each model
    Input: 
        - result_df: pandas dataframe of evaluation values
        - eval_method: evaluation method (accuracy, recall, etc.) to check
    Output: pandas dataframe of best classifier by each model
    '''
    
    tracker = {}
    index_list = []
    
    for index, row in result_df.iterrows():
        model = row['Model'][:23] 
        score = row[eval_method]
        if model not in tracker:
            tracker[model] = [(score, index)]
        else:
            if score > tracker[model][0][0]:
                tracker[model] = [(score, index)]
            if score == tracker[model][0][0]:
                tracker[model].append((score, index))
                
    for value in tracker.values():
        for tup in value: 
            index_list.append(tup[1])
    
    result_df = result_df.loc[index_list]
    
    return result_df


#CONTINUOUS = ['PENALTY_AMOUNT']

def transform(X_train, X_test):
    '''
    Standardize continuous variable values
    Input: X_train and X_test dataframes
    Output: Standardized dataframes
    '''
    
    for variable in ['PENALTY_AMOUNT']:
        
        scalar = preprocessing.RobustScaler()
        X_train[variable] = scalar.fit_transform(X_train[variable])
        
        #Since scalar object was fitted on X_train, X_test will be transformed on the same scale
        X_test[variable] = scalar.transform(X_test[variable])
    
    return X_train, X_test

def train_test_dates(year_lst):
    dictionaries = []
    for i in range(1, len(year_lst)):
        subdict ={}
        subdict['test'] = str(year_lst[i])
        sublist = []
        for j in range(0,i):      
            theyears = year_lst[j:i]
            if len(theyears) == 1:
                sub = (str(theyears[0]),str(theyears[0]))
                sublist.append(sub)
            else:
                year_start = theyears[0]
                year_end = theyears[-1]
                sub = (str(year_start), str(year_end))
                sublist.append(sub)
            subdict['train'] = sublist
        dictionaries.append(subdict)
    return dictionaries

def run_loops(year_list, models_to_run, clfs, grid):
    '''
    Input: 
        - year_list: list of dictionaries of test and train years
        - models_to_run: list of models to run
        - clfs: dictionary of classifiers
        - grid: dictionary of classifiers and parameter specifications
    Output: 
        - pandas dataframe of evaluation values per model/classifier
        - also prints time it took to run
    '''
    start_time = time.time()
    final_df = pd.DataFrame()
    
    master_feature = generate_features('2000','2016')
    master_label = generate_label('2000','2016')
    merged = pd.merge(master_feature.dropna(), master_label, how='inner', left_on=['PGM_SYS_ID','HPV_DAYZERO_DATE_year'], right_on = ['PGM_SYS_ID','ACTUAL_END_DATE_year'])
    
    
    for each_set in year_list:
        
        #Get train and test sets
        start_date = each_set['train'][0][0]
        end_date = each_set['train'][0][1]
        test_date = each_set['test']
        YEAR = 'HPV_DAYZERO_DATE_year'

        training = merged[merged[YEAR]>= start_date]
        training = training[training[YEAR] <= end_date]
        testing = merged[merged[YEAR] == test_date]

        (training.shape)

        X_train = training.iloc[:,2:].drop('Outcome', axis = 1)
        
        y_train = training[['Outcome']]

        X_test = testing.iloc[:,2:].drop('Outcome', axis = 1)
        y_test = testing[['Outcome']]
    
        #run once standardized (continuous variables) and once with undersampling + standarization
        
        specification = '_Standardized'
        train_test_year = 'TR:' + str(start_date) + '-' + str(end_date) + '&TS:' + str(test_date) + '_'
        # standardize
        X_train_st, X_test_st = transform(X_train, X_test)  #Need to hardcode continuous variables
        results_df_1 = clf_loop(models_to_run, clfs, grid, X_train_st, X_test_st, y_train, y_test, train_test_year, specification)

        final_df = final_df.append(results_df_1)

    print('Took ', (time.time() - start_time), ' seconds to run models')
    return final_df


