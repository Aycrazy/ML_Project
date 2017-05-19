
# coding: utf-8

# ## Creating a machine learning pipeline

# In[2]:

#Load libraries

import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
import sklearn as sk 
import json 
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


# ### Select data and features

# In[3]:



FEATURES = []
LABEL = ['First_period_enrolled']



# ### Assign global variables

# In[4]:

CLFS = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
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

GRID = { 
'RF':{'n_estimators': [1,10], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
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

N_FOLDS = 3

MODELS =['LR','NB','DT','RF']


# ### Load Functions

# In[5]:

#READ FILE - (RW) To handle most typical data format... should we include pickle?
def read_file(file_name):
	'''
	Given a 'xls', 'csv', or 'xlsx' file, read file into a pandas dataframe

	Input: (String) file name

	Output: (df) pandas dataframe 
	'''
	pattern = r'[(?!=.)]([a-z]*)'
	file_type = re.findall(pattern, file_name)[0]

	if file_type == 'csv':
		data_file = pd.read_csv(file_name)

	elif file_type in ['xls', 'xlsx']:
		data_file = pd.read_excel(file_name)

	return data_file

# Transform column name into a uniformed format 'flat' or 'snake_case' 
##(RW: So next time owen doesn't have to map it one by one - See below)
def transform_colnames (data, method):
    '''
    Transform column name to a uniform format
        - flat: turning 'Column_Names-123' to 'columnnames123'
        - cameltosnake: turning 'ColumnNames123' to 'column_names_123'
    '''
    def flat_case (colname):
        s = re.sub('[^A-Za-z0-9]+', '', colname)
        return s.lower()

    def cameltosnake_case(colname):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', colname)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    col_dict = {}

    for col in data.columns:
        c = ''
        if method == 'flat':
            c = flat_case(col)
        elif method == "cameltosnake":
            c = cameltosnake_case(col)
        else:
            return "method is not supported"
        
        col_dict[col] = c
        
    data.rename(columns = col_dict, inplace = True)


#DESCRIBE data_file 
def obtain_basic_statistics(data_file):
	'''
	Provide descriptive statistics and save them in
	a csv
	'''
	stats = data_file.describe().round(2)
	missing_data_file = data_file.iloc[0,:].apply(lambda x : len(data_file) - x) ## RW: what is this for?
	missing_data_file.name = "missing values"
	stats = stats.append(missing_data_file)
	stats.to_csv("descriptive_statistics.csv")


#Create folder to save graphs output (RW)
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def generate_histogram(data_file):
	'''
	Generate histogram graphs and save them in png files
	'''
	for field in data_file.describe().keys():
		#Determine number of bins based on number of different values
		unique_vals = len(data_file[field].value_counts())
		if unique_vals == 2:
			data_file.groupby(field).size().plot(kind='bar')
		elif unique_vals < 15:
			bins = unique_vals	
			data_file[field].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
		else:
			data_file[field].hist()

		hist_dir = "histogram" # RW: I added this so the graphs will be in its own folder instead of crowding
		mkdir_p(hist_dir)

		pylab.title("Distribution of {0}".format(field))
		pylab.xlabel(field)
		pylab.ylabel("Counts")
		pylab.savefig(corr_dir + "/histograms/" + field + ".png")
		pylab.close()


def generate_correlations(data_file):
	corr_dir = "corr_plot"
	mkdir_p(corr_dir)
       
	correlations = data_file.corr()
	correlations.to_csv(corr_dir + "/correlations.csv".format(corr_dir))

	for x in data_file.describe().keys():
		for y in [i for i in data_file.describe().keys() if i != x]:
			plt.scatter(data_file[x],data_file[y])
			plt.xlabel(x)
			plt.ylabel(y)
			plt.savefig(corr_dir"/"+x+"-"+y+"_correlation.png", bbox_inches = "tight")
			plt.close()


#PRE-PROCESS data_file
def replace_with_value(data_file, variables, values):
	'''
	'''
	for variable in variables:
		value = values[variables.index(variable)]
		data_file[variable] = data_file[variable].fillna(value)


def replace_with_mean(data_file, variables):
	'''
	Fill in missing values 
	'''
	for variable in variables:
		data_file[variable] = data_file[variable].fillna(data_file[variable].mean())


def replace_with_median(data_file, variables):
	'''
	Fill in missing values 
	'''
	for variable in variables:
		data_file[variable] = data_file[variable].fillna(data_file[variable].median())


def replace_with_weighted_mean(data_file, variable_interest, cols):
	l2 = itertools.combinations(cols,2)
	cols += l2 
	for variable in variable_interest:
		for comb in cols:
			value = data_filefile[variable].groupby(comb).transform("mean")
			data_file[variable+comb].fillna(value, inplace = True)


def replace_kneighbors(data_file, variable, cols):
	'''
	Function takes a data_fileframe and a variables that matches this criteria 
	as well as a list of columns to calibrate with and uses nearest neighbors 
	to impute the null values.
	'''
	# Transform data_file into test and train
	splitter_criteria = np.random.uniform(0, 1, len(data_file)) > 0.75
	train = data_file[splitter_criteria==False]
	test = data_file[splitter_criteria==True]
	
	## Create a k-nearest neighbor regressor classifier 
	imputer = KNeighborsRegressor(n_neighbors=1)

	# Split data_file into null and not null for given variable
	train_not_null = train[train[variable].isnull()==False]
	train_null = train[train[variable].isnull()==True]

	# Replace missing values
	imputer.fit(train_not_null[cols], train_not_null[variable])
	new_values = imputer.predict(train_null[cols])
	train_null[variable] = new_values

	# Combine Training data_file Back Together
	train = train_not_null.append(train_null)

	# Apply Nearest Neighbors to Validation data_file
	new_var_name = variable + 'Imputed'
	test[new_var_name] = imputer.predict(test[cols])
	test[variable] = np.where(test[variable].isnull(), test[new_var_name],
								test[variable])


	# Drop Imputation Column & Combine Test & Validation
	test.drop(new_var_name, axis=1, inplace=True)
	data_file = train.append(test)


def replace_missing_values(data_file,features,method,columns_interest = []):
	if method == "mean":
		replace_with_mean(data_file,features)
	if method == "median":
		replace_with_median(data_file,features)
	if method == "weighted_mean":
		replace_with_weighted_mean(data_file,features,columns_interest)
	if method == "knn":
		for feature in features:
			replace_kneighbors(data_file,feature,columns_interest)

	data_file.to_csv("clean_data_file.csv")


#FEATURE GENERATION 
def obtain_log(data_file, column):
    log_col = 'log_' + column
    data_file[log_col] = data_file[column].apply(lambda x: np.log(x + 1))

    return data_file


def generate_discrete_variable(data_file, criteria_dict):
	'''
	Write a sample function that can discretize a continuous variable 
	'''
	#Generate categorical values from continous variable 
	for column, criteria in criteria_dict.items():
		#The parameter list contains the labels
		parameter_list = []
		#The range set contains the values for the different parameters
		range_set = set()
		for parameter in range(len(criteria)):
			parameter_list.append(criteria[parameter][0])
			range_set.add(criteria[parameter][1])
			range_set.add(criteria[parameter][2])

		range_list = list(range_set)
		range_list.sort()

		#Generate categorical variables, the "right" option
		#creates set [a,b) to satisfy greater or equal restriction
		#for lower limit. 
		data_file[column+"cat"] = pd.cut(data_file[column],range_list,
						right = False, labels = parameter_list)
		
		#Drop rows that did not have a categorical match
		data_file = data_file[~data_file[column].isnull()]

	return data_file 


def generate_continous_variable(data_file, variable_list):
	'''
	function that can take a categorical variable and create 
	binary variables from it
	'''
	for variable in variable_list:
		list_values = list(data_file.groupby(variable).groups.keys())
		for i,value in enumerate(list_values):
			data_file[variable].replace(value,i)

	return data_file 


def create_percentiles(data_file, column, percentile):
    '''
    '''
    new_col = 'bins_' + str(column)

    data_file[new_col] = pd.qcut(data_file[column], q = percentile,  labels = list(range(percentile)))

    return data_file


#BUILD AND EVALUATE CLASSIFIERS 

def model_loop(models_to_run, data_file, label, n_folds): 
    
    features = list(data_file.columns)
    best_overall_model = None
    best_overall_area_under = 0
    best_overall_params = None

    # create dictionary results
    results = {}
    
    for i,clf in enumerate([CLFS[x] for x in models_to_run]):
        model = {} 
        current_model = models_to_run[i]
        parameter_values = GRID[current_model]
    
        for j,p in enumerate(ParameterGrid(parameter_values)):
            print(current_model+str(j))
            print(p)
            area_under_per_fold = []

            kf = KFold(len(data_file), n_folds=n_folds)
            
            for train_i, test_i in kf: 
                test = data_file[:len(test_i)]
                train = data_file[:len(train_i)]
            
                clf.set_params(**p)
                
                clf.fit(train[features], train[label])
                
                if hasattr(clf, 'predict_proba'):
                    y_pred_probs = clf.predict_proba(test[features])[:,1] 
                else:
                    y_pred_probs = clf.decision_function(test[features])
                
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test[label], y_pred_probs)
                precision = precision_curve[:-1]
                recall = recall_curve[:-1]
               	
               	try:
                    AUC = auc(recall, precision)
                    area_under_per_fold.append(AUC)
                except:
                	pass

                avg_model_auc = np.mean(area_under_per_fold)
                sd_model_auc = np.std(area_under_per_fold)

            model["Model"] = current_model
            model["Parameters"] = p
            model["Average AUC"] = avg_model_auc
            model["SD AUC"] = sd_model_auc
            results[current_model+str(j)] = model
            
        
        print(current_model+"-Finished")
        # find best model with params overall
        if avg_model_auc > best_overall_area_under:
            best_overall_area_under = avg_model_auc
            best_overall_model = current_model
            best_overall_params = p

    with open("results.json", 'w') as outfile:
        json.dump(results, outfile)

    return best_overall_model, best_overall_params

def evaluate_model(test_data, label, predicted_values):
    '''
    '''
    accuracy = accuracy_score(test_data[label], predicted_values) 
    precision = precision_score(test_data[label], predicted_values) 
    recall = recall_score(test_data[label], predicted_values) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(test_data[label], predicted_values) 

    return accuracy, precision, recall, f1


def plot_precision_recall(y_true, y_prob, model_name, model_params):
    
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]
    plt.clf()
    plt.plot(recall, precision, label='%s' % model_params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.savefig(model_name)
    plt.legend(loc="lower right")


#Helpers for final model 

def clean_data(data_file,features): 
	replace_missing_values(data_file,features,method = "mean")
	data_file = data_file.dropna()
	return data_file


def generate_features(data_file):
    '''
    Generate features for train and test sets.
    '''

    decile_features = ["age","MonthlyIncome"]
    for i in decile_features:
    	data_file = create_percentiles(data_file,i,10)

    return data_file


def evaluate_final_model(train,test,best_overall_model, best_overall_params):
    final_model = {}
    params = best_overall_params
    clf = CLFS[best_overall_model]  
    clf.set_params(**params)
    clf.fit(train[FEATURES], train[LABEL])
    predicted_values = clf.predict(test[FEATURES])

    accuracy, precision, recall, f1 = evaluate_model(test, LABEL, predicted_values)

    try: 
        y_pred_probs = clf.predict_proba(test[FEATURES])[:,1]
    except:
        y_pred_probs = None

    if y_pred_probs != None:
        plot_precision_recall(test[LABEL], y_pred_probs,best_overall_model,best_overall_params)

    final_model["params"] = best_overall_params
    final_model["model"] = best_overall_model
    final_model["accuracy"] = accuracy
    final_model["precision"] = precision
    final_model["recall"] = recall
    final_model["f1"] = f1
    
    with open("final_model.json", 'w') as outfile:
        json.dump(final_model, outfile)

    

