import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from upload_and_vizualize import camel_to_snake
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
import seaborn as sn
from sklearn import tree

def split_data_cust(df, use_seed=None):
    '''
    Given a pandas datframe sorted by column to have the first column be
    the outcome column, and the following columns to be potential features
    split the fame into training sets and testing sets
    '''
    if use_seed:
        np.random.seed(seed=20)
    test_set = np.random.uniform(0,1,len(df)) > .75
    X_train = df[test_set==False].ix[:,1:]
    X_test = df[test_set==True].ix[:,1:]
    y_train = df[test_set==False].ix[:,1]
    y_test = df[test_set==True].ix[:,1]
    return X_train,X_test,y_train,y_test

def process_categorical(df,cat_col_list):
    '''
    Given a pandas dataframe. This function will process continuous variables
    and transform them into categorical variables
    Inputs:
        df = pandas dataframe
        cat_col_list = list of columns you want categorical variables for
    '''

    cat_col_num_list = [loc for loc,c_name in enumerate(list(X.columns)) if c_name in cat_col_list]
    X = np.array(df)
    labelencoder_X = LabelEncoder()
    for cat_col in cat_col_num_list:
        X[:,cat_col] = labelencoder_X.fit_transform(X[:,cat_col])
    return X

def preprocess_by_mean(df):
    '''
    Given a pandas dataframe, this function will fillna with the mean
    '''
    df = df.fillna(df.mean())
    return df


def preprocess_imputer(df,by_strategy):
    '''
    Given a pandas dataframe and a imputation method,
    this will impute values for NaN values with values according the strategy
    (ie 'median')
    '''
    imputer = Imputer(missing_values='NaN', strategy = by_strategy, axis = 0)
    imputer = imputer.fit(df)
    df= imputer.transform(df)
    return pd.DataFrame(df).reset_index(drop=True)

def cross_vectors(df, var1, var2):
    '''
    Given a pandas dataframe and variable names
    create a crosstable dataframe
    '''
    return pd.crosstab(df[var1], df[var2])

def split_data(X,y,size):
    '''
    Given two dataframes ouptut a training and testing set given size
    '''

    #Output:X_train, X_test, y_train, y_test 
    return train_test_split(X, y, test_size = size, random_state = 0)


def model_logistic(X_train, y_train, X_test):

    '''
    With training and testing data and the data's features and label, select the best
    features with recursive feature elimination method, then
    fit a logistic regression model and return predicted values on the test data
    and a list of the best features used.
    '''
    
    model = LogisticRegression()
    rfe = RFE(model)
    rfe = rfe.fit(X_train, y_train)
    predicted = rfe.predict(X_test)
    best_features = rfe.get_support(indices=True)
    return predicted, best_features

def do_learning(X_training, Y_training, X_test, Y_test, reference_dic, model_class):

    '''
    credit: Juan Arroyo-Miranda & Dani Alcala

    With training and testing data select the best
    features with recursive feature elimination method, then
    fit a classifier and return a tuple containing the predicted values on the test data
    and a list of the best features used.
    '''
    
    model = model_class
    # Recursive Feature Elimination
    rfe = RFE(model)
    rfe = rfe.fit(X_training, Y_training)
    
    best_features = rfe.get_support(indices=True)

    best_features_names = [reference_dic[i] for i in best_features]

    predicted = rfe.predict(X_test)
    expected = Y_test

    accuracy = accuracy_score(expected, predicted)
    return (expected, predicted, best_features_names, accuracy)


from sklearn.metrics import accuracy_score 

#currently borrowed from Hector
def accuracy(observed, predicted):
    '''
    Takes:
    predicted, a list with floats or integers with predicted values
    observed, a list with floats or integers with observed values 
    Calculates the accuracy of the predicted values.
    '''
    return accuracy_score(observed, predicted)

def plot_confusion_matrix(data, col_name, labels, model_name):
    '''
    Given a pandas dataframe with a confusion confusion_matrix
    and a list of axis lables plot the results
    '''
    sn.set(font_scale=1.4)#for label size

    xticks =  labels
    yticks =  labels
    ax = plt.axes()
    sn.heatmap(data, annot=True,annot_kws={"size": 16}, linewidths=.5, xticklabels = xticks,  
              yticklabels = yticks, fmt = '')
    ax.set_title('Confusion Matrix for' + ' ' + model_name + col_name)

def create_confusion_matrix(df_y_test,df_y_pred, col_name, labels, model_name):
    '''
    Given an actual set of y values (based on the test set) and a predicted set of y values 
    (based on the test set), a column name, and a column name this function will produce
    a confusion matrix and then plot that matrix, utilizing the plot_confusion_matrix function.
    '''

    actual = pd.Series(df_y_test[col_name], name = 'Actual')
    predicted = pd.Series(df_y_pred[col_name], name='Predicted')
    array = confusion_matrix(actual, predicted)
    df_cxm = pd.DataFrame(array, range(2), range(2))
    plot_confusion_matrix(df_cxm,col_name, labels, model_name)

