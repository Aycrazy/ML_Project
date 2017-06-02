import pandas as pd
%matplotlib inline
%run Pipeline//upload_and_vizualize 
%run Pipeline//classify_and_evaluate 
%run Pipeline//aux_1
%run Pipeline//ULAB_ML_Pipeline
%run Pipeline//processing
%run Pipeline//cleaning
%run Pipeline//magicloops.py

years = list(range(2000,2016))  #the years are inclusive (the last year will be the ultimate test year and not included in model generation)

year_list = train_test_dates(years) #calling the function that creates our train test data

models_to_run = ['RF', 'LR', 'ET','AB','GB','NB','DT','SVM','KNN']

#models with base parameters
clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)}
#The small grid seemed most appropriate for our purposes since we end up with only 4000 rows of data

small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
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

#runs the small grid with the years dictionary declared above, our selected models, and standard model parameters
result_df = run_loops(year_list, models_to_run, clfs, small_grid)

#view results dataframe in csv
result_df.to_csv('results.csv')

year_list =[{'test': '2016',
  'train': [('2000', '2015'),
   ('2001', '2015'),
   ('2002', '2015'),
   ('2003', '2015'),
   ('2004', '2015'),
   ('2005', '2015'),
   ('2006', '2015'),
   ('2007', '2015'),
   ('2008', '2015'),
   ('2009', '2015'),
   ('2010', '2015'),
   ('2011', '2015'),
   ('2012', '2015'),
   ('2013', '2015'),
   ('2014', '2015'),
   ('2015', '2015')]}]

# we wanted to run a version of our best performing model (Gradient Boosted Decision Trees) with the most complete set of
#train and test set years
result_df = run_loops(year_list, ['GB'], clfs, small_grid)

#view results dataframe in csv
result_df.to_csv('results_final.csv')

##fin!
