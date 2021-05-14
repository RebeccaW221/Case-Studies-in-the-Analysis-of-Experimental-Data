# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:57:38 2021

@author: wille
"""

# import relevant modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # for plots
from pprint import pprint # for pretty printing
import os
os.getcwd()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer  # feature selector
from sklearn.feature_extraction.text import TfidfTransformer # feature selector
from sklearn.linear_model import SGDClassifier   # classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

# load in the data set 'Facial Burns'
burns_df = pd.read_csv(r'C:\Users\wille\OneDrive\Documenten\UGent\CAED\Github\Dataset\FacialBurns_all.csv') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
burns_df.head() # look at first 5 rows


# =============================================================================
#  SEPARATE data set into response variable and feature variables
# =============================================================================
X = burns_df.drop('Selfesteem', axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X_list = list(X.columns) # Saving feature names for later use
X = np.array(X) # Convert to numpy array
y = np.array(burns_df['Selfesteem']) # Convert to numpy array

X[:5] # check first 5 rows
y[:5]


#train/test split for randomized search
X_train, X_test, y_train, y_test = train_test_split(X, y)

# EXPLORATORY, compare grid search with randomized search
rf = RandomForestRegressor(random_state = 42)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Random search
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid) # 2 * 12 * 2 * 3 * 3 * 10 = 4320 settings

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

pprint(random_grid)




# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)
rf_random.best_params_
{'n_estimators': 1800,
 'min_samples_split': 2,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 90,
 'bootstrap': True}




def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42) 
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test) #model without hyper parameter tuning
#Model Performance
#Average Error: 3.7580 degrees.
#Accuracy = 88.84%.
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test) # model with hyper parameting tuning (randomized search)
Model Performance
Average Error: 3.1167 degrees.
Accuracy = 90.61%.
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# grid search
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110, 120],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [1, 2, 3],
    'n_estimators': [200, 300, 400], #[1800, 1900, 2000, 2100] # 1 * 4 * 2 * 3 * 3 * 4 = 288
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)


# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 90,
 'max_features': 'sqrt',
 'min_samples_leaf': 4,
 'min_samples_split': 3,
 'n_estimators': 1800}
best_grid = grid_search.best_estimator_
best_grid
grid_accuracy = evaluate(best_grid, X_test, y_test)
Model Performance
Average Error: 3.1034 degrees.
Accuracy = 90.67%.
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - random_accuracy) / random_accuracy))



###### Feature selection

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
FI = rf.feature_importances_
print(FI)
plt.barh(X_list, rf.feature_importances_) # most to least important: HADS, TBSA, RUM, AGE, SEX
X_list


X2 = burns_df.drop(['Selfesteem', 'Sex'], axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X2_list = list(X2.columns) # Saving feature names for later use
X2 = np.array(X2) # Convert to numpy array
y = np.array(burns_df['Selfesteem']) # Convert to numpy array

X2[:5] # check first 5 rows
y[:5]
X2_list

burns_df.head()
#train/test split for randomized search
X2_train, X2_test, y_train, y_test = train_test_split(X2, y)

rf = RandomForestRegressor()
rf.fit(X2_train, y_train)
FI = rf.feature_importances_
print(FI)
plt.barh(X2_list, rf.feature_importances_) # most to least important: HADS, AGE, TBSA, RUM, SEX



