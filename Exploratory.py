# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:57:38 2021

@author: wille
"""

# =============================================================================
# # import relevant modules
# =============================================================================
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # for plots
from pprint import pprint # for pretty printing
import os
#from os import chdir, getcwd # to set the working directory DELETE AT END
os.chdir(r'C:\Users\wille\OneDrive\Documenten\UGent\CAED\Github') # to set the working directory DELETE AT END
os.getcwd()


from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
#from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression



#%% ===========================================================================
#                     Preprocessing
# =============================================================================

# load in the data set 'Facial Burns'
burns_df = pd.read_csv(r'Dataset\FacialBurns_all.csv') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'


#------SEPARATE data set into response variable and feature variables--------#
X = burns_df.drop('Selfesteem', axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X_list = list(X.columns) # Saving feature names for later use
X = np.array(X) # Convert to numpy array
y = np.array(burns_df['Selfesteem']) # Convert to numpy array


#train/test split for feature selection and randomized search
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#%% ===========================================================================
# Feature selection using SelectKBest
# =============================================================================

#------feature selection using regression function----------------#
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))


#%%
# plot the scores
brplot=plt.bar(X_list, fs.scores_)
#brplot[0].set_color('lightblue')
#plt.title('Feature importance scores using a linear regression function', fontsize=30)
plt.tick_params(labelsize=25) # increase font size of ticks
plt.xlabel('Features', fontsize=30, fontweight ='bold', labelpad=25)
plt.ylabel('Scores', fontsize=30, fontweight ='bold',labelpad=25)

#%%
plt.close() # close plot (clean memory for next plot)


#------feature selection using mutual info regression----------------#
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%%
# plot the scores
brplot=plt.bar(X_list, fs.scores_)
#brplot[0].set_color('lightblue')
#plt.title('Feature importance scores using a linear regression function', fontsize=30)
plt.tick_params(labelsize=25) # increase font size of ticks
plt.xlabel('Features', fontsize=30, fontweight ='bold', labelpad=25)
plt.ylabel('Scores', fontsize=30, fontweight ='bold',labelpad=25)

#%%

##------feature selection using stepwise ----------------#

#%% ===========================================================================
# # EXPLORATORY: compare grid search with randomized search
# =============================================================================
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
#Model Performance
#Average Error: 3.1167 degrees.
#Accuracy = 90.61%.
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
#Model Performance
#Average Error: 3.1034 degrees.
#Accuracy = 90.67%.
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - random_accuracy) / random_accuracy))





#%% VIF: check multicollinearity
#pip install statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = burns_df[["HADS", "Age", "RUM", "TBSA", "Sex"]]
X = burns_df[["HADS", "Age", "RUM", "TBSA"]]

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
    
calc_vif(X)    # all variables below 10, Age: 5.9. Rum: 8.8. Sex: 9.4


