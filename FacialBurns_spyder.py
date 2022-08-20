# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:41:52 2021

@author: wille
"""

#%% import relevant modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # for plots
#from pprint import pprint # for pretty printing
import sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

#%% ===========================================================================
#                     Preprocessing
# =============================================================================
# load in the data set 'Facial Burns'
burns_df = pd.read_csv(r'Dataset\FacialBurns_all.csv') 
burns_df.head() # look at first 5 rows
burns_df.info() # check variable types, check null values: no null values present

# Separate data set into response variable and feature variables
X = burns_df.drop('Selfesteem', axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X_list = list(X.columns) # Saving feature names for later use
X = np.array(X) # Convert to numpy array
y = np.array(burns_df['Selfesteem']) # Convert to numpy array
y_list = ['Selfesteem']

#X[:5] # check first 5 rows
#y[:5]
#%% Build preprocessing pipeline (DOESN'T WORK)
preprocessing_pipe = ColumnTransformer( #define column transformer
                    transformers=[
                        ('ss', StandardScaler(), [0, 1, 2, 3]), #apply standard scaling to numerical features (HADS, Age, TBSA, RUM)
                        ('ohe', OneHotEncoder(), [4])],) # apply one hot encoding to categorical feature (sex)

#%% ===========================================================================
#     Model 1: Linear Regression
# =============================================================================
# Define pipeline
lr_pipeline = Pipeline([
    ('lr', LinearRegression()) # Define model.  # default scorer is r²
])

#Pipeline with preprocessor, literally no effect so not used
#lr_pipeline = Pipeline(steps=[('preprocessor', preprocessing_pipe),
#                      ('lr', LinearRegression())])


# CV
cv = KFold(n_splits=7, shuffle = True, random_state=42) # divisible by 77 

#scores = cross_val_score(lr_pipeline, X=X, y=y, cv=cv, scoring='r2') #default scorer is r2
scores = cross_validate(lr_pipeline, X=X, y=y, cv=cv, scoring='r2', return_train_score=True)
#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#0.40 accuracy with a standard deviation of 0.14 SCALING, one hot encoding. all var


#look at underfitting/overfitting
trsc = scores['train_score']
tesc = scores['test_score']
print(trsc.mean()) #0.48
print(tesc.mean()) #0.39. no overfitting but possible underfitting

#%% Exploratory: checking each predictor separately -> sex and age poor accuracy
X = np.array(burns_df['HADS']) #0.35 accuracy with a standard deviation of 0.09
X = np.array(burns_df['TBSA']) # 0.16 accuracy with a standard deviation of 0.11
X = np.array(burns_df['RUM']) #0.18 accuracy with a standard deviation of 0.18
X = np.array(burns_df['Age']) #-0.02 accuracy with a standard deviation of 0.01
X = np.array(burns_df['Sex']) # 0.01 accuracy with a standard deviation of 0.00
X =X.reshape(-1, 1)

# checking model with HADS, TBSA and RUM -> no effect of leaving age and sex out
X = np.array(burns_df[['HADS', 'RUM', 'TBSA']]) #0.38 accuracy with a standard deviation of 0.14
y = np.array(burns_df['Selfesteem'])


lr_pipeline = Pipeline([
    ('scale', StandardScaler()), # Apply standard scaling to get optimized results. doesn't do much for linear model but good practice anyway
    ('lr', LinearRegression()) # Define model.  # default scorer is r²
])

cv = KFold(n_splits=3, shuffle = True, random_state=42) 

scores = cross_val_score(lr_pipeline, X=X, y=y, cv=cv)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#%% ===========================================================================
# Model 2: Ridge Regression
# =============================================================================
rdg_pipeline = Pipeline([
    ('scale', StandardScaler()), # Apply standard scaling to get optimized results. doesn't do much for linear model but good practice anyway
    ('rdg', Ridge()) # Define model.  # default scorer is r²
])

cv = KFold(n_splits=7, shuffle = True, random_state=42) # divisible by 77 

#scores = cross_val_score(rdg_pipeline, X=X, y=y, cv=cv, scoring = 'explained_variance') #default scoring ='r2')
scores = cross_validate(rdg_pipeline, X=X, y=y, cv=cv, return_train_score=True) #default scoring ='r2')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#0.39 accuracy with a standard deviation of 0.28 R2 
#0.42 accuracy with a standard deviation of 0.29 explained var


trsc = scores['train_score']
tesc = scores['test_score']
print(trsc.mean()) #0.48
print(tesc.mean()) #0.39. no overfitting but possible underfitting, no diff with lin: no collin in data
#%% ===========================================================================
#     Model 3: Random Forest Regressor
# =============================================================================
#Pipeline
pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                      ('rfr', RandomForestRegressor())])

#pipeline = Pipeline(steps=[('preprocessor', preprocessing_pipe), #Doesn't work here, not sure why
#                      ('rfr', RandomForestRegressor())])

#rfr.get_params().keys() # shows possible parameters

#Define parameter grid
par_grid = { # final grid after multiple searches: total combinations = 9
    'rfr__max_depth': [1, 2, 3], #best param: 2
    'rfr__n_estimators': [200, 300, 400],} # best param: 300 (weird because so low?)


#par_grid = { # Base grid: total combinations = 324 (Warning: computation time very long)
#    'rfr__bootstrap': [True, False], #default = True 
#    'rfr__max_depth': [40, 50, 60], #nr of splits of each tree
#    'rfr__max_features': ['auto', 'sqrt'], #default = auto (uses all features)
#    'rfr__min_samples_leaf': [3, 4, 5],
#    'rfr__min_samples_split': [2, 3, 4],
#    'rfr__n_estimators': [600, 800, 1000],} #how many trees to build, the higher the better

#par_grid = { # parameter grid to illustrate effect of high max_depth
#    'rfr__max_depth': [40, 50, 60], #best param: 2
#    'rfr__n_estimators': [200, 300, 400],} # best param: 300 (weird because so low?)


#%%==== Nested cross validation ========

# configure inner loop of the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=42) # shuffle to rule out any biases in the order of how data was collected

# define search grid 
grid_search_forest = GridSearchCV(estimator=pipeline, param_grid=par_grid, n_jobs=1, cv=cv_inner, verbose=2, scoring='explained_variance')
#Refit=True: refit an estimator using the best found parameters on the whole (training?) dataset. default = true

# configure outer loop of the cross-validation procedure
cv_outer = KFold(n_splits=7, shuffle=True, random_state=42) 


#%% fit to get the best parameters
grid_search_forest.fit(X,y) 
bestpar = grid_search_forest.best_params_
print(bestpar)
#%% execute the nested cross-validation
# cross_val score
#scores = cross_val_score(grid_search_forest, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3,
#                                       scoring='explained_variance')
#
## report performance
#print('Error mean: %.3f (%.3f)' % (np.mean(scores), np.std(scores))) # the closer to 0 the better, higher than 1 is bad
#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# cross_validate: to access training scores and estimator
scores_val = cross_validate(grid_search_forest, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3,
                                      scoring='explained_variance', return_train_score=True,) #return_estimator = True
print('test')
#%% Report training and testing scores (cross_validate)
train = scores_val['train_score']
test = scores_val['test_score']
print('Average training score: ', train.mean()) # 0.8 something
print('Average test score: ', test.mean()) #0.3 or 0.4 something (for first param grid) OVERFITTING

#%%   Feature selection using random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
FI = rf.feature_importances_
print(FI)
plt.barh(X_list, rf.feature_importances_) # most to least important: HADS, AGE, TBSA, RUM, SEX


#%%


#second paramgrid: bootstrap = false: even more overfitting: 1 vs 0.04 wtf 
# limiting max depth tree [5, 10, 15], bootstrap = true, no effect, 0.86 vs 0.29
# max depth: 2, n_estimators 300: 52 vs 32, less overfitting but possible more underfitting, wel fuck me


#scoring='neg_mean_squared_error'
# returns array of scores of the estimator for each run of the (outer) cross validation.


#NMSE
# -25, SE ~ hads, rum, age, TBSA, sex
# -25, SE ~ hads, rum, age, TBSA

# Explained variance
# 0.124 with sex
# 0.163 without sex

# Error mean: 0.307 (0.372) PREPROCESSER

#Results Eplained var
#Error mean: 0.308 (0.371) WITHOUT PREPROCESSOR, makes no difference wtf
#0.29 accuracy with a standard deviation of 0.37    
#0.32 accuracy with a standard deviation of 0.35 LOWER CAUSE LESS OVERFITTING?








