# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:41:52 2021

@author: wille
"""

#%%

# import relevant modules ORGANIZE AT END, remove redundant and set everything on same line with comma
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # for plots
from pprint import pprint # for pretty printing

import sklearn
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


#%% TO DO
Choose second model
Feature selection
figure out how to interpret scores
use linear regression as base model? Expect linear relationship?
check correlation between variables, maybe ridge regression?


# report mean of models skills scores + standard deviation or standard error of this mean
# choose k value, recommended 10, (not sure about leave one out), sample must be split evenly (divisible by 77 (7 or 11))
# structure (levels and sublevels, use lines and numbers) and comment code
# report time to compute each step

# INFO
#Models
# 1. random forest
# 2.
#3.

# R² selfesteem - HADS: 4%
# R² selfesteem - TBSA: -5%
# R² selfesteem - RUM: 2%

Correlations
Selfesteem - HADS: -0.613
Selfesteem - TBSA: -0.3845
Selfesteem - RUM: -0.467
Selfesteem - Age: -0.0098
Selfesteem - Sex: -0.119

# correlation matrix
burns_df.corr()

plt.matshow(burns_df.corr())
plt.show()

corrM = burns_df.corr()
corrM.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps
corrM # doesn't work okikkkk

#Common examples of optimization algorithms include grid search and random search
#It is common to use k=10 for the outer loop and a smaller value of k for the inner loop, such as k=3 or k=5.

#THINGS THAT CAN VARY
#- which model (3 models)
#- # of folds in inner loop
#- # of folds in outer loop

#- feature selector
# feature importance? Random forest inherently does this

Scoring metric: explained variance or mean squared error (most used)
3 diff algorithms, each 3 diff models (diff features, diff hyperparameter tuning)

#THINGS THAT DONT VARY
#- grid search hyperparameter tuning


#%%

# =============================================================================
#                     Preprocessing
# =============================================================================


# =============================================================================
# Feature selection ????
# outliers??


# load in the data set 'Facial Burns'
burns_df = pd.read_csv(r'C:\Users\wille\OneDrive\Documenten\UGent\CAED\Dataset\FacialBurns_all.csv') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
burns_df.head() # look at first 5 rows
burns_df.info() # All variables have the correct scale, no null values present

# Get a series containing minimum value of each column
minValuesObj = burns_df.min()
print('minimum value in each column : ')
print(minValuesObj)

# Maximum value of each column
maxValuesObj = burns_df.max()
print('maximum value in each column : ')
print(maxValuesObj)

# Average value of each column
avgValuesObj = burns_df.mean()
print('average value in each column : ')
print(avgValuesObj)


burns_df.std()
burns_df.groupby('Sex').std()
burns_df.GroupBy.std('Sex')
burns_df.groupby('Sex').agg(np.std)
# recode sex (optional)


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

# =============================================================================
#                      Visualizing the data
# =============================================================================

# Selfesteem & HADS
burns_df.plot(kind='scatter', x='Selfesteem', y='HADS')
plt.show() # small neg correlation

x = burns_df['Selfesteem']
y = burns_df['HADS']
x.corr(y)   # -0.613

#Selfesteem & TBSA
burns_df.plot(kind='scatter', x='Selfesteem', y='TBSA')
plt.show() # small neg correlation

x = burns_df['Selfesteem']
y = burns_df['TBSA']
x.corr(y)   # -0.3845

#Selfesteem & RUM
burns_df.plot(kind='scatter', x='Selfesteem', y='RUM')
plt.show() # small neg correlation

x = burns_df['Selfesteem']
y = burns_df['RUM']
x.corr(y)   # -0.467

x = burns_df['Selfesteem']
y = burns_df['Age']
x.corr(y)   # -0.0098

x = burns_df['Selfesteem']
y = burns_df['Sex']
x.corr(y)   # -0.1189

#Selfesteem & HADS, TBSA and RUM

# AGE as continuous variable
fig=plt.figure(figsize=(20,8),facecolor='white')
gs=fig.add_gridspec(1,1)
ax=[None]
ax[0]=fig.add_subplot(gs[0,0])

sns.kdeplot(data = burns_df['Age'],ax=ax[0],shade=True, color='gold', alpha=0.6,zorder=3,linewidth=5,edgecolor='black')

sns.kdeplot(data = burns_df['Age'],shade=True, color='lightblue', alpha=0.6,zorder=3,linewidth=5)



# AGE AS categorical variable
def age_cat(age): #min age = 18, max age is 66
    
    if age >= 10 and age < 20:
        return '10-20'
    if age >= 20 and age < 30:
        return '20-30'
    if age >= 30 and age < 40:
        return '30-40'
    if age >= 40 and age < 50:
        return '40-50'
    if age >= 50 and age < 60:
        return '50-60'
    if age >= 60 and age < 70:
        return '60-70'


burns_df['Age_cat']=burns_df.Age.apply(lambda x: age_cat(x))
burns_df=burns_df.dropna()
burns_df=burns_df.sort_values('Age')


fig=plt.figure(figsize=(20,8),facecolor='white')
gs=fig.add_gridspec(1,1)

ax=[None]
ax[0]=fig.add_subplot(gs[0,0])

ax[0].text(-0.5, 63, 
         'Participant"s age', 
         fontsize=25, 
         fontweight='bold', 
         fontfamily='monospace'
        )

#ax[0].text(-0.5, 60, 
#         'Most of the passengers were between 30 to 40 years old', 
#         fontsize=15, 
#         fontweight='light', 
#         fontfamily='monospace'
#        )

ax[0].grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


colormap = ["lightblue" for _ in range(9)]
colormap[1] = "blue"

sns.countplot(data=burns_df,x='Age_cat',ax=ax[0],palette=colormap,alpha=1,zorder=2)


for direction in ['top','right','left']:
    ax[0].spines[direction].set_visible(False)
    
    
ax[0].set_xlabel('Age of passengers',fontsize=12, fontweight='bold')

ax[0].set_yticklabels([])
ax[0].tick_params(axis='y',length=0)
ax[0].set_ylabel('',)






# SEX





# =============================================================================
#    Define pipeline
# =============================================================================
pipeline = Pipeline([ # scaler: check if robust scaler is better (is robust to outliers)
    ('scale', StandardScaler()), # Apply standard scaling to get optimized results (necessary as scales have different ranges).  # rescales between -1 and 1. training set is fitted and transformed, test set is transformed. No data leakage between training and test set
    ('rfr', RandomForestRegressor()) # Define model.  # default scorer is r²

])


#pipeline = Pipeline([
#    ('vect', CountVectorizer()), # feature selector
#    ('tfidf', TfidfTransformer()), # feature selector
#    ('clf', SGDClassifier()), # model
#])

# =============================================================================
#    Define parameter grid 
# =============================================================================
par_grid = {
    'rfr__bootstrap': [True],
    'rfr__max_depth': [80, 90, 100],
#    'rfr__max_features': ['auto', 'sqrt'],
#    'rfr__min_samples_leaf': [3, 4, 5],
#    'rfr__min_samples_split': [2, 3, 4],
    'rfr__n_estimators': [200, 300, 400], #[1700, 1800, 1900], # 1 * 4 * 2 * 3 * 3 * 4 = 288
}

print('test')

#param_grid = {
#    'vect__max_df': (0.5, 0.75, 1.0),
#    # 'vect__max_features': (None, 5000, 10000, 50000),
#    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#    # 'tfidf__use_idf': (True, False),
#    # 'tfidf__norm': ('l1', 'l2'),
#    'clf__max_iter': (20,),
#    'clf__alpha': (0.00001, 0.000001),
#    'clf__penalty': ('l2', 'elasticnet'),
#    # 'clf__max_iter': (10, 50, 80),
#}

# =============================================================================
#      Somethingh
# =============================================================================
rfr = RandomForestRegressor()

rfr.get_params().keys()



# configure the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=1) # shuffle to rule out any biases in the order of how data was collected

# define search
grid_search_forest = GridSearchCV(estimator=pipeline, param_grid=par_grid, n_jobs=1, cv=cv_inner, verbose=2, scoring='neg_mean_squared_error')
#Refit=True: refit an estimator using the best found parameters on the whole (training?) dataset. default = true
grid_search_forest.fit(X,y) # not sure why necessary if refit = True?

#rf_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 100, cv = cv_inner, verbose=1, n_jobs = -1) # TAKES HELLA LONG
#rf_random.fit(X,y)
#bestpar = rf_random.best_params_
#print(bestpar)
#bestest = rf_random.best_estimator_
#print(bestest)

bestest = grid_search_forest.best_estimator_
print(bestest)
bestpar = grid_search_forest.best_params_
print(bestpar)

bestest = grid_search_forest.best_estimator_
print(bestest)

bestscore = grid_search_forest.best_score_
print(bestscore)

# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1) 
# execute the nested cross-validation
generalization_error = cross_val_score(grid_search_forest, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='neg_mean_squared_error')
#generalization_error = cross_val_score(rf_random, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='neg_mean_squared_error') TAKES HELLA LONG



print('test')

#scoring='neg_mean_squared_error'
# returns array of scores of the estimator for each run of the (outer) cross validation.
#N_jobs: number of jobs to run in parallel. None = 1. -1 means use all processors
# report performance
print('Error mean: %.3f (%.3f)' % (np.mean(generalization_error), np.std(generalization_error))) # the closer to 0 the better, higher than 1 is bad







# Predict certain values?

# =============================================================================
# Second pipeline for second model
# =============================================================================



# =============================================================================
# Third pipeline for Third model
# =============================================================================





