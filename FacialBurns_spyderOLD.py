# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:41:52 2021

@author: wille
"""

#%%

# import relevant modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # for plots
from pprint import pprint # for pretty printing
from os import chdir, getcwd # to set the working directory DELETE AT END
os.chdir(r'C:\Users\wille\OneDrive\Documenten\UGent\CAED\Github')
os.getcwd()

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
outliers??

AT END
- all paths relative
- check: names, comments, structure
- delete unused modules, set on same line with comma



# choose k value, recommended 10, (not sure about leave one out), sample must be split evenly (divisible by 77 (7 or 11))
# structure (levels and sublevels, use lines and numbers) and comment code
# report time to compute each step

# INFO
#Models
# 1. random forest
# 2.
# 3.

# R² selfesteem - HADS: 4%
# R² selfesteem - TBSA: -5%
# R² selfesteem - RUM: 2%


#THINGS THAT CAN VARY
#- which model (3 models)
#- # of folds in inner loop
#- # of folds in outer loop


Scoring metric: explained variance or mean squared error (most used)
3 diff algorithms, each 3 diff models (diff features, diff hyperparameter tuning)

# Info for others
#random_state is set to 42 so code can be reproduced


#%%

# =============================================================================
#                     Preprocessing
# =============================================================================

# load in the data set 'Facial Burns'
burns_df = pd.read_csv(r'C:\Users\wille\OneDrive\Documenten\UGent\CAED\Github\Dataset\FacialBurns_all.csv') # DELETE AT END
#burns_df = pd.read_csv(r'Dataset\FacialBurns_all.csv') 
burns_df.head() # look at first 5 rows
burns_df.info() # check variable types (correct), check null values (none present)

# recode sex DUMMY? necessary for linear models but not for random forest 



#------SEPARATE data set into response variable and feature variables--------#

X = burns_df.drop('Selfesteem', axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X_list = list(X.columns) # Saving feature names for later use
X = np.array(X) # Convert to numpy array
y = np.array(burns_df['Selfesteem']) # Convert to numpy array

X2 = burns_df.drop(['Selfesteem', 'Sex'], axis=1)  # Remove the labels from the features, all features minus target (selfesteem), axis 1 refers to the columns
X2_list = list(X2.columns) # Saving feature names for later use
X2 = np.array(X2) # Convert to numpy array



X[:5] # check first 5 rows
y[:5]

#train/test split for randomized search CHECK AT ENDDDDD
X_train, X_test, y_train, y_test = train_test_split(X, y)

# =============================================================================
#                      Exploring and visualizing the data
# =============================================================================

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

# SD value per column and for each gender
burns_df.std()
burns_df.groupby('Sex').std()



# Correlation matrix (heatmap)
corr = burns_df.corr()
sns.set(font_scale=2)
corr_heatmap= sns.heatmap(corr, 
                          xticklabels=corr.columns.values, yticklabels=corr.columns.values, 
                          annot = True, annot_kws={"size": 22})
corr_heatmap.set_title('Correlation Matrix of Facial Burns Data',fontsize = 25)
plt.xticks(rotation=0)
plt.show()


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
cv_inner = KFold(n_splits=3, shuffle=True, random_state=42) # shuffle to rule out any biases in the order of how data was collected
cv_inner = KFold(n_splits=3, shuffle=False, random_state=42) # shuffle to rule out any biases in the order of how data was collected
CHECK RANDOM STATE AND SHUFFLE reproduce code

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
cv_outer = KFold(n_splits=10, shuffle=True, random_state=42) 
# execute the nested cross-validation
generalization_error = cross_val_score(grid_search_forest, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='neg_mean_squared_error')
generalization_error = cross_val_score(grid_search_forest, X=X2, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='neg_mean_squared_error')
#generalization_error = cross_val_score(rf_random, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='neg_mean_squared_error') TAKES HELLA LONG
generalization_error = cross_val_score(grid_search_forest, X=X, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='explained_variance')
generalization_error = cross_val_score(grid_search_forest, X=X2, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='explained_variance')
generalization_error = cross_val_score(grid_search_forest, X=X3, y=y, cv=cv_outer, n_jobs=-1,verbose=3, scoring='explained_variance')


print('test')

#scoring='neg_mean_squared_error'
# returns array of scores of the estimator for each run of the (outer) cross validation.
#N_jobs: number of jobs to run in parallel. None = 1. -1 means use all processors
# report performance
print('Error mean: %.3f (%.3f)' % (np.mean(generalization_error), np.std(generalization_error))) # the closer to 0 the better, higher than 1 is bad

#NMSE
# -25, SE ~ hads, rum, age, TBSA, sex
# -25, SE ~ hads, rum, age, TBSA

# Explained variance
# 0.124 with sex
# 0.163 without sex





# Predict certain values?

# =============================================================================
# Second pipeline for second model
# =============================================================================



# =============================================================================
# Third pipeline for Third model
# =============================================================================





