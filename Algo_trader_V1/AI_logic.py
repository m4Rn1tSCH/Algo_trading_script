# -*- coding: utf-8 -*-
"""
Created on 24 4/24/2020 5:25 PM 2020

@author: bill-
"""
#packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np

import os
import matplotlib.pyplot as plt


from sklearn.feature_selection import SelectKBest , chi2, f_classif, RFE, RFECV


from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
'''
This module contains the AI/ML packages to take preprocessed data, find informative features
and facilitate their prediction
'''

'''
            Setting up a pipeline
'''
#SelectKBest picks features based on their f-value to
#find the features that can optimally predict the labels
#F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on
#the f-values between features & labels;
#Chi2: for regression tasks; requires non-neg values
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression

#Create pipeline with feature selector and regressor
#replace with gradient boosted at this point or regressor
#TODO
##turn this into a function
def set_pipeline_KNN():
    '''
    Pipeline 6 - SelectKBest and K Nearest Neighbor
    ##########
    Pipeline 6; 2020-04-27 11:00:27
    {'clf__n_neighbors': 7, 'feature_selection__k': 3}
    Best accuracy with parameters: 0.5928202115158637
    '''
    #Create pipeline with feature selector and classifier
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', KNeighborsClassifier())])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERs NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
        'clf__n_neighbors':[2, 3, 4, 5, 6, 7, 8]}

    #Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline 6; {dt.today()}")
    print(grid_search.fit(X_train, y_train).best_params_)
    print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
def set_pipeline_Reg():
    '''
    Pipeline 4 - Logistic Regression and Support Vector Kernel
    '''
    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', SVR(kernel = 'linear'))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERs NEED TO HAVE THE SAME LENGTH
    #C regularization parameter that is applied to all terms
    #to push down their individual impact and reduce overfitting
    #Epsilon tube around actual values; threshold beyond which regularization is applied
    #the more features picked the more prone the model is to overfitting
    #stricter C and e to counteract
    params = {
        'feature_selection__k':[4, 6, 7, 8, 9],
        'reg__C':[1.0, 0.1, 0.01, 0.001],
        'reg__epsilon':[0.30, 0.25, 0.15, 0.10],
        }

    #Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid = params)

    #best combination of feature selector and the regressor
    #grid_search.best_params_
    #best score
    #grid_search.best_score_
    #Fit it to the data and print the best value combination
    print(f"Pipeline 4; {dt.today()}")
    print(grid_search.fit(X_train, y_train).best_params_)
    print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
'''
        Application of Recursive Feature Extraction - Cross Validation
        IMPORTANT
        Accuracy: for classification problems
        Mean Squared Error(MSE); Root Mean Squared Error(RSME); R2 Score: for regression
TEST RESULTS

'''
#TODO
##turn RFE into a function
#Use the Cross-Validation function of the RFE modul
#accuracy describes the number of correct classifications
#LOGISTIC REGRESSION
est_logreg = LogisticRegression(max_iter = 2000)
#SGD REGRESSOR
est_sgd = SGDRegressor(loss='squared_loss',
                            penalty='l1',
                            alpha=0.001,
                            l1_ratio=0.15,
                            fit_intercept=True,
                            max_iter=1000,
                            tol=0.001,
                            shuffle=True,
                            verbose=0,
                            epsilon=0.1,
                            random_state=None,
                            learning_rate='constant',
                            eta0=0.01,
                            power_t=0.25,
                            early_stopping=False,
                            validation_fraction=0.1,
                            n_iter_no_change=5,
                            warm_start=False,
                            average=False)
#SUPPORT VECTOR REGRESSOR
est_svr = SVR(kernel = 'linear',
                  C = 1.0,
                  epsilon = 0.01)

#WORKS WITH LOGREG(pick r2), SGDRregressor(r2;rmse)
rfecv = RFECV(estimator = est_logreg,
              step = 2,
#cross_calidation determines if clustering scorers can be used or regression based!
#needs to be aligned with estimator
              cv = None,
              scoring = 'completeness_score')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

#plot number of features VS. cross-validation scores
plt.figure(figsize = (10,7))
plt.suptitle(f"{RFECV.get_params}")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()