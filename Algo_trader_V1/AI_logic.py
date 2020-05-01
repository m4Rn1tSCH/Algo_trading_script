# -*- coding: utf-8 -*-
"""
Created on 24 4/24/2020 5:25 PM 2020

@author: bill-
"""
# packages
import pandas as pd

pd.set_option('display.width', 1000)
from datetime import datetime as dt
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

'''
This module contains the AI/ML packages to take preprocessed data, find informative features
and facilitate their prediction
'''

'''
            Setting up the pipeline

SelectKBest picks features based on their f-value to
find the features that can optimally predict the labels
F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on
the f-values between features & labels;
Chi2: for regression tasks; requires non-neg values
other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
'''


# Create pipeline with feature selector and regressor

# TODO
# define/import X_train/y_train correctly
def set_pipeline_knn(X=X_train, y=y_train):
    """
    Pipeline - SelectKBest and K Nearest Neighbor
    """
    # Create pipeline with feature selector and classifier
    # replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('clf', KNeighborsClassifier())])

    # Create a parameter grid

    params = {
        'feature_selection__k': [1, 2, 3, 4, 5, 6, 7],
        'clf__n_neighbors': [2, 3, 4, 5, 6, 7, 8]}

    # Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid=params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline 6; {dt.today()}")
    print(grid_search.fit(X, y).best_params_)
    print(f"Best accuracy with parameters: {grid_search.best_score_}")

    return grid_search.best_score_

#TODO
#define X_train , y_train correctly
def set_pipeline_reg():
    '''
    Pipeline - Logistic Regression and Support Vector Kernel
    '''

    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('reg', SVR(kernel='linear'))
    ])

    # Create a parameter grid
    # C regularization parameter that is applied to all terms
    # to push down their individual impact and reduce overfitting

    # Epsilon tube around actual values; threshold beyond which regularization is applied
    # the more features picked the more prone the model is to overfitting
    # stricter C and e to counteract
    params = {
        'feature_selection__k': [4, 6, 7, 8, 9],
        'reg__C': [1.0, 0.1, 0.01, 0.001],
        'reg__epsilon': [0.30, 0.25, 0.15, 0.10],
    }

    # Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid=params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline 4; {dt.today()}")
    print(grid_search.fit(X_train, y_train).best_params_)
    print(f"Best accuracy with parameters: {grid_search.best_score_}")

    return grid_search.best_score_


# %%
def set_RFE_cross_val():
    '''
        Application of Recursive Feature Extraction - Cross Validation
        IMPORTANT
        Accuracy: for classification problems
        Mean Squared Error(MSE); Root Mean Squared Error(RSME); R2 Score: for regression
    '''
    # TODO
    ##turn RFE into a function
    # Use the Cross-Validation function of the RFE module
    # accuracy describes the number of correct classifications
    # LOGISTIC REGRESSION
    est_logreg = LogisticRegression(max_iter=2000)
    # SGD REGRESSOR
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
    # SUPPORT VECTOR REGRESSOR
    est_svr = SVR(kernel='linear',
                  C=1.0,
                  epsilon=0.01)

    # WORKS WITH LOGREG(pick r2), SGDRregressor(r2;rmse)
    rfecv = RFECV(estimator=est_logreg,
                  step=1,
                  # cross_calidation determines if clustering scorers can be used or regression based!
                  # needs to be aligned with estimator
                  cv=None,
                  scoring='completeness_score')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

    # plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 7))
    plt.suptitle(f"{RFECV.get_params}")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return rfecv.grid_scores_
#%%
'''
        Usage of a Pickle Model -Storage of a trained Model
'''
def store_pickle(model):
    model_file = "gridsearch_model.sav"
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)
    return model_file
#%%
'''
        Usage of a Pickle Model -Loading of a Pickle File

model file can be opened either with FILE NAME
open_pickle(model_file="gridsearch_model.sav")
INTERNAL PARAMETER
open_pickle(model_file=model_file)
'''

def open_pickle(model_file):
    with open(model_file, mode='rb') as m_f:
        grid_search = pickle.load(m_f)
        result = grid_search.score(X_test, y_test)
        print("Employed Estimator:", grid_search.get_params)
        print("--------------------")
        print("BEST PARAMETER COMBINATION:", grid_search.best_params_)
        print("Training Accuracy Result: %.4f" %(result))
        return 'grid_search parameters loaded'