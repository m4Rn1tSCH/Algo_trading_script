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
import pickle

from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier

'''
This module contains the AI/ML packages to take preprocessed data, find informative features
and facilitate their prediction
1.) split data
2.) apply standard scaler, minmax scaler and principal component analysis
3.) OPTIONAL plot the PCA 
4.) feed test and training data to pipeline
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
# for tests only
# df = intra_df
# label_col = 'open'
# Create pipeline with feature selector and regressor/classifier

# TODO

# set up fitting with scaled values as well
def pipeline_knn(df, pca_plot=False):

    """
    Pipeline - SelectKBest and K Nearest Neighbor
    Pipeline KNN; 2020-05-14 16:35:40
    {'clf__n_neighbors': 7, 'feature_selection__k': 1}
    Best accuracy with parameters: 0.6242857142857143
    GridSearchCV(cv=None, error_score=nan,
             estimator=Pipeline(memory=None,
                                steps=[('feature_selection',
                                        SelectKBest(k=10,
                                        score_func=chi2)),
    """

    model_features = df.drop(columns=label_col, axis=1, inplace=False)
    model_label = df[label_col]

    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")

    # datetime object is pushed back to be index again before split
    # this way scaling is possible + index allows plotting later on
    if df['date'].dtype == 'datetime64[ns]':
        model_features.set_index('date', drop=True, inplace=True)
    else:
        print("datetime object still in df; scaling will fail")

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle=True,
                                                        test_size=0.4)


    # create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    # Principal Component Reduction
    # keep the most important features of the data
    pca = PCA(n_components=int(len(df.columns) / 2))
    # fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    # transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))


    if pca_plot:
        '''
                    Plotting of PCA/ Cluster Pairs

        '''
        #Kmeans clusters to categorize groups WITH SCALED DATA
        #determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=600,squeeze=False)
        #styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style='oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style='oblique')
        ax[1].legend(test_clusters.l1abels_)

    else:
        pass

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
    grid_search_knn = GridSearchCV(pipe, param_grid=params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline KNN; {dt.today()}")
    print(grid_search_knn.fit(X_train, y_train).best_params_)
    print(f"Best accuracy with parameters: {grid_search_knn.best_score_}")

    return grid_search_knn


# TODO
# set up fitting with scaled values as well
def pipeline_reg(df, pca_plot=False):
    """
    Pipeline - Logistic Regression and Support Vector Kernel
    """

    model_features = df.drop(columns=label_col, axis=1, inplace=False)
    model_label = df[label_col]

    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")

    # datetime object is pushed back to be index again before split
    # this way scaling is possible + index allows plotting later on
    if df['date'].dtype == 'datetime64[ns]':
        model_features.set_index('date', drop=True, inplace=True)
    else:
        print("datetime object still in df; scaling will fail")

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle=True,
                                                        test_size=0.4)

    # create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    # fit the scaler to the training data first
    # standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    # Principal Component Reduction
    # keep the most important features of the data
    pca = PCA(n_components=int(len(df.columns) / 2))
    # fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    # transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))


    if pca_plot:

        '''
                    Plotting of PCA/ Cluster Pairs

        '''
        #Kmeans clusters to categorize groups WITH SCALED DATA
        #determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=600,squeeze=False)
        #styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style='oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style='oblique')
        ax[1].legend(test_clusters.l1abels_)
    else:
        pass

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
    grid_search_svr = GridSearchCV(pipe, param_grid=params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline SVR; {dt.today()}")
    print(grid_search_svr.fit(X_train, y_train).best_params_)
    print(f"Best accuracy with parameters: {grid_search_svr.best_score_}")

    return grid_search_svr


#TODO
# set up fitting with scaled values

def pipeline_rfr(df, pca_plot=False):
    """
    Pipeline  - SelectKBest and Random Forest Regressor
    REQUIRES FLOAT32 OR INT32 VALUES AS LABELS
    Pipeline RFR; 2020-05-14 16:28:25
    {'feature_selection__k': 3, 'reg__min_samples_split': 4, 'reg__n_estimators': 200}
    Overall score: 0.9796
    Best accuracy with parameters: 0.9823479626506231
    """

    model_features = df.drop(columns=label_col, axis=1, inplace=False)
    model_label = df[label_col]

    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")

    # datetime object is pushed back to be index again before split
    # this way scaling is possible + index allows plotting later on
    if df['date'].dtype == 'datetime64[ns]':
        model_features.set_index('date', drop=True, inplace=True)
    else:
        print("datetime object still in df; scaling will fail")

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle=True,
                                                        test_size=0.4)

    # create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    # Principal Component Reduction
    # keep the most important features of the data
    pca = PCA(n_components=int(len(df.columns) / 2))
    # fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    # transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))


    if pca_plot:
        '''
                    Plotting of PCA/ Cluster Pairs

        '''
        #Kmeans clusters to categorize groups WITH SCALED DATA
        #determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=600,squeeze=False)
        #styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style='oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style='oblique')
        ax[1].legend(test_clusters.l1abels_)
    #principal components of bank panel has better results than card panel with clearer borders
    else:
        pass

    # Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('reg', RandomForestRegressor(max_depth=len(df.columns) / 2,
                                      random_state=8))
    ])

    # Create a parameter grid
    params = {
        'feature_selection__k': [2, 3, 4, 5],
        'reg__n_estimators': [75, 100, 150, 200],
        'reg__min_samples_split': [4, 8, 10, 15],
    }

    # Initialize the grid search object
    grid_search_rfr = GridSearchCV(pipe, param_grid=params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline RFR; {dt.today()}")
    print(grid_search_rfr.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" % (grid_search_rfr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_rfr.best_score_}")

    return grid_search_rfr


def pipeline_trans_reg():

    '''
            Application of Transformed Linear Regression

    #n_quantiles needs to be smaller than the number of samples (standard is 1000)

    '''
    transformer = QuantileTransformer(n_quantiles=750, output_distribution='normal')
    regressor = LinearRegression()
    regr = TransformedTargetRegressor(regressor=regressor,
                                       transformer=transformer)

    regr.fit(X_train, y_train)

    TransformedTargetRegressor(...)
    print('q-t R2-score: {0:.3f}'.format(regr.score(X_test, y_test)))

    raw_target_regr = LinearRegression().fit(X_train, y_train)
    print('unprocessed R2-score: {0:.3f}'.format(raw_target_regr.score(X_test, y_test)))
    return regr, raw_target_regr

def pipeline_mlp(label_col, df, pca_plot=False):

    '''
    Pipeline - SelectKBest and Multi-Layer Perceptron
    ##########

    '''

    model_features = df.drop(columns=label_col, axis=1, inplace=False)
    model_label = df[label_col]

    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")

    # datetime object is pushed back to be index again before split
    # this way scaling is possible + index allows plotting later on
    if df['date'].dtype == 'datetime64[ns]':
        model_features.set_index('date', drop=True, inplace=True)
    else:
        print("datetime object still in df; scaling will fail")

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle=True,
                                                        test_size=0.4)

    # create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    # Principal Component Reduction
    # keep the most important features of the data
    pca = PCA(n_components=int(len(df.columns) / 2))
    # fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    # transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))

    if pca_plot:
        '''
                    Plotting of PCA/ Cluster Pairs

        '''
        # Kmeans clusters to categorize groups WITH SCALED DATA
        # determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=600, squeeze=False)
        # styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style='oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style='oblique')
        ax[1].legend(test_clusters.l1abels_)
    # principal components of bank panel has better results than card panel with clearer borders
    else:
        pass

    #Create pipeline with feature selector and classifier
    #replace with classifier or regressor
    #learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('clf', MLPClassifier(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    #Parameter explanation
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7],
        'clf__max_iter':[1500, 2000],
        'clf__alpha':[0.0001, 0.001]}

    #Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline; {dt.today()}")
    print(grid_search_mlp.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_mlp.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlp.best_score_}")
    return grid_search_mlp


def set_rfe_cross_val(pca_plot=False):
    """
        Application of Recursive Feature Extraction - Cross Validation
        IMPORTANT
        Accuracy: for classification problems
        Mean Squared Error(MSE); Root Mean Squared Error(RSME); R2 Score: for regression
    """
    df = df
    label_col = 'open'

    model_features = df.drop(columns=label_col, axis=1, inplace=False)
    model_label = df[label_col]

    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")

    # datetime object is pushed back to be index again before split
    # this way scaling is possible + index allows plotting later on
    if df['date'].dtype == 'datetime64[ns]':
        model_features.set_index('date', drop=True, inplace=True)
    else:
        print("datetime object still in df; scaling will fail")

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle=True,
                                                        test_size=0.4)

    # create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    # Principal Component Reduction
    # keep the most important features of the data
    pca = PCA(n_components=int(len(df.columns) / 2))
    # fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    # transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))

    if pca_plot:
        '''
                    Plotting of PCA/ Cluster Pairs

        '''
        # Kmeans clusters to categorize groups WITH SCALED DATA
        # determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=600,squeeze=False)
        # styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style='oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style='oblique')
        ax[1].legend(test_clusters.l1abels_)

    else:
        pass

    # Use the Cross-Validation function of the RFE module
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
    rfecv.fit(x, y)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(x.columns[rfecv.support_]))

    # plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 7))
    plt.suptitle(f"{RFECV.get_params}")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return rfecv_obj


'''
        Usage of a Pickle Model -Storage of a trained Model
'''
def store_pickle(model):
    model_file = "gridsearch_model.sav"
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)
    return model_file


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
        print("Training Accuracy Result: %.4f" % (result))
        return 'grid_search parameters loaded'
