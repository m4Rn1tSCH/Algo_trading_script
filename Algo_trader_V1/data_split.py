# -*- coding: utf-8 -*-
"""
Created on 25 4/25/2020 10:57 PM 2020

@author: bill-
"""
'''
This module contains the function to split the data in a test and a training set
'''
from sklearn.model_selection import GridSearchCV, train_test_split

def split_data(df, label_col, plot = True):
    '''


    Parameters
    ----------
    df : TYPE
        DATAFRAME WITH NUMERIC VALUES ONLY.
    label_col : TYPE
        COLUMN THAT IS TO BE PREDICTED.
    plot : TYPE, optional
        SPECIFIES IF A SCATTERPLOT SHOULD BE GENERATED. The default is True.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    #specify the label to be predicted
    model_features = df.drop([label_col], axis = 1, inplace = False)
    model_label = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle = True,
                                                        test_size = 0.3)

    #create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")
    #STD SCALING - does not work yet
    #fit the scaler to the training data first
    #standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    #transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)
    scaler_mean = scaler.mean_
    stadard_scale = scaler.scale_
    #%%
    #MINMAX SCALING - works with Select K Best
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)

    X_test_minmax = min_max_scaler.transform(X_test)
    minmax_scale = min_max_scaler.scale_
    min_max_minimum = min_max_scaler.min_
    #%%
    #Principal Component Reduction
    #first scale
    #then reduce
    #keep the most important features of the data
    pca = PCA(n_components = int(len(bank_df.columns) / 2))
    #fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    #transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))

    if plot == True:
    '''
                PLotting of PCA/ Cluster Pairs

    '''
        #Kmeans clusters to categorize groups WITH SCALED DATA
        #determine number of groups needed or desired for
        kmeans = KMeans(n_clusters = 5, random_state = 10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters = 5, random_state = 10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10), dpi = 600)
        #styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c = train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAIN DATA', style = 'oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style = 'oblique')
        ax[1].legend(test_clusters.l1abels_)
    #principal components of bank panel has better results than card panel with clearer borders
    else:
        pass

    return df