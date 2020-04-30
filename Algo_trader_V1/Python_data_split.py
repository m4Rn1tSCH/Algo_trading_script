# -*- coding: utf-8 -*-
"""
Created on 25 4/25/2020 10:57 PM 2020

@author: bill-
"""
'''
This module contains the function to split the data in a test and a training set
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    #TODO
    ##variables arent being passed correctly
    #specify the label to be predicted
    model_features = df.drop([label_col], axis = 1, inplace = False)
    model_label = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        shuffle = True,
                                                        test_size = 0.4)

    #create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    model_features.set_index('date', drop=True, inplace=True)
    
    #TODO
    #fit the scaler to the training data first
    #standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    #transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)
    #%%
    #Principal Component Reduction
    #keep the most important features of the data
    pca = PCA(n_components = int(len(df.columns) / 2))
    #fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    #transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))

    if plot == True:
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        '''
                    Plotting of PCA/ Cluster Pairs
    
        '''
        #Kmeans clusters to categorize groups WITH SCALED DATA
        #determine number of groups needed or desired for
        kmeans = KMeans(n_clusters=5, random_state=10)
        train_clusters = kmeans.fit(X_train_scaled)

        kmeans = KMeans(n_clusters=5, random_state=10)
        test_clusters = kmeans.fit(X_test_scaled)

        fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 10), dpi = 600)
        #styles for title: normal; italic; oblique
        ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c = train_clusters.labels_)
        ax[0].set_title('Plotted Principal Components of TRAINING DATA', style = 'oblique')
        ax[0].legend(train_clusters.labels_)
        ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = test_clusters.labels_)
        ax[1].set_title('Plotted Principal Components of TEST DATA', style = 'oblique')
        ax[1].legend(test_clusters.l1abels_)
    #principal components of bank panel has better results than card panel with clearer borders
    else:
        pass

    return df