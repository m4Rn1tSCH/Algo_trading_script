# -*- coding: utf-8 -*-
"""
Created on 25 4/25/2020 10:57 PM 2020

@author: bill-
"""
'''
This module contains the function to split the data in a test and a training set
'''
from sklearn.model_selection import GridSearchCV, train_test_split

def split_data(df, label_col):
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
    return df