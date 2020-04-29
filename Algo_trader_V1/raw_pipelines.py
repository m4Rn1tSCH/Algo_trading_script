# -*- coding: utf-8 -*-
"""
Created on 28 4/28/2020 10:34 AM 2020

@author: bill-
"""
#TODO
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
#PARAMETERS NEED TO HAVE THE SAME LENGTH
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
'''
Pipeline 7 - SelectKBest and K Nearest Neighbor
##########
Pipeline 7; 2020-04-28 10:22:10
{'clf__C': 100, 'clf__gamma': 0.1, 'feature_selection__k': 5}
Best accuracy with parameters: 0.6742596944770858
'''
#Create pipeline with feature selector and classifier
#replace with classifier or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', SVC())])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
#Parameter explanation
#   C: penalty parameter
#   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
#
params = {
    'feature_selection__k':[4, 5, 6, 7, 8, 9],
    'clf__C':[0.01, 0.1, 1, 10, 100],
    'clf__gamma':[0.1, 0.01, 0.001]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 7; {dt.today()}")
print(grid_search.fit(X_train_scaled, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
'''
Pipeline 8 - SelectKBest and Multi-Layer Perceptron
##########

'''
#Create pipeline with feature selector and classifier
#replace with classifier or regressor
#learning_rate = 'adaptive'; when solver='sgd'
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = chi2)),
    ('clf', MLPClassifier(activation='relu',
                          solver='adam',
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
    'clf__max_iter':[1500, 2000, 2500],
    'clf__alpha':[0.0001, 0.001, 0.01]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 7; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print("Overall score: %.4f" %(grid_search.score(X_test, y_test)))
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
#accuracy negative; model toally off
#n_quantiles needs to be smaller than the number of samples (standard is 1000)
transformer = QuantileTransformer(n_quantiles=750, output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,
                                   transformer=transformer)

regr.fit(X_train, y_train)

TransformedTargetRegressor(...)
print('q-t R2-score: {0:.3f}'.format(regr.score(X_test, y_test)))


raw_target_regr = LinearRegression().fit(X_train, y_train)
print('unprocessed R2-score: {0:.3f}'.format(raw_target_regr.score(X_test, y_test)))