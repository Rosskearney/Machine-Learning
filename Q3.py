#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:44:43 2022

@author: rosskearney
"""

import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split

# =============================================================================
#       READ IN CSV
# =============================================================================
df = pd.read_csv('/Users/rosskearney/Desktop/Machine Learning/GroupProject/European_bank_marketing.csv')

# =============================================================================
#   CREATE DUMMY VARIABLES TO DEAL WITH CATAGORICAL VARIABLES
#   AND SET DEPENDANT VARIABLE 'term_deposit' AND INDEPENDANT VARIABLES
# =============================================================================
dummydf = pd.get_dummies(df)

y = dummydf['term_deposit']
x = dummydf.drop('term_deposit', axis=1).astype('float64')
# Brief says to discard 'duration' as it should only be included for benchmark purposes
x = x.drop('duration', axis=1).astype('float64')

# SCALE/STANDARDISE PREDICTORS
x_scaled = preprocessing.scale(x)

y_list = y.to_list()

# =============================================================================
# #     SPLIT DATAFRAME INTO TRAINING AND TESTING 
# =============================================================================
x_train, x_test = train_test_split(x_scaled, test_size=0.3, shuffle = False)
y_train, y_test = train_test_split(y_list, test_size=0.3, shuffle = False)


# =============================================================================
#   CREATE KNN = 1 PREDICTION SCORE
# =============================================================================
clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform')
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
score = clf.score(x_test, y_test)

score

# =============================================================================
#       CREATE FUNCTION FOR KNN 
# =============================================================================
def knn(n_neighbors=1, weight='uniform'):
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print(score)
    cm_df = pd.DataFrame(confusion_matrix(y_test, pred).T, index=clf.classes_, columns=clf.classes_)
    print(cm_df)
    return pred


knn(1);

knn(3);

knn(5);

knn(10);


# Vertical axis = predicted, Horizonal axis = Actual
# 1 = Yes, 0 = No -- 'Has client subscribed to a term deposit?'

# =============================================================================
# # 
# #                     Actual Value
# #                       0      1
# #   Predicted   0     8700   2520 
# #       Value   1      623    514
# # 
# # 
# =============================================================================
