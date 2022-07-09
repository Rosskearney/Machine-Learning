#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:36:16 2022

@author: rosskearney
"""


import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing


# =============================================================================
#       READ IN CSV FILE
# =============================================================================

df = pd.read_csv('/Users/rosskearney/Desktop/Machine Learning/GroupProject/European_bank_marketing.csv')

# =============================================================================
#   CREATE DUMMY VARIABLES TO DEAL WITH CATAGORICAL VARIABLES
# =============================================================================

dummydf = pd.get_dummies(df)

# =============================================================================
# SET DEPENDANT VARIABLE 'term_deposit'
# =============================================================================

y = dummydf['term_deposit']
x = dummydf.drop('term_deposit', axis=1)
x = x.drop('duration', axis=1)

# SCALE/STANDARDISE PREDICTORS      NOT SURE IF NECESSARY AFTER Q3, BREAKS CODE FURTHER DOWN
# ==========================================
# x = preprocessing.scale(x)
# y = y.to_list()
# ==========================================


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



# =============================================================================
#               SUBSCRIPTION DATA STATISTICS 
# =============================================================================

print("length of the data is ",len(x))
print("Number of no subscriptions in data",len(y[y==0]))
print("Number of subscriptions in data",len(y[y==1]))
print("Proportion of no subscription data in data is ",len(y[y==0])/len(x))
print("Proportion of subscription data in data is ",len(y[y==1])/len(x))

# =============================================================================
#                       RESULTS
# =============================================================================

  # logit results, ugly as we left all data variables in
# =======================================
# logit_model=sm.Logit(y,x)
# result=logit_model.fit()
# print(result.summary2())
# =======================================



logreg = LogisticRegression(max_iter=10000)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(logreg.score(x_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)




