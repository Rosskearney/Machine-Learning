#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:46:45 2022

@author: rosskearney
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# =============================================================================
#                                IMPORT DATA
# =============================================================================

df = pd.read_csv('/Users/rosskearney/Desktop/Machine Learning/GroupProject/European_bank_marketing.csv')

# =============================================================================
#   CREATE DUMMY VARIABLES TO DEAL WITH CATAGORICAL VARIABLES
# =============================================================================

dummydf = pd.get_dummies(df)

# =============================================================================
#                   SET DEPENDANT VARIABLE 'term_deposit'
# =============================================================================

y = dummydf['term_deposit']
x = dummydf.drop('term_deposit', axis=1)
x = x.drop('duration', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)

columns = x_train.columns

# =============================================================================
#                               OVERFITTING
# =============================================================================

os = SMOTE(random_state=0)


os_data_X,os_data_y=os.fit_resample(x_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['term_deposit'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['term_deposit']==0]))
print("Number of subscription",len(os_data_y[os_data_y['term_deposit']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['term_deposit']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['term_deposit']==1])/len(os_data_X))


# =============================================================================
#                            PRINT RESULTS
# =============================================================================

X=os_data_X
y=os_data_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(logreg.score(X_test, y_test)))


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
