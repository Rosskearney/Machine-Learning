# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:43:33 2022

@author: brian
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_roc_curve, plot_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

os.chdir("C:\\Documents\\College\\Master's\\Spring\\FIN42100 ML for Finance\\Group Project")

data = pd.read_csv("European_bank_marketing.csv")







#%% AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# categorical variables with more than one category
# need to assign at least two dummy variables for each one
job_cats = data["job"].unique()
mar_cats = data["marital"].unique()
ed_cats = data["education"].unique()
def_cats = data["default"].unique()
house_cats = data["housing"].unique()
loan_cats = data["loan"].unique()
mon_cats = data["month"].unique()
day_cats = data["day_of_week"].unique()
p_cats = data["poutcome"].unique()

# categorical with just two outcomes
cont_cat = data["contact"].unique()

#%%

# data that need to be changed from categorical to binary
categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "month",
    "day_of_week",
    "poutcome",
    "contact"
    ]

#%%

def create_dummies(data, column):
    
    col = data[column]
    
    # get unique values from column
    unique_cats = col.unique()
    
    # drop one to avoid dummy variable trap
    unique_cats = unique_cats[0:-1]
    
    # array of column
    array = np.array( data[column] )
    
    # loop through all possible categories in column
    for cat in unique_cats:
        binary_col = np.where( array==cat, 1, 0)
        new_col_name = f"{column}_{cat}"
        data[new_col_name] = binary_col
     
    # drop original column from data
    data.drop(labels=column, inplace=True, axis=1)
    
    pass
    
#%%

for cat in categorical_cols:
    create_dummies(data, cat)

#%%

dataTypeSeries = data.dtypes

for t in dataTypeSeries:
    print(t)

# shows all data types are either integer or float
    
#%%

# set dependent and independetn variables
y = np.array( data["term_deposit"] )
X = np.array( data.drop(["term_deposit"], axis=1) )

#%%

# create model
log_model = LogisticRegression()

# fit to variables
log_model.fit(X, y)

# predict y probabilities
log_pred = log_model.predict_proba(X)

# create dataframe
log_df = pd.DataFrame(log_pred)

# drop prob of 0 column
log_df.drop([0], axis=1, inplace=True)

# threshold values of 20%, 35% and 50%
thresh_values = [0.2, 0.35, 0.5]

# array of probabilities of being classified as 1
prob_array = np.array(log_df[1])

# loop through threshold values
for val in thresh_values:
    # where prob is greater than threshold value, =1, else 0
    class_array = np.where( prob_array >= val, 1, 0 )
    
    # add back to dataframe
    log_df[f"{val*100}%"] = class_array

# actual class
log_df["actual"] = data["term_deposit"]

#%%

# actual
y_act = log_df["actual"]

# predicted at different thresholds
y_20 = log_df["20.0%"]
y_35 = log_df["35.0%"]
y_50 = log_df["50.0%"]

# confusion matrix for 20%
conf_20 = confusion_matrix(y_act, y_20)
# true neg, false pos, false neg, true pos
tn20, fp20, fn20, tp20 = confusion_matrix(y_act, y_20).ravel()

# confusion matrix for 35%
conf_35 = confusion_matrix(y_act, y_35)
# true neg, false pos, false neg, true pos
tn35, fp35, fn35, tp35 = confusion_matrix(y_act, y_35).ravel()

# confusion matrix for 50%
conf_50 = confusion_matrix(y_act, y_50)
# true neg, false pos, false neg, true pos
tn50, fp50, fn50, tp50 = confusion_matrix(y_act, y_50).ravel()






#%% BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# data already cleaned etc., so just need to create new model and fit to training data
# new model with _tt - train-test
# create model
log_model_tt = LogisticRegression()

# fit to variables
log_model_tt.fit(X_train, y_train)

# predict y probabilities of test set
log_pred_tt = log_model.predict_proba(X_test)

# create dataframe
log_df_tt = pd.DataFrame(log_pred_tt)

# drop prob of 0 column
log_df_tt.drop([0], axis=1, inplace=True)

# array of probabilities of being classified as 1
prob_array_tt = np.array(log_df_tt[1])

# loop through threshold values
for val in thresh_values:
    # where prob is greater than threshold value, =1, else 0
    class_array = np.where( prob_array_tt >= val, 1, 0 )
    
    # add back to dataframe
    log_df_tt[f"{val*100}%"] = class_array

# actual class
log_df_tt["actual"] = y_test

#%%

# predicted at different thresholds
y_20_tt = log_df_tt["20.0%"]
y_35_tt = log_df_tt["35.0%"]
y_50_tt = log_df_tt["50.0%"]

# confusion matrix for 20%
conf_20_tt = confusion_matrix(y_test, y_20_tt)
# true neg, false pos, false neg, true pos
tn20_tt, fp20_tt, fn20_tt, tp20_tt = confusion_matrix(y_test, y_20_tt).ravel()

# confusion matrix for 35%
conf_35_tt = confusion_matrix(y_test, y_35_tt)
# true neg, false pos, false neg, true pos
tn35_tt, fp35_tt, fn35_tt, tp35_tt = confusion_matrix(y_test, y_35_tt).ravel()

# confusion matrix for 50%
conf_50_tt = confusion_matrix(y_test, y_50_tt)
# true neg, false pos, false neg, true pos
tn50_tt, fp50_tt, fn50_tt, tp50_tt = confusion_matrix(y_test, y_50_tt).ravel()






#%% CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

# roc curve plot for test set
plot_roc_curve(log_model_tt, X_test, y_test)
plt.show()

#%% 

# test set area under curve
tt_auc = roc_auc_score(y_test, log_model_tt.predict_proba(X_test)[:, 1])





