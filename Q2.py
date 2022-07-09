# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:41:28 2022

@author: brian
"""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_roc_curve, plot_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

os.chdir("C:\\Documents\\College\\Master's\\Spring\\FIN42100 ML for Finance\\Group Project")

data = pd.read_csv("European_bank_marketing.csv")

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

# set dependent and independetn variables
y = np.array( data["term_deposit"] )
X = np.array( data.drop(["term_deposit"], axis=1) )

#%%

# create model
dec_tree = DecisionTreeClassifier()

# fit to variables
dec_tree.fit(X, y)

# predict y probabilities
dec_pred = dec_tree.predict(X)

# add actual and predicted to df
dec_df = pd.DataFrame()
dec_df["Predicted"] = dec_pred
dec_df["Actual"] = data["term_deposit"]

# confusion matrix and true/false pos/neg
dt_conf = confusion_matrix(dec_df["Actual"], dec_df["Predicted"])
dt_tn, dt_fp, dt_fn, dt_tp = confusion_matrix(dec_df["Actual"], dec_df["Predicted"]).ravel()

#%%

# create model
bag_tree = BaggingClassifier()

# fit to variables
bag_tree.fit(X, y)

# predict y probabilities
bag_pred = bag_tree.predict(X)

# add actual and predicted to df
bag_df = pd.DataFrame()
bag_df["Predicted"] = bag_pred
bag_df["Actual"] = data["term_deposit"]

# confusion matrix and true/false pos/neg
bag_conf = confusion_matrix(bag_df["Actual"], bag_df["Predicted"])
bag_tn, bag_fp, bag_fn, bag_tp = confusion_matrix(bag_df["Actual"], bag_df["Predicted"]).ravel()

#%%

# create model
rand_for = RandomForestClassifier()

# fit to variables
rand_for.fit(X, y)

# predict y probabilities
rf_pred = rand_for.predict(X)

# add actual and predicted to df
rf_df = pd.DataFrame()
rf_df["Predicted"] = rf_pred
rf_df["Actual"] = data["term_deposit"]

# confusion matrix and true/false pos/neg
rf_conf = confusion_matrix(rf_df["Actual"], rf_df["Predicted"])
rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(rf_df["Actual"], rf_df["Predicted"]).ravel()

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#%%

# create model
dec_tree_tt = DecisionTreeClassifier()

# fit to variables
dec_tree_tt.fit(X_train, y_train)

# predict y probabilities
dec_pred_tt = dec_tree_tt.predict(X_test)

# add actual and predicted to df
dec_df_tt = pd.DataFrame()
dec_df_tt["Predicted"] = dec_pred_tt
dec_df_tt["Actual"] = y_test

# confusion matrix and true/false pos/neg
dt_conf_tt = confusion_matrix(dec_df_tt["Actual"], dec_df_tt["Predicted"])
dt_tn_tt, dt_fp_tt, dt_fn_tt, dt_tp_tt = confusion_matrix(dec_df_tt["Actual"], dec_df_tt["Predicted"]).ravel()

#%%

# create model
bag_tree_tt = BaggingClassifier()

# fit to variables
bag_tree_tt.fit(X_train, y_train)

# predict y probabilities
bag_pred_tt = bag_tree_tt.predict(X_test)

# add actual and predicted to df
bag_df_tt = pd.DataFrame()
bag_df_tt["Predicted"] = bag_pred_tt
bag_df_tt["Actual"] = y_test

# confusion matrix and true/false pos/neg
bag_conf_tt = confusion_matrix(bag_df_tt["Actual"], bag_df_tt["Predicted"])
bag_tn_tt, bag_fp_tt, bag_fn_tt, bag_tp_tt = confusion_matrix(bag_df_tt["Actual"], bag_df_tt["Predicted"]).ravel()

#%%

# create model
rf_tt = RandomForestClassifier()

# fit to variables
rf_tt.fit(X_train, y_train)

# predict y probabilities
rf_pred_tt = rf_tt.predict(X_test)

# add actual and predicted to df
rf_df_tt = pd.DataFrame()
rf_df_tt["Predicted"] = rf_pred_tt
rf_df_tt["Actual"] = y_test

# confusion matrix and true/false pos/neg
rf_conf_tt = confusion_matrix(rf_df_tt["Actual"], rf_df_tt["Predicted"])
rf_tn_tt, rf_fp_tt, rf_fn_tt, rf_tp_tt = confusion_matrix(rf_df_tt["Actual"], rf_df_tt["Predicted"]).ravel()

#%%

# decision tree has best true positive
print(dt_tp_tt*100/len(y_test))
print(bag_tp_tt*100/len(y_test))
print(rf_tp_tt*100/len(y_test))

#%%

# random forest has best true true_negative
print(dt_tn_tt*100/len(y_test))
print(bag_tn_tt*100/len(y_test))
print(rf_tn_tt*100/len(y_test))

#%%

# random forest has highest accuracy
print(dec_tree_tt.score(X_test,y_test))
print(bag_tree_tt.score(X_test,y_test))
print(rf_tt.score(X_test,y_test))

#%%

# choose random forest
importances = rf_tt.feature_importances_
feature_names = [f"feature {i}" for i in range(X.shape[1])]

fig, ax = plt.subplots()
ax.bar(x=range(len(importances)), height=importances)
ax.set_title("Random Forest Feature Importance")
ax.set_ylabel("Mean decrease in impurity")
ax.set_xticks(ticks=range(len(data.drop(["term_deposit"], axis=1).columns)))
ax.set_xticklabels( labels=list(data.drop(["term_deposit"], axis=1).columns), rotation=90 )
fig.tight_layout()

#%%

# should plot both logistic and rf on same graph to compare
plot_roc_curve(rf_tt, X_test, y_test)
plt.show()

#%%

# test set area under curve
rf_tt_auc = roc_auc_score(y_test, rf_tt.predict_proba(X_test)[:, 1])

#%%

# can get accuracy from conf matrix as well
rf_tt_conf = confusion_matrix(y_test, rf_pred_tt)

