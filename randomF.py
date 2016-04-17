# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 15:19:42 2016

@author: SAMSUNG
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

## S1Q6A HIGHEST GRADE OR YEAR OF SCHOOL COMPLETED
## S1Q11A TOTAL FAMILY INCOME IN LAST 12 MONTHS
## MAJORDEPLIFE' MAJOR DEPRESSION IN LAST 12 MONTHS
## SOCPDLIFE SOCIAL PHOBIA - LIFETIME (NON-HIERARCHICAL)
## GENAXLIFE GENERALIZED ANXIETY DISORDER - LIFETIME
## HISTDX2 HISTRIONIC PERSONALITY DISORDER (LIFETIME DIAGNOSIS)
## S11BQ1 (BLOOD/NATURAL FATHER EVER HAD BEHAVIOR PROBLEMS)

nesarc = pd.read_csv('nesarc_pds.csv', low_memory=False)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

dataClean = nesarc.dropna()

#subset data to adolescent age 18 to 28 and  girls with BLOOD/NATURAL MOTHER having BEHAVIOR PROBLEMS
subset_data = dataClean[(dataClean['AGE']>=18) & (dataClean['AGE']<=28) & (dataClean['S11BQ2']==1) & (dataClean['SEX']==2)]

#make a copy of my new subsetted data
subset = subset_data.copy()
subset.dtypes
subset.describe()

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = subset[['MAJORDEPLIFE', 'HISTDX2', 'S11BQ1', 'S1Q11A', 'SOCPDLIFE', 'GENAXLIFE', 'S1Q6A', 'REGION']]

targets = subset.ANTISOCDX2

pred_train, pred_test, tar_train, tar_test  =  train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for node in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=node + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[node]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)