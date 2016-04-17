# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:16:11 2016

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

predictors = subset[['MAJORDEPLIFE', 'HISTDX2']]

targets = subset.ANTISOCDX2

pred_train, pred_test, tar_train, tar_test  =  train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

with open('decisionTree.png', 'wb') as f:
    f.write(graph.create_png())