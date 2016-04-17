# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:48:28 2016

@author: SAMSUNG
"""
#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
 
## S1Q11A TOTAL FAMILY INCOME IN LAST 12 MONTHS
## MAJORDEPLIFE' MAJOR DEPRESSION IN LAST 12 MONTHS
## SOCPDLIFE SOCIAL PHOBIA - LIFETIME (NON-HIERARCHICAL)
## GENAXLIFE GENERALIZED ANXIETY DISORDER - LIFETIME
## HISTDX2 HISTRIONIC PERSONALITY DISORDER (LIFETIME DIAGNOSIS)
## S11BQ1 (BLOOD/NATURAL FATHER EVER HAD BEHAVIOR PROBLEMS)
## AVOIDPDX2 AVOIDANT PERSONALITY DISORDER (LIFETIME DIAGNOSIS)
## DEPPDDX2 DEPENDENT PERSONALITY DISORDER (LIFETIME DIAGNOSIS)
## ANTISOCDX2 ANTISOCIAL PERSONALITY DISORDER (WITH CONDUCT DISORDER
## OBCOMDX2 OBSESSIVE-COMPULSIVE PERSONALITY DISORDER
## PARADX2 PARANOID PERSONALITY DISORDER (LIFETIME DIAGNOSIS)
## SCHIZDX2 SCHIZOID PERSONALITY DISORDER (LIFETIME DIAGNOSIS) 
## S1Q1C HISPANIC OR LATINO ORIGIN
## S1Q1D1 "AMERICAN INDIAN OR ALASKA NATIVE" CHECKED IN MULTIRACE CODE
## S1Q1D2 "ASIAN" CHECKED IN MULTIRACE CODE
## S1Q1D3 "BLACK OR AFRICAN AMERICAN" CHECKED IN MULTIRACE CODE
## S1Q1D4 "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER" CHECKED IN MULTIRACE CODE
## S1Q1D5 "WHITE" CHECKED IN MULTIRACE CODE
## S1Q1F BORN IN UNITED STATES
## NUMREL NUMBER OF RELATED PERSONS IN HOUSEHOLD, INCLUDING SAMPLE PERSON
 
#Load the dataset
nesarc = pd.read_csv('nesarc_pds.csv', low_memory=False)
nesarc.head(n=5)
nesarc.shape

#upper-case all DataFrame column names
nesarc.columns = map(str.upper, nesarc.columns)

#subset data to girls with BLOOD/NATURAL MOTHER having BEHAVIOR PROBLEMS
subset_nesarc = nesarc[(nesarc['S11BQ2']==1) & (nesarc['SEX']==2)]

#make a copy of my new subsetted data
subset = subset_nesarc.copy()

# recode my missing values to python missing (NaN)
subset['S11BQ2']= subset['S11BQ2'].replace(9, np.nan)
subset['S11BQ1']= subset['S11BQ1'].replace(9, np.nan)
subset['S1Q1F']= subset['S1Q1F'].replace(9, np.nan)

dataClean = subset.dropna()

predvar = dataClean[['MAJORDEPLIFE', 'HISTDX2', 'S11BQ1', 'S1Q11A', 'SOCPDLIFE', 'GENAXLIFE', 'AGE', 
'ANTISOCDX2', 'AVOIDPDX2', 'DEPPDDX2', 'OBCOMDX2', 'PARADX2', 'SCHIZDX2', 'S1Q1C','S1Q1D1','S1Q1D2', 
 'S1Q1D3','S1Q1D4', 'S1Q1D5', 'S1Q1F']]

target = dataClean.NUMREL

# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()

from sklearn import preprocessing
predictors['MAJORDEPLIFE']=preprocessing.scale(predictors['MAJORDEPLIFE'].astype('float64'))
predictors['HISTDX2']=preprocessing.scale(predictors['HISTDX2'].astype('float64'))
predictors['S11BQ1']=preprocessing.scale(predictors['S11BQ1'].astype('float64'))
predictors['S1Q11A']=preprocessing.scale(predictors['S1Q11A'].astype('float64'))
predictors['SOCPDLIFE']=preprocessing.scale(predictors['SOCPDLIFE'].astype('float64'))
predictors['AGE']=preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['GENAXLIFE']=preprocessing.scale(predictors['GENAXLIFE'].astype('float64'))
predictors['ANTISOCDX2']=preprocessing.scale(predictors['ANTISOCDX2'].astype('float64'))
predictors['AVOIDPDX2']=preprocessing.scale(predictors['AVOIDPDX2'].astype('float64'))
predictors['DEPPDDX2']=preprocessing.scale(predictors['DEPPDDX2'].astype('float64'))
predictors['OBCOMDX2']=preprocessing.scale(predictors['OBCOMDX2'].astype('float64'))
predictors['PARADX2']=preprocessing.scale(predictors['PARADX2'].astype('float64'))
predictors['SCHIZDX2']=preprocessing.scale(predictors['SCHIZDX2'].astype('float64'))
predictors['S1Q1C']=preprocessing.scale(predictors['S1Q1C'].astype('float64'))
predictors['S1Q1D1']=preprocessing.scale(predictors['S1Q1D1'].astype('float64'))
predictors['S1Q1D2']=preprocessing.scale(predictors['S1Q1D2'].astype('float64'))
predictors['S1Q1D3']=preprocessing.scale(predictors['S1Q1D3'].astype('float64'))
predictors['S1Q1D4']=preprocessing.scale(predictors['S1Q1D4'].astype('float64'))
predictors['S1Q1D5']=preprocessing.scale(predictors['S1Q1D5'].astype('float64'))
predictors['S1Q1F']=preprocessing.scale(predictors['S1Q1F'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)