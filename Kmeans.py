# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:23:44 2016

@author: SAMSUNG
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

## S1Q11A TOTAL FAMILY INCOME IN LAST 12 MONTHS
## S1Q10A TOTAL PERSONAL INCOME IN LAST 12 MONTHS
## NBMCS NORM-BASED MENTAL DISABILITY SCALE (SF12-V2R)
## NBPCS NORM-BASED PHYSICAL DISABILITY SCALE (SF12-V2R)
## MAJORDEPLIFE' MAJOR DEPRESSION IN LAST 12 MONTHS
## NBS4 NORM-BASED GENERAL HEALTH SCALE (SF12-V2R)
## ANTISOCDX2 ANTISOCIAL PERSONALITY DISORDER (WITH CONDUCT DISORDER
## NBS6 NORM-BASED SOCIAL FUNCTIONING SCALE (SF12-V2R)
## NBS7 NORM-BASED ROLE EMOTIONAL SCALE (SF12-V2R)
## NBS8 NORM-BASED MENTAL HEALTH SCALE (SR-V2R)
## S1Q24LB WEIGHT: POUNDS

"""
Data Management
"""
nesarc = pd.read_csv('nesarc_pds.csv', low_memory=False)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)
#upper-case all DataFrame column names
nesarc.columns = map(str.upper, nesarc.columns)

#subset data to adolescent age 18 to 28 and  girls with BLOOD/NATURAL MOTHER having BEHAVIOR PROBLEMS
subset_nesarc = nesarc[(nesarc['S11BQ2']==1) & (nesarc['SEX']==2)]

#make a copy of my new subsetted data
subset = subset_nesarc.copy()
dataClean = subset.dropna()

# subset clustering variables
cluster = dataClean[['MAJORDEPLIFE', 'ANTISOCDX2', 'S1Q11A', 'NBMCS', 'NBPCS', 'NBS4', 
'NBS6','NBS7','NBS8', 'S1Q24LB']]

cluster.describe()
cluster.head(n=5)

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['MAJORDEPLIFE']=preprocessing.scale(clustervar['MAJORDEPLIFE'].astype('float64'))
clustervar['ANTISOCDX2']=preprocessing.scale(clustervar['ANTISOCDX2'].astype('float64'))
clustervar['S1Q11A']=preprocessing.scale(clustervar[ 'S1Q11A'].astype('float64'))
clustervar['NBMCS']=preprocessing.scale(clustervar['NBMCS'].astype('float64'))
clustervar['NBPCS']=preprocessing.scale(clustervar['NBPCS'].astype('float64'))
clustervar['NBS4']=preprocessing.scale(clustervar['NBS4'].astype('float64'))
clustervar['NBS6']=preprocessing.scale(clustervar['NBS6'].astype('float64'))
clustervar['NBS7']=preprocessing.scale(clustervar['NBS7'].astype('float64'))
clustervar['NBS8']=preprocessing.scale(clustervar['NBS8'].astype('float64'))
clustervar['S1Q24LB']=preprocessing.scale(clustervar['S1Q24LB'].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

# plot clusters
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

# Interpret 2 cluster solution
model2=KMeans(n_clusters=2)
model2.fit(clus_train)
clusassign=model2.predict(clus_train)

# plot clusters
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model2.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
smo_data = dataClean['S1Q10A']
# split GPA data into train and test sets
smo_train, smo_test = train_test_split(smo_data, test_size=.3, random_state=123)
smo_train1=pd.DataFrame(smo_train)
smo_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(smo_train1, merged_train, on='index')
sub1 = merged_train_all[['S1Q10A', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

smomod = smf.ols(formula='S1Q10A ~ C(cluster)', data=sub1).fit()
print (smomod.summary())

print ('means for income by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for income by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['S1Q10A'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())