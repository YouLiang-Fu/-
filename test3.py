# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:03:44 2023

@author: 88697
"""
#本档案放在目录: \机器学习2023\codes\2.clustering\kmeans
#来读取 kmeansSamples.txt

#K均档案放在此目录:值聚类算法评价: 时间& 轮廓系数

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.loadtxt("kmeansSamples.txt")
y = KMeans(n_clusters=3,init="k-means++").fit(X)
plt.scatter(X[:,0], X[:,1],c=y.labels_)
plt.scatter(y.cluster_centers_[:,0],y.cluster_centers_[:,1],c='r',marker='^')

plt.show()

import time
t0=time.time()
y=KMeans(n_clusters=3,init="k-means++").fit(X)
t1=time.time()
tn=t1-t0
print(tn)

from sklearn.metrics import silhouette_score
sc_value=silhouette_score(X,y.labels_)
print(sc_value)
