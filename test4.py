# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:00:01 2023

@author: 88697
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:00:17 2023

@author: 88697
"""
#应用K均值聚类、DESCAN、OPTICS、AGNES（AgglomerativeClustering）算法聚类两同心圆

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])

from sklearn.cluster import KMeans
y1=KMeans(n_clusters=2,init="k-means++").fit(X)
plt.scatter(X[:,0], X[:,1],c=y1.labels_)

from sklearn.cluster import DBSCAN
y2=DBSCAN(eps=0.2, min_samples=10).fit(X) 
plt.scatter(X[:,0], X[:,1],c=y2.labels_)

from sklearn.cluster import OPTICS
y3=OPTICS(eps=0.2, min_samples=10,cluster_method='dbscan').fit(X) 
plt.scatter(X[:,0], X[:,1],c=y3.labels_)

from sklearn.cluster import AgglomerativeClustering
y4=AgglomerativeClustering(n_clusters=2,linkage='single').fit(X) 
plt.scatter(X[:,0], X[:,1],c=y4.labels_)