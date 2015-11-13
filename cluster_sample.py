# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:26:56 2015

@author: felix
"""

from sklearn.cluster import KMeans
import numpy as np


C = np.array([[1,1], [1,2], [2,1], [2,2], [5,1], [6,1], [5,2]])
centers = [[3,0], [5,0]]

clf = KMeans(init='k-means++', n_clusters=2, n_init=5)
clf.cluster_centers_ = centers

clf.fit(C)

print "centros: ", clf.cluster_centers_