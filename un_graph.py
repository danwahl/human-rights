# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 22:45:16 2017

@author: dan
"""

import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import pandas as pd

pd.read_csv('wikisurvey_10205_votes_2017-06-20T16-27-32Z.csv', index_col=[0])

t = 100 # number of teams
n = 3 # number of groups
d = 100000 # number of matches

# create random teams
#T = np.random.rand(d, n)
#T = np.transpose(np.transpose(T)/np.sum(T, axis=1))
#T = np.zeros((t, n))
#T[np.arange(0, t), np.random.randint(0, n, t)] = 1
T = np.random.randint(0, n, t)

# generate win prob matrix
w = np.random.rand(comb(n, 2, exact=True))
W = np.zeros((n, n))
W[np.tril_indices(n, -1)] = w
W[np.triu_indices(n, 1)] = 1 - w
np.fill_diagonal(W, 0.5)

# generate matches
m = np.random.randint(0, t, (d, 2))

# play games
r = W[list(T[m].transpose())]
res = np.zeros(len(r))

# team matrix
t = np.zeros((t, t), dtype='int')
#t[list(m.transpose())] += (r > np.random.rand(r.size)).astype(int)
for i in range(len(r)):
    res[i] = int(r[i] > np.random.rand())
    t[m[i, 0], m[i, 1]] += int(res[i])

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(t.transpose())
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=T,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

# predict using kmeans
y_pred = KMeans(n_clusters=n).fit_predict(X_reduced)

# rebuild prediction matrix
w_pred = np.zeros((n, n))
c = np.zeros((n, n))
for i in range(len(r)):
    g1, g2 = y_pred[m[i, :]]
    w_pred[g1, g2] += res[i]
    w_pred[g2, g1] += 1 - res[i]
    c[g1, g2] += 1
    c[g2, g1] += 1

W_pred = w_pred/c