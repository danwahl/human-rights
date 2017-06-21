# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:41:50 2017

@author: dan
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':
    # read csv
    df = pd.read_csv('wikisurvey_10205_votes_2017-06-21T04-52-39Z.csv', index_col=[0])
    
    # collect unique ids
    names = dict()
    for index, row in df.iterrows():
        if not names.has_key(row['Winner ID']):
            names[row['Winner ID']] = row['Winner Text']
        if not names.has_key(row['Loser ID']):
            names[row['Loser ID']] = row['Loser Text']
    ids = names.keys()
    n = len(ids)
    
    # create table of results
    table = np.zeros((n, n), dtype='int')
    for index, row in df.iterrows():
        i1 = ids.index(row['Winner ID'])
        i2 = ids.index(row['Loser ID'])
        if table[i1, i2] > 2:
            print names[row['Winner ID']] + ', ' + names[row['Loser ID']]
        table[i1, i2] += 1
    
    # run pca
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(table)
    
    # predict using kmeans
    m = 5
    y_pred = SpectralClustering(n_clusters=m).fit_predict(X_reduced)
        
    # plot pca with kmeans prediction
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_pred, cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
    # rebuild prediction matrix
    w_pred = np.zeros((m, m))
    c = np.zeros((m, m))
    for index, row in df.iterrows():
        j1 = y_pred[ids.index(row['Winner ID'])]
        j2 = y_pred[ids.index(row['Loser ID'])]
        w_pred[j1, j2] += 1
        c[j1, j2] += 1
        c[j2, j1] += 1
    
    W_pred = w_pred/c
    
    res = sorted(zip(y_pred, names.values()))