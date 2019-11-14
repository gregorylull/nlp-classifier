# this file is for my own understanding of when and how StandardScaler and Normalizer affects the outcome.
# conclusion is that kmeans is very good at separating in terms of euclidean distance.
# dbscan is not, and the params are difficult to tune, without a good metric for book rec i'm not sure how to use it.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import Normalizer, StandardScaler
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors

import glob
import pprint
import math

from collections import namedtuple

def display_cluster(X,km=[],num_clusters=0, save_fig=False):
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        color = np.random.rand(3,)
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        cluster_range = range(num_clusters)
        for i in cluster_range:
            color = [list(np.random.rand(3,))]
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color,alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color, marker = 'x', s = 100)
    
    if save_fig:
        plt.savefig('modeling_clusters.png')

def display_dbscan(X,km=[],num_clusters=0, save_fig=False):
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        color = np.random.rand(3,)
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        cluster_range = range(num_clusters)
        for i in cluster_range:
            color = [list(np.random.rand(3,))]
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color,alpha = alpha,s=s)
    
    if save_fig:
        plt.savefig('modeling_clusters.png')

# Toy 1 doc_topics, 6 clusters
doc_topic1 = np.array([ 
    # diagonal axis
    np.random.rand(10, 2) * 10,
    (np.random.rand(10, 2) * 20) + 20,
    (np.random.rand(10, 2) * 30) + 40,
    (np.random.rand(10, 2) * 40) + 60,

    # vertical, top left, bottom right
    (np.random.rand(10, 2) * [15, 60]) + [0, 50],
    (np.random.rand(10, 2) * 30) + [70, 0]

]).flatten().reshape(60, 2)

n_clusters = 6

# no scaling
m1 = KMeans(n_clusters).fit(doc_topic1)
display_cluster(doc_topic1, m1, n_clusters)
print('kmeans', m1.labels_)
m1df = pd.DataFrame(
    cosine_similarity(doc_topic1)
)

# normalizer will change eucilidean distance, but
# the cosine_similarity should still be the same.
# This seems to mess up KMeans 
doc_topic1_norm = Normalizer().fit_transform(doc_topic1)
m1_norm = KMeans(n_clusters).fit(doc_topic1_norm)
display_cluster(doc_topic1_norm, m1_norm, n_clusters)
print('kmeans norm', m1_norm.labels_)
m1df_norm = pd.DataFrame(
    cosine_similarity(doc_topic1_norm)
)

# standard scaler
# KMeans clustering seems fine, but the cosine_similarity is now not as informative.
doc_topic1_std = StandardScaler().fit_transform(doc_topic1)
m1_std = KMeans(n_clusters).fit(doc_topic1_std)
display_cluster(doc_topic1_std, m1_std, n_clusters)
print('kmeans norm', m1_std.labels_)
m1df_std = pd.DataFrame(
    cosine_similarity(doc_topic1_std)
)

# cosine similarity
cosine_similarity(np.array([
    [1, 1, 1],
    [1, 1, 1],

    # these will all have the same values because they match in one single col.
    [1, 1, 0],  
    [0, 1, 1],
    [1, 0, 1],

    # these will have 
    [5, 5, 0], # will have a cos_sim of 1 with [1, 1, 0]
    [0, 5, 5],
    [5, 0, 5]
])).round(1)



# more dimensions
doc_topic2 = np.array([ 
    # diagonal axis
    np.random.rand(1000, 100) * 10,
    (np.random.rand(1000, 100) * 20) + 20,
    (np.random.rand(1000, 100) * 30) + 40,
    (np.random.rand(1000, 100) * 40) + 60,

    # vertical, top left, bottom right
    (np.random.rand(1000, 100) * [15, 60, *np.arange(98)]) + [0, 50, *np.arange(98)],
    (np.random.rand(1000, 100) * 30) + [70, 0, *np.arange(98)]

]).flatten().reshape(6000, 100)

n_components=30
lsa = TruncatedSVD(n_components=n_components).fit(doc_topic2)
doc_topic2_lsa = lsa.transform(doc_topic2)
plt.plot(range(1, n_components+1), np.cumsum(lsa.explained_variance_ratio_))

# no scaling
m2 = KMeans(n_clusters).fit(doc_topic2)
display_cluster(doc_topic2, m2, n_clusters)
print('kmeans', m2.labels_)
m2df = pd.DataFrame(
    cosine_similarity(doc_topic2)
)
m2_counter = Counter(list(m2.labels_))
print('m2_counter', m2_counter)

m2_lsa = KMeans(n_clusters).fit(doc_topic2_lsa)
display_cluster(doc_topic2_lsa, m2_lsa, n_clusters)
m2_lsa_counter = Counter(list(m2_lsa.labels_))
print('m2_lsa_counter', m2_lsa_counter)

# normalizer will change eucilidean distance, but
# the cosine_similarity should still be the same.
# This seems to mess up KMeans 
doc_topic2_norm = Normalizer().fit_transform(doc_topic2)
m2_norm = KMeans(n_clusters).fit(doc_topic2_norm)
display_cluster(doc_topic2_norm, m2_norm, n_clusters)
print('kmeans norm', m2_norm.labels_)
m2df_norm = pd.DataFrame(
    cosine_similarity(doc_topic2_norm)
)
m2_norm_counter = Counter(list(m2_norm.labels_))
print('m2_norm_counter', m2_norm_counter)

doc_topic2_lsa_norm = Normalizer().fit_transform(doc_topic2_lsa)
m2_lsa_norm = KMeans(n_clusters).fit(doc_topic2_lsa_norm)
display_cluster(doc_topic2_lsa_norm, m2_lsa_norm, n_clusters)
m2_lsa_norm_counter = Counter(list(m2_lsa_norm.labels_))
print('m2_lsa_norm_counter', m2_lsa_norm_counter)

# the same results
cosine_similarity(doc_topic2_norm).round(2) == cosine_similarity(doc_topic2).round(2)


# standard scaler
# KMeans will not work if i reduce dimension and then standardscaler
doc_topic2_std = StandardScaler().fit_transform(doc_topic2)
m2_std = KMeans(n_clusters).fit(doc_topic2_std)
display_cluster(doc_topic2_std, m2_std, n_clusters)
print('kmeans norm', m2_std.labels_)
m2df_std = pd.DataFrame(
    cosine_similarity(doc_topic2_std)
)
m2_std_counter = Counter(list(m2_std.labels_))
print('m2_std_counter', m2_std_counter)

doc_topic2_lsa_std = StandardScaler().fit_transform(doc_topic2_lsa)
m2_lsa_std = KMeans(n_clusters).fit(doc_topic2_lsa_std)
display_cluster(doc_topic2_lsa_std, m2_lsa_std, n_clusters)
m2_lsa_std_counter = Counter(list(m2_lsa_std.labels_))
print('m2_lsa_std_counter', m2_lsa_std_counter)


# Norm is really bad at separating the clusters
doc_topic2_norm = Normalizer().fit_transform(doc_topic2)
for min_sample in np.linspace(3, 10, 3).round(2):
    for i in np.linspace(0.05, 0.4, 10).round(2):
        db2 = DBSCAN(eps=i, min_samples=min_sample, n_jobs=-1).fit(doc_topic2_norm)
        # display_dbscan(doc_topic2_norm, db2, n_clusters)
        db2_counter = Counter(list(db2.labels_))
        print(f'db2_norm_counter min: {min_sample}, eps: {i}:', db2_counter)


# stdscaler not much better
doc_topic2_std = StandardScaler().fit_transform(doc_topic2)
for min_sample in np.linspace(3, 20, 5).round(2):
    for i in np.linspace(0.01, 3, 20).round(2):
        db2 = DBSCAN(eps=i, min_samples=min_sample, n_jobs=-1).fit(doc_topic2_std)
        # display_dbscan(doc_topic2_std, db2, n_clusters)
        db2_counter = Counter(list(db2.labels_))
        print(f'db2_std_counter min: {min_sample}, eps: {i}:', db2_counter)

# lsa norm
doc_topic2_lsa_norm = Normalizer().fit_transform(doc_topic2_lsa)
for min_sample in np.linspace(3, 20, 5).round(2):
    for i in np.linspace(0.01, 3, 20).round(2):
        db2 = DBSCAN(eps=i, min_samples=min_sample, n_jobs=-1).fit(doc_topic2_lsa_norm)
        # display_dbscan(doc_topic2_lsa_norm, db2, n_clusters)
        db2_counter = Counter(list(db2.labels_))
        print(f'db2_lsa_norm_counter min: {min_sample}, eps: {i}:', db2_counter)

# lsa std
doc_topic2_lsa_std = StandardScaler().fit_transform(doc_topic2_lsa)
for min_sample in np.linspace(3, 20, 5).round(2):
    for i in np.linspace(0.01, 3, 20).round(2):
        db2 = DBSCAN(eps=i, min_samples=min_sample, n_jobs=-1).fit(doc_topic2_lsa_std)
        # display_dbscan(doc_topic2_lsa_std, db2, n_clusters)
        db2_counter = Counter(list(db2.labels_))
        print(f'db2_lsa_std_counter min: {min_sample}, eps: {i}:', db2_counter)


