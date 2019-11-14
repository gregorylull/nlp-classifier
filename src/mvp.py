# this file is just for the MVP portion of this project,
# I plan to extract the functions to other files.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors

import glob
import pprint
import math

from collections import namedtuple

# my functions
from src import configs
from src.utilities import doc_utils as gldocs

clean_config = configs.clean
model_config = configs.model
corpus_config = configs.corpus

ROOT = 'src/'
DATA = 'data/ebook_output/'
CURRENT = f'{ROOT}'
FIGURES = f'{ROOT}figures/'

pp = pprint.PrettyPrinter(indent = 4)

# AB* is about 50 books
GLOB_PATH = f'{DATA}**/{corpus_config.books_glob}'

USE_MEMO_CACHE = True

PATH_TO_BOOK = "data/ebook_output/Douglas Adams - Dirk Gently 01 Dirk Gently's Holistic Detective Agency.txt"

# for lsa you don't want to transform first


def tokenize(doc_array, tokenizer_type='count'):

    if tokenizer_type == 'count':
        vectorizer = CountVectorizer(stop_words='english')
        doc_words = vectorizer.fit_transform(doc_array)
        print('count vectorizer shape:', doc_words.shape)

    return vectorizer, doc_words

def get_docs_from_file(pathname):
    with open(pathname, 'r') as readfile:
        lines = readfile.readlines()
    
    return lines

def get_docs(glob_paths):
    results = []
    for glob_path in glob_paths:
        lines = get_docs_from_file(glob_path)
        partitioned = partition_doc(lines, clean_config)
        doc = clean_doc(partitioned, clean_config)
        results.append(doc)

    return results

def partition_doc(text_array, clean_config):
    # skip first 1000 and last 1000 lines, publisher info and appendix.
    middle = text_array

    if len(text_array) > clean_config.start + clean_config.end:
        middle = text_array[clean_config.start:-clean_config.end]

    # take a certain percentage (slices) of the pages
    total_lines = len(text_array)
    results = []
    if clean_config.percentages and total_lines > clean_config.minimum_lines:
        for start_percentage, end_percentage in clean_config.percentages:
            start_index = math.floor(total_lines * start_percentage)
            end_index = math.floor(total_lines * end_percentage)

            results = [*results, *middle[start_index:end_index]]
    else:
        results = middle
    
    return results

def clean_doc(text_array, clean_config):
    doc = ' '.join(text_array)

    return doc

def reduce_dimension(docs, reducer_type='truncated_svd', model_config=model_config):
    if reducer_type == 'truncated_svd':
        reducer_model = TruncatedSVD(n_components = model_config.n_components)
        doc_topic = reducer_model.fit_transform(docs)
        explained_ratio = reducer_model.explained_variance_ratio_

    print(f'{reducer_type} (n={model_config.n_components}) explained ratio:')
    pp.pprint(explained_ratio)
    return reducer_model, doc_topic
    
# pulled directly from lecture project 4 - Topic_Modeling_LSA_NMF
def display_topics(model, feature_names, no_top_words = 10, no_top = 10, topic_names=None):
    for ix, topic in enumerate(model.components_[:no_top]):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# (glull) pulled directly from lecture project 4 - KMeansClustering
# helper function that allows us to display data in 2 dimensions and highlights the clusters
def display_cluster(X,km=[],num_clusters=0, save_fig=f'{FIGURES}modeling_clusters.png'):
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
        plt.savefig(save_fig)

# (glull) directly from lecture
def fit_model(X):
    num_clusters = model_config.knn_clusters

    # n_init, number of times the K-mean algorithm will run
    km = KMeans(n_clusters=num_clusters, n_jobs=-1) 
    X_trans = km.fit_transform(X)
    display_cluster(X,km,num_clusters)

    return km, X_trans

def get_doc_cluster(model, model_transformed_docs, target_cluster):
    doc_bools = model.labels_ == target_cluster
    cluster = model_transformed_docs[doc_bools, ]

    return doc_bools, cluster


def sort_doc_cluster(indexes, cluster_reduced_dim, metric='cosine_similarity'):
    sorted_results = []

    if metric == 'cosine_similarity':
        similarity = cosine_similarity(cluster)


    return sorted_results


def main():
    glob_paths = glob.glob(GLOB_PATH, recursive=True)
    print(f'\nLooking at {len(glob_paths)} docs.')
    pp.pprint(glob_paths[:3])

    docs = get_docs(glob_paths)

    vect, doc_word = tokenize(docs)

    reducer_model, docs_reduced = reduce_dimension(doc_word)

    display_topics(reducer_model, vect.get_feature_names())

    # noramlized
    normalizer = Normalizer()


    model, doc_word_fit_transformed = fit_model(docs_reduced)


    # given book
    docs_given = get_docs([PATH_TO_BOOK])
    doc_word_given = vect.transform(docs_given)
    docs_reduced_given = reducer_model.transform(doc_word_given)
    docs_word_given_transformed = model.transform(docs_reduced_given)

    return model, doc_word_fit_transformed, docs_word_given_transformed
    

if __name__ == '__main__':
    print(f'\nRunning MVP code\n')
    model, doc, doc_given = main()
    print('\nfinished\n')
