# this was mvp but has evolved into the final project

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors

import glob
import pprint
import math
import pickle

from collections import namedtuple
from sklearn.pipeline import Pipeline

# my functions
from src.utilities import doc_utils as gldocs
from src.utilities import clean as glclean
from src.utilities import dimension as gldim
from src.utilities import model as glmodel
from src.utilities import file_utils as glfile

from src import configs
clean_config = configs.clean
model_config = configs.model
CONFIG_RESULTS = configs.results_string

ROOT = 'src/'
DATA = 'data/ebook_output/'
CURRENT = f'{ROOT}'
FIGURES = f'{ROOT}figures/'

# set to True to tune model params
TUNE = False
TOKENIZER_TYPE = 'count'  # count | tfidf
DIM_REDUCER_TYPE = 'nmf'  # lsa | nmf
MODEL_TYPE = 'kmeans'    # kmeans | dbscan

pp = pprint.PrettyPrinter(indent=4)

# AB* is about 50 books
GLOB_PATH = f'{DATA}**/{clean_config.books_glob}'

# when getting 10% of a book can just save it as a pickle file
USE_DOC_RETRIEVAL_CACHE = True

PATH_TO_BOOK = "data/ebook_output/Douglas Adams - Hitchhiker 02 The Restaurant at the End of the Universe.txt"
PATH_TO_TEST_BOOKS = "data/ebook_output_test/*.txt"

# (glull) pulled directly from lecture project 4 - Topic_Modeling_LSA_NMF


def display_topics(model, feature_names, no_top_words=10, no_top=10, topic_names=None):
    for ix, topic in enumerate(model.components_[:no_top]):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix], "'")
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))

# (glull) pulled directly from lecture project 4 - KMeansClustering
# helper function that allows us to display data in 2 dimensions and highlights the clusters


def display_cluster(X, km=[], num_clusters=0, save_fig=f'{FIGURES}modeling_clusters.png'):
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        color = np.random.rand(3,)
        plt.scatter(X[:, 0], X[:, 1], c=color[0], alpha=alpha, s=s)
    else:
        cluster_range = range(num_clusters)
        for i in cluster_range:
            color = [list(np.random.rand(3,))]
            plt.scatter(X[km.labels_ == i, 0], X[km.labels_ ==
                                                 i, 1], c=color, alpha=alpha, s=s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[
                        i][1], c=color, marker='x', s=100)

    if save_fig:
        plt.savefig(save_fig)

# (glull) directly from lecture
# def get_doc_cluster(model, model_transformed_docs, target_cluster):
#     doc_bools = model.labels_ == target_cluster
#     cluster = model_transformed_docs[doc_bools, ]

#     return doc_bools, cluster


# def sort_doc_cluster(indexes, cluster_reduced_dim, metric='cosine_similarity'):
#     sorted_results = []

#     if metric == 'cosine_similarity':
#         similarity = cosine_similarity(cluster)


#     return sorted_results

def pipeline_fitting(glob_paths, tokenizer_type, reducer_type, model_type, model_config, tune=TUNE):

    prefix = 'pipeline_fitting'
    intermediary = ''.join(glob_paths) + CONFIG_RESULTS
    postfix = f'{tokenizer_type}_{reducer_type}_{model_type}'
    path_exists, path_to_file, filename, pickled_file = glfile.cache_file(
        prefix, intermediary, postfix)

    if path_exists:
        print('\ngetting cached pipeline_fitting')
        return pickled_file

    docs_raw = gldocs.get_docs(
        glob_paths, clean_config, USE_DOC_RETRIEVAL_CACHE)

    vect, doc_word = glclean.tokenize(docs_raw, tokenizer_type, model_config)

    reducer_model, docs_reduced = gldim.reduce_dimension(
        doc_word, reducer_type, model_config, tune)

    model, X_transformed = glmodel.fit_model(
        docs_reduced, model_type, model_config, tune)

    display_topics(reducer_model, vect.get_feature_names(), 10, 5)

    pipeline = Pipeline([
        (tokenizer_type, vect),
        (reducer_type, reducer_model),
        (model_type, model),
    ])

    results = {

        # tokenize
        'tokenizer': vect,
        'tokenized_doc': doc_word,

        # reduce dim
        'dim_reducer': reducer_model,
        'docs_reduced': docs_reduced,

        # model
        'cluster_model': model,
        'X_transformed': X_transformed,

        # pipeline
        'pipeline': pipeline
    }

    # if the initial hashing is a non existant file, then write to file
    with open(path_to_file, 'wb') as writefile:
        print('  saving cached pipeline', path_to_file)
        pickle.dump(results, writefile)

    return results


def pipeline_transform(glob_paths, tokenizer, dim_reducer, cluster_model):
    docs = gldocs.get_docs(glob_paths, clean_config, USE_DOC_RETRIEVAL_CACHE)

    doc_word = tokenizer.transform(docs)
    docs_reduced = dim_reducer.transform(doc_word)
    X_transformed = cluster_model.transform(docs_reduced)
    predicted = cluster_model.predict(docs_reduced)

    display_topics(dim_reducer, tokenizer.get_feature_names(), 20, 10)

    return {
        # tokenize
        'tokenized_doc': doc_word,

        # reduce dim
        'docs_reduced': docs_reduced,

        # model
        'X_transformed': X_transformed,

        # predicted
        'predicted': predicted
    }


def main():
    glob_paths = glob.glob(GLOB_PATH, recursive=True)
    author_titles = gldocs.get_author_titles(glob_paths)
    print(f'\nLooking at {len(glob_paths)} docs.')
    pp.pprint(glob_paths[:3])

    pipeline_fitted = pipeline_fitting(
        glob_paths,
        TOKENIZER_TYPE,
        DIM_REDUCER_TYPE,
        MODEL_TYPE,
        model_config,
        TUNE
    )

    # tokenize
    tokenizer = pipeline_fitted['tokenizer']
    tokenized_doc = pipeline_fitted['tokenized_doc']

    # reduce dim
    dim_reducer = pipeline_fitted['dim_reducer']
    docs_reduced = pipeline_fitted['docs_reduced']

    # model
    cluster_model = pipeline_fitted['cluster_model']
    X_transformed = pipeline_fitted['X_transformed']

    # pipeline
    pipeline = pipeline_fitted['pipeline']

    # given book
    pipeline_given = pipeline_transform(
        [PATH_TO_BOOK],
        tokenizer,
        dim_reducer,
        cluster_model
    )

    # tokenize
    tokenized_doc_given = pipeline_given['tokenized_doc']

    # reduce dim
    docs_reduced_given = pipeline_given['docs_reduced']

    # model
    X_transformed_given = pipeline_given['X_transformed']

    # predicted
    predicted_given = pipeline_given['predicted']

    # get the cluster of the books.
    # get the top 10 cosine_sim of the book in that cluster
    # get the top 10 in that cluster ordered by goodreads rating

    return pipeline_fitted, pipeline_given, author_titles


if __name__ == '__main__':
    print(f'\nRunning MVP code\n')
    pipeline_fitted, pipeline_given, author_titles = main()

    tokenizer = pipeline_fitted['tokenizer']
    tokenized_doc = pipeline_fitted['tokenized_doc']

    # reduce dim
    dim_reducer = pipeline_fitted['dim_reducer']
    docs_reduced = pipeline_fitted['docs_reduced']

    # model
    cluster_model = pipeline_fitted['cluster_model']
    X_transformed = pipeline_fitted['X_transformed']

    # pipeline
    pipeline = pipeline_fitted['pipeline']

    tokenized_doc_given = pipeline_given['tokenized_doc']

    # reduce dim
    docs_reduced_given = pipeline_given['docs_reduced']

    # model
    X_transformed_given = pipeline_given['X_transformed']

    # predicted
    predicted_given = pipeline_given['predicted']

    print('\nfinished\n')
