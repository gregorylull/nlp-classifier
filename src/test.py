import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob

TUNE = False
TOKENIZER_TYPE = 'tfidf'  # count | tfidf
DIM_REDUCER_TYPE = 'lsa'  # lsa | nmf
MODEL_TYPE = 'kmeans'    # kmeans | dbscan

ROOT = 'src/'
DATA = 'data/ebook_output/'
CURRENT = f'{ROOT}'
FIGURES = f'{ROOT}figures/'


# AB* is about 50 books
GLOB_PATH = f'{DATA}/*.txt'

# when getting 10% of a book can just save it as a pickle file
USE_DOC_RETRIEVAL_CACHE = True

PATH_TO_BOOK = "data/ebook_output_test/Douglas Adams - Hitchhiker 02 The Restaurant at the End of the Universe.txt"
PATH_TO_TEST_BOOKS = "data/ebook_output_test/*.txt"

pickle_file = 'src/cached_docs/pipeline_fitting_tfidf_lsa_kmeans_0ec544cb97f02256c1cea9e3564a64e696f8e65d.pkl'

with open(pickle_file, 'rb') as readfile:
    pipeline = pickle.load(readfile)



