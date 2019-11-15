
# some stats:
# 550 books at 100% document produces CountVectorized (550, 240000)
# 50 components and 10 clusters

import numpy as np
import pprint
import re

pp = pprint.PrettyPrinter(indent=4)


class CleanConfig:
    def __init__(self):

        # all books
        self.books_glob = '*.txt'

        # books that start with A-<character> only
        # as reference A-F is about 250 files
        # self.books_glob = '[a-aA-A]*.txt'

        # skip publisher info, table of contents, forewords, etc.
        self.start = 50

        # don't include appendix
        self.end = 50

        # don't use percentages to take slices if doc has less than 500 lines of text
        self.minimum_lines = 500

        # all of the book
        self.percentages = []

        # 20% near beginning
        # On amazon you can get samples for books which cover about
        # 10% of the whole book. So a 200 page book has a ~20 page sampler
        # (includes cover, meta information, table of contents, etc.)
        self.percentages = [
            (0.05, 0.25),
        ]

        # 45% beginning, middle, and end
        # self.percentages = [
        #     (0.0, 0.15),
        #     (.5, .65),
        #     (.8, .95)
        # ]

        # 60% middle
        # self.percentages = [
        #     (0.1, 0.3),
        #     (0.4, 0.6),
        #     (0.7, 0.9),
        # ]

    def get_param_postfix(self):
        results = []

        for percentage in self.percentages:
            start = percentage[0]
            end = percentage[1]
            results.append(str(start))
            results.append(str(end))

        results.append(self.books_glob)

        joined = '_'.join(results)

        clean_filename = re.sub(r'[^\w]', '', joined)

        return clean_filename


clean = CleanConfig()


class ModelConfig:
    def __init__(self):

        # n_components could be a range, 1% - 10% of features,
        # for 40k words that's 400 components 4000 components.
        # However, I only have 1000 documents.
        self.lsa__n_components = 100
        self.lsa__n_components_tune = 200

        self.nmf__n_components = 100

        self.kmeans__cluster_num = 20
        # self.kmeans__cluster_num_tune = np.arange(5, 100, 5)
        self.kmeans__cluster_num_tune = np.arange(5, 50, 5)

        self.tfidf__ngram_range = (1, 2)
        self.count__ngram_range = (1, 2)


model = ModelConfig()

all_configs = [clean, model]

results = []
for config in all_configs:
    for key, val in vars(config).items():
        results.append((key, val))

results = sorted(results, key=lambda x: x[0])
results_string = '__'.join([f'{pair[0]}-{pair[1]}' for pair in results])

pp.pprint(results)

print('\n', results_string)
