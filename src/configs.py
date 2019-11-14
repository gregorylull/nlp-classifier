
# some stats:
# 550 books at 100% document produces CountVectorized (550, 240000)
# 50 components and 10 clusters


class CorpusConfig():
    def __init__(self):

        # all books
        self.books_glob = '*.txt'

        # books that start with A only
        # self.books_glob = '[aA]*.txt'

corpus = CorpusConfig()

class CleanConfig: 
    def __init__(self):
        # skip publisher info, table of contents, forewords, etc.
        self.start = 100

        # don't include appendix
        self.end = 100

        # don't use percentages to take slices if doc has less than 500 lines of text
        self.minimum_lines = 500

        # all of the book
        self.percentages = []

        # 30% beginning, middle, and end
        # self.percentages = [
        #     (0, 0.1),
        #     (.4, .5),
        #     (.8, .9)
        # ]

        # 10% near beginning
        self.percentages = [
            (0.1, 0.2),
        ]

        # 50% middle
        # self.percentages = [
        #     (0.25, 0.75),
        # ]

clean = CleanConfig()

class ModelConfig:
    def __init__(self):
        # n_components could be a range, 1% - 10% of features,
        # for 40k words that's 400 components 4000 components 
        self.n_components = 50

        self.knn_clusters = 8

model = ModelConfig()

all_configs = [corpus, clean, model]

results = []
for config in all_configs:
    for key,val in vars(config).items():
        results.append((key, val))

results = sorted(results, key=lambda x: x[0])

results_string = '__'.join([f'{pair[0]}-{pair[1]}' for pair in results])

print(results_string)





