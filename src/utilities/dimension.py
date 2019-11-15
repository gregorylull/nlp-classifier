
from sklearn.decomposition import TruncatedSVD, NMF

from matplotlib import pyplot as plt
import numpy as np

ROOT = 'src/'
CURRENT = f'{ROOT}'
FIGURES = f'{ROOT}figures/'

def reduce_dimension(docs, reducer_type, model_config, tune):

    if reducer_type == 'lsa':
        if tune:
            reducer_model = TruncatedSVD(n_components = model_config.lsa__n_components_tune)
            doc_reduced = reducer_model.fit_transform(docs)
            print('plotting lsa components:', model_config.lsa__n_components_tune)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(range(model_config.lsa__n_components_tune), np.cumsum(reducer_model.explained_variance_ratio_))
            plt.title('Components for dimension reduction')
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative variance explained')
            plt.savefig(
                f'{FIGURES}lsa_components.png',
                dpi=1200,
                bbox_inches='tight',
            )
            plt.close(fig)

        
        else:
            reducer_model = TruncatedSVD(n_components = model_config.lsa__n_components)


    elif reducer_type == 'nmf':
        reducer_model = NMF(n_components = model_config.n_components)

    doc_reduced = reducer_model.fit_transform(docs)
    explained_ratio = reducer_model.explained_variance_ratio_

    print(f'{reducer_type} explained ratio:')
    print(explained_ratio.round(2)[:20])
    return reducer_model, doc_reduced
    