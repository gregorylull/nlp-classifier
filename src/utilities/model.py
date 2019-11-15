from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

ROOT = 'src/'
CURRENT = f'{ROOT}'
FIGURES = f'{ROOT}figures/'

def fit_model(X, model_type, model_config, tune):

    if model_type == 'kmeans':

        if tune:
            inertias = []
            for num_clusters in model_config.kmeans__cluster_num_tune:
                model = KMeans(
                    n_clusters=num_clusters,
                    n_init=50,
                    max_iter=1000,
                    n_jobs=-1
                ) 
                model.fit(X)
                inertias.append(model.inertia_)
            
            print('plotting inertias:', inertias)
            fig = plt.figure(figsize=(10, 5))
            plt.plot(model_config.kmeans__cluster_num_tune, inertias)
            plt.title('Selecting K clusters for KMeans')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.savefig(
                f'{FIGURES}kmeans_clusters.png',
                dpi=1200,
                bbox_inches='tight'
            )
            plt.close(fig)

        else:
            num_clusters = model_config.kmeans__cluster_num
            model = KMeans(n_clusters=num_clusters, n_jobs=-1) 

    X_trans = model.fit_transform(X)

    return model, X_trans
