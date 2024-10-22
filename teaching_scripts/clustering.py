"""
A simple script to illustrate the application of different clustering methods.

author: Fabrizio Musacchio
date: Oct 18, 2024

Adopted from: https://scikit-learn.org/1.5/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATHS
RESULTSPATH = '../results/teaching_material/'
# check whether the results path exists, if not, create it:
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)
# %% FUNCTIONS
# helper-function to plot the silhouette score:
def plot_silhouette_score(fitted_model, PCA_model_S3_fit, method='kmeans', n_clusters=2):
    
    # get the cluster labels:
    labels = fitted_model.labels_
    n_clusters = len(np.unique(labels))

    # compute the silhouette scores for each sample:
    silhouette_vals = silhouette_samples(PCA_model_S3_fit, labels)
    """ 
    With silhouette_samples, we can compute the silhouette score for each sample. The silhouette
    score is a measure of how similar an object is to its own cluster (cohesion) compared to other
    clusters (separation). The silhouette ranges from -1 to 1, where a high value indicates that the
    object is well matched to its own cluster and poorly matched to neighboring clusters. 

    The silhouette_samples function computes the silhouette coefficient for each sample. The Silhouette 
    Coefficient is a measure of how well samples are clustered with sample 10 that are similar to 
    themselves. Clustering models with a high silhouette coefficient are said to be dense, where samples 
    in the same cluster are similar to each other, and well separated, where samples in different 
    clusters are not very similar to each other.

    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean 
    nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is 

            (b - a) / max(a, b)). 

    To clarify, b is the distance between a sample and the nearest cluster that the sample is not a 
    part of. Note that Silhouette Coefficient is only defined if number of labels is 

            2 < n_labels < n_samples - 1.

    """

    # compute the mean silhouette score:
    silhouette_avg = silhouette_score(PCA_model_S3_fit, labels)
    print(f"Mean silhouette score: {silhouette_avg}")
    """ 
    silhouette_score returns the mean Silhouette Coefficient over all samples.
    """

    # Create a silhouette plot
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 6)

    ax1.set_xlim([-0.1, 1]) # the silhouette coefficient can range from -1, 1
    ax1.set_ylim([0, len(PCA_model_S3_fit) + (n_clusters + 1) * 10])
    # the (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters

    y_lower = 10
    for i in range(n_clusters):
        # aggregate the silhouette scores for samples belonging to cluster i, and sort them:
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()

        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.viridis(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=1.0)

        # label the silhouette plots with their cluster numbers at the middle:
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # compute the new y_lower for next plot:
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # plot a vertical line for the average silhouette score of all the values:
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTSPATH, f'{method} silhouette_plot ({n_clusters} clusters).png'), dpi=300)
    plt.show()
# %% GENERATE SOME TEST DATA
# for reproducibility, we set the seed:
np.random.seed(0)

# noisy_circles and noisy_moons:
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# anisotropicly distributed data with different variances:
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances:
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# plot all datasets:
datasets_all = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]
dataset_names = ['Noisy circles', 'Noisy moons', 'Varied variances', 'Anisotropic', 'Blobs', 'No structure']
fig, ax = plt.subplots(1, 6, figsize=(20, 4))
# iterate over datasets:
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    # standardize the data:
    X = StandardScaler().fit_transform(X)
    ax[idx].scatter(X[:, 0], X[:, 1], s=10)
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    ax[idx].set_title(f'{dataset_names[idx]}')
# add a main title:
plt.suptitle('Different, unclustered datasets')
plt.tight_layout()
plt.savefig(RESULTSPATH + 'clustering_datasets.png', dpi=300)
plt.show()

# %% KMEANS CLUSTERING
# set the number of clusters:
n_clusters = 2

# iterate over datasets and collect k-means labels for each dataset:
kmeans_labels = []
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    # standardize the data:
    X = StandardScaler().fit_transform(X)
    # create the KMeans object:
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed)
    y_pred = kmeans.fit_predict(X)
    kmeans_labels.append(y_pred)

# plot the kmeans clustering results:
fig, ax = plt.subplots(1, 6, figsize=(20, 4))
# iterate over datasets:
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    ax[idx].scatter(X[:, 0], X[:, 1], c=kmeans_labels[idx], s=10)
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    ax[idx].set_title(f'{dataset_names[idx]}')
# add a main title:
plt.suptitle(f'KMeans clustering ({n_clusters} clusters)')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'clustering_kmeans_({n_clusters} clusters).png', dpi=300)
plt.show()

# plot the silhouette score for the blobs dataset:
X, y = blobs
X = StandardScaler().fit_transform(X)
kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(X)
plot_silhouette_score(kmeans, X, method='clustering_kmeans_blobs', n_clusters=n_clusters)

# plot the silhouette score for the noisy circles dataset:
X, y = noisy_circles
X = StandardScaler().fit_transform(X)
kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(X)
plot_silhouette_score(kmeans, X, method='clustering_kmeans_noisy_circles', n_clusters=n_clusters)

# plot the silhouette score for the anisotropic dataset:
X, y = aniso
X = StandardScaler().fit_transform(X)
kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(X)
plot_silhouette_score(kmeans, X, method='clustering_kmeans_anisotropic', n_clusters=n_clusters)
# %% AGGLOMERATIVE CLUSTERING
# set Agglomerative clustering parameters:
n_clusters = 2

# iterate over datasets and collect Agglomerative labels for each dataset:
agg_labels = []
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    # standardize the data:
    X = StandardScaler().fit_transform(X)
    # create the Agglomerative object:
    agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = agg.fit_predict(X)
    agg_labels.append(y_pred)
    
# plot the Agglomerative clustering results:
fig, ax = plt.subplots(1, 6, figsize=(20, 4))
# iterate over datasets:
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    ax[idx].scatter(X[:, 0], X[:, 1], c=agg_labels[idx], s=10)
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    ax[idx].set_title(f'{dataset_names[idx]}')
# add a main title:
plt.suptitle(f'Agglomerative clustering ({n_clusters} clusters)')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'clustering_agg_(n_clusters={n_clusters}).png', dpi=300)
plt.show()

# for the blobs dataset, we can also plot the dendrogram:
X, y = blobs
X = StandardScaler().fit_transform(X)
agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
agg.fit(X)

# calculate the linkage matrix:
Z = linkage(X, 'ward')
# plot the dendrogram:
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
dendrogram(Z, ax=ax)
plt.title('Dendrogram')
plt.savefig(RESULTSPATH + f'clustering_agg_dendrogram_blobs_dataset.png', dpi=300)
plt.show()
    
# plot the silhouette score for the blobs dataset:
X, y = blobs
X = StandardScaler().fit_transform(X)
agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
agg.fit(X)
plot_silhouette_score(agg, X, method='clustering_agg_blobs', n_clusters=n_clusters)

# plot the silhouette score for the noisy circles dataset:
X, y = noisy_circles
X = StandardScaler().fit_transform(X)
agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
agg.fit(X)
plot_silhouette_score(agg, X, method='clustering_agg_noisy_circles', n_clusters=n_clusters)

# plot the silhouette score for the anisotropic dataset:
X, y = aniso
X = StandardScaler().fit_transform(X)
agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
agg.fit(X)
plot_silhouette_score(agg, X, method='clustering_agg_anisotropic', n_clusters=n_clusters)



# %% DBSCAN CLUSTERING
# set DBSCAN parameters:
eps = 0.3
min_samples = 10

# iterate over datasets and collect DBSCAN labels for each dataset:
dbscan_labels = []
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    # standardize the data:
    X = StandardScaler().fit_transform(X)
    # create the DBSCAN object:
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X)
    dbscan_labels.append(y_pred)
    
# plot the DBSCAN clustering results:
fig, ax = plt.subplots(1, 6, figsize=(20, 4))
# iterate over datasets:
for idx, dataset in enumerate(datasets_all):
    X, y = dataset
    ax[idx].scatter(X[:, 0], X[:, 1], c=dbscan_labels[idx], s=10)
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    ax[idx].set_title(f'{dataset_names[idx]}')
# add a main title:
plt.suptitle(f'DBSCAN clustering (eps={eps}, min_samples={min_samples})')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'clustering_dbscan_(eps={eps}, min_samples={min_samples}).png', dpi=300)
plt.show()

# plot the silhouette score for the blobs dataset:
X, y = blobs
X = StandardScaler().fit_transform(X)
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)
plot_silhouette_score(dbscan, X, method='clustering_dbscan_blobs', n_clusters=len(np.unique(dbscan_labels[-2])))

# plot the silhouette score for the noisy circles dataset:
X, y = noisy_circles
X = StandardScaler().fit_transform(X)
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)
plot_silhouette_score(dbscan, X, method='clustering_dbscan_noisy_circles', n_clusters=len(np.unique(dbscan.labels_)))

# plot the silhouette score for the anisotropic dataset:
X, y = aniso
X = StandardScaler().fit_transform(X)
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)
plot_silhouette_score(dbscan, X, method='clustering_dbscan_anisotropic', n_clusters=len(np.unique(dbscan.labels_)))
# %% END