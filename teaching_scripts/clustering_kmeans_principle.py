"""
A simple script to illustrate the principle of k-means clustering 

author: Fabrizio Musacchio
date: Oct 24, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn import datasets

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATHS
RESULTSPATH = '../results/teaching_material/kmeans_principle/'
# check whether the results path exists, if not, create it:
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)
# %% GENERATE SOME TEST DATA
# for reproducibility, we set the seed:
np.random.seed(0)

# create a dataset with three clusters, each with a different standard deviation:
n_samples = 500
random_state = 111
blobs_data = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# plot the data:
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(blobs_data[0][:, 0], blobs_data[0][:, 1], c='black', s=10)
plt.title('Data with three clustered blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(RESULTSPATH + 'clustering_kmeans_principle_data.png', dpi=300, bbox_inches='tight')
plt.show()
# %% K-MEANS CLUSTERING EXAMPLE
# fit a k-means model to the data:
kmeans = KMeans(n_clusters=3, random_state=random_state)
kmeans.fit(blobs_data[0])

# replot the data with the cluster centers:
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(blobs_data[0][:, 0], blobs_data[0][:, 1], c=kmeans.labels_, s=10, cmap=cm.tab10)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, marker='X')
plt.title('Data with three clustered blobs and k-means cluster centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(RESULTSPATH + 'clustering_kmeans_principle_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
# %% K-MEANS ITERATIONS DEMO

# Plot the initial centroids
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(blobs_data[0][:, 0], blobs_data[0][:, 1], c='black', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X')
plt.title('Initial random cluster centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(RESULTSPATH + 'clustering_kmeans_initial_centroids.png', dpi=300, bbox_inches='tight')
plt.show()

# %% ITERATIVE PROCESS OF K-MEANS
# for reproducibility, we set the seed:
np.random.seed(0)

n_clusters = 3
max_iter = 10  # Set a maximum number of iterations

# Manually initialize the centroids by selecting random points from the data:
initial_centroids_idx = np.random.choice(n_samples, n_clusters, replace=False)
centroids = blobs_data[0][initial_centroids_idx]

# Plot the initial centroids:
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(blobs_data[0][:, 0], blobs_data[0][:, 1], c='black', s=4)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X')
plt.title('Initial cluster centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(RESULTSPATH + 'clustering_kmeans_initial_centroids.png', dpi=300, bbox_inches='tight')
plt.show()

for iteration in range(max_iter):
    # Step 1: Assign points to the nearest centroid
    distances = np.sqrt(((blobs_data[0][:, np.newaxis] - centroids)**2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    
    # Plot the data and current centroids (before updating centroids)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(blobs_data[0][:, 0], blobs_data[0][:, 1], c=labels, s=4, cmap=cm.tab10)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', zorder=10)
    
    # Draw lines to indicate distances from each point to its currently assigned centroid
    for i in range(n_samples):
        plt.plot([blobs_data[0][i, 0], centroids[labels[i], 0]], 
                 [blobs_data[0][i, 1], centroids[labels[i], 1]], 
                  linestyle='--', linewidth=0.5,
                  c=cm.tab10(labels[i]))

    plt.title(f'Iteration {iteration+1} of k-means clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(RESULTSPATH + f'clustering_kmeans_iteration_{iteration+1}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Step 2: Recalculate centroids
    new_centroids = np.array([blobs_data[0][labels == i].mean(axis=0) for i in range(n_clusters)])

    # Check if centroids have changed; if not, stop the algorithm
    if np.all(centroids == new_centroids):
        print(f'Convergence reached after {iteration+1} iterations.')
        break

    # Update centroids
    centroids = new_centroids

# %% END