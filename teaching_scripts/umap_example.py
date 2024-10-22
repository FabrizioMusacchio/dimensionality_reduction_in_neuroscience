"""
A simple script to illustrate embedding high-dimensional data into a latent space representation using UMAP.
We apply the UMAP algorithm to the MNIST dataset

author: Fabrizio Musacchio
date: Oct 18, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

import umap

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

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
# %% LOAD THE DATASET
# we will load the digits dataset and only use six first of the ten available classes:
digits = load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n_neighbors = 30

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(5, 5))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
fig.suptitle(f"A selection from the 64-dimensional\ndigits dataset", fontsize=16)
plt.tight_layout()
plt.savefig(RESULTSPATH + f'umap_digits_samples.png', dpi=300)
plt.show()

# standardize the data:
X = MinMaxScaler().fit_transform(X)
# %% UMAP
# UMAP embedding of the digits dataset
umap_model = umap.UMAP(n_components=2, random_state=0)
X_umap = umap_model.fit_transform(X)

# %% PLOT THE EMBEDDED DATA
DIGIT_COLORS = {
    "0": "#1f77b4",
    "1": "#ff7f0e",
    "2": "#2ca02c",
    "3": "#d62728",
    "4": "#9467bd",
    "5": "#8c564b",
    #"6": "#e377c2",
    #"7": "#7f7f7f",
    #"8": "#bcbd22",
    #"9": "#17becf"
    }

# plot the UMAP embedding w/o ground truth color coding:
plt.figure(figsize=(5.15, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c="k", cmap=plt.cm.tab10, alpha=0.5, s=2)
plt.title("UMAP embedding of the digits dataset\nwithout ground truth labeling")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(RESULTSPATH + f'umap_digits.png', dpi=300)
plt.show()

# plot the UMAP embedding with ground truth color coding:
plt.figure(figsize=(6.25, 6))
colors = plt.cm.tab10(y)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap=plt.cm.tab10, s=2, alpha=0.6)
plt.title("UMAP embedding of the digits dataset\nwith ground truth labeling")
plt.xticks([])
plt.yticks([])
# add a 'colorbar' that matches the cell types:
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
           for label, color in DIGIT_COLORS.items()]
plt.legend(handles=handles, title="digits", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(RESULTSPATH + f'umap_digits_GT_labeling.png', dpi=300)
plt.show()

# apply k-means clustering to the UMAP embedding:
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans_labels = kmeans.fit_predict(X_umap)

# plot the UMAP embedding with k-means cluster labels:
plt.figure(figsize=(6.25, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=kmeans_labels, cmap=plt.cm.tab10, s=2, alpha=0.6)
plt.title("UMAP embedding of the digits dataset\nwith KMeans clustering")
plt.xticks([])
plt.yticks([])
# add a 'colorbar' that matches the cell types:
# create a dictionary to assign the cluster labels the tab10 colors:
cluster_to_color = {i: plt.cm.tab10(i) for i in range(6)}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), markersize=8, 
                      label=f"{i}")
           for i in range(6)]
plt.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(RESULTSPATH + f'umap_digits_kmeans.png', dpi=300)
plt.show()
# %% PCA COMPARISON
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# plot the PCA embedding with ground truth color coding:
plt.figure(figsize=(6, 6))
colors = plt.cm.tab10(y)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap=plt.cm.tab10, s=2, alpha=0.6)
plt.title("PCA embedding of the digits dataset\nwith ground truth labeling")
plt.xticks([])
plt.yticks([])
# add a 'colorbar' that matches the cell types:
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
           for label, color in DIGIT_COLORS.items()]
plt.legend(handles=handles, title="digits", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(RESULTSPATH + f'umap_pca_digits_GT_labeling.png', dpi=300)
plt.show()
# %% END