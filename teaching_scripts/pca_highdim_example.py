"""
A simple script to illustrate the application of PCA to a high-dimensional dataset using the Wine dataset and
scikit-learn. 

author: Fabrizio Musacchio
date: Oct 18, 2024
"""
# %% IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
# %% LOADING THE DATASET AND TAKE A FIRST LOOK AT IT
input_data = load_wine()
X = input_data.data

feature_names = input_data.feature_names

# Many dimensionality reduction algorithms are affected by scale, so we need to scale
# the features in our data before applying. e.g., PCA or FA. We can use StandardScaler 
# to standardize the data set’s features onto unit scale (mean = 0 and variance = 1).
# If we don’t scale our data, it can have a negative effect on our algorithm:
X_Factor = StandardScaler().fit_transform(X)
df = pd.DataFrame(data=X_Factor, columns=input_data.feature_names)

df_full = df.copy()
df_full['Target'] = input_data.target
target_names = input_data.target_names

# let's explore the data:
print(f"Shape of the dataset: {df.shape} (13 features with 178 samples)")
print(f"Feature names: {feature_names}")
print(f"Target names: {target_names}")

# %% PCA

# Explore the number of necessary PCA:
pca_explore = PCA(n_components=13)
pcas_explore = pca_explore.fit_transform(X_Factor)
ev = pca_explore.singular_values_

# Scree plot for validating the number of factors:
plt.figure(figsize=(6, 4))
plt.scatter(range(1, df.shape[1]+1), ev)
plt.plot(range(1, df.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('PC#')
plt.ylabel('Singular Value')
plt.ylim(0, 30)
plt.xticks(np.arange(1, df.shape[1]+1, 1))
plt.grid()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_scree_plot.png', dpi=300)
plt.show() 


"""
The explained variance tells us how much information (variance) can be attributed 
to each of the principal components. This is important because while we can convert 
high-dimensional space to a two- or three-dimensional space, we lose some of the variance 
(information). By using the attribute explained_variance_ratio, we can see that the 
first principal component contains XY percent of the variance,  the second XY percent and
the third XY percent of the variance. Together, the three components contain XY percent of the 
information.

btw., variance_explained_ratio = eigenvalues / np.sum(eigenvalues)
"""
plt.figure(figsize=(6, 5))
var=np.cumsum(np.round(pca_explore.explained_variance_ratio_, decimals=3) *100)
plt.plot(var)
plt.ylabel("% Variance Explained")
plt.xlabel("# of PCs") 
plt.title ("PCA Variance Explained")
plt.ylim(min(var), 100.5) 
#plt.style.context ('seaborn-whitegrid') 
plt.axhline(y=80, color='r', linestyle='--')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_variance_explained.png', dpi=300)
plt.show()

print(f"Explained variances for all 13 PCs:\n {pca_explore.explained_variance_ratio_}\n")
print(f"Cumulative explained variance for the first 3 PCs: {np.sum(pca_explore.explained_variance_ratio_[0:3])}")


# %% PERFORM PCA AND VISUALIZE THE RESULTS
# perform PCA with 3 components:
pca = PCA(n_components=3)
pcas = pca.fit_transform(X_Factor)

# create a dataframe with the 3 components and the target variable:
principal_df = pd.DataFrame(data=pcas, columns=['PC1', 'PC2', 'PC3'])
final_df = pd.concat([principal_df, pd.DataFrame(data=input_data.target, columns=['target'])], axis=1)

# visualize the 3 components:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcas[:, 0], pcas[:, 1], pcas[:, 2], c='k',  alpha=0.6)
ax.view_init(elev=35, azim=45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad=-5.5
ax.set_title('PCA of Wine Dataset')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_wine_dataset.png', dpi=300)
plt.show()


# visualize the 3 components with their groundtruth labels:
colors = ['r', 'g', 'b']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    ax.scatter(pcas[indices, 0], pcas[indices, 1], pcas[indices, 2], c=color, label=target)
ax.view_init(elev=35, azim=45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad=-5.5
ax.set_title('PCA of Wine Dataset with GT labels')
ax.legend()
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_wine_dataset_with_GT_labels.png', dpi=300)
plt.show()

# projections:
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
# XY projection: 
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[0].scatter(pcas[indices, 0], pcas[indices, 1], c=color, label=target)
axs[0].set_xlabel('PCA 1')
axs[0].set_ylabel('PCA 2')
axs[0].set_title('PC1 vs PC2')

# XZ projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[1].scatter(pcas[indices, 0], pcas[indices, 2], c=color, label=target)
axs[1].set_xlabel('PCA 1')
axs[1].set_ylabel('PCA 3')
axs[1].set_title('PC1 vs PC3')

# YZ projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[2].scatter(pcas[indices, 1], pcas[indices, 2], c=color, label=target)
axs[2].set_xlabel('PCA 2')
axs[2].set_ylabel('PCA 3')
axs[2].set_title('PC2 vs PC3')

plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_wine_dataset_projections.png', dpi=300)
plt.show()



# %% END