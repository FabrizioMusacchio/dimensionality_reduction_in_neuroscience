"""
A simple script to illustrate principles of PCA.

author: Fabrizio Musacchio
date: Oct 18, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

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
# %% SIMPLE EXAMPLE ON 2-FEATURE TOY DATA
# generate random two-dimensional data:
np.random.seed(1)
mean = [0, 0]
cov = [[1.5, 0.75], [0.75, 1.5]]
X = np.random.multivariate_normal(mean, cov, 100)

# visualization of the data:
plt.figure(figsize=(6, 7))
plt.scatter(X[:, 0], X[:, 1], zorder=2)
plt.title('Original data: Firing rates of\ntwo toy neurons')
plt.xlabel('Neuron 1')
plt.ylabel('Neuron 2')
plt.xticks(np.arange(-3,3.01,1))
plt.yticks(np.arange(-3,3.01,1))
plt.grid(True)
# make x and y axis have the same scale:
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_original_data.png', dpi=300)


# First, we standardize the data:

# centering the data:
X_centered = X - np.mean(X, axis=0)

# scaling of the data:
X_scaled = X_centered / np.std(X_centered, axis=0)


# Next, we calculate the covariance matrix:
cov_matrix = np.cov(X_scaled.T)


# Then, we determine the eigenvalues and eigenvectors of the covariance matrix 
# and sort the eigenvectors in descending order:

# calculation of the eigenvalues and eigenvectors:
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# sorting the eigenvectors according to descending eigenvalues:
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# The sorted eigenvectors represent the principal components of our data. 


# plot the two principle components in the original space of data:
plt.figure(figsize=(6, 7))
plt.scatter(X[:, 0], X[:, 1], zorder=2)
plt.quiver(np.mean(X[:, 0]), np.mean(X[:, 1]), eigenvectors[0, 0], eigenvectors[1, 0], 
           color='red', scale=2, label='Principal Component 1')
plt.quiver(np.mean(X[:, 0]), np.mean(X[:, 1]), eigenvectors[0, 1], eigenvectors[1, 1], 
           color='blue', scale=3, label='Principal Component 2')
plt.title('Principal components in\noriginal data space')
plt.xlabel('Neuron 1')
plt.ylabel('Neuron 2')
plt.legend()
plt.xticks(np.arange(-3,3.01,1))
plt.yticks(np.arange(-3,3.01,1))
plt.grid(True)
# make x and y axis have the same scale:
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_principal_components_original_data.png', dpi=300)
plt.show()



# %% PLOT THE ORTHOGONAL DISTANCES OF THE DATA POINTS TO THE FIRST PRINCIPAL COMPONENT
""" 
By projecting the data onto the principal components, we can illustrate the variance of the data
along the principal components. PCA aims to maximize the covered variance along the first principal component
(PC1) and then along the second principal component (PC2). 
"""

# project the data onto the first two principal components (PC1 and PC2):
X_pca1 = np.dot(X_scaled, eigenvectors[:, 0])
X_pca2 = np.dot(X_scaled, eigenvectors[:, 1])

# reconstruct the points projected on PC1 and PC2:
reconstructed_pc1 = np.outer(X_pca1, eigenvectors[:, 0])
reconstructed_pc2 = np.outer(X_pca2, eigenvectors[:, 1])

# create figure and plot data points along with PC1, PC2 and projections on each:
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Projection on PC1
axs[0].scatter(X_scaled[:, 0], X_scaled[:, 1], label='Original data', zorder=2)
for i in range(len(X_scaled)):
    axs[0].plot([X_scaled[i, 0], reconstructed_pc1[i, 0]], [X_scaled[i, 1], reconstructed_pc1[i, 1]], '--', 
                c="orange", lw=0.8, zorder=1)
axs[0].arrow(-2, -2, eigenvectors[0, 0]*6, eigenvectors[1, 0]*6, 
             color='red', head_width=0.1, head_length=0.1, lw=2, label='PC1')
axs[0].arrow(1.5, -1.5, eigenvectors[0, 1]*4, eigenvectors[1, 1]*4, 
             color='blue', head_width=0.1, head_length=0.1, lw=2, label='PC2')
axs[0].set_title('Data projection on PC1')
axs[0].set_xlabel('Neuron 1')
axs[0].set_ylabel('Neuron 2')
axs[0].grid(True)
axs[0].set_xlim(-3, 3)
axs[0].set_ylim(-3, 3)
# make x and y axis have the same scale:
axs[0].set_aspect('equal', adjustable='box')
axs[0].legend()

# Plot 2: Projection on PC2
axs[1].scatter(X_scaled[:, 0], X_scaled[:, 1], label='Original data', zorder=2)
for i in range(len(X_scaled)):
    axs[1].plot([X_scaled[i, 0], reconstructed_pc2[i, 0]], [X_scaled[i, 1], reconstructed_pc2[i, 1]], '--', 
                c="cyan", lw=0.8, zorder=1)
axs[1].arrow(-2, -2, eigenvectors[0, 0]*6, eigenvectors[1, 0]*6, 
             color='red', head_width=0.1, head_length=0.1, lw=2, label='PC1')
axs[1].arrow(1.5, -1.5, eigenvectors[0, 1]*4, eigenvectors[1, 1]*4, 
             color='blue', head_width=0.1, head_length=0.1, lw=2, label='PC2')
axs[1].set_title('Data projection on PC2')
axs[1].set_xlabel('Neuron 1')
axs[1].set_ylabel('Neuron 2')
axs[1].grid(True)
axs[1].set_xlim(-3, 3)
axs[1].set_ylim(-3, 3)
axs[1].set_aspect('equal', adjustable='box')
axs[1].legend()

plt.tight_layout()
plt.savefig(RESULTSPATH + 'pca_projections.png', dpi=300)
plt.show()


# %% PLOT THE ORTHOGONAL DISTANCES OF THE DATA POINTS TO THE FIRST PRINCIPAL COMPONENT (ROTATED)
""" 
If the principal components are note properly aligned with the data, the overall variance of the data
is not captured efficiently, i.e., the variance along the principal components is not maximized. 
This can be seen in the following example, where we rotate the principal components by 20 degrees and 
calculate the projections of the data onto the rotated PCs.
"""
 
# Let's rotate both principal components by 10 degrees and calculate the projections 
# of the data onto the rotated PCs:
theta = np.radians(20) # rotation angle of 20 degrees
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# rotate the eigenvectors (principal components):
rotated_eigenvectors = np.dot(eigenvectors, rotation_matrix)

# project the data onto the rotated principal components:
X_pca1_rotated = np.dot(X_scaled, rotated_eigenvectors[:, 0])
X_pca2_rotated = np.dot(X_scaled, rotated_eigenvectors[:, 1])

# reconstruct the points projected on the rotated PC1 and PC2:
reconstructed_pc1_rotated = np.outer(X_pca1_rotated, rotated_eigenvectors[:, 0])
reconstructed_pc2_rotated = np.outer(X_pca2_rotated, rotated_eigenvectors[:, 1])

# create figure and plot data points along with rotated PCs and projections:
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Projection on PC1
axs[0].scatter(X_scaled[:, 0], X_scaled[:, 1], label='Original data', zorder=2)
for i in range(len(X_scaled)):
    axs[0].plot([X_scaled[i, 0], reconstructed_pc1_rotated[i, 0]], [X_scaled[i, 1], reconstructed_pc1_rotated[i, 1]], '--', 
                c="orange", lw=0.8, zorder=1)
axs[0].arrow(-1.40, -3, rotated_eigenvectors[0, 0]*6.25, rotated_eigenvectors[1, 0]*6.25, 
             color='red', head_width=0.1, head_length=0.1, lw=2, label='PC1')
axs[0].arrow(2.1, -1.0, rotated_eigenvectors[0, 1]*4.5, rotated_eigenvectors[1, 1]*4.5, 
             color='blue', head_width=0.1, head_length=0.1, lw=2, label='PC2')
axs[0].set_title('Data projection on PC1')
axs[0].set_xlabel('Neuron 1')
axs[0].set_ylabel('Neuron 2')
axs[0].grid(True)
axs[0].set_xlim(-3, 3)
axs[0].set_ylim(-3, 3)
# make x and y axis have the same scale:
axs[0].set_aspect('equal', adjustable='box')
axs[0].legend()

# Plot 2: Projection on PC2
axs[1].scatter(X_scaled[:, 0], X_scaled[:, 1], label='Original data', zorder=2)
for i in range(len(X_scaled)):
    axs[1].plot([X_scaled[i, 0], reconstructed_pc2_rotated[i, 0]], [X_scaled[i, 1], reconstructed_pc2_rotated[i, 1]], '--', 
                c="cyan", lw=0.8, zorder=1)
axs[1].arrow(-1.40, -3, rotated_eigenvectors[0, 0]*6.25, rotated_eigenvectors[1, 0]*6.25, 
             color='red', head_width=0.1, head_length=0.1, lw=2, label='PC1')
axs[1].arrow(2.1, -1.0, rotated_eigenvectors[0, 1]*4.5, rotated_eigenvectors[1, 1]*4.5, 
             color='blue', head_width=0.1, head_length=0.1, lw=2, label='PC2')
axs[1].set_title('Data projection on PC2')
axs[1].set_xlabel('Neuron 1')
axs[1].set_ylabel('Neuron 2')
axs[1].grid(True)
axs[1].set_xlim(-3, 3)
axs[1].set_ylim(-3, 3)
axs[1].set_aspect('equal', adjustable='box')
axs[1].legend()

plt.tight_layout()
plt.savefig(RESULTSPATH + 'pca_projections_rotated.png', dpi=300)
plt.show()
# %% PROJECT THE DATA ONTO THE FIRST TWO PRINCIPAL COMPONENTS (TRANSFORMED DATA)
# Finally, we project the data onto the principal components and visualize the result. 

# projecting the data onto the principal components:
projected_data = np.dot(X_scaled, eigenvectors[:, :2])

# visualization of the data projected on the first two main components:
fig = plt.figure(figsize=(6, 7))
plt.scatter(projected_data[:, 0], projected_data[:, 1], zorder=2)
plt.title('Data projected onto first two\nprincipal components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# indicate first PC direction by horizontal blue line:
plt.axhline(0, color='red', linestyle='-', label='PC 1 axis')
# indicate second PC direction by vertical red line:
plt.axvline(0, color='blue', linestyle='-', label='PC 2 axis')
plt.xticks(np.arange(-3,3.01,1))
plt.yticks(np.arange(-3,3.01,1))
plt.grid(True)
plt.legend()
# make x and y axis have the same scale:
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'pca_projected_data.png', dpi=300)
plt.show()
# %% END