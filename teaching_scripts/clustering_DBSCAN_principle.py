# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.cluster import DBSCAN
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
blobs_data = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)[0]

# plot the data:
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(blobs_data[:, 0], blobs_data[:, 1], c='black', s=10)
plt.title('Data with three clustered blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(RESULTSPATH + 'clustering_dbscan_principle_data.png', dpi=300, bbox_inches='tight')
plt.show()
# %% DBSCAN CLUSTERING EXAMPLE

# Apply DBSCAN
epsilon = 1.0
db = DBSCAN(eps=epsilon, min_samples=10).fit(blobs_data)
labels = db.labels_

# Core samples are labeled as True, non-core as False
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Plot the result
fig, ax = plt.subplots(figsize=(8, 5))
unique_labels = set(labels)
colors = [plt.cm.tab10(each) for each in np.linspace(0, 0.5, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    #print(k, col)
    if k == -1:
        # violet for noise
        col = [0.5, 0.5, 0.5, 0.5]
        curr_label = 'Noise\n(cluster -1)'
    else:
        curr_label = f'cluster {k}'

    class_member_mask = (labels == k)

    # Plot core points
    xy = blobs_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14, label=f'Core points {curr_label}',
             alpha=0.75)

    # plot border points:
    xy = blobs_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6, label=f'Border points {curr_label}',
             alpha=0.75)
    
    """ # Draw eps-circles around core points
    xy_core = blobs_data[class_member_mask & core_samples_mask]
    for point in xy_core:
        circle = Circle(point, epsilon, color=tuple(col), fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(circle) """


plt.title(f'DBSCAN: Core points, border points, and noise\neps={epsilon}, min_samples=10')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(fontsize=12, bbox_to_anchor=(1.00, 1))
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# make axes equal:
#plt.axis('equal')
plt.tight_layout()
plt.savefig(RESULTSPATH + f'clustering_dbscan_principle_eps{epsilon}.png', dpi=300, bbox_inches='tight')
plt.show()
# %% END