"""
A simple script to illustrate Kullback-Leibler divergence.

author: Fabrizio Musacchio
date: Oct 30, 2024

"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import entropy

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
# %% KL DIVERGENCE
def kl_divergence(p, q):
    """Compute KL divergence between two probability distributions."""
    return entropy(p, q)
# %% EXAMPLES
# Generate x values
x = np.linspace(-10, 10, 1000)

# Define two Gaussian distributions with different parameters
# Case 1: Distributions closer together
mean_p1, std_p1 = 0, 1
mean_q1, std_q1 = 5, 1.5
p1 = norm.pdf(x, mean_p1, std_p1)
q1 = norm.pdf(x, mean_q1, std_q1)
kl_div1 = kl_divergence(p1, q1)

# Case 2: Distributions further apart
mean_p2, std_p2 = 0, 1
mean_q2, std_q2 = 0.5, 1.5
p2 = norm.pdf(x, mean_p2, std_p2)
q2 = norm.pdf(x, mean_q2, std_q2)
kl_div2 = kl_divergence(p2, q2)

# Plot the two cases
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.25))

# Plot for Case 1 (distributions closer together)
ax1.plot(x, p1, label=f'$P$ ($\mu$={mean_p1}, $\sigma$={std_p1})', color='turquoise')
ax1.plot(x, q1, label=f'$Q$ ($\mu$={mean_q1}, $\sigma$={std_q1})', color='hotpink')
ax1.fill_between(x, p1, q1, color="gray", alpha=0.2)
ax1.set_title(f"KL Divergence = {kl_div1:.3f}")
ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend(loc='upper left', fontsize=12)

# Plot for Case 2 (distributions further apart)
ax2.plot(x, p2, label=f'$P$ ($\mu$={mean_p2}, $\sigma$={std_p2})', color='turquoise')
ax2.plot(x, q2, label=f'$Q$ ($\mu$={mean_q2}, $\sigma$={std_q2})', color='hotpink')
ax2.fill_between(x, p2, q2, color="gray", alpha=0.2)
ax2.set_title(f"KL Divergence = {kl_div2:.3f}")
ax2.set_xlabel('x')
#ax2.set_ylabel('Density')
ax2.legend(loc='upper left', fontsize=12)

# add super title:
fig.suptitle(f'KL divergence of two example Gaussian distributions $P$ and $Q$')

plt.tight_layout()
plt.savefig(RESULTSPATH + f'KL_divergence_demonstrated.png', dpi=300)
plt.show()
# %% END