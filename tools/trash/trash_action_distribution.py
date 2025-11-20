import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

bin_count = 50
data = h5py.File("logs/data_0604/raw_right_data.hdf5", "r")["data"]

action_buffer = []
for index in range(len(data)):
    action_buffer.append(data[f"demo_{index}"]["actions"])

action_buffer = np.concatenate(action_buffer, axis=0)[...,
                                                      -16:].reshape(-1, 16)
action_mean = action_buffer.mean(0)


def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    ncols = 4
    nrows = math.ceil(n_elems / ncols)

    fig = plt.figure(tag, figsize=(5 * ncols, 4 * nrows))
    for elem in range(n_elems):
        plt.subplot(nrows, ncols, elem + 1)
        plt.hist(vector[:, elem], bins=bin_count)
        plt.title(f"{vector_mean[elem]:.2f}")
    plt.tight_layout()
    plt.show()


vis_stats(action_buffer, action_mean, 'action_stats')
# plt.savefig("rl_action_stats.png")
plt.show()

from sklearn.neighbors import KernelDensity
import numpy as np

from sklearn.cluster import KMeans


def fit_and_resample_gaussian(X, n_samples, spread_scale=4.0, bins=bin_count):
    n, d = X.shape
    new_data = []
    all_debug_info = []

    for i in range(d):
        x_min, x_max = X[:, i].min(), X[:, i].max()

        mu = (x_min + x_max) / 2
        sigma = (x_max - x_min) / spread_scale
        # mu = np.mean(X[:, i])
        # sigma = np.std(X[:, i])

        samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        new_data.append(samples)

        counts, bin_edges = np.histogram(samples,
                                         bins=bins,
                                         range=(x_min, x_max))
        all_debug_info.append({
            'samples': samples,
            'counts': counts,
            'bin_edges': bin_edges,
            'mu': mu,
            'sigma': sigma
        })

    return np.stack(new_data, axis=1), all_debug_info  # Shape: (n_samples, d)


#####
def match_histogram_bin_counts(original_data, debug_infos, target_dim=-3):
    """
    For each dimension, sample from original_data using bin counts from debug_infos.
    Ensures same length per dimension so they can be stacked.
    """
    n, d = original_data.shape
    matched_samples = []
    sample_lengths = []

    x = original_data[:, target_dim]
    bin_edges = debug_infos[target_dim]['bin_edges']
    bin_counts = debug_infos[target_dim]['counts']

    # Assign each value to a bin
    counts, data_bin_edges = np.histogram(x, bins=len(bin_counts))
    bin_indices = np.digitize(x, bins=data_bin_edges) - 1

    dim_samples = []

    for b, count in enumerate(bin_counts):
        indices_in_bin = np.where(bin_indices == b)[0]
        if len(indices_in_bin) == 0:
            continue  # no data in this bin

        sample_indices = np.random.choice(indices_in_bin,
                                          size=count,
                                          replace=(len(indices_in_bin)
                                                   < count))

        dim_samples.append(original_data[sample_indices])

    dim_samples = np.concatenate(dim_samples, axis=0)
    return dim_samples
    # sample_lengths.append(len(dim_samples))
    # matched_samples.append(dim_samples)

    # # Truncate all to same min length
    # min_len = min(sample_lengths)

    # matched_samples = [x[:min_len] for x in matched_samples]
    # return np.stack(matched_samples, axis=1)  # shape: (min_len, d)


resampled_actions, debug_infos = fit_and_resample_gaussian(
    action_buffer, action_buffer.shape[0], bins=bin_count)
matched_actions = match_histogram_bin_counts(action_buffer, debug_infos)
vis_stats(matched_actions,
          matched_actions.mean(0),
          tag="matched_by_gaussian_bins")

plt.show()

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

num_clusters = 100
# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(action_buffer)
labels = kmeans.labels_  # (batch,)
centers = kmeans.cluster_centers_  # (5, 16)

# Dimensionality reduction to 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(action_buffer)
centers_3d = pca.transform(centers)

# 3D plotting
action_np = np.array(action_buffer)

dim = 16
bins = 40

# Create a grid: 5 rows (clusters) × 16 columns (dimensions)
fig, axes = plt.subplots(num_clusters,
                         dim,
                         figsize=(4 * dim, 2.5 * num_clusters),
                         constrained_layout=True)

for cluster_id in range(num_clusters):
    cluster_data = action_np[labels ==
                             cluster_id]  # Filter data in this cluster
    for d in range(dim):
        ax = axes[cluster_id, d]
        ax.hist(cluster_data[:, d], bins=bins, color='tab:blue', alpha=0.7)
        if cluster_id == 0:
            ax.set_title(f"Dim {d}")
        if d == 0:
            ax.set_ylabel(f"Cluster {cluster_id}")
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

plt.suptitle("Distribution of 16 Action Dimensions per Cluster", fontsize=18)
plt.show()
