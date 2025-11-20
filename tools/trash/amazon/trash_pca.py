import h5py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

import copy


def h5py2zarr(path):

    root = {'meta': {}, 'data': {}}
    action_dim = 0
    with h5py.File(path, "r") as f:

        demo_buffer = f['data']

        episode_ends = []
        episode_count = 0
        action_buffer = []
        state_buffer = []
        obs_buffer = []
        for key in demo_buffer.keys():

            if "obs" in demo_buffer[key].keys():
                import pdb
                pdb.set_trace()

                action = np.array(demo_buffer[key]['actions'])
                state = np.array(demo_buffer[key]['obs']["state"])

            else:
                action = demo_buffer[key]['actions'][1:, -16:]
                state = demo_buffer[key]['actions'][:-1, -16:]
            num_steps = action.shape[0]

            episode_ends.append(copy.deepcopy(num_steps + episode_count))
            episode_count += num_steps
            action_buffer.append(copy.deepcopy(action).clip(-1, 1))

            state_buffer.append(copy.deepcopy(state))
            obs_buffer.append(copy.deepcopy(state))
        action_dim = action.shape[-1]

        root['meta']['episode_ends'] = np.array(episode_ends, dtype=np.int64)
        action_buffer = np.concatenate(action_buffer, axis=0)
        state_buffer = np.concatenate(state_buffer, axis=0)
        obs_buffer = np.concatenate(obs_buffer, axis=0)
        root['data']['action'] = action_buffer
        root['data']['state'] = state_buffer
        root['data']['hand_joints'] = obs_buffer

    return root, action_dim


root, action_dim = h5py2zarr(
    path="logs/data_0705/retarget_visionpro_data/retarget_visionpro_data.hdf5")

state = root['data']['state']
hand_joints = root['data']['hand_joints']
action = root['data']['action']
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter


# Step 1: Summarize each motion trajectory to a fixed-length feature
def summarize_motion(motion):  # motion shape: (T, D)
    mean_pose = motion.mean(axis=0)  # (D,)
    mean_velocity = np.diff(motion, axis=0).mean(axis=0)  # (D,)
    return np.concatenate([mean_pose, mean_velocity])  # (2*D,)


# features = np.array([summarize_motion(m) for m in motions])  # shape: (B, 2*D)

# # Step 2: Normalize the features
# scaler = StandardScaler()
# features_norm = scaler.fit_transform(features)

# # Step 3: Run KMeans clustering
# n_clusters = 100
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_ids = kmeans.fit_predict(features_norm)  # shape: (B,)

# # Step 4: Analyze cluster distribution
# cluster_counts = Counter(cluster_ids)
# print("Cluster distribution:", dict(cluster_counts))

# # Step 5: Visualize clusters using t-SNE (optional)
# tsne = TSNE(n_components=2, perplexity=10, random_state=42)

# features_2d = tsne.fit_transform(features_norm)

# plt.figure(figsize=(8, 6))
# plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_ids, cmap='tab10')
# plt.colorbar(label="Cluster ID")
# plt.title("t-SNE Visualization of Hand Motion Clusters")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.tight_layout()
# plt.show()


def compute_eigengrasps(X,
                        num_components=6,
                        save_path="eigengrasps_info_pca_9.npy"):
    """
    Args:
        X: (N, D) torch.Tensor, hand pose dataset
        num_components: int, number of principal components to keep
        save_path: str, path to save the eigengrasp information
    """

    print(f"Original data shape: {X.shape}")  # (N, D)

    # Step 1: Normalize the data
    D_mean = X.mean(dim=0)
    D_std = X.std(dim=0)
    X_norm = (X - D_mean) / D_std

    print("Data normalized.")

    # Step 2: Perform PCA using SVD
    U, S, Vh = torch.linalg.svd(X_norm, full_matrices=False)  # Vh: (D, D)
    eigen_vectors = Vh[:num_components]  # (K, D)

    print(f"Computed top-{num_components} eigengrasps.")

    # Step 3: Project data onto eigenvectors (eigengrasp values)
    eigengrasp_values = torch.matmul(X_norm, eigen_vectors.T)  # (N, K)

    # Step 4: Compute min and max of eigengrasp values
    # lower_values = torch.min(eigengrasp_values, dim=0).values  # (K,)
    # upper_values = torch.max(eigengrasp_values, dim=0).values  # (K,)
    lower_values = torch.quantile(eigengrasp_values, 0.20, dim=0)
    upper_values = torch.quantile(eigengrasp_values, 0.80, dim=0)

    print("Computed 90% projection value range.")

    # Step 5: Save everything
    eigengrasps_info = {
        "eigen_vectors": eigen_vectors.cpu().numpy(),  # (K, D)
        "lower_values": lower_values.cpu().numpy(),  # 5th percentile
        "upper_values": upper_values.cpu().numpy(),  # 95th percentile
        "D_mean": D_mean.cpu().numpy(),  # (D,)
        "D_std": D_std.cpu().numpy(),  # (D,)
    }
    os.makedirs(dir, exist_ok=True)

    np.save(f"{dir}/right_pca", eigengrasps_info)

    print(f"Saved eigengrasps info to {save_path}")

    eigengrasp_values = torch.matmul(X_norm, eigen_vectors.T)

    # Assuming eigengrasp_values is a torch tensor of shape (N, 9)
    proj = eigengrasp_values.cpu().numpy()

    plt.figure(figsize=(12, 4))
    for i in range(proj.shape[1]):
        plt.hist(proj[:, i], bins=40, alpha=0.6, label=f"z[{i}]")
    plt.title(f"Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # # 2D Scatter plot
    # plt.figure(figsize=(6, 5))
    # plt.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.6)
    # plt.title("Projection onto First Two Eigengrasps (PC1 vs PC2)")
    # plt.xlabel("Eigengrasp 1")
    # plt.ylabel("Eigengrasp 2")
    # plt.grid(True)
    # plt.show()

    return eigen_vectors, eigengrasp_values, lower_values, upper_values, D_mean, D_std


dir = "logs/data_0705/retarget_visionpro_data"
eigen_vectors, eigengrasp_values, lower_values, upper_values, D_mean, D_std = compute_eigengrasps(
    torch.as_tensor(action)[..., -16:], num_components=9)

import torch
import numpy as np


# -----------------------
# Decode function
# -----------------------
def reconstruct_hand_pose_from_normalized_action(a_normalized, eigen_vectors,
                                                 min_values, max_values,
                                                 D_mean, D_std):
    """
    Args:
        a_normalized: (B, K) or (K,), RL output in [-1, 1]
        eigen_vectors: (K, D)
        min_values: (K,)
        max_values: (K,)
        D_mean: (D,)
        D_std: (D,)
    
    Returns:
        x: (B, D) if batched, or (D,) if single
    """
    # If a_normalized is (K,), unsqueeze to (1, K)
    # single_input = False
    # if a_normalized.dim() == 1:
    #     a_normalized = a_normalized.unsqueeze(0)
    #     single_input = True

    # Step 1: Scale normalized action to eigengrasp range

    if isinstance(a_normalized, np.ndarray):
        # NumPy version
        a_scaled = (a_normalized + 1) / 2 * (max_values -
                                             min_values) + min_values  # (B, K)
        x_norm = a_scaled @ eigen_vectors  # (B, D)
        x = (x_norm * D_std + D_mean)[0]  # (B, D)
    else:
        # PyTorch version
        a_scaled = (a_normalized + 1) / 2 * (max_values -
                                             min_values) + min_values  # (B, K)
        x_norm = torch.matmul(a_scaled, eigen_vectors)  # (B, D)
        x = x_norm * D_std + D_mean  # (B, D)
    return x


# -----------------------
# Demo usage
# -----------------------

# Example 1: single latent action
a_normalized = torch.rand(9) * 2 - 1  # random in [-1, 1]

# Example 2: batch of latent actions
a_normalized_batch = torch.rand(100, 9) * 2 - 1  # (5, K)
hand_actions = reconstruct_hand_pose_from_normalized_action(
    a_normalized_batch, eigen_vectors, lower_values, upper_values, D_mean,
    D_std)

print("\nBatch hand actions shape:", hand_actions.shape)
print("Batch hand actions:\n", hand_actions)
