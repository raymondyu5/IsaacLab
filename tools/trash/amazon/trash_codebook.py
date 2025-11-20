import sys

sys.path.append("submodule/benchmark_VAE/src")
import yaml
from pythae.models import AutoModel
import os
import torch
import numpy as np
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

device = "cuda"

vae_path = "logs/teleop_0523/right_vae/VQVAE_latent_2_minmax_codebook8/"

all_dirs = [
    d for d in os.listdir(vae_path) if os.path.isdir(os.path.join(vae_path, d))
]
last_training = sorted(all_dirs)[-1]

vae_model = AutoModel.load_from_folder(os.path.join(vae_path, last_training,
                                                    'final_model'),
                                       device="cuda")
vae_model.eval()

with open(f"{vae_path}/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

    action_mean = torch.as_tensor(model_config["action_mean"]).to(device)
    action_std = torch.as_tensor(model_config["action_std"]).to(device)
    data_normalizer = model_config["data_normalizer"]
    max_latent_value = np.array(model_config["max_latent_value"])
    min_latent_value = np.array(model_config["min_latent_value"])
    latent_dim = model_config["latent_dim"]


def index_to_codebook_combination(index, base, length):
    combo = []
    for _ in range(length):
        combo.append(index % base)
        index //= base
    return list(reversed(combo))  # to match product ordering


batch_size = 1024
num_embeddings = vae_model.quantizer.num_embeddings
total_combinations = num_embeddings**latent_dim
reconstructed_all = []

with torch.no_grad():
    for start in range(0, total_combinations, batch_size):
        batch_indices = range(start, min(start + batch_size,
                                         total_combinations))
        batch_combinations = [
            index_to_codebook_combination(idx, num_embeddings, latent_dim)
            for idx in batch_indices
        ]
        batch_combinations = torch.tensor(batch_combinations, device=device)
        decoded = vae_model.decode_action_index(batch_combinations)
        reconstructed = dataset_denrormalizer(decoded, action_mean, action_std)
        reconstructed_all.append(reconstructed.cpu())

reconstructed_hand = torch.cat(reconstructed_all, dim=0)

# Cluster them into K groups (e.g., K = 4)
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_ids = kmeans.fit_predict(reconstructed_hand.cpu().numpy())  # (16,)

# Optional: visualize with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
reduced = pca.fit_transform(reconstructed_hand.cpu().numpy())
from scipy.spatial import ConvexHull

plt.figure(figsize=(7, 6))

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

for cluster_id in range(4):
    mask = cluster_ids == cluster_id
    points = reduced[mask]

    # Scatter points
    plt.scatter(points[:, 0],
                points[:, 1],
                color=colors[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.6)

    # Draw convex hull around each cluster
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = np.append(hull.vertices,
                                hull.vertices[0])  # close the loop
        plt.plot(points[hull_points, 0],
                 points[hull_points, 1],
                 color=colors[cluster_id],
                 linestyle='--',
                 linewidth=1.5)

plt.title("VQ Codebook Embeddings (Separated Clusters)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.figure(figsize=(6, 5))
# plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap='tab10')
# plt.title("VQ Codebook Embeddings (clustered)")
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.colorbar(label="Cluster ID")
# plt.grid(True)
# plt.show()
