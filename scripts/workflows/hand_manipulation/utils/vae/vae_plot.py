import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import yaml
import argparse
import gym
import os
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import QuantileTransformer

from sklearn.manifold import TSNE
import scipy.stats as stats

import numpy as np

from scipy.stats import norm, zscore, gaussian_kde, probplot


def plot_action(mu_action, log_dir, model_type):
    # Ensure it's a NumPy array
    if not isinstance(mu_action, np.ndarray):
        mu_action = mu_action.cpu().numpy()

    num_dims = mu_action.shape[1]

    # Create 3-column layout: Raw, KDE, Q-Q Plot
    fig, axes = plt.subplots(num_dims,
                             3,
                             figsize=(18, 3 * num_dims),
                             constrained_layout=True)

    for i in range(num_dims):
        raw_flat = mu_action[:, i].flatten()

        for j, label in enumerate(
            ['Raw', 'Gaussian KDE', 'Histogram Gaussian Fit']):
            ax = axes[i, j] if num_dims > 1 else axes[j]

            if label == "Raw":
                ax.hist(raw_flat,
                        bins=40,
                        alpha=0.6,
                        density=True,
                        label="Raw")
                x = np.linspace(min(raw_flat), max(raw_flat), 100)
                ax.plot(x,
                        norm.pdf(x, loc=raw_flat.mean(), scale=raw_flat.std()),
                        linestyle='--',
                        linewidth=2,
                        label="Gaussian Fit")

            elif label == "Gaussian KDE":
                ax.hist(raw_flat,
                        bins=40,
                        alpha=0.6,
                        density=True,
                        label="KDE")
                x = np.linspace(min(raw_flat), max(raw_flat), 100)
                kde = gaussian_kde(raw_flat)
                ax.plot(x, kde(x), linestyle='--', linewidth=2, label="KDE")

            elif label == "Histogram Gaussian Fit":
                # Histogram
                counts, bin_edges, _ = ax.hist(raw_flat,
                                               bins=40,
                                               alpha=0.6,
                                               density=True,
                                               label="Raw")
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                # Estimate mean and std from histogram bins
                hist_mean = np.sum(bin_centers * counts) / np.sum(counts)
                hist_var = np.sum(
                    counts * (bin_centers - hist_mean)**2) / np.sum(counts)
                hist_std = np.sqrt(hist_var)

                # Plot Gaussian based on histogram fit
                x = np.linspace(min(raw_flat), max(raw_flat), 100)

                ax.plot(x,
                        norm.pdf(x, loc=hist_mean, scale=hist_std),
                        linestyle='--',
                        linewidth=2,
                        label=f"Hist Gaussian Fit")

            ax.set_title(f"z[{i}] - {label}")

    plt.savefig(
        f"{log_dir}/vae_action_distribution_{model_type}.png",
        dpi=300,
        bbox_inches='tight',
    )


def visualize_benchmark_latent_space(model,
                                     data_loader,
                                     device,
                                     log_dir,
                                     name="train"):

    all_mu = []

    with torch.no_grad():

        mu = model(data_loader).embedding
        all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    latent_dim = all_mu.shape[1]

    # Print range summary
    print(
        f"{name} latent mean stats: mean={all_mu.mean():.3f}, std={all_mu.std():.3f}, "
        f"min={all_mu.min():.3f}, max={all_mu.max():.3f}")

    plt.figure(figsize=(12, 4))
    for i in range(min(16, latent_dim)):
        plt.hist(all_mu[:, i], bins=40, alpha=0.6, label=f"z[{i}]")
    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()

    # plt.show()
    plt.savefig(os.path.join(log_dir, f"{name}_latent_distribution.png"))

    return all_mu.max(dim=0).values.cpu().numpy(), all_mu.min(
        dim=0).values.cpu().numpy()


def visualize_latent_space(model, data_loader, device, log_dir, name="train"):

    all_mu = []

    with torch.no_grad():
        for state, action in data_loader:  # assuming the loader yields (state, action)
            state = state.to(device)
            mu = model.encoder(state.to(device), action.to(device))[0]
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    latent_dim = all_mu.shape[1]

    # Print range summary
    print(
        f"{name} latent mean stats: mean={all_mu.mean():.3f}, std={all_mu.std():.3f}, "
        f"min={all_mu.min():.3f}, max={all_mu.max():.3f}")

    plt.figure(figsize=(12, 4))
    for i in range(min(16, latent_dim)):
        plt.hist(all_mu[:, i], bins=40, alpha=0.6, label=f"z[{i}]")
    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()

    # plt.show()
    plt.savefig(os.path.join(log_dir, f"{name}_latent_distribution.png"))

    return all_mu.max(dim=0).values.cpu().numpy(), all_mu.min(
        dim=0).values.cpu().numpy()


def plot_action_distribution(action_tensor,
                             save_path,
                             name,
                             max_dims=16,
                             title="Action Distribution"):
    action_tensor = action_tensor  # if it's on GPU
    dim = action_tensor.shape[1]
    dims_to_plot = min(dim, max_dims)

    plt.figure(figsize=(16, 4))
    for i in range(dims_to_plot):
        plt.subplot(2, (dims_to_plot + 1) // 2, i + 1)
        plt.hist(action_tensor[:, i], bins=40, alpha=0.7)
        plt.title(f"Dim {i}")
        plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    # plt.show()
    plt.savefig(
        os.path.join(save_path, f"{name}.png"),
        dpi=300,
        bbox_inches='tight',
    )


def visualize_benchmark_latent_space(model,
                                     data_dict,
                                     device,
                                     log_dir,
                                     name="train",
                                     batch_size=32):

    action_chunk = data_dict["action_chunk"]
    state = data_dict["state"]

    assert len(action_chunk) == len(state), "Mismatched lengths"

    all_mu = []

    with torch.no_grad():
        for start in range(0, len(action_chunk), batch_size):
            end = start + batch_size
            batch_action = action_chunk[start:end].to(device)
            batch_state = state[start:end].to(device)

            result = model({
                "action_chunk": batch_action,
                "state": batch_state
            })

            mu = result.embedding  # e.g., from encoder's latent mean
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0).reshape(len(action_chunk), -1)
    latent_dim = all_mu.shape[-1]

    # Print summary
    print(
        f"{name} latent mean stats: mean={all_mu.mean():.3f}, std={all_mu.std():.3f}, "
        f"min={all_mu.min():.3f}, max={all_mu.max():.3f}")

    # Plot
    plt.figure(figsize=(12, 4))
    for i in range(min(16, latent_dim)):
        plt.hist(all_mu[:, i], bins=40, alpha=0.6, label=f"z[{i}]")

    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, f"{name}_latent_distribution.png"))
    plt.close()

    return all_mu.max(dim=0).values.numpy(), all_mu.min(dim=0).values.numpy()


def visualize_single_action_latent_space(model,
                                         data_dict,
                                         device,
                                         log_dir,
                                         name="train",
                                         batch_size=2048):

    action = data_dict

    all_mu = []

    with torch.no_grad():
        for start in range(0, len(action), batch_size):
            end = start + batch_size
            batch_action = action[start:end].to(device)

            result = model(batch_action)

            mu = result.embedding  # e.g., from encoder's latent mean
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0).reshape(len(action), -1)
    latent_dim = all_mu.shape[-1]

    # Print summary
    print(
        f"{name} latent mean stats: mean={all_mu.mean():.3f}, std={all_mu.std():.3f}, "
        f"min={all_mu.min():.3f}, max={all_mu.max():.3f}")

    # Plot
    plt.figure(figsize=(12, 4))
    for i in range(min(16, latent_dim)):
        plt.hist(all_mu[:, i], bins=40, alpha=0.6, label=f"z[{i}]")

    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, f"{name}_latent_distribution.png"))
    plt.close()

    return all_mu.max(dim=0).values.numpy(), all_mu.min(dim=0).values.numpy()


def visualize_vq_latent_space(model,
                              data_dict,
                              device,
                              log_dir,
                              name="train",
                              batch_size=1024):

    action_chunk = data_dict["data"]["action_chunk"]
    state = data_dict["data"]["state"]

    assert len(action_chunk) == len(state), "Mismatched lengths"

    all_indices = []

    with torch.no_grad():
        for start in range(0, len(action_chunk), batch_size):
            end = start + batch_size
            batch_action = action_chunk[start:end].to(device)
            batch_state = state[start:end].to(device)

            result = model(
                {"data": {
                    "action_chunk": batch_action,
                    "state": batch_state
                }})

            indices = result.quantized_indices  # shape: (B, latent_dim)
            all_indices.append(indices.cpu())

    result = torch.cat(all_indices, dim=0)
    latent_dim = result.shape[-1]
    all_mu = result.reshape(result.shape[0], -1).numpy()

    # Plot histogram
    plt.figure(figsize=(12, 4))
    for i in range(min(16, latent_dim)):
        plt.hist(all_mu[:, i], bins=40, alpha=0.6, label=f"z[{i}]")

    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, f"{name}_latent_vq_distribution.png"))
    plt.close()


def plot_action_distribution(action_tensor,
                             save_path,
                             name,
                             max_dims=16,
                             title="Action Distribution"):
    action_tensor = action_tensor  # if it's on GPU
    dim = action_tensor.shape[1]
    dims_to_plot = min(dim, max_dims)

    plt.figure(figsize=(16, 4))
    for i in range(dims_to_plot):
        plt.subplot(2, (dims_to_plot + 1) // 2, i + 1)
        plt.hist(action_tensor[:, i], bins=40, alpha=0.7)
        plt.title(f"Dim {i}")
        plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    # plt.show()
    plt.savefig(
        os.path.join(save_path, f"{name}.png"),
        dpi=300,
        bbox_inches='tight',
    )
