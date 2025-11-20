import numpy as np
import h5py

import torch
import collections
import math

import torch
import math

import yaml
import sys

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.model.common.normalizer import LinearNormalizer
import isaaclab.utils.math as math_utils


class TemporalEnsembleImageObservation:

    def __init__(
        self,
        num_envs: int,
        horizon_K: int,
        obs_dim: int,
        m: float = 0.4,
        device="cuda",
    ):
        self.B = num_envs
        self.K = horizon_K
        self.D = obs_dim
        self.m = m
        self.device = device

    def reset(self, num_timesteps: int = 0, image_dim=None, num_envs=None):
        if num_envs is not None:
            self.B = num_envs

        # Initialize empty buffers with size 0
        self.buffer = torch.empty((self.B, num_timesteps, *image_dim),
                                  device=self.device)  # (B, T, D)
        self.steps = torch.empty((self.B, num_timesteps),
                                 dtype=torch.long,
                                 device=self.device)  # (B, T)

    def add_obs(self, current_steps: int, predicted_chunks: torch.Tensor):

        # Append new predictions
        self.buffer[:, current_steps] = predicted_chunks  # (B, T, D)
        self.steps[:, current_steps] = current_steps
        self.current_steps = current_steps
        # print("add obs current steps:", self.current_steps)

    def compute_obs(self, ):
        # print("current steps:", self.current_steps)
        start_time = max(self.current_steps - self.K, 0)
        chunk_indices = self.current_steps + 1 if self.current_steps <= self.K - 1 else start_time + self.K

        temporal_obs = self.buffer[:, start_time:
                                   chunk_indices, :]  # shape: (B, T, D)

        if temporal_obs.shape[1] < self.K:
            B, T, D = temporal_obs.shape
            pad_len = self.K - T
            first_value = temporal_obs[:,
                                       0:, :]  # shape: [B, 1, D] # pad first value
            padding = first_value.repeat(1, pad_len,
                                         1)  # shape: [B, pad_len, D]
            temporal_obs = torch.cat([padding, temporal_obs], dim=1)

        return temporal_obs  # (B,T, D)


class TemporalEnsembleBufferObservation:

    def __init__(self,
                 num_envs: int,
                 horizon_K: int,
                 obs_dim: int,
                 m: float = 0.4,
                 device="cuda"):
        self.B = num_envs
        self.K = horizon_K
        self.D = obs_dim
        self.m = m
        self.device = device

    def reset(self, num_timesteps: int = 0, num_envs=None):
        if num_envs is not None:
            self.B = num_envs

        # Initialize empty buffers with size 0
        self.buffer = torch.empty((self.B, num_timesteps, self.D),
                                  device=self.device)  # (B, T, D)
        self.steps = torch.empty((self.B, num_timesteps),
                                 dtype=torch.long,
                                 device=self.device)  # (B, T)

    def add_obs(self, current_steps: int, predicted_chunks: torch.Tensor):

        # Append new predictions
        self.buffer[:, current_steps] = predicted_chunks  # (B, T, D)
        self.steps[:, current_steps] = current_steps
        self.current_steps = current_steps

    def compute_obs(self, ):
        start_time = max(self.current_steps - self.K, 0)
        chunk_indices = self.current_steps + 1 if self.current_steps <= self.K - 1 else start_time + self.K

        temporal_obs = self.buffer[:, start_time:
                                   chunk_indices, :]  # shape: (B, T, D)

        if temporal_obs.shape[1] < self.K:
            B, T, D = temporal_obs.shape
            pad_len = self.K - T
            first_value = temporal_obs[:,
                                       0:, :]  # shape: [B, 1, D] # pad first value
            padding = first_value.repeat(1, pad_len,
                                         1)  # shape: [B, pad_len, D]
            temporal_obs = torch.cat([padding, temporal_obs], dim=1)

        return temporal_obs  # (B,T, D)


class TemporalEnsembleBufferAction:

    def __init__(self,
                 num_envs: int,
                 horizon_K: int,
                 action_dim: int,
                 execution_steps: int = 2,
                 m: float = 1.0,
                 device="cuda"):
        self.B = num_envs
        self.K = horizon_K
        self.D = action_dim
        self.m = m
        self.device = device
        self.execution_steps = execution_steps

    def reset(self, num_timesteps: int = 0, num_envs=None):
        if num_envs is not None:
            self.B = num_envs

        # Initialize empty buffers with size 0
        self.buffer = torch.empty((self.B, num_timesteps, self.K, self.D),
                                  device=self.device)  # (B, T, K, D)
        self.steps = torch.empty((self.B, num_timesteps),
                                 dtype=torch.long,
                                 device=self.device)  # (B, T)

    def add_prediction(self, current_steps: int,
                       predicted_chunks: torch.Tensor):

        # Append new predictions
        self.buffer[:, current_steps] = predicted_chunks  # (B, T, K, D)
        self.steps[:, current_steps] = current_steps
        self.current_steps = current_steps

    def compute_action(self, ):
        start_time = max(self.current_steps - self.execution_steps, 0)
        chunk_indices = self.current_steps + 1 if self.current_steps <= self.execution_steps - 1 else start_time + self.execution_steps

        buffer_slice = self.buffer[:, start_time:
                                   chunk_indices, :, :]  # shape: (B, T, K, D)

        #
        k_indices = torch.arange(buffer_slice.shape[1] - 1,
                                 -1,
                                 -1,
                                 device=buffer_slice.device)

        B, T, _, D = buffer_slice.shape

        index = k_indices.view(1, T, 1, 1).expand(B, T, 1, D)

        temporal_actions = torch.gather(buffer_slice, dim=2,
                                        index=index).squeeze(
                                            2)  # shape: (B, T, K', D)

        weights = torch.exp(
            -self.m *
            torch.arange(T - 1, -1, -1, device=buffer_slice.device))  # (T,)
        weights = (weights / weights.sum()).view(1, T, 1).repeat_interleave(
            self.B, 0).repeat_interleave(buffer_slice.shape[-1],
                                         2)  # shape: (B, T, action_dim)

        weighted_actions = temporal_actions * weights
        return weighted_actions.sum(dim=1)  # (B, D)


def temporal_ensemble_finger_joints(
    joints,
    gain: float = 0.01,
):

    exp_weights = torch.exp(-gain * torch.arange(
        joints.shape[1],
        dtype=joints.dtype,
    )).to(joints.device)
    exp_weights = (exp_weights / exp_weights.sum()).view(1, -1, 1)
    action = (joints * exp_weights).sum(dim=1)
    return action


def extract_finger_joints(joints, joint_limits):

    shape = [1] * (joints.ndim - 1) + [joint_limits.shape[0], 2]
    if isinstance(joint_limits, np.ndarray):
        joint_limits = joint_limits.reshape(*shape)
    else:
        joint_limits = joint_limits.view(*shape)

    jmin = joint_limits[..., 0]
    jmax = joint_limits[..., 1]

    real_joints = (joints + 1) / 2 * (jmax - jmin) + jmin
    return real_joints


def dataset_minmax_normalizer(actions, quantile=1):

    if quantile == 1:
        min_val = np.min(actions, axis=0)
        max_val = np.max(actions, axis=0)
    else:
        min_val = np.quantile(actions, 1 - quantile, axis=0)
        max_val = np.quantile(actions, quantile, axis=0)

    mean = (min_val + max_val) / 2
    std = (max_val - min_val) / 2
    return mean, std, max_val, max_val


def dataset_denrormalizer(actions, mean, std):
    return actions * std + mean


def dataset_normalizer(actions, mean, std):
    return (actions - mean) / std


def sliding_chunks(raw_actions, chunk_size):
    total_steps, action_dim = raw_actions.shape
    num_chunks = total_steps - chunk_size + 1

    # Pre-allocate the output array
    actions_chunks = np.stack(
        [raw_actions[i:i + chunk_size] for i in range(1, num_chunks)], axis=0)
    state = np.stack(
        [raw_actions[i:i + chunk_size] for i in range(0, num_chunks - 1)],
        axis=0)

    return actions_chunks, state


def init_chunk_data(data_dir, num_hand_joints, chunk_size, eval_percent,
                    device):
    data = h5py.File(data_dir, "r")["data"]

    all_actions = []
    all_states = []

    for index in range(len(data)):

        raw_actions = data[f"demo_{index}"]["actions"][..., -num_hand_joints:]
        chunks_actions, state = sliding_chunks(raw_actions, chunk_size)

        all_actions.append(chunks_actions)
        all_states.append(state)

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    print("num actions:", all_actions.shape[0])

    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)

    action_min = np.min(all_actions, axis=0)
    action_max = np.max(all_actions, axis=0)

    num_actions = all_actions.shape[0]

    actions_chunk = torch.as_tensor(all_actions).to(device).to(torch.float32)
    state = torch.as_tensor(all_states).to(device).to(torch.float32)
    # normalizer = LinearNormalizer()
    # normalizer.fit(data={"action": actions_chunk},
    #                last_n_dims=1,
    #                mode='limits')
    # actions_chunk = normalizer['action'].normalize(actions_chunk)

    train_dataset = [
        actions_chunk[:int(num_actions * eval_percent)],
        state[:int(num_actions * eval_percent)]
    ]
    eval_dataset = [
        actions_chunk[int(num_actions * eval_percent):],
        state[int(num_actions * eval_percent):]
    ]
    input_dim = actions_chunk.shape[-1]
    return train_dataset, eval_dataset, action_mean, action_std, action_min, action_max, input_dim


def load_config(vae_checkpoint, device="cuda", to_torch=False):

    with open(f"{vae_checkpoint}/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

        action_mean = torch.as_tensor(model_config["action_mean"]).to(device)
        action_std = torch.as_tensor(model_config["action_std"]).to(device)
        data_normalizer = model_config["data_normalizer"]
        max_latent_value = np.array(model_config["max_latent_value"])
        min_latent_value = np.array(model_config["min_latent_value"])
        embedding_dim = model_config["embedding_dim"]
        if model_config["model_type"] in ["VQVAE"]:
            embedding_dim = 1
            max_latent_value = np.ones(embedding_dim)
            min_latent_value = -np.ones(embedding_dim)
        if to_torch:
            action_mean = action_mean.to(device).to(torch.float32)
            action_std = action_std.to(device).to(torch.float32)
            min_latent_value = torch.as_tensor(min_latent_value).to(device).to(
                torch.float32)
            max_latent_value = torch.as_tensor(max_latent_value).to(device).to(
                torch.float32)
    return [
        min_latent_value, max_latent_value, data_normalizer, action_mean,
        action_std, embedding_dim
    ]


def load_data(data_dir, device="cuda", num_hand_joints=16):
    data = h5py.File(data_dir, "r")["data"]

    all_actions = []
    all_raw_actions = []

    for index in range(len(data)):

        raw_actions = data[f"demo_{index}"]["actions"][..., -num_hand_joints:]
        raw_actions = (raw_actions + np.pi) % (2 * np.pi) - np.pi
        all_actions.append(raw_actions)

        all_raw_actions.append(
            data[f"demo_{index}"]["actions"][:, -num_hand_joints:])

    all_actions = np.concatenate(all_actions, axis=0)

    all_actions = ((all_actions + np.pi) % (2 * np.pi) - np.pi)
    all_actions = torch.as_tensor(all_actions).to(torch.float32)

    return all_actions.to(device), all_raw_actions


def h5py_group_to_dict(h5_group):
    result = {}
    for key in h5_group:
        item = h5_group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[(
            )]  # Load dataset into memory (as NumPy array or scalar)
        elif isinstance(item, h5py.Group):
            result[key] = h5py_group_to_dict(item)  # Recursive call
    return result


def load_latent_action(latent_action_dir, device="cuda"):

    with h5py.File(f"{latent_action_dir}/latent_action.hdf5", 'r') as file:
        latent_data = h5py_group_to_dict(file)

    return latent_data["data"]


def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_and_resample_gaussian(X, n_samples, spread_scale=4.0, bins=50):
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


def resample_data(data, bin_size):

    resampled_actions, debug_infos = fit_and_resample_gaussian(data,
                                                               data.shape[0],
                                                               bins=bin_size)

    matched_actions = match_histogram_bin_counts(data, debug_infos)
    np.random.shuffle(matched_actions)

    return matched_actions
