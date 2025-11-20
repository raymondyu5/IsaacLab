"""Zarr-based dataset loader for inverse dynamics training.

Loads trajectory data from Zarr episodes and converts to (state, action, next_state) transitions.
Supports both low-dimensional observations and pointcloud observations with augmentation.
"""

import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class ZarrInverseDynamicsDataset(Dataset):
    """Universal dataset for loading inverse dynamics training data from Zarr episodes.

    Supports both low-dimensional observations only and multi-modal (low-dim + pointcloud).
    The output format adapts automatically based on whether image_keys is provided:
    - If image_keys is None/empty → returns {'state': tensor, 'next_state': tensor, 'action': tensor}
    - If image_keys provided → returns {'state': dict, 'next_state': dict, 'action': tensor}
    """

    def __init__(
        self,
        data_path: str,
        obs_keys: List[str],
        image_keys: Optional[List[str]] = None,
        action_key: str = "actions",
        num_episodes: int = 1000,
        val_ratio: float = 0.1,
        train: bool = True,
        seed: int = 42,
        # Pointcloud parameters (only used if image_keys is provided)
        downsample_points: int = 2048,
        pcd_noise: float = 0.02,
        noise_extrinsic: bool = True,
        noise_extrinsic_parameter: Optional[List[float]] = None,
        # Low-dim augmentation
        noise_keys: Optional[List[str]] = None,
        noise_scale: Optional[Dict[str, float]] = None,
    ):
        """Initialize Zarr inverse dynamics dataset.

        Args:
            data_path: Path to directory containing episode_N.zarr folders
            obs_keys: List of low-dim observation keys (e.g., ['right_hand_joint_pos', 'right_ee_pose'])
            image_keys: List of pointcloud keys (e.g., ['seg_pc']). If None, loads low-dim only.
            action_key: Key for actions in zarr data
            num_episodes: Maximum number of episodes to load
            val_ratio: Fraction of data for validation
            train: Whether this is training set (vs validation)
            seed: Random seed for train/val split
            downsample_points: Number of points to sample from pointcloud (default: 2048)
            pcd_noise: Noise scale for pointcloud XYZ coordinates (default: 0.02)
            noise_extrinsic: Whether to apply rotation/translation augmentation (default: True)
            noise_extrinsic_parameter: [translation_std, rotation_std] (default: [0.05, 0.2])
            noise_keys: Low-dim obs keys to apply noise to (subset of obs_keys)
            noise_scale: Dictionary mapping noise_keys to noise scale
        """
        super().__init__()

        self.data_path = data_path
        self.obs_keys = obs_keys
        self.image_keys = image_keys or []
        self.use_pointcloud = len(self.image_keys) > 0
        self.action_key = action_key
        self.num_episodes = num_episodes
        self.seed = seed
        self.train = train

        # Pointcloud augmentation settings
        self.downsample_points = downsample_points
        self.pcd_noise = pcd_noise
        self.noise_extrinsic = noise_extrinsic
        self.noise_extrinsic_parameter = noise_extrinsic_parameter or [0.05, 0.2]

        # Low-dim augmentation settings
        self.noise_keys = noise_keys or []
        self.noise_scale = noise_scale or {}

        # Load episodes
        print("=" * 80)
        if self.use_pointcloud:
            print("LOADING ZARR DATASET (WITH POINTCLOUDS)")
        else:
            print("LOADING ZARR DATASET (LOW-DIM ONLY)")
        print("=" * 80)
        print(f"Data path: {data_path}")
        print(f"Obs keys (low-dim): {obs_keys}")
        if self.use_pointcloud:
            print(f"Image keys (PCD): {image_keys}")
            print(f"Downsample points: {downsample_points}")
            print(f"PCD noise: {pcd_noise}")
            print(f"Extrinsic augmentation: {noise_extrinsic}")
            if noise_extrinsic:
                print(f"  Translation std: {self.noise_extrinsic_parameter[0]:.3f} m")
                print(f"  Rotation std: {self.noise_extrinsic_parameter[1]:.3f} rad (~{np.degrees(self.noise_extrinsic_parameter[1]):.1f}°)")
        print(f"Max episodes: {num_episodes}")
        print("=" * 80)

        self._load_episodes()

        # Split train/val
        np.random.seed(seed)
        n_episodes_loaded = len(self.episode_slices)
        indices = np.arange(n_episodes_loaded)
        np.random.shuffle(indices)

        n_val = int(n_episodes_loaded * val_ratio)
        val_indices = set(indices[:n_val])

        # Filter transitions by train/val split
        filtered_transitions = []
        for ep_idx, (start, end) in enumerate(self.episode_slices):
            is_val = ep_idx in val_indices
            if (train and not is_val) or (not train and is_val):
                filtered_transitions.extend(range(start, end))

        self.transition_indices = filtered_transitions

        split_name = "TRAIN" if train else "VALIDATION"
        print(f"\n{split_name} DATASET:")
        print(f"  Episodes: {n_episodes_loaded - n_val if train else n_val}")
        print(f"  Transitions: {len(self.transition_indices):,}")
        if self.use_pointcloud:
            print(f"  Low-dim obs: {self.low_dim_obs_dim}")
            print(f"  PCD obs: {len(self.image_keys)} pointclouds × {downsample_points} points")
        else:
            print(f"  Obs dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print("=" * 80 + "\n")

    def _load_episodes(self):
        """Load all episodes from Zarr directories."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        all_files = os.listdir(self.data_path)
        episode_dirs = sorted(
            [f for f in all_files if f.startswith('episode_') and f.endswith('.zarr')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        episode_dirs = episode_dirs[:self.num_episodes]
        print(f"Found {len(episode_dirs)} episode directories")

        # Buffers
        low_dim_obs_buffer = []
        pcd_obs_buffers = {key: [] for key in self.image_keys} if self.use_pointcloud else {}
        action_buffer = []
        self.episode_slices = []
        transition_count = 0
        loaded_count = 0

        for ep_dir in tqdm(episode_dirs, desc="Loading episodes"):
            zarr_path = os.path.join(self.data_path, ep_dir)

            try:
                data = zarr.open(zarr_path, mode='r')

                # Load actions
                actions = np.array(data[f'data/{self.action_key}'])
                episode_len = len(actions)

                # Load low-dim observations
                obs_list = []
                for key in self.obs_keys:
                    obs = np.array(data[f'data/{key}'])
                    assert len(obs) == episode_len, f"Length mismatch for {key}"
                    obs_list.append(obs)

                low_dim_obs = np.concatenate(obs_list, axis=-1)

                # Load pointcloud observations if needed
                if self.use_pointcloud:
                    pcd_obs = {}
                    for key in self.image_keys:
                        pcd_data = np.array(data[f'data/{key}'])  # (T, N, 3)
                        assert len(pcd_data) == episode_len, f"Length mismatch for {key}"
                        pcd_obs[key] = pcd_data

                # Store episode data
                if episode_len > 1:
                    num_transitions = episode_len - 1

                    low_dim_obs_buffer.append(low_dim_obs)
                    action_buffer.append(actions)

                    if self.use_pointcloud:
                        for key in self.image_keys:
                            pcd_obs_buffers[key].append(pcd_obs[key])

                    self.episode_slices.append((transition_count, transition_count + num_transitions))
                    transition_count += num_transitions
                    loaded_count += 1

            except Exception as e:
                print(f"Error loading {zarr_path}: {e}")
                continue

        # Concatenate all episodes
        self.low_dim_observations = np.concatenate(low_dim_obs_buffer, axis=0)
        self.actions = np.concatenate(action_buffer, axis=0)

        if self.use_pointcloud:
            self.pcd_observations = {}
            for key in self.image_keys:
                self.pcd_observations[key] = np.concatenate(pcd_obs_buffers[key], axis=0)
            self.low_dim_obs_dim = self.low_dim_observations.shape[1]
        else:
            self.obs_dim = self.low_dim_observations.shape[1]

        self.action_dim = self.actions.shape[1]

        print(f"Successfully loaded {loaded_count} episodes with {transition_count} timesteps")

    def _augment_pointcloud(self, pcd: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single pointcloud.

        Args:
            pcd: Pointcloud of shape (N, 3) where N is number of points

        Returns:
            Augmented pointcloud of shape (downsample_points, 3)
        """
        # Step 1: Random point sampling (downsample)
        num_points = pcd.shape[0]
        if num_points > self.downsample_points:
            indices = np.random.permutation(num_points)[:self.downsample_points]
            pcd_sampled = pcd[indices]
        elif num_points < self.downsample_points:
            indices = np.random.choice(num_points, self.downsample_points, replace=True)
            pcd_sampled = pcd[indices]
        else:
            pcd_sampled = pcd.copy()

        # Step 2: Extrinsic augmentation (rotation + translation)
        if self.noise_extrinsic and self.train:
            # Random rotation
            euler_angles = (np.random.rand(3) * 2 - 1) * self.noise_extrinsic_parameter[1]
            rotation = R.from_euler('xyz', euler_angles)
            rotation_matrix = rotation.as_matrix()

            # Apply rotation
            pcd_sampled = pcd_sampled @ rotation_matrix.T

            # Random translation
            translation = (np.random.rand(3) * 2 - 1) * self.noise_extrinsic_parameter[0]
            pcd_sampled = pcd_sampled + translation

        # Step 3: Point noise
        if self.pcd_noise > 0 and self.train:
            noise = (np.random.rand(*pcd_sampled.shape) * 2 - 1) * self.pcd_noise
            pcd_sampled = pcd_sampled + noise

        return pcd_sampled.astype(np.float32)

    def __len__(self) -> int:
        return len(self.transition_indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get a single transition.

        Returns:
            If use_pointcloud=False:
                {'state': tensor(obs_dim), 'next_state': tensor(obs_dim), 'action': tensor(action_dim)}
            If use_pointcloud=True:
                {'state': {'low_dim': tensor, 'seg_pc': tensor}, 'next_state': {...}, 'action': tensor}
        """
        transition_idx = self.transition_indices[idx]

        # Get low-dim observations
        low_dim_state = self.low_dim_observations[transition_idx].copy()
        low_dim_next_state = self.low_dim_observations[transition_idx + 1].copy()

        # Apply low-dim noise if configured
        if self.train and len(self.noise_keys) > 0:
            for key in self.noise_keys:
                if key in self.noise_scale:
                    noise_std = self.noise_scale[key]
                    noise = (np.random.rand(*low_dim_state.shape) * 2 - 1) * noise_std
                    low_dim_state = low_dim_state + noise
                    noise = (np.random.rand(*low_dim_next_state.shape) * 2 - 1) * noise_std
                    low_dim_next_state = low_dim_next_state + noise
                    break  # Apply once to entire obs

        # Get action
        action = self.actions[transition_idx].copy()

        if not self.use_pointcloud:
            # Low-dim only format
            return {
                'state': torch.FloatTensor(low_dim_state),
                'next_state': torch.FloatTensor(low_dim_next_state),
                'action': torch.FloatTensor(action),
            }
        else:
            # Multi-modal format with pointclouds
            state_dict = {'low_dim': torch.FloatTensor(low_dim_state)}
            next_state_dict = {'low_dim': torch.FloatTensor(low_dim_next_state)}

            for key in self.image_keys:
                # Get raw pointcloud data (N, 3)
                pcd_state = self.pcd_observations[key][transition_idx]
                pcd_next_state = self.pcd_observations[key][transition_idx + 1]

                # Apply augmentation (includes downsampling, rotation, translation, noise)
                pcd_state_aug = self._augment_pointcloud(pcd_state)  # (downsample_points, 3)
                pcd_next_state_aug = self._augment_pointcloud(pcd_next_state)  # (downsample_points, 3)

                # Convert to (3, num_points) format for PointNet
                pcd_state_tensor = torch.FloatTensor(pcd_state_aug.T)  # (3, downsample_points)
                pcd_next_state_tensor = torch.FloatTensor(pcd_next_state_aug.T)  # (3, downsample_points)

                state_dict[key] = pcd_state_tensor
                next_state_dict[key] = pcd_next_state_tensor

            return {
                'state': state_dict,
                'next_state': next_state_dict,
                'action': torch.FloatTensor(action),
            }

    def get_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute normalization statistics (mean, std) for low-dim observations.

        Note: Pointcloud data doesn't need normalization (already in metric coordinates).

        Returns:
            Dictionary with keys 'state' and 'action', values are (mean, std) tuples
        """
        state_mean = np.mean(self.low_dim_observations, axis=0)
        state_std = np.std(self.low_dim_observations, axis=0) + 1e-8

        action_mean = np.mean(self.actions, axis=0)
        action_std = np.std(self.actions, axis=0) + 1e-8

        return {
            'state': (state_mean, state_std),
            'action': (action_mean, action_std),
        }