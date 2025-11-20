from typing import Dict
import torch
import numpy as np
import copy
import sys

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, get_val_mask,
                                             downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from scripts.workflows.hand_manipulation.utils.diffusion.diffusion_hand_dataset import ReplayBuffer


class HandLowdimDataset(BaseLowdimDataset):

    def __init__(self,
                 data_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='keypoint',
                 state_key='state',
                 action_key='action',
                 noise_key='hand_joints',
                 noise_scale=0.05,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()

        self.replay_buffer = ReplayBuffer(data_path, )
        self.action_dim = self.replay_buffer.action_dim
        self.noise_key = noise_key
        self.noise_scale = noise_scale

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                val_ratio=val_ratio,
                                seed=seed)

        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask,
                                     max_n=max_train_episodes,
                                     seed=seed)

        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                       sequence_length=horizon,
                                       pad_before=pad_before,
                                       pad_after=pad_after,
                                       episode_mask=train_mask)
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                          sequence_length=self.horizon,
                                          pad_before=self.pad_before,
                                          pad_after=self.pad_after,
                                          episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', latent_dim=None, **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()

        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]

        obs = keypoint.reshape(keypoint.shape[0], -1)
        global_cond_noise = (np.random.rand(*obs.shape) * 2 -
                             1) * self.noise_scale

        data = {
            'obs': obs + global_cond_noise,  # T, D_o
            'action': sample[self.action_key],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class LatentHandLowdimDataset(BaseLowdimDataset):

    def __init__(self,
                 data_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='keypoint',
                 state_key='state',
                 action_key='action',
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 latent_model=None):
        super().__init__()

        self.replay_buffer = ReplayBuffer(data_path, )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                val_ratio=val_ratio,
                                seed=seed)
        self.latent_model = latent_model

        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask,
                                     max_n=max_train_episodes,
                                     seed=seed)

        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                       sequence_length=horizon,
                                       pad_before=pad_before,
                                       pad_after=pad_after,
                                       episode_mask=train_mask)
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                          sequence_length=self.horizon,
                                          pad_before=self.pad_before,
                                          pad_after=self.pad_after,
                                          episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', latent_dim=None, **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()

        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]

        obs = keypoint.reshape(keypoint.shape[0], -1)

        data = {
            'obs': obs,
            'action': sample[self.action_key],
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data
