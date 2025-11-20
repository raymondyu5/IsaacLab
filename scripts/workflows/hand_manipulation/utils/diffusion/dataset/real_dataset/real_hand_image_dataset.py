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
from diffusion_policy.dataset.base_dataset import BaseImageDataset

import imageio
from typing import Tuple
import cv2
import math
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.real_dataset.real_image_dataset_loader import RealImageDatasetLoader
from torchvision.transforms import v2 as T
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.real_dataset.sample_image_wrapper import RealImageSequenceSampler


class RealHandImageDataset(BaseImageDataset):

    def __init__(
        self,
        data_path,
        load_list,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=list,
        noise_key=list,
        noise_scale=0.05,
        seed=42,
        crop_region=None,
        val_ratio=0.0,
        image_dim=(1920, 1080, 3),
        resize_shape=(256, 256),
        num_demo=100,
        max_train_episodes=None,
        add_randomizer=False,
        camera_id=["CL838420160"],
    ):
        super().__init__()

        self.image_tf = T.Compose([
            T.Resize(resize_shape,
                     interpolation=T.InterpolationMode.BICUBIC,
                     antialias=True),
            T.ToDtype(torch.float32, scale=True),  # replaces ToTensor in v2
        ])
        self.add_randomizer = add_randomizer
        self.image_key = ["image"]

        self.replay_buffer = RealImageDatasetLoader(data_path,
                                                    load_list,
                                                    obs_key,
                                                    image_tf=self.image_tf,
                                                    num_demo=num_demo,
                                                    crop_region=crop_region,
                                                    camera_id=camera_id)
        self.action_dim = self.replay_buffer.action_dim
        self.low_obs_dim = self.replay_buffer.low_obs_dim
        self.crop_region = crop_region

        self.raw_image_dim = image_dim

        self.image_dim = resize_shape + (3, )

        self.noise_key = noise_key
        self.noise_scale = noise_scale

        self.data_path = data_path

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                val_ratio=val_ratio,
                                seed=seed)

        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask,
                                     max_n=max_train_episodes,
                                     seed=seed)

        self.sampler = RealImageSequenceSampler(
            replay_buffer=self.replay_buffer,
            path=data_path,
            image_tf=self.image_tf,
            image_dim=self.raw_image_dim,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            load_list=self.replay_buffer.load_list,
            add_randomizer=self.add_randomizer,
            crop_region=crop_region,
        )
        self.obs_key = obs_key

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = RealImageSequenceSampler(
            replay_buffer=self.replay_buffer,
            path=self.data_path,
            image_tf=self.image_tf,
            image_dim=self.raw_image_dim,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
            load_list=self.replay_buffer.load_list,
            add_randomizer=self.add_randomizer,
            crop_region=self.crop_region,
        )
        val_set.train_mask = ~self.train_mask

        return val_set

    def get_normalizer(self, mode='limits', latent_dim=None, **kwargs):
        agent_pos_list = []
        for key in self.obs_key:
            agent_pos_list.append(self.replay_buffer[key])
        agent_pos = np.concatenate(agent_pos_list, axis=-1)

        data = {'action': self.replay_buffer['action'], 'agent_pos': agent_pos}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.image_key:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        obs = []
        for key in self.obs_key:
            if key in self.noise_key:
                noise = (np.random.rand(*sample[key].shape) * 2 -
                         1) * self.noise_scale
                obs.append(sample[key] + noise)
            else:
                obs.append(sample[key])

        obs = np.concatenate(obs, axis=-1)
        image_obs = {}
        for image_key in self.image_key:

            image_obs[image_key] = sample[image_key]

        data = {
            "obs": {
                "agent_pos": obs
            } | image_obs,
            'action': sample["action"],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


if __name__ == "__main__":

    dataset = RealHandImageDataset(
        "/home/weirdlab/Documents/droid/logs/teleop",
        load_list=["bunny"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=[
            'cartesian_position',
            'joint_positions',
        ],
        noise_key=["joint_positions"],
        noise_scale=0.05,
        seed=42,
        val_ratio=0.0,
        num_demo=10,
        resize_shape=(128, 128),
        max_train_episodes=None,
        add_randomizer=True,
        crop_region=[
            300,
            50,
            -500,
            -100,
        ]  # [x_min, y_min, x_max, y_max]
    )
    video = imageio.get_writer("logs/test_video.mp4", fps=30)

    images = []
    for i in range(0, 200):
        sample = dataset[i]

        # image = sample["obs"]["rgb_0"][0].cpu().numpy()

        image = (sample["obs"]["image"][0].cpu().numpy() * 255).astype(
            np.uint8)

        video.append_data(image.transpose(1, 2, 0))
