from typing import Dict
import torch
import numpy as np
import copy
import sys
import os

# Add diffusion_policy to path (try both locations)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
diffusion_policy_path = os.path.join(SCRIPT_DIR, "..", "..", "diffusion_policy")
if os.path.exists(diffusion_policy_path):
    sys.path.insert(0, diffusion_policy_path)
else:
    # Fallback to submodule
    sys.path.append("submodule/diffusion_policy")

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, get_val_mask,
                                             downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.zarr_buffer import ZarrBuffer

from scripts.workflows.hand_manipulation.utils.diffusion.dataset.pcd_dataset_sampler import PCDSequenceSampler
import imageio
from typing import Tuple
import cv2
import math
try:
    from tools.visualization_utils import vis_pc, visualize_pcd
except:
    pass
from typing import Union, Dict
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


def get_pcd_range_normalizer():
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    stat = {
        'min': np.array([-10], dtype=np.float32),
        'max': np.array([10], dtype=np.float32),
        'mean': np.array([0.0], dtype=np.float32),
        'std': np.array([1], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(scale=scale,
                                                     offset=offset,
                                                     input_stats_dict=stat)


class PickandPlacePCDDataset(BaseImageDataset):

    def __init__(
        self,
        data_path,
        load_list,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=list,
        image_key=list,
        action_key=list,
        noise_key=list,
        noise_scale=0.05,
        seed=42,
        val_ratio=0.0,
        downsample_points=2048,
        num_demo=150,
        pcd_noise=0.01,
        noise_extrinsic=False,
        noise_extrinsic_parameter=[0.05, 0.02],
        max_train_episodes=None,
    ):
        super().__init__()

        self.replay_buffer = ZarrBuffer(
            data_path,
            load_list,
            obs_key,
            image_key,
            num_demo=num_demo,
            downsample_points=downsample_points,
        )
        self.noise_extrinsic = noise_extrinsic
        self.noise_extrinsic_parameter = noise_extrinsic_parameter

        self.action_dim = self.replay_buffer.action_dim
        self.low_obs_dim = self.replay_buffer.low_obs_dim

        self.downsample_points = downsample_points
        self.pcd_noise = pcd_noise

        self.noise_key = noise_key
        self.noise_scale = noise_scale
        self.image_key = image_key
        self.data_path = data_path

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                val_ratio=val_ratio,
                                seed=seed)

        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask,
                                     max_n=max_train_episodes,
                                     seed=seed)

        self.sampler = PCDSequenceSampler(
            replay_buffer=self.replay_buffer,
            image_key=image_key,
            downsample_points=self.downsample_points,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            noise_extrinsic=self.noise_extrinsic,
            noise_extrinsic_parameter=self.noise_extrinsic_parameter,
        )
        self.obs_key = obs_key

        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = PCDSequenceSampler(
            replay_buffer=self.replay_buffer,
            image_key=self.image_key,
            downsample_points=self.downsample_points,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            noise_extrinsic=self.noise_extrinsic,
            noise_extrinsic_parameter=self.noise_extrinsic_parameter,
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
        normalizer['seg_pc'] = get_pcd_range_normalizer()
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
                         1) * self.noise_scale[key]

                obs.append(sample[key] + noise)

            else:
                obs.append(sample[key])

        obs = np.concatenate(obs, axis=-1)
        image_obs = {}
        for image_key in self.image_key:
            noise = (np.random.rand(*sample[image_key].shape) * 2 -
                     1) * self.pcd_noise

            image_obs[image_key] = sample[image_key][:, :3]

            image_obs[image_key][:, :3] += noise[:, :3]  # add noise to xyz

        data = {
            "obs": {
                "agent_pos": obs
            } | image_obs,
            'action': sample[self.action_key],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


if __name__ == "__main__":

    dataset = PickandPlacePCDDataset(
        "logs/data_0705/retarget_visionpro_data/rl_data/image",
        load_list=["tomato_soup_can"],
        horizon=2,
        pad_before=0,
        pad_after=0,
        obs_key=[
            'right_ee_pose', 'right_hand_joint_pos',
            'right_manipulated_object_pose'
        ],
        image_key=["seg_pc"],
        action_key='action',
        noise_key=["right_hand_joint_pos", "right_ee_pose"],
        noise_scale={
            "right_hand_joint_pos": 0.05,
            "right_ee_pose": 0.02
        },
        seed=42,
        val_ratio=0.1,
        num_demo=3,
        downsample_points=2048,
        pcd_noise=0.0,
        noise_extrinsic=True,
        noise_extrinsic_parameter=[0.06, 0.2],
        max_train_episodes=None,
    )
    # video = imageio.get_writer("logs/test_video.mp4", fps=10)

    for i in range(0, 1000):
        sample = dataset[i]["obs"]
        pcd = sample["seg_pc"].permute(0, 2, 1)[0].cpu().numpy()

        # o3d = vis_pc(pcd[:, :3])
        # visualize_pcd([o3d])

    #     video.append_data((sample["obs"]["rgb"][0].cpu().numpy() * 255).astype(
    #         np.uint8).transpose(1, 2, 0))
