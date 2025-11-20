import sys
import random

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, get_val_mask,
                                             downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import numpy as np
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.real_dataset.real_hand_pcd_dataset import RealHandPCDDataset
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.pickandplace_pcd_dataset import PickandPlacePCDDataset, get_pcd_range_normalizer
import copy


class COHandDataset:
    """
    Co-training dataset wrapper that mixes real and sim datasets
    according to a specified ratio, with independent argument sets.
    """

    def __init__(self,
                 real_dataset: dict,
                 sim_dataset: dict,
                 real_ratio: float = 0.5):
        """
        Args:
            real_args: dict of arguments for RealHandPCDDataset
            sim_args: dict of arguments for PickandPlacePCDDataset
            real_ratio: probability of sampling from real dataset [0,1]
        """
        self.real_dataset = RealHandPCDDataset(**real_dataset)
        self.sim_dataset = PickandPlacePCDDataset(**sim_dataset)

        self.real_len = len(self.real_dataset)
        self.sim_len = len(self.sim_dataset)
        self.total_len = self.real_len + self.sim_len

        self.real_ratio = real_ratio

    def __len__(self):
        # A "virtual length" — you can choose total_len or something else
        return self.total_len

    def __getitem__(self, idx):
        # If idx falls into real dataset range
        if idx < self.real_len:
            return self.real_dataset[idx]
        else:
            sim_idx = idx - self.real_len
            return self.sim_dataset[sim_idx]

    def get_validation_dataset(self):
        """
        Return validation version of the same COHandDataset,
        without reloading datasets.
        """
        # shallow copy of self
        val_set = copy.copy(self)

        # swap in validation datasets for real/sim
        val_set.real_dataset = self.real_dataset.get_validation_dataset()
        val_set.sim_dataset = self.sim_dataset.get_validation_dataset()

        # update lengths
        val_set.real_len = len(val_set.real_dataset)
        val_set.sim_len = len(val_set.sim_dataset)
        val_set.total_len = val_set.real_len + val_set.sim_len

        return val_set

    def get_normalizer(self, mode='limits', latent_dim=None, **kwargs):
        # ---- Real agent_pos ----
        real_agent_pos_list = []
        for key in self.real_dataset.obs_key:
            real_agent_pos_list.append(self.real_dataset.replay_buffer[key])
        real_agent_pos = np.concatenate(real_agent_pos_list, axis=-1)

        # ---- Sim agent_pos ----
        sim_agent_pos_list = []
        for key in self.sim_dataset.obs_key:
            sim_agent_pos_list.append(self.sim_dataset.replay_buffer[key])
        sim_agent_pos = np.concatenate(sim_agent_pos_list, axis=-1)

        # ---- Concatenate real + sim ----
        agent_pos = np.concatenate([real_agent_pos, sim_agent_pos], axis=0)
        actions = np.concatenate([
            self.real_dataset.replay_buffer['action'],
            self.sim_dataset.replay_buffer['action']
        ],
                                 axis=0)

        # ---- Fit joint normalizer ----
        data = {'action': actions, 'agent_pos': agent_pos}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer['seg_pc'] = get_pcd_range_normalizer()

        return normalizer


if __name__ == "__main__":
    real_args = dict(
        data_path="logs/trash/teleop_data",
        load_list=["bunny"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=['joint_positions', "gripper_position"],
        noise_key=["joint_positions"],
        noise_scale={"joint_positions": 0.05},
        seed=42,
        val_ratio=0.0,
        num_demo=200,
        max_train_episodes=None,
        downsample_points=2048,
        pcd_noise=0.0,
        noise_extrinsic=True,
        noise_extrinsic_parameter=[0.06, 0.2],
        camera_id="CL838420160",
        crop_region=[-0.20, -0.40, 0.02, 0.85, 0.40, 0.70],
    )

    sim_args = dict(
        data_path="logs/data_0705/retarget_visionpro_data/rl_data/image",
        load_list=["tomato_soup_can"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=[
            'right_hand_joint_pos',
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

    dataset = COHandDataset(real_dataset=real_args,
                            sim_dataset=sim_args,
                            real_ratio=0.2)

    # print("Total length:", len(dataset))
    # for i in range(500):
    #     sample = dataset[i]
    #     # print("Sample keys:", sample.keys())
    from torch.utils.data import WeightedRandomSampler, DataLoader

    real_ratio = 0.2

    weights = [real_ratio] * len(
        dataset.real_dataset) + [1 - real_ratio] * len(dataset.sim_dataset)
    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(weights),
                                    replacement=True)

    train_dataloader = DataLoader(dataset,
                                  sampler=sampler,
                                  batch_size=512,
                                  num_workers=32,
                                  pin_memory=True,
                                  persistent_workers=False)
    normalizer = dataset.get_normalizer()

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()

    weights = [real_ratio] * len(val_dataset.real_dataset) + \
            [1 - real_ratio] * len(val_dataset.sim_dataset)

    val_sampler = WeightedRandomSampler(weights,
                                        num_samples=len(weights),
                                        replacement=True)

    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
    )
    for batch in train_dataloader:
        obs = batch["obs"]  # dict of observations
        action = batch["action"]  # ground truth actions
