import torch

import pdb
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.envs import ManagerBasedRLEnv


class LeapXarmApproachReward:

    def __init__(self, config, rewards):
        self.config = config
        self.rewards = rewards
        self.target_object_name = self.config["params"]["grasper"][
            "target_object_name"]
        self.ee_name = self.config["params"]["grasper"]["ee_name"]

        self.config_reward()

    def config_reward(self):

        setattr(
            self.rewards, "object_is_approach",
            RewTerm(
                func=self.object_is_approach,
                weight=1.0,
                params={
                    "object_name": self.target_object_name,
                    "ee_name": self.ee_name,
                },
            ))

    def object_is_approach(self, env: ManagerBasedRLEnv, object_name: str,
                           ee_name: str):

        object_pose = env.scene[object_name]._data.root_state_w
        ee_pose = env.scene[ee_name]._data.root_state_w

        ee_object_dist = torch.norm(ee_pose[:3] - object_pose[:3], dim=1)

        return torch.clip(1 / ee_object_dist, 0, 10) * 1 / env.step_dt
