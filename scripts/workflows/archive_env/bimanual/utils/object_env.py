import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym
from scripts.workflows.bimanual.utils.rigid_bodies_env import RigidbodiesEnv
from scripts.workflows.bimanual.utils.deformable_bodies_env import DeformableBodiesEnv
from scripts.workflows.bimanual.utils.grippers_env import GrippersEnv


class ObjectEnv(RigidbodiesEnv, DeformableBodiesEnv, GrippersEnv):

    def __init__(self):
        super().__init__()

        self.deformable_asset = self.env.scene["deform_object"]
        self.env_ids = torch.arange(self.num_envs, device=self.device)

    def new_episode_training_start(self,
                                   explore_type,
                                   explore_action_index,
                                   reset_gripper,
                                   repeat=True):

        self.reset_deformable_object_pose(explore_type, explore_action_index,
                                          reset_gripper, repeat)
