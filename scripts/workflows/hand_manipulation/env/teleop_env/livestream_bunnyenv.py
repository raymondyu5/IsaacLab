import pickle
from pathlib import Path
from typing import List
import torch
import numpy as np
import tqdm
import tyro
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import isaaclab.utils.math as math_utils

from scripts.workflows.hand_manipulation.utils.visionpro_utils import *
from scripts.workflows.hand_manipulation.env.teleop_env.bunny_listener import RealTimeHandSubscriber
import rclpy

import threading
from scripts.workflows.hand_manipulation.env.teleop_env.bunny_retarget import BunnyRetargetEnv


class BunnyVisionProEnv(BunnyRetargetEnv):

    def __init__(self, env, args_cli, env_config):
        super().__init__(env, args_cli, env_config)

        # Init env etc.
        self.env = env
        self.args_cli = args_cli
        self.env_config = env_config
        self.device = self.env.device

    def run(self):
        result = self.get_hand_info()

        if result is not None:
            actions = []
            if self.add_left_hand:
                left_wrist_pose = torch.zeros((1, 6)).to(self.device)
                left_hand_action = torch.as_tensor(
                    result["retargeted_left_qpos"]).unsqueeze(0).to(
                        self.device)
                actions.append(left_wrist_pose)
                actions.append(left_hand_action)

            if self.add_right_hand:
                right_wrist_pose = self.init_ee_pose.unsqueeze(0)
                right_hand_action = torch.as_tensor(
                    result["retargeted_right_qpos"]).unsqueeze(0).to(
                        self.device)
                actions.append(right_wrist_pose)
                actions.append(right_hand_action)
            actions = torch.cat(actions, dim=1)
            self.env.step(actions)
