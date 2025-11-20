import torch
import zmq
import json
from scripts.workflows.hand_manipulation.teleoperation.mano.mano_detection_client import ManoClient

from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import yaml

import numpy as np

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget


class ManoTOhand:

    def __init__(self, env, arg_cli, env_config, port=1024, host="localhost"):
        self.env = env
        self.arg_cli = arg_cli
        self.env_config = env_config
        self.add_right_hand = arg_cli.add_right_hand
        self.add_left_hand = arg_cli.add_left_hand
        self.device = env.device
        self.client = ManoClient(
            port=port,
            host=host,
        )

        if self.add_right_hand:
            self.hand_side = "right"
        elif self.add_left_hand:
            self.hand_side = "left"

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.init_ee_pose = torch.as_tensor(
            self.env_config["params"]["init_ee_pose"]).to(self.device)
        self.init_setting()

    def init_setting(self):
        self.left_robot_joint_names, self.right_robot_joint_names = None, None

        self.init_retargeting_config()

    def init_retargeting_config(self):

        init_leap_hand_retarget(self, )
        self.robot_joint_name = get_hand_joint_names(
            self,
            self.hand_side,
        )[-self.num_hand_joints:]

        self.retarget2sim = [
            self.retargeting.optimizer.target_joint_names.index(j)
            for j in self.robot_joint_name
        ]

    def update_last_retargeted_qpos(self, joint):
        retargeting = self.retargeting
        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]

            ref_value = joint[task_indices, :] - joint[origin_indices, :]
        # ref_value = joint[indices, :]

        qpos = retargeting.retarget(ref_value)[self.retarget2sim]

        # retargeted_qpos = retargeting.retarget(ref_value)[self.robot_reorder]
        return qpos

    def get_hand_info(self):

        result = self.client.get_latest_info()

        if result is None:
            return None

        retargeted_qpos = self.update_last_retargeted_qpos(np.array(result))
        return retargeted_qpos

    def run(self):
        retargeted_qpos = self.get_hand_info()

        actions = []
        if self.add_left_hand:
            left_wrist_pose = torch.zeros((1, 6)).to(self.device)
            if retargeted_qpos is not None:
                left_hand_action = torch.as_tensor(retargeted_qpos).unsqueeze(
                    0).to(self.device)
            else:
                left_hand_action = torch.zeros(
                    (1, self.num_hand_joints)).to(self.device)
            actions.append(left_wrist_pose)
            actions.append(left_hand_action)

        if self.add_right_hand:
            right_wrist_pose = torch.zeros((1, 6)).to(self.device)

            if retargeted_qpos is not None:
                right_hand_action = torch.as_tensor(retargeted_qpos).unsqueeze(
                    0).to(self.device)
            else:
                right_hand_action = torch.zeros(
                    (1, self.num_hand_joints)).to(self.device)
            actions.append(right_wrist_pose)
            actions.append(right_hand_action)
        actions = torch.cat(actions, dim=1)
        self.env.step(actions)
