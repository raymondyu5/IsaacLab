from scripts.workflows.hand_manipulation.env.teleop_env.bimanual_teleop_client import TeleopClient
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path

import yaml

import torch


class BunnyRetargetEnv:

    def __init__(self, env, args_cli, env_confi):
        self.env = env
        self.device = self.env.device
        self.args_cli = args_cli
        self.env_config = env_confi
        self.add_right_hand = args_cli.add_right_hand
        self.add_left_hand = args_cli.add_left_hand

        self.init_setting()

        port_num = 5500
        server_address = "localhost"
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.init_ee_pose = torch.as_tensor(
            self.env_config["params"]["init_ee_pose"]).to(self.device)

        self.teleop_client = TeleopClient(
            port=port_num,
            host=server_address,
            right_robot_joint_names=self.
            right_robot_joint_names[-self.num_hand_joints:]
            if self.right_robot_joint_names is not None else None,
            left_robot_joint_names=self.
            left_robot_joint_names[-self.num_hand_joints:]
            if self.left_robot_joint_names is not None else None,
            init_ee_pose=self.init_ee_pose.cpu().numpy(),
        )

    def init_setting(self):
        self.left_robot_joint_names, self.right_robot_joint_names = None, None

        if self.add_left_hand:
            self.left_robot_joint_names = self.get_hand_joint_names("left", )
        if self.add_right_hand:
            self.right_robot_joint_names = self.get_hand_joint_names("right")

    def get_hand_joint_names(self, hand_side):
        robot = self.env.scene[f"{hand_side}_hand"]
        robot_joint_names = []

        for action_name in self.env.action_manager._terms.keys():
            if hand_side in action_name:
                control_joint_names = self.env.action_manager._terms[
                    action_name].cfg.joint_names
                robot_joint_names += robot.find_joints(control_joint_names)[1]

        return robot_joint_names

    def get_hand_info(self):

        result = self.teleop_client.get_teleop_cmd()
        return result
