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


class BunnyVisionProEnv:

    def __init__(self, env, args_cli, env_config):
        self.env = env
        self.args_cli = args_cli
        self.env_config = env_config
        self.hand_offset = torch.as_tensor([0.5, 0.0, 0.4]).to(self.env.device)

        if self.args_cli.free_hand:

            self.run = self.run_free_hand
        else:
            self.run = self.run_arm_hand

        self.delta_quat = torch.as_tensor([[0.707, 0.0, 0.0,
                                            -0.707]]).to(self.env.device)
        self.transformation = torch.eye(4).to(self.env.device)

        self.init_setting()

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        robot_dir = "source/assets/dex-urdf/robots/hands"

        robot_name = RobotName.leap
        RetargetingConfig.set_default_urdf_dir(robot_dir)
        left_config_path = get_default_config_path(robot_name,
                                                   RetargetingType.dexpilot,
                                                   HandType.left)
        right_config_path = get_default_config_path(robot_name,
                                                    RetargetingType.dexpilot,
                                                    HandType.right)

        left_retargeting = RetargetingConfig.load_from_file(
            left_config_path).build()
        right_retargeting = RetargetingConfig.load_from_file(
            right_config_path).build()

        data = np.load(self.args_cli.data_dir, allow_pickle=True)
        # data = filter_data(data, fps=30, duration=15)
        length = len(data)

        self.left_qpos_list = []
        self.right_qpos_list = []
        left_pose_list = []
        right_pose_list = []
        retarget2leftisaac = None
        retarget2rightisaac = None
        head_mat = []

        if self.args_cli.add_left_hand:
            isaac_joint_names = self.env.scene["left_hand"].joint_names

            isaac_joint_names = isaac_joint_names[-self.num_hand_joints:]

            retarget2leftisaac = np.array([
                left_retargeting.joint_names.index(joint)
                for joint in isaac_joint_names
            ]).astype(int)

        if self.args_cli.add_right_hand:
            isaac_joint_names = self.env.scene["right_hand"].joint_names

            isaac_joint_names = isaac_joint_names[-self.num_hand_joints:]

            retarget2rightisaac = np.array([
                right_retargeting.joint_names.index(joint)
                for joint in isaac_joint_names
            ]).astype(int)

        for i in range(length):
            single_data = data[i]
            for hand_num, retargeting in enumerate(
                [left_retargeting, right_retargeting]):
                if hand_num == 0:
                    joint_pose = two_mat_batch_mul(single_data["left_fingers"],
                                                   OPERATOR2AVP_LEFT.T)
                else:
                    joint_pose = two_mat_batch_mul(
                        single_data["right_fingers"], OPERATOR2AVP_RIGHT.T)

                joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]

                indices = retargeting.optimizer.target_link_human_indices
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[
                    origin_indices, :]

                if hand_num == 0:
                    qpos = retargeting.retarget(ref_value)[retarget2leftisaac]
                    self.left_qpos_list.append(torch.as_tensor(qpos))

                    wrist_pose = single_data["left_wrist"][0].copy()
                    wrist_pose[:3, :3] = wrist_pose[:3, :3] @ OPERATOR2AVP_LEFT
                    wrist_pose = self.transformation @ torch.as_tensor(
                        wrist_pose, dtype=torch.float32).to(self.env.device)

                    left_pose_list.append(wrist_pose.clone())
                else:
                    qpos = retargeting.retarget(ref_value)[retarget2rightisaac]
                    self.right_qpos_list.append(torch.as_tensor(qpos))
                    wrist_pose = single_data["right_wrist"][0].copy()
                    wrist_pose[:3, :
                               3] = wrist_pose[:3, :3] @ OPERATOR2AVP_RIGHT
                    wrist_pose = self.transformation @ torch.as_tensor(
                        wrist_pose, dtype=torch.float32).to(self.env.device)

                    right_pose_list.append(wrist_pose.clone())

            head_pose = single_data["head"][0]
            head_pose[:3, :3] = OPERATOR2AVP_CAM[:3, :3] @ head_pose[:3, :3]

            heat_pose = math_utils.pose_from_transformations(
                torch.as_tensor(head_pose).unsqueeze(0).to(
                    self.env.device,
                    dtype=torch.float32)).to(self.env.device,
                                             dtype=torch.float32)

            head_mat.append(torch.as_tensor(heat_pose)[0])
        self.right_qpos_list = torch.stack(self.right_qpos_list, dim=0)
        self.left_qpos_list = torch.stack(self.left_qpos_list, dim=0)

        self.head_mat = torch.stack(head_mat, dim=0)
        left_pose_list = torch.stack(left_pose_list, dim=0)
        right_pose_list = torch.stack(right_pose_list, dim=0)

        right_hand_quat = math_utils.quat_from_matrix(
            torch.as_tensor(right_pose_list[:, :3, :3]))
        self.right_pose = torch.cat(
            [torch.as_tensor(right_pose_list[:, :3, 3]), right_hand_quat],
            dim=1).to(self.env.device, dtype=torch.float32)

        left_hand_quat = math_utils.quat_from_matrix(
            torch.as_tensor(left_pose_list[:, :3, :3]))
        self.left_pose = torch.cat(
            [torch.as_tensor(left_pose_list[:, :3, 3]), left_hand_quat],
            dim=1).to(self.env.device, dtype=torch.float32)

    def run_free_hand(self):

        self.env.reset()

        for i in range(self.right_qpos_list.shape[0]):

            actions = []

            if self.args_cli.add_left_hand:
                ee_pose = torch.zeros((1, 6)).to(self.env.device)
                # ee_pose[:, 3] = 1

                actions.append(ee_pose)
                actions.append(
                    torch.tensor(self.left_qpos_list[i],
                                 device=self.env.device))
            if self.args_cli.add_right_hand:
                actions.append(
                    torch.tensor(self.right_pose[i], device=self.env.device))
                actions.append(
                    torch.tensor(self.right_qpos_list[i],
                                 device=self.env.device))

            actions = torch.cat(actions, dim=0).unsqueeze(0)

            # if self.args_cli.add_right_hand:

            #     self.apply_hand_pose(self.right_pose[i],
            #                          hand_name="right_hand")
            # if self.args_cli.add_left_hand:
            #     self.apply_hand_pose(self.left_pose[i], hand_name="left_hand")

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)
            head_quat = self.head_mat[i, 3:7].unsqueeze(0)

            # head_quat = math_utils.quat_mul(
            #     torch.as_tensor([[0.707, 0.707, 0.0,
            #                       0.0]]).to(self.env.device), head_quat)

            # # delta_quat = math_utils.quat_mul(
            # #     torch.as_tensor([[0.707, 0.707, 0.0, 0.0]]),
            # #     torch.as_tensor([[0.707, 0.0, -0.707, 0.0]]))
            # math_utils.matrix_from_quat(
            #     torch.as_tensor([[0.707, 0.707, 0.0, 0.0]]))
            # import pdb
            # pdb.set_trace()

            # self.env.scene["camera_01"]._view.set_world_poses(
            #     self.head_mat[i, 0:3].unsqueeze(0), head_quat)

    def run_arm_hand(self):
        self.env.reset()

        for i in range(self.right_qpos_list.shape[0]):

            actions = []

            if self.args_cli.add_left_hand:

                actions.append(
                    torch.as_tensor(self.left_pose[i], device=self.env.device))
                actions.append(
                    torch.tensor(self.left_qpos_list[i],
                                 device=self.env.device))

            if self.args_cli.add_right_hand:

                actions.append(
                    torch.as_tensor(self.right_pose[i],
                                    device=self.env.device))
                actions.append(
                    torch.tensor(self.right_qpos_list[i],
                                 device=self.env.device))

            actions = torch.cat(actions, dim=0).unsqueeze(0)

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

            # np.copyto(self.tv.img_array, obs["policy"]["rgb"][0,
            #                                                   0].cpu().numpy())
