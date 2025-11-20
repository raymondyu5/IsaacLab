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

import isaaclab.utils.math as math_utils

from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

import pickle
import time

import tyro

from pathlib import Path

import numpy as np

from bunny_teleop.bimanual_teleop_client import TeleopClient
from bunny_teleop.init_config import BimanualAlignmentMode

from scripts.workflows.hand_manipulation.env.teleop_env.stream_visionpro import VisionProStreamer
from scripts.workflows.hand_manipulation.utils.visionpro_utils import *

from scripts.workflows.hand_manipulation.env.teleop_env.LivestreamPC import LiveStreamPCViewer


class BunnyEnv:

    def __init__(self, env, args_cli, save_config, render_viser=False):
        self.env = env
        self.args_cli = args_cli
        self.env_config = save_config
        self.device = env.device
        self.init_robot_setting()
        self.render_viser = render_viser
        if self.render_viser:
            self.livestream = LiveStreamPCViewer(env, save_config)
            # from scripts.workflows.hand_manipulation.utils.keyboard_interface.dex_retargeting import Se3Keyboard
            # self.keyboard_interface = Se3Keyboard()
            # self.keyboard_interface.reset()

        self.hand_offset = torch.as_tensor([0.2, 0.0,
                                            -0.5]).to(self.env.device)

        self.delta_quat = torch.as_tensor([[0.707, 0.0, 0.0,
                                            -0.707]]).to(self.env.device)
        self.transformation = torch.eye(4).to(self.env.device)
        self.transformation[:3, :3] = math_utils.matrix_from_quat(
            self.delta_quat)
        self.env_ids = torch.arange(self.env.num_envs).to(self.env.device)

        if "Play" in self.args_cli.task:
            self.play_mode = True
        else:
            self.play_mode = False

        if self.args_cli.proccess_raw:
            self.init_raw_setting()
        else:
            self.init_visionpro_setting()
            self.await_teleop_client()

    def await_teleop_client(self):
        print("Waiting for camera adjustment.")
        actions = []
        if self.args_cli.add_left_hand:
            actions.append(self.init_robot_qpos)
        if self.args_cli.add_right_hand:
            actions.append(
                torch.tensor(self.init_robot_qpos, device=self.env.device))
        actions = torch.cat(actions, dim=1)

        print("Waiting for teleop client to start...")

        print(
            f"Place your two hands under your Apple Vision Pro. Make sure you all your fingers are in flatten pose."
        )

        while not self.teleop_client.started:
            self.env.step(actions)
            if self.render_viser:
                self.livestream.update()
            continue
        print(f"Begin teleoperation initialization")

    def robot_setting(self, robot, init_robot_qpos, hand_side="left"):
        robot_dof = len(robot.joint_names)
        robot_robot_base_pose = robot._data.root_state_w[0, :7].cpu().numpy()
        if self.play_mode:
            robot_joint_names = []

            for action_name in self.env.action_manager._terms.keys():
                if hand_side in action_name:
                    control_joint_names = self.env.action_manager._terms[
                        action_name].cfg.joint_names

                    robot_joint_names += robot.find_joints(
                        control_joint_names)[1]

        else:
            robot_joint_names = robot.joint_names

        robot._data.reset_joint_pos = init_robot_qpos
        init_robot_qpos = robot._data.reset_joint_pos
        robot.root_physx_view.set_dof_positions(robot._data.reset_joint_pos,
                                                indices=self.env_ids)

        retarget2isaac = torch.as_tensor([
            self.init_joint_names.index(joint)
            for joint in robot_joint_names if joint in self.init_joint_names
        ],
                                         dtype=torch.int32).to(self.env.device)

        return robot_dof, robot_robot_base_pose, robot_joint_names, retarget2isaac

    def init_robot_setting(self, ):
        self.init_qpos = torch.as_tensor([
            self.env_config["params"]["grasper"]["init_robot_qos"]
        ]).to(self.env.device)
        self.init_joint_names = self.env_config["params"]["grasper"][
            "joint_names"]

        self.num_hands = 0
        self.num_joints = self.env_config["params"][
            "num_hand_joints"] + self.env_config["params"]["num_arm_joints"]
        self.num_hands_joints = self.env_config["params"]["num_hand_joints"]

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_config,
        )

        init_ee_pose = torch.as_tensor(
            self.env_config["params"]["init_ee_pose"]).to(
                self.device).unsqueeze(0)

        init_arm_qpos = self.arm_motion_env.ik_plan_motion(
            init_ee_pose).repeat_interleave(self.env.num_envs, dim=0)
        init_hand_qpos = torch.zeros(
            (self.env.num_envs, self.num_hands_joints)).to(self.device)
        self.init_robot_qpos = torch.cat([init_arm_qpos, init_hand_qpos],
                                         dim=1).to(self.device)

    def init_visionpro_robot_setting(self, hand_side="left"):
        robot = self.env.scene[f"{hand_side}_hand"]
        setattr(self, f"{hand_side}_robot", robot)

        # reset robot pose
        robot_base_pose = torch.as_tensor(
            self.env_config["params"][f"{hand_side}_robot_pose"]).unsqueeze(
                0).to(self.env.device)
        robot.data.reset_root_state[:, :7] = robot_base_pose
        robot.write_root_pose_to_sim(
            robot_base_pose,
            env_ids=torch.arange(self.env.num_envs).to(self.env.device))

        _, _, _, retarget2isaac = self.robot_setting(robot,
                                                     self.init_robot_qpos,
                                                     hand_side=hand_side)
        setattr(self, f"{hand_side}_retarget2isaac", retarget2isaac)
        self.num_hands += 1

    def init_visionpro_setting(self):

        if self.args_cli.add_left_hand:
            self.init_visionpro_robot_setting(hand_side="left")
        if self.args_cli.add_right_hand:
            self.init_visionpro_robot_setting(hand_side="right")
        self.init_teleop_client()

    def init_teleop_client(self):
        # Teleoperation client
        port_num = 5500
        server_address = "localhost"

        self.teleop_client = TeleopClient(port=port_num,
                                          cmd_dims=(self.num_joints,
                                                    self.num_joints),
                                          host=server_address)

        left_robot_base_pose = np.array([0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        right_robot_base_pose = np.array([-0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.teleop_client.send_init_config(
            robot_base_pose=(left_robot_base_pose, right_robot_base_pose),
            init_qpos=(self.init_robot_qpos[0].cpu().numpy(),
                       self.init_robot_qpos[0].cpu().numpy()),
            joint_names=(self.init_joint_names, self.init_joint_names),
            align_gravity_dir=True,
            bimanual_alignment_mode=BimanualAlignmentMode.ALIGN_SEPARATELY,
        )

    def init_raw_setting(self):

        robot_dir = "source/assets/robot/dex-urdf/robots/hands"

        robot_name = RobotName.leap
        RetargetingConfig.set_default_urdf_dir(robot_dir)
        left_config_path = get_default_config_path(robot_name,
                                                   RetargetingType.dexpilot,
                                                   HandType.left)
        right_config_path = get_default_config_path(robot_name,
                                                    RetargetingType.dexpilot,
                                                    HandType.right)
        self.left_retargeting = RetargetingConfig.load_from_file(
            left_config_path).build()
        self.right_retargeting = RetargetingConfig.load_from_file(
            right_config_path).build()

        if self.args_cli.add_left_hand:
            isaac_joint_names = self.env.scene["left_hand"].joint_names
            if not self.args_cli.free_hand:
                isaac_joint_names = isaac_joint_names[-16:]

            self.retarget2leftisaac = np.array([
                self.left_retargeting.joint_names.index(joint[1])
                for joint in isaac_joint_names
            ]).astype(int)

        if self.args_cli.add_right_hand:
            isaac_joint_names = self.env.scene["right_hand"].joint_names
            if not self.args_cli.free_hand:
                isaac_joint_names = isaac_joint_names[-16:]

            self.retarget2rightisaac = np.array([
                self.right_retargeting.joint_names.index(joint[1])
                for joint in isaac_joint_names
            ]).astype(int)
        self.visionpro_stream = VisionProStreamer(self.args_cli.avp_ip)

    def apply_hand_pose(self, pose, hand_name="right_hand"):

        hand_pose = pose.unsqueeze(0).clone()

        self.env.scene[hand_name].write_root_pose_to_sim(
            hand_pose,
            env_ids=torch.arange(self.env.num_envs).to(self.env.device))

    def run_free_hand(self):
        self.env.reset()

        for i in range(500):
            latest_data = self.visionpro_stream.get_latest()
            left_hand_qpos, left_wrist_pose, right_hand_qpos, right_wrist_pose = self.process_data(
                latest_data)

            actions = []
            if self.args_cli.add_left_hand:

                actions.append(
                    torch.tensor(left_hand_qpos, device=self.env.device))
            if self.args_cli.add_right_hand:

                actions.append(
                    torch.tensor(right_hand_qpos, device=self.env.device))

            actions = torch.cat(actions, dim=0).unsqueeze(0)

            if self.args_cli.add_right_hand:
                right_wrist_pose[:, 2] -= 0.8

                self.apply_hand_pose(right_wrist_pose, hand_name="right_hand")
            if self.args_cli.add_left_hand:
                left_wrist_pose[:, 2] -= 0.8
                self.apply_hand_pose(left_wrist_pose, hand_name="left_hand")

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

    def run_raw_arm_hand(self):
        self.env.reset()

        for i in range(500):
            latest_data = self.visionpro_stream.get_latest()
            left_hand_qpos, left_wrist_pose, right_hand_qpos, right_wrist_pose = self.process_data(
                latest_data)
            actions = []
            left_wrist_pose[:, :3] -= self.hand_offset
            right_wrist_pose[:, :3] -= self.hand_offset
            left_wrist_pose[:, :3] = torch.as_tensor([[0.4, 0.0, 0.3]
                                                      ]).to(self.env.device)
            right_wrist_pose[:, :3] = torch.as_tensor([[0.4, 0.0, 0.3]
                                                       ]).to(self.env.device)

            if self.args_cli.add_left_hand:
                actions.append(
                    torch.tensor(left_hand_qpos, device=self.env.device))
                actions.append(
                    torch.tensor(left_wrist_pose[0], device=self.env.device))

            if self.args_cli.add_right_hand:

                actions.append(
                    torch.tensor(right_hand_qpos, device=self.env.device))
                actions.append(
                    torch.tensor(right_wrist_pose[0], device=self.env.device))
            print(right_wrist_pose)
            actions = torch.cat(actions, dim=0).unsqueeze(0)

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

    def process_data(self, data):

        for hand_num, retargeting in enumerate(
            [self.left_retargeting, self.right_retargeting]):
            if hand_num == 0:
                joint_pose = two_mat_batch_mul(data["left_fingers"],
                                               OPERATOR2AVP_LEFT.T)
            else:
                joint_pose = two_mat_batch_mul(data["right_fingers"],
                                               OPERATOR2AVP_RIGHT.T)

            joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]

            indices = retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[
                origin_indices, :]

            if hand_num == 0:
                left_hand_qpos = retargeting.retarget(ref_value)[
                    self.retarget2leftisaac]

                wrist_pose = data["left_wrist"][0].copy()
                wrist_pose[:3, :3] = wrist_pose[:3, :3] @ OPERATOR2AVP_LEFT
                left_wrist_pose = self.transformation @ torch.as_tensor(
                    wrist_pose, dtype=torch.float32).to(self.env.device)

                left_hand_quat = math_utils.quat_from_matrix(
                    torch.as_tensor(left_wrist_pose[:3, :3]))
                left_wrist_pose = torch.cat(
                    [torch.as_tensor(left_wrist_pose[:3, 3]), left_hand_quat
                     ], ).to(self.env.device, dtype=torch.float32).unsqueeze(0)

            else:
                right_hand_qpos = retargeting.retarget(ref_value)[
                    self.retarget2rightisaac]

                wrist_pose = data["right_wrist"][0].copy()
                wrist_pose[:3, :3] = wrist_pose[:3, :3] @ OPERATOR2AVP_RIGHT
                right_wrist_pose = self.transformation @ torch.as_tensor(
                    wrist_pose, dtype=torch.float32).to(self.env.device)

                right_hand_quat = math_utils.quat_from_matrix(
                    torch.as_tensor(right_wrist_pose[:3, :3]))
                right_wrist_pose = torch.cat([
                    torch.as_tensor(right_wrist_pose[:3, 3]), right_hand_quat
                ], ).to(self.env.device, dtype=torch.float32).unsqueeze(0)

        return left_hand_qpos, left_wrist_pose, right_hand_qpos, right_wrist_pose
