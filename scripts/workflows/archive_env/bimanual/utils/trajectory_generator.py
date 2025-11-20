import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym

from scripts.workflows.bimanual.utils.robot_trajectory import RobotTrajectoryGenerator

from scripts.workflows.bimanual.utils.gripper_trajectories import MultiGripperTrajectoryGenerator


class TrajectoryGenerator(RobotTrajectoryGenerator,
                          MultiGripperTrajectoryGenerator):

    def __init__(self) -> None:

        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.sample_attachment_points = None

        self.plan_trajectories = None

    def init_callback(self):
        if self.has_gripper:

            MultiGripperTrajectoryGenerator.__init__(self)

        if self.include_robot:
            RobotTrajectoryGenerator.__init__(self)

    def generate_trajectories(
        self,
        cur_attachment_points,
    ):

        actions_buffer = []

        if self.has_gripper:
            gripper_trajectories = self.gripper_generator_trajectories(
                cur_attachment_points)
            actions_buffer.append(gripper_trajectories)
            self.num_gripper_action_dim = gripper_trajectories.shape[-1]

        if self.include_robot:
            push_trajectories = self.generate_push_trajectories(
                cur_attachment_points)
            actions_buffer.append(push_trajectories)
            self.num_robot_action_dim = push_trajectories.shape[-1]

        self.plan_trajectories = torch.cat(actions_buffer, dim=-1)

        self.num_robot_actions = self.num_explore_actions - self.num_gripper_actions

        if self.include_robot and self.has_gripper:
            # set the robot action to zero
            self.plan_trajectories[:self.num_gripper_actions, ...,
                                   self.num_gripper_action_dim + 2] = 0.8
            self.plan_trajectories[:self.num_gripper_actions, ...,
                                   self.num_gripper_action_dim] = 0.0

            # set the gripper action to zero
            self.plan_trajectories[self.num_gripper_actions:, ..., 0] = 1.8
            self.plan_trajectories[self.num_gripper_actions:, ..., 1] = 1.8
            self.plan_trajectories[self.num_gripper_actions:, ...,
                                   self.num_gripper_action_dim -
                                   1] = 1  # set the gripper action to open

        return self.plan_trajectories

    def get_step_actions(self, explore_type, explore_action_index, **args):

        if explore_type == "train":
            return self.plan_trajectories[explore_action_index,
                                          self.env.episode_length_buf[0]]
        elif explore_type == "target" or explore_type == "eval":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]

            actions[:self.
                    num_explore_actions] = self.plan_trajectories[:, self.env.
                                                                  episode_length_buf[
                                                                      0],
                                                                  0].clone()

            return actions

    def reset_robot_manipulator(self, explore_type, reset_gripper,
                                explore_action_index):

        if self.has_gripper and reset_gripper:
            self.reset_robot_raw_pose(explore_type, explore_action_index)

        elif not self.has_gripper and self.include_robot:

            indices = torch.arange(self.num_envs,
                                   dtype=torch.int64,
                                   device=self.device)

            # robot_target_pose = torch.cat([robot_jpos.squeeze(1)], dim=-1)
            robot_target_pose = self.env.scene[
                "robot"]._data.default_joint_pos[:, :9]
            robot_target_pose[:, -2:] = -1

            self.env.scene["robot"].root_physx_view.set_dof_positions(
                robot_target_pose, indices)
            self.env.scene["robot"].root_physx_view.set_dof_velocities(
                robot_target_pose * 0, indices)

    def reset_robot_raw_pose(self, explore_type, explore_action_index):

        if self.include_robot:
            self.robot_traj_generator.reset_robot_joint(
                explore_type, explore_action_index)

    def random_orientation(self):

        random_orientation = torch.rand(
            self.num_explore_actions,
            device=self.env.device) * (self.random_orientation_range[1] -
                                       self.random_orientation_range[0]
                                       ) + self.random_orientation_range[1]
        random_orientation = torch.cat([
            torch.zeros(self.num_explore_actions, 2, device=self.env.device),
            random_orientation.unsqueeze(1)
        ],
                                       dim=1)

        random_target_orientation = math_utils.quat_from_euler_xyz(
            random_orientation[:, 0], random_orientation[:, 1],
            random_orientation[:, 2])

        # random_target_orientation = random_target_orientation.unsqueeze(
        #     1).repeat_interleave(self.num_envs, 1)
        return random_target_orientation.view(
            self.num_explore_actions,
            4).unsqueeze(-2).repeat_interleave(self.num_gripper, -2)
