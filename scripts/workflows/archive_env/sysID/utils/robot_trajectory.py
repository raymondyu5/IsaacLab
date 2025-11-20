import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym
from tools.curobo_planner import IKPlanner


class RobotTrajectoryGenerator:

    def __init__(self, env, num_explore_actions, num_gripper_actions) -> None:

        self.env = env
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.num_explore_actions = num_explore_actions
        self.sample_attachment_points = None
        self.num_gripper_actions = num_gripper_actions

        self.init_setting()

    def init_setting(self):

        robot_cfg = self.env.scene["robot"].cfg.articulation_cfg[
            "robot_setting"]
        self.end_frame = robot_cfg["motion_type"]["push"]["end_frame"]
        self.static_frame = robot_cfg["motion_type"]["push"]["static_frame"]

        self.init_ee_pose = torch.as_tensor([
            settings["init_pose"]
            for settings in robot_cfg["motion_type"].values()
        ]).to(self.device)
        self.ee_motion_vel = torch.as_tensor([
            settings["vel"] for settings in robot_cfg["motion_type"].values()
        ]).to(self.device)
        self.key_motions = list(robot_cfg["motion_type"].keys())

        self.has_gripper = False

        self.reset_robot_target_pose = self.env.scene[
            "robot"]._data.default_joint_pos[:, :9].clone()
        self.reset_robot_target_pose[:, -2:] = -1

        self.gripper_actions = torch.as_tensor([-1]).to(self.device)

        self.random_orientation_range = robot_cfg["motion_type"]["push"][
            "random_orientation_rang"]

        if "gripper" in self.env.scene.keys():

            init_pos = torch.as_tensor([0.3, 0.0, 0.5, 0.0, 1.0, 0.0,
                                        0.0]).to(self.device)

            curobo_ik = IKPlanner()
            self.frozen_robot_sol = curobo_ik.plan_motion(
                init_pos[:3].unsqueeze(0), init_pos[3:7].unsqueeze(0))
            self.reset_robot_target_pose = self.env.scene[
                "robot"]._data.default_joint_pos[:, :9].clone()
            self.reset_robot_target_pose[:, -2:] = -1

    def robot_generator_trajectories(self):

        push_trajectories = self.generate_push_trajectories()

        return push_trajectories

    def generate_push_trajectories(self, ):

        push_init_pose = torch.cat([
            self.init_ee_pose,
            self.gripper_actions.unsqueeze(0).repeat_interleave(
                len(self.init_ee_pose), 0).repeat_interleave(2, -1)
        ],
                                   dim=-1)

        if self.num_explore_actions > len(push_init_pose):
            push_init_pose = push_init_pose.repeat_interleave(
                self.num_explore_actions, 0)[:self.num_explore_actions]
            ee_motion_vel_expanded = self.ee_motion_vel.repeat_interleave(
                self.num_explore_actions, 0)[:self.num_explore_actions]
        else:
            ee_motion_vel_expanded = self.ee_motion_vel

        push_trajectories = push_init_pose.unsqueeze(1).repeat_interleave(
            self.env.max_episode_length, 1
        ).unsqueeze(2).repeat_interleave(
            self.num_envs, 2
        )  #(num_explore_actions,max_episode_length, num_envs, max_episode_length, 9)

        push_trajectories[:, :, :,
                          1] = self.sample_attachment_points[:self.
                                                             num_explore_actions,
                                                             1].unsqueeze(
                                                                 1
                                                             ).unsqueeze(
                                                                 2
                                                             ).repeat(
                                                                 1, self.env.
                                                                 max_episode_length,
                                                                 self.num_envs)

        motion_frames_tensor = torch.arange(self.end_frame - self.static_frame,
                                            device=self.device)  # Shape [15]

        push_trajectories[:, self.static_frame:self.
                          end_frame, :, :3] += motion_frames_tensor.view(
                              1, len(motion_frames_tensor), 1,
                              1) * ee_motion_vel_expanded.unsqueeze(
                                  1).repeat_interleave(
                                      len(motion_frames_tensor),
                                      1).unsqueeze(2).repeat_interleave(
                                          self.num_envs, 2)

        # push_trajectories[:, self.end_frame:, :, :
        #                   3] = push_trajectories[:, self.end_frame -
        #                                          1, :, :3].unsqueeze(1).repeat(
        #                                              1, self.env.
        #                                              max_episode_length -
        #                                              self.end_frame, 1, 1)

        return push_trajectories

    def random_orientation(self):
        random_orientation = torch.rand(
            self.num_envs * self.num_explore_actions,
            device=self.env.device) * (self.random_orientation_range[1] -
                                       self.random_orientation_range[0]
                                       ) + self.random_orientation_range[1]
        random_orientation = torch.cat([
            torch.zeros(self.num_envs * self.num_explore_actions,
                        2,
                        device=self.env.device),
            random_orientation.unsqueeze(1)
        ],
                                       dim=1)
        random_target_orientation = math_utils.quat_from_euler_xyz(
            random_orientation[:, 0], random_orientation[:, 1],
            random_orientation[:, 2])

        return random_target_orientation.view(self.num_explore_actions,
                                              self.num_envs, 4)

    def reset_robot_joint(self, explore_type, explore_action_index):

        indices = torch.arange(self.num_envs,
                               dtype=torch.int64,
                               device=self.device)

        robot_target_pose = self.reset_robot_target_pose.clone()

        if explore_type == "train":
            if explore_action_index < self.num_gripper_actions:  #avoid the collision with gripper
                robot_target_pose[:] = self.frozen_robot_sol[0][0][:9]

        elif explore_type == "target" or explore_type == "eval":  #avoid tjh collision with gripper

            robot_target_pose[:self.
                              num_gripper_actions] = self.frozen_robot_sol[0][
                                  0][:9]

        self.env.scene["robot"].root_physx_view.set_dof_positions(
            robot_target_pose, indices)
        self.env.scene["robot"].root_physx_view.set_dof_velocities(
            robot_target_pose * 0, indices)
        self.env.scene[
            "robot"]._data.default_joint_pos[:, :9] = robot_target_pose
