import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym


class FloatingGripperTrajectoryGenerator:

    def __init__(self, env, num_explore_actions, num_gripper_actions) -> None:

        self.env = env
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.num_explore_actions = num_explore_actions
        self.sample_attachment_points = None
        self.num_gripper_actions = num_gripper_actions
        self.init_setting()

    def init_setting(self):
        deform_object_cfg = self.env.scene["deform_object"].cfg.deform_cfg
        floating_gripper_setting = deform_object_cfg["floating_gripper"]

        self.static_frames = floating_gripper_setting["static_frames"]

        self.gripper_offset = torch.as_tensor(
            floating_gripper_setting["gripper_offset"]).to(self.device)
        self.gripper_offset_xyz = torch.as_tensor(
            floating_gripper_setting["gripper_offset_xyz"]).to(self.device)

        self.random_orientation_range = floating_gripper_setting[
            "random_orientation_range"]

        self.trajectories_dir = torch.as_tensor(
            floating_gripper_setting["random_position_dir"]).to(self.device)
        self.gripper_actions = torch.as_tensor(
            [floating_gripper_setting["gripper_actions"]]).to(self.device)

    def gripper_generator_trajectories(self, **args):

        gripper_offset_xyz = self.gripper_offset_xyz.clone()

        gripper_offset_xyz = self.sample_attachment_points[:self.
                                                           num_explore_actions].unsqueeze(
                                                               1
                                                           ).unsqueeze(
                                                               1
                                                           ).repeat(
                                                               1, self.env.
                                                               max_episode_length,
                                                               self.num_envs,
                                                               1)
        gripper_offset_xyz[..., :3] += self.gripper_offset
        self.random_target_orientation = self.random_orientation()
        plan_trajectories = self.heuristic_single_direction_trajectories()

        plan_trajectories[..., :3] += gripper_offset_xyz[..., :3]

        self.num_gripper_actions = self.trajectories_dir.shape[0]

        self.num_gripper_action_dim = plan_trajectories.shape[-1]

        return plan_trajectories

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

    def heuristic_single_direction_trajectories(self, ):
        """Generate heuristic gripper trajectories."""

        target_gripper_traj = torch.zeros(self.num_explore_actions,
                                          self.env.max_episode_length,
                                          self.num_envs,
                                          8,
                                          device=self.env.device)
        target_gripper_traj[..., -1] = self.gripper_actions
        target_gripper_traj[...,
                            3:7] = self.random_target_orientation.unsqueeze(
                                1).repeat_interleave(
                                    self.env.max_episode_length, 1)

        timestep = torch.arange(0,
                                self.env.max_episode_length -
                                self.static_frames,
                                device=self.env.device)

        if self.num_explore_actions > len(self.trajectories_dir):
            proposed_traj = self.trajectories_dir.repeat_interleave(
                self.num_explore_actions, 0)[:self.num_explore_actions]

        else:
            proposed_traj = self.trajectories_dir
        proposed_traj = proposed_traj.unsqueeze(1).repeat_interleave(
            self.env.max_episode_length - self.static_frames,
            1).unsqueeze(2).repeat_interleave(self.num_envs, 2)

        target_gripper_traj[:, self.static_frames:, :, :
                            3] = proposed_traj * timestep.view(
                                1, proposed_traj.shape[1], 1, 1)
        return target_gripper_traj
