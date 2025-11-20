import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym


class MultiGripperTrajectoryGenerator:

    def __init__(self, ) -> None:

        self.sample_attachment_points = None

        self.num_gripper = 1

    def gripper_generator_trajectories(
        self,
        cur_attachment_points,
    ):

        gripper_offset_xyz = cur_attachment_points[:self.
                                                   num_explore_actions].unsqueeze(
                                                       1).unsqueeze(1).repeat(
                                                           1, self.env.
                                                           max_episode_length,
                                                           self.num_envs, 1, 1)
        gripper_offset_xyz[..., :3] += self.gripper_offset

        self.random_target_orientation = self.random_orientation()

        random_target_orientation = self.random_target_orientation.unsqueeze(
            1).repeat_interleave(self.num_envs, 1)

        plan_trajectories = self.heuristic_single_direction_trajectories(
            random_target_orientation)

        plan_trajectories[..., :3] += gripper_offset_xyz[..., :3]

        self.num_gripper_action_dim = plan_trajectories.shape[-1]

        return plan_trajectories.view(self.num_explore_actions,
                                      self.env.max_episode_length,
                                      self.num_envs, -1)

    def heuristic_single_direction_trajectories(self,
                                                random_target_orientation):
        """Generate heuristic gripper trajectories."""
        target_gripper_traj = torch.zeros(self.num_explore_actions,
                                          self.env.max_episode_length,
                                          self.num_envs,
                                          self.num_gripper,
                                          8,
                                          device=self.env.device)
        target_gripper_traj[..., -1] = self.gripper_actions

        target_gripper_traj[..., 3:7] = random_target_orientation.unsqueeze(
            1).repeat_interleave(self.env.max_episode_length, 1)

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
            1).unsqueeze(2).repeat_interleave(
                self.num_envs,
                2).unsqueeze(-2).repeat_interleave(self.num_gripper, -2)

        target_gripper_traj[:, self.static_frames:,
                            ..., :3] = proposed_traj * timestep.view(
                                1, proposed_traj.shape[1], 1, 1,
                                1).repeat_interleave(len(proposed_traj), 0)
        return target_gripper_traj
