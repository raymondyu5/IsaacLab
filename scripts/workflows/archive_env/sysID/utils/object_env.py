import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym


class ObjectEnv:

    def __init__(self):
        self.deformable_asset = self.env.scene["deform_object"]
        self.env_ids = torch.arange(self.num_envs, device=self.device)

    def new_episode_training_start(self,
                                   explore_type,
                                   explore_action_index,
                                   reset_gripper,
                                   repeat=True):

        self.reset_deformable_object_pose(explore_type, explore_action_index,
                                          reset_gripper, repeat)

    def reset_deformable_object_pose(self,
                                     explore_type,
                                     explore_action_index,
                                     reset_gripper,
                                     repeat=True):
        # get default root state
        if self.has_gripper and reset_gripper:
            self.env.scene["deform_object"].remove_attachment()

        if explore_type == "target":
            if self.has_gripper:
                self.move_gripper_away()
            self.random_init_deformable_pose(repeat=repeat)

        elif explore_type == "train":
            self.reinit_deformable_object_pose(explore_action_index)
        elif explore_type == "eval":
            self.reinit_eval_deformable_object_pose(explore_action_index)

    def reinit_eval_deformable_object_pose(self, explore_action_index):
        random_pos = self.deformable_random_init_pose[
            explore_action_index].unsqueeze(0).repeat_interleave(
                self.num_envs, 0).clone()
        random_pos[:self.
                   num_explore_actions, :] = self.deformable_random_init_pose

        self.reset_deformable(random_pos)

    def reinit_deformable_object_pose(self, explore_action_index):
        random_pos = self.deformable_random_init_pose[
            explore_action_index].unsqueeze(0).repeat_interleave(
                self.num_envs, 0).clone()

        self.reset_deformable(random_pos)

    def random_init_deformable_pose(self, repeat=True):
        pose_range = self.deformable_asset.cfg.deform_cfg.get("pose_range", {})

        # Define ranges for x, y, z, roll, pitch, yaw
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device='cuda:0')

        # Sample random positions and orientations
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1], (len(self.env_ids), 6),
            device=self.deformable_asset.device)
        positions = rand_samples[:, :3]
        orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 3],
                                                      rand_samples[:, 4],
                                                      rand_samples[:, 5])

        random_pos = torch.cat([positions, orientations], dim=-1)

        if repeat:
            random_pos[self.num_explore_actions:, :] = random_pos[0, :]

        # Calculate nodal states exclude the env spacing
        self.deformable_random_init_pose = random_pos.clone(
        )[:self.num_explore_actions]

        self.reset_deformable(random_pos)

    def reset_deformable(self, random_pos):
        if self.has_gripper:

            self.env.sim.pause()

            self.reset_randomized_deformable_object_pose(random_pos)
            self.reset_randomized_deformable_object_nodal_state(random_pos)

        else:  # if just robot not need for reset
            self.reset_randomized_deformable_object_nodal_state(random_pos)
        self.env.sim.play()

    def reset_randomized_deformable_object_nodal_state(self, random_pos):
        # set the origns of the environment
        raw_nodal_w = (
            self.deformable_asset.data.
            default_nodal_state_w[:, :self.deformable_asset.
                                  max_simulation_mesh_vertices_per_body, :3].
            clone() - self.env.scene.env_origins[:, None, :].repeat_interleave(
                self.deformable_asset.max_simulation_mesh_vertices_per_body,
                1))
        # set the origns of the coordinates
        raw_nodal_w[:, :, :3] -= self.deformable_asset.data.default_root_state[
            ..., :3].unsqueeze(1)
        # apply the trasnformations
        init_nodal_w = math_utils.transform_points(raw_nodal_w,
                                                   random_pos[:, :3],
                                                   random_pos[:, 3:])
        # Set final nodal positions after applying the transformations
        tansformed_nodal_pos = init_nodal_w + self.env.scene.env_origins[:, None, :].repeat_interleave(
            self.deformable_asset.max_simulation_mesh_vertices_per_body, 1)
        tansformed_nodal_pos += self.deformable_asset.data.default_root_state[
            ..., :3].unsqueeze(1)
        # Set final nodal states
        tansformed_nodal_w = torch.cat(
            [tansformed_nodal_pos,
             torch.zeros_like(tansformed_nodal_pos)],
            dim=1)

        self.deformable_asset._data.reset_nodal_state_w = tansformed_nodal_w
        self.deformable_asset.write_root_state_to_sim(tansformed_nodal_w,
                                                      env_ids=self.env_ids)

    def reset_randomized_deformable_object_pose(self, random_pos):

        target_pose = self.deformable_asset.data.default_root_state.clone()
        target_pose[:, :7] = random_pos[:, :7].clone()
        self.deformable_asset.reset_default_root(target_pose)

    def move_gripper_away(self):

        target_root_state = self.env.scene[
            "gripper"].data.default_root_state.clone()

        target_root_state[:, 2] += 1.0
        self.env.scene[
            "gripper"].data.default_root_state = target_root_state.clone()

        self.env.scene["gripper"].reset_default_root_state(target_root_state)
