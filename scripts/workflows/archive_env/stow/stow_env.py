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
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.workflows.tasks.stow.placement_samplers import UniformRandomSampler, SequentialCompositeSampler, sample_positions, SequentialSampleObjects
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg


class StowEnv:

    def __init__(self, env, init_pos, shelf_info):
        self.env = weakref.proxy(env)
        self.num_envs = env.num_envs
        self.device = env.device
        self.env_indices = torch.arange(self.num_envs,
                                        dtype=torch.int64,
                                        device=self.device)

        self.shelf_info = shelf_info
        self.init_pos = init_pos
        self.init_robot(init_pos)
        self.load_objecs_size()
        self.init_placement_sampler()

    def init_placement_sampler(self):

        self.sequential_sampler = SequentialSampleObjects(
            self.shelf_info, self.object_bbox, self.num_deformable_objects,
            self.device)

    def load_objecs_size(self):

        self.object_bbox = []
        with open("source/assets/object_size.yaml", 'r') as file:
            info = yaml.safe_load(file)
        self.deform_object_names = [*self.env.scene._deformable_objects.keys()]

        self.rigid_object_names = [*self.env.scene._rigid_objects.keys()]

        self.num_deformable_objects = len(self.deform_object_names)
        self.num_rigid_objects = len(self.rigid_object_names)

        self.rigid_bodies = []
        self.deformable_bodies = []

        self.assest_raw_pose = []

        for name in (self.deform_object_names + self.rigid_object_names):

            rotation_matrix = math_utils.matrix_from_quat(
                self.env.scene[name]._data.default_root_state[0][3:7])
            self.assest_raw_pose.append(
                self.env.scene[name]._data.default_root_state[0][:7])
            if "rigid" in name:
                raw_bbox = info[
                    self.env.scene[name].cfg.rigid_cfg["name"]]["corners"]
                self.rigid_bodies.append(self.env.scene[name])
                corners = self.transform_bbox(raw_bbox, rotation_matrix)

            else:

                deform_object = self.env.scene[name]
                default_nodal_w = deform_object._data.default_nodal_state_w[:, :
                                                                            deform_object
                                                                            .
                                                                            max_simulation_mesh_vertices_per_body][
                                                                                0].clone(
                                                                                )
                default_nodal_w -= (
                    self.env.scene.env_origins[0] +
                    deform_object._data.default_root_state[0][:3])
                default_nodal_w *= 1.05
                # 4 corners on the bottom or top plane
                x_max, y_max, z_max = default_nodal_w.max(dim=0).values
                x_min, y_min, z_min = default_nodal_w.min(dim=0).values
                corners = torch.stack([
                    torch.as_tensor([x_min, y_min,
                                     z_min]),  # Bottom-left (or top-left)
                    torch.as_tensor([x_max, y_min,
                                     z_min]),  # Bottom-right (or top-right)
                    torch.as_tensor([x_max, y_max,
                                     z_max]),  # Top-right (or bottom-right)
                    torch.as_tensor([x_min, y_max,
                                     z_max]),  # Top-left (or bottom-left)
                ]).view(-1).to(self.device)

                self.deformable_bodies.append(self.env.scene[name])

                # # import open3d as o3d
                # pcd_nodal = o3d.geometry.PointCloud()

                # pcd_nodal.points = o3d.utility.Vector3dVector(
                #     default_nodal_w.cpu().numpy())

                # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                #     size=0.2, origin=[0.0, 0, 0])

                # o3d.visualization.draw_geometries(
                #     [pcd_nodal, coordinate_frame])

            self.object_bbox.append(corners)

        self.object_bbox = torch.stack(self.object_bbox)

    def init_robot(self, init_pos):

        curobo_ik = IKPlanner()

        init_qpos = curobo_ik.plan_motion(init_pos[:, :3], init_pos[:, 3:])
        self.robot_target_qpos = self.env.scene[
            "robot"]._data.default_joint_pos[:, :9].clone()
        self.env.scene[
            "robot"]._data.default_joint_pos[:, :9] = init_qpos.squeeze(1)

        self.robot_target_qpos[:] = init_qpos.squeeze(1)

    def step_env(self):

        self.reset_env()
        # self.env.reset()
        for i in range(10):
            self.env.sim.step(render=True)

        for k in range(self.env.max_episode_length):
            # sample actions from -1 to 1
            actions = torch.rand(self.env.action_space.shape,
                                 device=self.device) * 0
            actions[:, :7] = self.init_pos
            # if k > 30:
            #     actions[:, 0] = 0.5
            #     actions[:, 1] = 0.2 - 0.005 * (k - 30)
            #     actions[:, 2] = 0.04

            next_obs, reward, terminate, time_out, info = self.env.step(
                actions)

        self.env.scene["robot"].root_physx_view.set_dof_positions(
            self.robot_target_qpos, self.env_indices)
        self.env.scene["robot"].root_physx_view.set_dof_velocities(
            self.robot_target_qpos * 0, self.env_indices)

    def reset_env(self):

        sample_position = self.sequential_sampler.sample()

        transformed_pose = sample_position.unsqueeze(1).repeat_interleave(
            self.num_envs, 1)

        if self.deformable_bodies != []:
            self.reset_randomized_deformable_object_nodal_state(
                transformed_pose[:self.num_deformable_objects])

        if self.rigid_bodies != []:
            self.reset_rigid_object_raw_state(
                transformed_pose[self.num_deformable_objects:])

    def transform_bbox(self, raw_bbox, rotation_matrix):

        rotated_corners = torch.matmul(
            torch.as_tensor(raw_bbox).reshape(-1, 3).to(self.device),
            rotation_matrix.T)

        # Compute the new size

        new_size = rotated_corners.max(dim=0).values - rotated_corners.min(
            dim=0).values

        new_bbox = rotated_corners.view(-1)
        return new_bbox

    def reset_rigid_object_raw_state(self, random_pos):

        for index, rigid_object in enumerate(self.rigid_bodies):
            # set into the physics simulation
            root_states = rigid_object.data.default_root_state[
                self.env_indices].clone()
            positions = self.env.scene.env_origins[
                self.env_indices] + random_pos[index, :, 0:3]
            orientations = math_utils.quat_mul(
                random_pos[index, :, 3:7],
                root_states[:, 3:7],
            )

            rigid_object.data.reset_root_state[:, :7] = torch.cat(
                [positions, orientations], dim=-1)
            rigid_object.write_root_pose_to_sim(torch.cat(
                [positions, orientations], dim=-1),
                                                env_ids=self.env_indices)
            rigid_object.write_root_velocity_to_sim(root_states[:, 7:13] * 0,
                                                    env_ids=self.env_indices)

            # mdp.randomize_rigid_body_mass(
            #     self.env,
            #     self.env_indices,
            #     SceneEntityCfg(rigid_object.body_names[0]),
            #     mass_distribution_params=(0.5, 1.5),
            #     operation="scale",
            #     distribution="uniform",
            # )
            rigid_object.reset(self.env_indices)
            self.env.sim.step(render=True)

    def reset_randomized_deformable_object_nodal_state(self, random_pos):

        for index, deformable_object in enumerate(self.deformable_bodies):
            # set the origns of the environment
            raw_nodal_w = (
                deformable_object.data.
                default_nodal_state_w[:, :deformable_object.
                                      max_simulation_mesh_vertices_per_body, :
                                      3].clone() -
                self.env.scene.env_origins[:, None, :].repeat_interleave(
                    deformable_object.max_simulation_mesh_vertices_per_body,
                    1))
            # set the origns of the coordinates
            raw_nodal_w[:, :, :3] -= deformable_object.data.default_root_state[
                ..., :3].unsqueeze(1)

            random_pos[
                index,
                ..., :3] -= deformable_object.data.default_root_state[:, :3]
            # apply the trasnformations
            init_nodal_w = math_utils.transform_points(
                raw_nodal_w, random_pos[index, :, :3], random_pos[index, :,
                                                                  3:])
            # Set final nodal positions after applying the transformations
            tansformed_nodal_pos = init_nodal_w + self.env.scene.env_origins[:, None, :].repeat_interleave(
                deformable_object.max_simulation_mesh_vertices_per_body, 1)
            tansformed_nodal_pos += deformable_object.data.default_root_state[
                ..., :3].unsqueeze(1)

            tansformed_nodal_pos[...,
                                 2] -= torch.min(tansformed_nodal_pos[..., 2],
                                                 dim=1).values + 0.03
            # Set final nodal states
            tansformed_nodal_w = torch.cat(
                [tansformed_nodal_pos,
                 torch.zeros_like(tansformed_nodal_pos)],
                dim=1)

            deformable_object._data.reset_nodal_state_w = tansformed_nodal_w
            deformable_object.write_root_state_to_sim(tansformed_nodal_w,
                                                      env_ids=self.env_indices)
            deformable_object.reset(self.env_indices)
            self.env.sim.step(render=True)
