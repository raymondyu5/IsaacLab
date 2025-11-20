# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, DeformableObject
from isaaclab.sensors import Camera
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab.sensors.camera.batch_utils import create_pointcloud_from_rgbd_batch
import isaaclab.utils.math as math_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# # import open3d as o3d
from tools.visualization_utils import *


def robot_qpos(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    robot_assest = env.scene["robot"]

    return robot_assest.data.joint_pos


def robot_ee_pose(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    robot_assest = env.scene["robot"]
    body_ids, body_names = robot_assest.find_bodies("panda_hand")

    return robot_assest.data.body_state_w[:, body_ids[0]]


def object_physical_params(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    physical_params = []
    for name in env.scene.keys():
        if "deform" in name:
            deformable_object = env.scene[name]
            physical_params.append(deformable_object.data.physical_params)

    physical_params = torch.stack(physical_params)
    num_object = physical_params.shape[0]
    num_env = physical_params.shape[1]
    physical_params = physical_params.view(-1, physical_params.shape[2])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            physical_params.device)[0]

    return physical_params[pc_idx_per_env].view(num_env, num_object,
                                                physical_params.shape[1])


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, ) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    for name in env.scene.keys():
        if "deform" in name:

            deformable_object = env.scene[name]

            tube_front_positions = deformable_object.data.nodal_state_w[:, 200] - torch.mean(
                deformable_object.data.
                default_nodal_state_w[:, :deformable_object.
                                      max_simulation_mesh_vertices_per_body],
                dim=1)
            tube_back_positions = deformable_object.data.nodal_state_w[:,
                                                                       int(
                                                                           deformable_object
                                                                           .max_simulation_mesh_vertices_per_body
                                                                           / 2
                                                                       )] - torch.mean(
                                                                           deformable_object
                                                                           .
                                                                           data
                                                                           .
                                                                           default_nodal_state_w[:, :
                                                                                                 deformable_object
                                                                                                 .
                                                                                                 max_simulation_mesh_vertices_per_body],
                                                                           dim=1
                                                                       )
            robot_assest = env.scene["robot"]
            body_ids, body_names = robot_assest.find_bodies("panda_hand")
            gripper_site_pos = robot_assest.data.body_state_w[:,
                                                              body_ids[0]][:, :
                                                                           3]
            front_to_gripper = tube_front_positions - gripper_site_pos

    return torch.cat(
        [tube_front_positions, tube_back_positions, front_to_gripper], dim=-1)


def object_node_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, ) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    nodals_position = []
    default_root_states = []
    contact_ornot = []
    for name in env.scene.keys():
        if "deform" in name:

            deformable_object = env.scene[name]

            nodals_position.append(
                deformable_object.data.
                nodal_state_w[:, :deformable_object.
                              max_simulation_mesh_vertices_per_body])
            default_root_states.append(
                deformable_object.data.default_root_state)
            deformable_object.data.root_pos_w

            # collision_element_stresses = deformable_object.data.collision_element_deformation_gradients
            # collision_element_stresses = collision_element_stresses.view(
            #     collision_element_stresses.shape[0],
            #     collision_element_stresses.shape[1] * 9)
            # conact_env_index = torch.where(collision_element_stresses > 1)[0]
            # contact_array = torch.zeros(len(collision_element_stresses), 1).to(
            #     collision_element_stresses.device)

            # contact_array[conact_env_index] = 1

            # contact_ornot.append(contact_array)

    nodals_position = torch.stack(nodals_position)
    num_object = nodals_position.shape[0]
    num_env = nodals_position.shape[1]
    nodals_position = nodals_position.view(-1, nodals_position.shape[2],
                                           nodals_position.shape[3])
    default_root_states = torch.stack(default_root_states)
    default_root_states = default_root_states.view(
        -1, default_root_states.shape[2])
    nodals_position[:, :, :
                    2] -= default_root_states[:, :
                                              2][:, None, :].repeat_interleave(
                                                  nodals_position.shape[1], 1)
    # import pdb

    # pdb.set_trace()
    # contact_ornot = torch.stack(contact_ornot)[:, None, :].repeat_interleave(
    #     nodals_position.shape[1], 1)
    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            nodals_position.device)[0]

    return nodals_position[pc_idx_per_env].view(num_env, num_object,
                                                nodals_position.shape[1], 3)


def object_camera_observation(
        env: ManagerBasedRLEnv,
        camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    camera: Camera = env.scene[camera_cfg.name]
    camera_list = []
    for name in env.scene.keys():
        if "camera" in name:

            camera_list.append(env.scene[name].data)
    depth_data = []
    rgb_data = []
    num_camera = 0
    for data in camera_list:
        num_camera += 1
        num_env = len(data.intrinsic_matrices)

        depth_data.append(data.output["distance_to_image_plane"][..., None])
        rgb_data.append(data.output["rgb"])

    depth = torch.stack(depth_data).view(-1, depth_data[0].shape[1],
                                         depth_data[0].shape[2])
    rgb = torch.stack(rgb_data).view(-1, rgb_data[0].shape[1],
                                     rgb_data[0].shape[2],
                                     rgb_data[0].shape[3])

    rgbd = torch.cat([rgb, depth[..., None]], dim=-1)
    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            depth.device)[0]

    return rgbd[pc_idx_per_env].view(num_env, num_camera, rgbd.shape[1],
                                     rgbd.shape[2], rgbd.shape[3])


def object_pc(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera")
) -> torch.Tensor:
    # camera: Camera = env.scene[camera_cfg.name]
    intrinsic_matrices = []
    depth = []
    rgb = []
    position = []
    orientation = []
    num_camera = 0
    camera_list = []
    for name in env.scene.keys():

        if "camera" in name:
            num_camera += 1
            camera_list.append(env.scene[name].data)

    for data in camera_list:
        position.append(data.pos_local)
        orientation.append(data.quat_local_ros)

        num_env = len(data.intrinsic_matrices)

        intrinsic_matrices.append(data.intrinsic_matrices)

        depth.append(data.output["distance_to_image_plane"])
        rgb.append(data.output["rgb"])

    # points_xyz, points_rgb = create_pointcloud_from_rgbd(
    #     intrinsic_matrix=intrinsic_matrices[0][0],
    #     depth=depth[0][0],
    #     rgb=rgb[0][0],
    #     position=position[0][0],
    #     orientation=orientation[0][0],
    # )
    # return torch.cat([points_xyz, points_rgb], dim=-1)

    points_xyz, points_rgb = create_pointcloud_from_rgbd_batch(
        intrinsic_matrix=torch.stack(intrinsic_matrices).view(-1, 3, 3),
        depth=torch.stack(depth).view(-1, depth[0].shape[1],
                                      depth[0].shape[2]),
        rgb=torch.stack(rgb).view(-1, rgb[0].shape[1], rgb[0].shape[2],
                                  rgb[0].shape[3]),
        position=torch.stack(position).reshape(-1, 3),
        orientation=torch.stack(orientation).reshape(-1, 4))

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            points_xyz.device)[0]

    # rearrange color,pc

    points_xyz_rgb = torch.cat([points_xyz, points_rgb], dim=-1)
    points_xyz_rgb = points_xyz_rgb[pc_idx_per_env]

    combined_points_xyz_rgb = points_xyz_rgb.view(num_env, -1, 6)

    return combined_points_xyz_rgb
