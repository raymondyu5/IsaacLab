# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.sensors.camera.batch_utils import create_pointcloud_from_rgbd_batch, create_pointcloud_from_depth
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
import time
"""
Root state.
"""
import time
from tools.visualization_utils import vis_pc, visualize_pcd
import open3d as o3d

from torch_cluster import fps


class CameraObservation:

    def __init__(self, camera_setting, sample_points=False):

        self.extract_all_pc = camera_setting.get("extract_all_pc", False)
        self.extract_seg_pc = camera_setting.get("extract_seg_pc", False)
        self.extract_rgb = camera_setting.get("extract_rgb", False)
        self.extract_seg_rgb = camera_setting.get("extract_seg_rgb", False)
        self.segmentation_name = camera_setting.get("segmentation_name", None)
        self.pc_bbox = camera_setting.get("pc_bbox", None)
        self.align_robot_base = camera_setting.get("align_robot_base", True)
        self.downsampled_points = camera_setting.get("num_downsampled_points",
                                                     2048)
        self.extract_depth = camera_setting.get("extract_depth", False)
        self.robot_name = camera_setting.get("robot_name", "robot")
        self.sample_points = sample_points

        self.camera_name_list = list(camera_setting["cameras"].keys())
        self.num_cameras = len(self.camera_name_list)

        # self.fps_points = math_utils.fps_points

    def extract_raw_camera_data(self, env):
        rgb_data, segementation_data, intrinsic_params, depth, position, orientation, extrinsic_orientation,id2lables = [], [], [], [], [], [], [],[]

        for name in self.camera_name_list:
            env.scene[name]._update_poses(
                torch.arange(env.num_envs).to(env.device))

            data = env.scene[name].data
            if self.extract_rgb or self.extract_seg_rgb:  # extract rgb data if any of these flags are set

                rgb_data.append(data.output["rgb"])
            if self.extract_seg_pc or self.extract_seg_rgb:  # extract segmentation data if any of these flags are set
                segementation_data.append(data.output[self.segmentation_name])

            # Extract position, orientation, and intrinsic parameters
            cam_pos_w = data.pos_w.clone() - env.scene.env_origins

            position.append(cam_pos_w)
            orientation.append(data.quat_w_ros)

            extrinsic_orientation.append(data.quat_local_opengl)
            intrinsic_params.append(data.intrinsic_matrices)

            if self.extract_all_pc or self.extract_seg_pc:

                # try:
                depth.append(
                    data.output["distance_to_image_plane"]
                )  # extract depth data if any of these flags are set
                # except:  # for tiled camera
                #     depth.append(
                #         data.output["distance_to_camera"]
                #     )  # extract depth data if any of these flags are set

            if self.extract_seg_pc or self.extract_seg_rgb:

                if isinstance(data.info, list):
                    id2lables.append(
                        data.info[0][self.segmentation_name]["idToLabels"])
                else:
                    id2lables.append(
                        data.info[self.segmentation_name]["idToLabels"])

        return rgb_data, segementation_data, intrinsic_params, depth, position, orientation, extrinsic_orientation, id2lables

    def get_camera_indices(self, camera_list):
        num_env = len(camera_list[0].intrinsic_matrices)
        num_camera = len(camera_list)
        pc_idx_per_env = (
            torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
            torch.arange(0, num_env).repeat_interleave(num_camera)).to('cuda')
        return pc_idx_per_env[0], num_env, num_camera

    def process_rgb_and_segmentation(self, result, rgb_data, seg_data,
                                     pc_idx_per_env, num_env, num_camera):

        rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])

        rgb = rgb[pc_idx_per_env].view(num_env, num_camera,
                                       *rgb.shape[1:])[..., :3]
        if torch.max(rgb) < 1.2:
            rgb = (rgb * 255).to(torch.uint8)

        if self.extract_seg_rgb:
            seg = torch.stack(seg_data).view(-1, *seg_data[0].shape[1:])

            seg = seg[pc_idx_per_env].view(num_env, num_camera, *seg.shape[1:])
            background_mask = torch.all(seg.unsqueeze(-1) == 1, axis=-1)
            frontground_mask = ~background_mask
            front_ground = torch.zeros_like(rgb) + 255
            front_ground[frontground_mask.squeeze(-1)] = rgb[
                frontground_mask.squeeze(-1)]

            result['seg_rgb'] = torch.cat([front_ground, seg], dim=-1)
            result['segmentation'] = seg

        if self.extract_rgb:

            result['rgb'] = rgb

    def process_point_cloud(
        self,
        env,
        result,
        rgb_data,
        depth,
        intrinsic_params,
        position,
        orientation,
        pc_idx_per_env,
        num_env,
        num_camera,
        seg_data,
        id2lables,
    ):

        if self.extract_seg_pc:
            if self.extract_rgb or self.extract_seg_rgb:
                rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])
            else:
                rgb = None

            depth_data = torch.stack(depth).view(
                -1, *depth[0].shape[1:]).squeeze(-1)

            # Stack and flatten seg_data to match depth_data's first dimension
            # Segmentation data may have multiple channels (e.g., RGBA), take only the first channel
            seg_mask_tensor = torch.stack(seg_data).view(-1, *seg_data[0].shape[1:])
            if seg_mask_tensor.dim() == 4 and seg_mask_tensor.shape[-1] > 1:
                # Take only the first channel if there are multiple channels (e.g., RGBA -> R)
                seg_mask_tensor = seg_mask_tensor[..., 0]

            points_xyz_rgb = create_pointcloud_from_rgbd_batch(
                intrinsic_matrix=torch.stack(intrinsic_params).view(-1, 3, 3),
                depth=depth_data,
                rgb=rgb,
                position=torch.stack(position).reshape(-1, 3),
                orientation=torch.stack(orientation).reshape(-1, 4),
            )

            num_cams_per_env = points_xyz_rgb.shape[0] // env.num_envs
            num_envs = env.num_envs

            points_xyz_rgb = points_xyz_rgb.view(
                num_cams_per_env, num_envs,
                *points_xyz_rgb.shape[-2:]).permute(1, 0, 2, 3)

            # Store the H*W dimension before reshaping
            hw_dim = points_xyz_rgb.shape[2]
            points_xyz_rgb = points_xyz_rgb.reshape(
                num_envs * num_cams_per_env, hw_dim, points_xyz_rgb.shape[-1])

            # pcd = points_xyz_rgb[:, :, :6].cpu().numpy()
            # for i in range(pcd.shape[0]):

            #     o3d_pc = vis_pc(pcd[i, :, :3], pcd[i, :, 3:6])
            #     o3d_pc.remove_non_finite_points()
            #     o3d.visualization.draw_geometries([o3d_pc])
            # Apply segmentation mask if provided
            B, H, W = depth_data.shape

            valid_mask = (~torch.isnan(points_xyz_rgb).any(dim=-1)
                          & ~torch.isinf(points_xyz_rgb).any(dim=-1))

            if seg_mask_tensor is not None:
                if self.downsampled_points > 0:
                    # seg_mask_tensor is already flattened to match depth_data: (B, H, W)
                    # Flatten H*W dimension
                    seg_mask_flatten = seg_mask_tensor.reshape(seg_mask_tensor.shape[0], -1)  # (B, H*W)

                    # Now apply the same permutation as points_xyz_rgb
                    # First reshape to match the view operation done on points_xyz_rgb
                    seg_mask_flatten = seg_mask_flatten.view(
                        num_cams_per_env, num_envs, -1).permute(1, 0, 2)  # (num_envs, num_cams_per_env, H*W)
                    seg_mask_flatten = seg_mask_flatten.reshape(num_envs * num_cams_per_env, -1)  # (num_envs * num_cams_per_env, H*W)

                    # Filter out background (R=0)
                    bg_mask = (seg_mask_flatten == 0)
                    valid_mask = ~bg_mask & valid_mask  # Keep only hand and object points

                    # Append segmentation ID to points_xyz_rgb
                    points_xyz_rgb = torch.cat(
                        [points_xyz_rgb,
                         seg_mask_flatten.unsqueeze(-1)],
                        dim=-1)  # (B, H*W, 4 or 7)

                else:
                    seg_mask_flatten = seg_mask_tensor.view(
                        B, -1, 1)  # Flatten the segmentation mask
                    #Append segmentation ID to points_xyz_rgb
                    points_xyz_rgb = torch.cat(
                        [points_xyz_rgb, seg_mask_flatten],
                        dim=-1)  # (B, H*W, 4 or 7)

            # Apply bounding box filter if provided
            if self.align_robot_base:
                points_xyz_rgb = align_pc_to_robot_base(
                    env, points_xyz_rgb, self.robot_name)

            points_xyz_rgb = points_xyz_rgb.reshape(num_env, -1,
                                                    points_xyz_rgb.shape[-1])
            valid_mask = valid_mask.reshape(num_env, -1)

            if self.pc_bbox is not None or "table_block" in env.scene.keys():
                if isinstance(self.pc_bbox, list):

                    self.pc_bbox = torch.as_tensor(self.pc_bbox).to(
                        env.device).unsqueeze(0).repeat_interleave(
                            env.num_envs, dim=0)
                if "table_block" in env.scene.keys():

                    self.pc_bbox[:, -1] = env.scene[
                        "table_block"]._data.root_state_w[:, 3].clone()

                valid_mask = crop_point_cloud(points_xyz_rgb, self.pc_bbox,
                                              valid_mask)

            if self.extract_all_pc:
                result['whole_pc'] = points_xyz_rgb.view(
                    num_env, num_camera, *points_xyz_rgb.shape[-2:])

            fps_points = []

            shapes = []

            for i in range(points_xyz_rgb.shape[0]):

                valid_points = torch.masked_select(points_xyz_rgb[i, :, :3],
                                                   valid_mask[i].unsqueeze(-1))
                valid_points = valid_points.view(-1, 3)
                points_index = torch.randperm(valid_points.shape[0])

                if self.sample_points:

                    fps_points.append(
                        math_utils.fps_points(valid_points,
                                              self.downsampled_points))
                else:

                    fps_points.append(valid_points[points_index].unsqueeze(0))

            # o3d_pc = vis_pc(fps_points[-1][..., :3].cpu().numpy().reshape(
            #     -1, 3))

            # visualize_pcd([o3d_pc])

            if self.sample_points:

                result['seg_pc'] = torch.cat(fps_points,
                                             dim=0).permute(0, 2, 1)
            else:
                result['seg_pc'] = fps_points

            if self.extract_depth:

                result['depth'] = depth_data[pc_idx_per_env].view(
                    num_env, num_camera, *depth_data.shape[1:])

    def process_camera_data(self, env: ManagerBasedRLEnv) -> dict:
        import time
        start_time = time.time()

        rgb_data, segementation_data, intrinsic_params, depth, position, orientation, extrinsic_orientation, id2lables = self.extract_raw_camera_data(
            env)

        pc_idx_per_env = (torch.arange(0, env.num_envs * self.num_cameras,
                                       env.num_envs).repeat(1, env.num_envs) +
                          torch.arange(0, env.num_envs).repeat_interleave(
                              self.num_cameras)).to('cuda')

        result = {}

        if self.extract_seg_rgb or self.extract_rgb:  #or self.extract_rgb:
            rgb_start = time.time()
            self.process_rgb_and_segmentation(result, rgb_data,
                                              segementation_data,
                                              pc_idx_per_env, env.num_envs,
                                              self.num_cameras)

        if self.extract_all_pc or self.extract_seg_pc:
            pc_start = time.time()
            self.process_point_cloud(env, result, rgb_data, depth,
                                     intrinsic_params, position, orientation,
                                     pc_idx_per_env, env.num_envs,
                                     self.num_cameras, segementation_data, id2lables)

        # process_camera_parameters(result, position, orientation,
        #                           extrinsic_orientation, intrinsic_params,
        #                           pc_idx_per_env, env.num_envs,
        #                           self.num_cameras)

        torch.cuda.empty_cache()
        # result["id2lables"] = id2lables
        del rgb_data, segementation_data, intrinsic_params, depth, position, orientation, extrinsic_orientation, id2lables

        return result


def align_pc_to_robot_base(env, points_xyz_rgb, robot_name="robot"):
    root_pose = env.scene[robot_name]._data.root_state_w
    # translate_root, quat_root = math_utils.subtract_frame_transforms(
    #     root_pose[:, :3], root_pose[:, 3:7],
    #     torch.zeros_like(root_pose[:, :3]),
    #     torch.tensor([[1., 0., 0.,
    #                    0.]]).to(env.device).repeat_interleave(env.num_envs, 0))
    # translate_root -= env.scene.env_origins
    translate_root = root_pose[:, :3].clone() - env.scene.env_origins
    quat_root = root_pose[:, 3:7].clone()

    num_cam = points_xyz_rgb.shape[0] // env.num_envs
    translate_root_expanded = translate_root.repeat_interleave(num_cam, dim=0)
    quat_root_expanded = quat_root.repeat_interleave(num_cam, dim=0)

    transformed_xyz = math_utils.transform_points(points_xyz_rgb[..., :3],
                                                  translate_root_expanded,
                                                  quat_root_expanded)

    return torch.cat([transformed_xyz, points_xyz_rgb[..., 3:]],
                     dim=-1).reshape(env.num_envs, -1,
                                     points_xyz_rgb.shape[-1])


def sample_points(points_xyz_rgb, max_length, valid_mask):

    if max_length > 0:

        valid_points_mask = valid_mask.clone()
        # valid_index = torch.stack(torch.where(valid_points_mask),dim=1)

        # For each batch, create a valid distribution and sample from it
        valid_sample_indices = torch.multinomial(
            torch.arange(max_length).unsqueeze(0).repeat_interleave(
                points_xyz_rgb.shape[0], dim=0).to(torch.float32),
            max_length).to(valid_mask.device)

        valid_mask[:] = False  # Reset valid_mask to all False
        for i in range(valid_mask.shape[0]):
            valid_id = torch.where(valid_points_mask[i])[0]

            valid_mask[i, valid_id[valid_sample_indices[i]]] = True

    return valid_mask


def crop_point_cloud(points_xyz_rgb, bbox, valid_mask):
    """
    points_xyz_rgb: (B, N, 6) or (N, 6)   [x,y,z,r,g,b]
    bbox:           (B, 6)   or (6,)      [xmin,ymin,zmin,xmax,ymax,zmax]
    valid_mask:     (B, N)   or (N,)      bool
    Returns:        mask with same shape as valid_mask
    """
    # Normalize shapes to batched form
    squeeze_out = False
    if points_xyz_rgb.dim() == 2:  # (N, 6)
        points = points_xyz_rgb[:, :3].unsqueeze(0)  # (1, N, 3)
        valid = valid_mask.unsqueeze(0)  # (1, N)
        bbox = bbox.unsqueeze(0) if bbox.dim() == 1 else bbox  # (1, 6)
        squeeze_out = True
    else:  # (B, N, 6)
        points = points_xyz_rgb[..., :3]  # (B, N, 3)
        valid = valid_mask  # (B, N)
        if bbox.dim() == 1:
            # expand single bbox to all batches
            bbox = bbox.unsqueeze(0).expand(points.size(0), -1)  # (B, 6)

    # Split mins/maxs
    mins = bbox[..., :3]  # (B, 3)
    maxs = bbox[..., 3:6]  # (B, 3)

    # Broadcast compare: (B, N, 3) vs (B, 1, 3)
    in_min = points >= mins.unsqueeze(1)
    in_max = points <= maxs.unsqueeze(1)
    bbox_mask = (in_min & in_max).all(dim=-1)  # (B, N)

    out = valid & bbox_mask

    return out[0] if squeeze_out else out


def process_camera_parameters(result, position, orientation,
                              extrinsic_orientation, intrinsic_params,
                              pc_idx_per_env, num_env, num_camera):
    extrinsic_position = torch.stack(position).view(-1, *position[0].shape[1:])
    extrinsic_orientation = torch.stack(extrinsic_orientation).view(
        -1, *orientation[0].shape[1:])

    extrinsic_orientation = math_utils.matrix_from_quat(extrinsic_orientation)

    extrisic_transformation = torch.eye(4).unsqueeze(0).repeat_interleave(
        len(extrinsic_position), 0).to(extrinsic_orientation.device)
    extrisic_transformation[:, :3, :3] = extrinsic_orientation
    extrisic_transformation[:, :3, 3] = extrinsic_position

    result['extrinsic_params'] = extrisic_transformation[pc_idx_per_env].view(
        num_env, num_camera, *extrisic_transformation.shape[1:3])

    intrinsic_params = torch.stack(intrinsic_params).view(
        -1, *intrinsic_params[0].shape[1:3])
    result['intrinsic_params'] = intrinsic_params[pc_idx_per_env].view(
        num_env, num_camera, *intrinsic_params.shape[1:3])
