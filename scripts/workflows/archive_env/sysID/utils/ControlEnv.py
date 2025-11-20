import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym


class ControlEnv:

    def __init__(self):
        pass

    def init_callback(self, **args):
        """Initialize callback to move gripper and extract point cloud data."""

        from isaaclab.sensors.camera.tiled_camera import TiledCamera
        deform_object_cfg = self.env.scene["deform_object"].cfg.deform_cfg
        self.num_segmentation_objects = self.env.scene.num_objects_in_scene + 1  # +1 for background
        self.target_object_seg_id = (torch.arange(self.num_envs)).to(
            self.device) * self.num_segmentation_objects + deform_object_cfg[
                "camera_obs"]["segmentation_id"]

        # inital object point cloud
        # self.init_object_pc()
        # initialize robot
        # self.init_robot()

    def init_robot(self):
        if self.trajectory_generator.include_robot:
            for i in range(self.num_envs):
                self.hide_prim(self.env.scene.stage,
                               f"/World/envs/env_{i}/gripper")

    def hide_prim(self, stage: Usd.Stage, prim_path: str):
        """Hide a prim by setting its visibility attribute to 'invisible'."""
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        visibility_attribute = prim.GetAttribute(
            "visibility") if prim.IsValid() else None
        if visibility_attribute:
            visibility_attribute.Set("invisible")

    def filter_points(self):
        for i in range(self.num_explore_actions):
            filtered_pc = self.object_color_pc[i]

            xy_coords = filtered_pc[..., :2]
            dist = torch.norm(xy_coords.unsqueeze(2) - xy_coords.unsqueeze(1),
                              dim=-1)

            radius = self.env.scene["deform_object"].cfg.deform_cfg[
                "camera_obs"]["filter_xy_radius"]
            neighbor_mask = dist <= radius

            z_coords = filtered_pc[..., 2]
            z_neighbors_max = torch.max(z_coords.unsqueeze(1) * neighbor_mask,
                                        dim=-1).values

            threshold = self.env.scene["deform_object"].cfg.deform_cfg[
                "camera_obs"]["filter_z_threshold"]
            z_diff = z_coords - z_neighbors_max

            mask = abs(z_diff) >= threshold

            self.object_color_pc[i] = self.object_color_pc[i][mask]
            # self.visualize_point_cloud(i)
            # self.object_color_pc = self.object_color_pc[:, mask[0], :]

    def init_object_pc(self):

        obs = self.env.reset()[0]

        # sim_nodal_pc = self.env.scene[
        #     "deform_object"].root_physx_view.get_sim_nodal_positions(
        #     ) - self.env.scene.env_origins[:,
        #                                    None, :]  #-self.env.scene["deform_object"].data.default_root_state[:,:3]

        # # import open3d as o3d
        # pcd_nodal = o3d.geometry.PointCloud()

        # pcd_nodal.points = o3d.utility.Vector3dVector(
        #     sim_nodal_pc[0].cpu().numpy())

        # pcd_pc = o3d.geometry.PointCloud()

        # pcd_pc.points = o3d.utility.Vector3dVector(
        #     obs["policy"]["seg_pc"][0][..., :3].cpu().numpy())

        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.2, origin=[0.0, 0, 0])

        # o3d.visualization.draw_geometries(
        #     [pcd_nodal, pcd_pc, coordinate_frame])

        if self.require_segmentation:
            object_color_pc, object_seg_id = obs["policy"]["seg_pc"][
                ..., :3], obs["policy"]["seg_pc"][..., -1]
            self.batch_size, self.num_pc, _ = object_color_pc.shape

            if self.target_object_seg_id is not None:

                self.pc_mask = []
                self.object_color_pc = []
                for i in range(self.num_explore_actions):
                    mask = object_seg_id[i] == self.target_object_seg_id[i]
                    if not mask.any().all():
                        raise ValueError("Invalid target_object_seg_id")

                    self.object_color_pc.append(object_color_pc[i][mask])

                if self.env.scene["deform_object"].cfg.deform_cfg[
                        "camera_obs"]["filter_xy_radius"]:
                    self.filter_points()

                if self.debug_vis:
                    self.visualize_point_cloud()
            else:
                raise ValueError("required target_object_seg_id")

    def visualize_point_cloud(self, i):
        # import open3d as o3d
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            self.object_color_pc[i].cpu().numpy())
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0.5, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    def get_sample_attachment_points(self, explore_type, explore_action_index):
        if explore_type == "train":
            sample_attachment_points = self.sample_attachment_points[
                explore_action_index].unsqueeze(0).repeat_interleave(
                    self.env.num_envs, 0)
            if not self.render_all:
                self.reset_deformable_visual_obs(boolen=False)
        elif explore_type == "eval":

            sample_attachment_points = self.sample_attachment_xyz()
            sample_attachment_points[:self.
                                     num_explore_actions] = self.sample_attachment_points
        elif explore_type == "target":
            if self.env.scene["deform_object"].cfg.deform_cfg["camera_obs"][
                    "reset_pc_everytime"]:
                self.env.scene["deform_object"].remove_attachment()
                self.init_object_pc()

            sample_attachment_points = self.sample_attachment_xyz()

            target_env_attachment_points = self.sample_attachment_xyz(
                self.num_explore_actions)
            sample_attachment_points[:min(
                self.num_explore_actions, self.
                num_envs)] = target_env_attachment_points[:min(
                    self.num_explore_actions, self.num_envs)]
            self.sample_attachment_points = sample_attachment_points[:min(
                self.num_explore_actions, self.num_envs)]

        return sample_attachment_points

    def sample_attachment_xyz(self, target_num=None):

        sequeece_grasp_points = []
        for i in range(self.num_explore_actions):

            sequeece_grasp_index = torch.multinomial(torch.ones(
                self.object_color_pc[i].shape[0]),
                                                     num_samples=1,
                                                     replacement=False)
            sequeece_grasp_points.append(
                self.object_color_pc[i][sequeece_grasp_index])

        return torch.cat(sequeece_grasp_points)

    def reset_deformable_visual_obs(self, boolen=False):

        self.env.env.set_rendering = boolen

        for name in self.reset_camera_obs_list:
            self.env.scene["deform_object"].cfg.deform_cfg["camera_obs"][
                name] = boolen

    def randomize_deformable_properties(self, random_method, sample_parms):

        self.env.scene[
            "deform_object"].parames_generator.random_method = random_method
        self.env.scene[
            "deform_object"].parames_generator.params_range = sample_parms

    def on_step(self, explore_type, explore_action_index):

        actions = self.trajectory_generator.get_step_actions(
            explore_type, explore_action_index)

        return actions

    def step_env(self, explore_action_index, next_obs, explore_type,
                 save_interval, transition):
        images_buffer = []
        success = True

        init_pose = next_obs["policy"]["deformable_pose"][:self.
                                                          num_explore_actions]
        for _ in range(self.env.max_episode_length):

            # generate next actions
            actions = self.on_step(explore_type, explore_action_index)

            if explore_type == "target" or explore_type == "eval":

                if self.require_segmentation and "seg_rgb" in next_obs[
                        "policy"].keys():

                    images_buffer.append(next_obs["policy"]["seg_rgb"].cpu())
                elif "rgb" in next_obs["policy"].keys():
                    images_buffer.append(next_obs["policy"]["rgb"].cpu())

            next_obs, reward, terminate, time_out, info = self.env.step(
                actions)

            if self.env.episode_length_buf[
                    0] > self.env.max_episode_length - 4:
                self.reset_deformable_visual_obs(boolen=True)
            if self.env.episode_length_buf[
                    0] % save_interval == 0 and self.buffer is not None and self.env.episode_length_buf[
                        0] > 0:
                transition = self.buffer._cache_transition(
                    transition,
                    next_obs,
                    target_count=self.num_explore_actions
                    if explore_type in ["target", "eval"] else None,
                    reset=(self.env.episode_length_buf[0] == save_interval))

                if explore_type == "target" and self.trajectory_generator.has_gripper:

                    lift_or_not = next_obs["policy"][
                        "deformable_pose"][:self.trajectory_generator.
                                           num_gripper_actions][:, -1] > 0.2

                    if not lift_or_not.any().all():

                        print("The object is not lifted")
                        success = False
                        # if explore_type == "target" and self.trajectory_generator.include_robot:

                        #     cur_robot_deformable_pose = next_obs["policy"]["deformable_pose"][self.trajectory_generator.num_gripper_actions:self.trajectory_generator.num_robot_actions]

                        #     move_or_not = torch.norm(cur_robot_deformable_pose[:, :3] - init_pose[:, :3], dim=-1) > 0.1

                        #     if not move_or_not.any().all():

                        #         print("The object is not moved")
                        success = False
        return images_buffer, success

    def technical_reset(self, explore_type, reset_gripper,
                        explore_action_index):
        if self.trajectory_generator.has_gripper and reset_gripper:
            self.trajectory_generator.reset_gripper(explore_type,
                                                    explore_action_index)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.env.scene["deform_object"].random_physical_properties(env_ids)

        for i in range(2):
            obs = self.env.reset()
        for i in range(2):

            self.trajectory_generator.technical_reset(explore_type,
                                                      reset_gripper,
                                                      explore_action_index)
            self.env.sim.step(render=True)

        return self.env.observation_manager.compute()
