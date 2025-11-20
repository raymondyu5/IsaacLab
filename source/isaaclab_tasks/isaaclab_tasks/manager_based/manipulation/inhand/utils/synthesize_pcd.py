import isaaclab.utils.math as math_utils

from tools.visualization_utils import vis_pc, visualize_pcd
import open3d as o3d
import trimesh
import os
import time
import pymeshlab
import shutil
import copy
import pickle
import numpy as np
import torch
try:
    from isaaclab.managers import ObservationTermCfg as ObsTerm
except:
    print("ObservationTermCfg not found, using ObsTerm instead.")
from collections import defaultdict


def configure_sysnthesize_robot_pc(
    obs_policy,
    env_cfg,
):

    if env_cfg["params"]["add_right_hand"]:
        setattr(obs_policy, "right_hand_sythesizer", "right_hand_sythesizer")

        right_hand_sythesizer = SynthesizeEnvPC(env_cfg, "right")
        obs_policy.right_hand_sythesizer = ObsTerm(
            func=right_hand_sythesizer.synthesize_env, )

    if env_cfg["params"]["add_left_hand"]:
        setattr(obs_policy, "left_hand_sythesizer", "left_hand_sythesizer")
        left_hand_sythesizer = SynthesizeEnvPC(env_cfg, "left")
        obs_policy.left_hand_sythesizer = ObsTerm(
            func=right_hand_sythesizer.synthesize_env, )


class SynthesizeEnvPC:

    def __init__(self, env_cfg, hand_side):

        self.env_cfg = env_cfg
        self.num_arm_pcd = env_cfg["params"].get("num_arm_pcd", 64)
        self.num_hand_pcd = env_cfg["params"].get("num_hand_pcd", 64)
        self.num_object_pcd = env_cfg["params"].get("num_object_pcd", 512)
        self.num_downsample_points = env_cfg["params"].get(
            "num_downsample_points", 2048)
        self.hand_side = hand_side
        self.mesh_init = False

    def synthesize_env(self, env):

        if not self.mesh_init:
            self.init_mesh(env)

        robot_link_state = env.scene[
            f"{self.hand_side}_hand"]._data.body_pose_w.clone()

        robot_link_state[:, :, :3] -= env.scene.env_origins.unsqueeze(
            1).repeat_interleave(robot_link_state.shape[1], dim=1)
        arm_link_state = robot_link_state[
            :,
            :8,
        ]
        hand_link_state = robot_link_state[:, 8:]

        arm_vertices = math_utils.transform_points(
            self.arm_mesh.reshape(-1, self.arm_mesh.shape[-2],
                                  3), arm_link_state[..., :3].reshape(-1, 3),
            arm_link_state[..., 3:7].reshape(-1, 4))
        hand_vertices = math_utils.transform_points(
            self.hand_mesh.reshape(-1, self.hand_mesh.shape[-2],
                                   3), hand_link_state[..., :3].reshape(-1, 3),
            hand_link_state[..., 3:7].reshape(-1, 4))

        arm_vertices = arm_vertices.reshape(env.num_envs, -1, 3)
        hand_vertices = hand_vertices.reshape(env.num_envs, -1, 3)
        object_state = env.scene[
            f"{self.hand_side}_hand_object"]._data.root_pose_w.clone()
        object_state[:, :3] -= env.scene.env_origins
        object_vertices = math_utils.transform_points(self.object_mesh,
                                                      object_state[..., :3],
                                                      object_state[..., 3:7])
        all_pcd = torch.cat([arm_vertices, hand_vertices, object_vertices],
                            dim=1)
        points_index = torch.randperm(all_pcd.shape[1]).to(env.device)

        sampled_pcd = all_pcd[:, points_index[:self.num_downsample_points]]

        return {
            "seg_pc": sampled_pcd.permute(0, 2, 1),
        }

    def init_mesh(self, env):

        self.arm_mesh_dir = self.env_cfg["params"]["arm_mesh_dir"]
        self.hand_mesh_dir = self.env_cfg["params"]["hand_mesh_dir"]
        self.target_manipulated_object = self.env_cfg['params'][
            "multi_cluster_rigid"][f"{self.hand_side}_hand_object"][
                "objects_list"]
        self.num_envs = env.num_envs
        self.arm_names = env.scene["right_hand"].body_names[:8]
        self.hand_names = env.scene["right_hand"].body_names[8:]

        self.load_arm_mesh(env, )
        self.load_hand_mesh(env, )
        self.laod_object_mesh(env)
        self.mesh_init = True

    def load_arm_mesh(self, env):

        self.arm_mesh = torch.zeros(
            (env.num_envs, len(self.arm_names), self.num_arm_pcd, 3),
            device=env.device,
            dtype=torch.float32)
        for index, name in enumerate(self.arm_names):
            arm_mesh = trimesh.load(
                os.path.join(self.arm_mesh_dir, f"{name}.obj"))
            arm_mesh = trimesh.util.concatenate(arm_mesh)

            vertices = torch.tensor(arm_mesh.vertices,
                                    dtype=torch.float32).to(env.device)
            self.arm_mesh[:, index] = math_utils.fps_points(
                vertices.unsqueeze(0), self.num_arm_pcd)

    def load_hand_mesh(self, env):
        self.hand_mesh = torch.zeros(
            (env.num_envs, len(self.hand_names), self.num_hand_pcd, 3),
            device=env.device,
            dtype=torch.float32)
        for index, link_name in enumerate(self.hand_names):

            if "palm_lower" in link_name:
                mesh_name = self.hand_side + "_" + link_name
            elif "thumb" in link_name:

                mesh_name = link_name
            else:
                parts = link_name.split("_")
                if parts[-1].isdigit():
                    mesh_name = "_".join(
                        parts[:-1])  # Drop the last numeric part
                else:
                    mesh_name = "_".join(parts)
            try:
                hand_mesh = trimesh.load(
                    os.path.join(self.hand_mesh_dir, f"{mesh_name}.obj"))
            except Exception as e:
                import pdb
                pdb.set_trace()

            hand_mesh = trimesh.util.concatenate(hand_mesh)

            mesh = pymeshlab.Mesh(vertex_matrix=np.array(hand_mesh.vertices, ),
                                  face_matrix=np.array(hand_mesh.faces,
                                                       dtype=np.int32))

            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh, 'my_mesh')
            ms.meshing_remove_duplicate_faces()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_surface_subdivision_midpoint(iterations=3)
            current_mesh = ms.current_mesh()

            vertices = current_mesh.vertex_matrix()
            faces = current_mesh.face_matrix()

            vertices = torch.tensor(hand_mesh.vertices,
                                    dtype=torch.float32).to(env.device)
            self.hand_mesh[:, index] = math_utils.fps_points(
                vertices.unsqueeze(0), self.num_hand_pcd)

    def laod_object_mesh(self, env):
        self.object_mesh = torch.zeros((env.num_envs, self.num_object_pcd, 3),
                                       dtype=torch.float32).to(env.device)
        num_object = len(self.target_manipulated_object)
        env_ids = torch.arange(env.num_envs).to(env.device)

        for index, obj_name in enumerate(self.target_manipulated_object):
            if index > env.num_envs:
                break
            object_setting = self.env_cfg["params"]["RigidObject"][obj_name]
            usd_path = object_setting.get("path", False)
            mesh_path = "/".join(
                usd_path.split("/")[:-1]) + "/textured_recentered.obj"

            obj_mesh = trimesh.load(mesh_path)
            obj_mesh = trimesh.util.concatenate(obj_mesh)
            vertices = torch.tensor(obj_mesh.vertices,
                                    dtype=torch.float32).to(env.device)
            downsample_vertices = math_utils.fps_points(
                vertices.unsqueeze(0), self.num_object_pcd)
            mask = (env_ids % num_object) == index

            self.object_mesh[mask] = downsample_vertices
