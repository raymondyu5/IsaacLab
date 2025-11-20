import torch
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import isaacsim.core.utils.prims as prim_utils

from curobo.util.usd_helper import UsdHelper
from torch_cluster import fps
import isaaclab.utils.math as math_utils

from tools.visualization_utils import vis_pc, visualize_pcd
import open3d as o3d
import trimesh
import os
import time
import pymeshlab
import shutil

COLOR_LIST = {
    "LINK": [0.0, 0.0, 1.0],
    "FINGER": [0.0, 0.0, 1.0],
    "PALM": [0.0, 0.0, 1.0],
    "RIGID_BODIES": ([[1.0, 0.27, 0.27]])
}


class SynthesizePC:

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.spawn_mesh_list = []
        self.colorize_pc = env_cfg["params"]["spawn_robot"]["colorize_pc"]

        if env_cfg["params"]["add_right_hand"]:
            self.init_settings("right")
            self.add_right_hand = True
        if env_cfg["params"]["add_left_hand"]:
            self.init_settings("left")
            self.add_left_hand = True
        self.synthesize_rigid_objects = False

        if env_cfg["params"]["spawn_rigid_objects"]["init"]:
            self.spawn_mesh_list += env_cfg["params"]["spawn_rigid_objects"][
                "spawn_list"].copy()
            self.rigid_object_list = env_cfg["params"]["spawn_rigid_objects"][
                "spawn_list"].copy()
            self.synthesize_rigid_objects = True
        else:
            self.synthesize_rigid_objects = False
            self.rigid_object_list = []

        self.cache_mesh()

        self.init_mesh = False
        self.num_downsample_points = env_cfg["params"]["spawn_robot"][
            "num_downsample_points"]

    def init_settings(self, hand_side):
        spawn_mesh_list = []

        spawn_mesh_list += self.env_cfg["params"]["spawn_robot"][
            "spawn_hand_list"].copy()

        self.hand_mesh_path = self.env_cfg["params"]["hand_mesh_dir"]

        if self.env_cfg["params"]["arm_type"] is not None and self.env_cfg[
                "params"]["spawn_robot"]["spawn_arm"]:
            self.spawn_arm_list = self.env_cfg["params"]["spawn_robot"][
                "spawn_arm_list"].copy()
            self.arm_type = self.env_cfg["params"]["arm_type"]
            spawn_mesh_list += self.spawn_arm_list
            self.arm_mesh_path = self.env_cfg["params"]["arm_mesh_dir"]

        self.spawn_mesh_list = [
            f'{hand_side}_{mesh_name.replace(".*", hand_side)}'
            for mesh_name in spawn_mesh_list
        ]

    def cache_mesh(self):
        timestamp = int(time.time())
        self.cache_dir = (f"logs/trash/{timestamp}")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(f"{self.cache_dir}/raw_mesh", exist_ok=True)
        self.save_data(self.hand_mesh_path, f"{self.cache_dir}/raw_mesh")
        if self.env_cfg["params"]["arm_type"] is not None and self.env_cfg[
                "params"]["spawn_robot"]["spawn_arm"]:
            self.save_data(self.arm_mesh_path, f"{self.cache_dir}/raw_mesh")
        if self.synthesize_rigid_objects:
            self.save_data(
                self.env_cfg["params"]["spawn_rigid_objects"]
                ["object_mesh_dir"], f"{self.cache_dir}/raw_mesh")

    def save_data(self, source_path, target_path):
        # print(os.listdir(source_path))
        for filename in os.listdir(source_path):
            src_file = os.path.join(source_path, filename)
            dst_file = os.path.join(target_path, filename)
            if os.path.isfile(src_file):  # optionally skip subdirectories
                shutil.copy(src_file, dst_file)

    def prepare_color_list(self, env, link_name):
        self.color_list = []
        rigid_id = 0

        if "palm" in link_name:
            color = COLOR_LIST["PALM"]
        elif "thumb" in link_name or "finger" in link_name:
            color = COLOR_LIST["FINGER"]
        elif link_name in self.rigid_object_list:

            color = COLOR_LIST["RIGID_BODIES"][rigid_id]

            rigid_id += 1
        else:
            color = COLOR_LIST["LINK"]
        all_link_color = torch.as_tensor(color).unsqueeze(0).repeat_interleave(
            self.num_downsample_points, dim=0).to(env.device)

        self.mesh_dict[link_name] = torch.cat(
            [self.mesh_dict[link_name],
             all_link_color.unsqueeze(0)], dim=2)

    def augment_mesh(self, mesh):

        mesh = trimesh.util.concatenate(mesh)

        vertices = mesh.vertices
        faces = mesh.faces
        # vertices = fps_points(vertices, downsample_points=2048)

        mesh = pymeshlab.Mesh(vertex_matrix=np.array(vertices, ),
                              face_matrix=np.array(faces, dtype=np.int32))

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh, 'my_mesh')
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_surface_subdivision_midpoint(iterations=3)
        current_mesh = ms.current_mesh()

        vertices = current_mesh.vertex_matrix()
        faces = current_mesh.face_matrix()

        # if self.downsample_points is not None:

        #     vertices = math_utils.fps_points(vertices, downsample_points=2048)

        return vertices, faces

    def process_mesh(self, env, mesh_name, link_name):
        if env is not None:
            device = env.device
            num_envs = env.num_envs
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            num_envs = 1
        mesh = trimesh.load(f"{self.cache_dir}/raw_mesh/{mesh_name}.obj")

        if mesh_name in self.spawn_arm_list:
            mesh = trimesh.util.concatenate(mesh)
            vertices = torch.as_tensor(mesh.vertices).unsqueeze(0).to(device)
        else:
            vertices, faces = self.augment_mesh(mesh)
        vertices = torch.as_tensor(vertices).unsqueeze(0).to(device)
        vertices = math_utils.fps_points(
            vertices, downsample_points=self.num_downsample_points)

        self.mesh_dict[link_name] = vertices.type(torch.float32)
        # if self.colorize_pc:
        #     self.prepare_color_list(env, link_name)

    def load_mesh(self, env=None):
        if env is not None:
            device = env.device
            num_envs = env.num_envs
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            num_envs = 1

        self.mesh_dict = {}

        for index, link_name in enumerate(self.spawn_mesh_list):

            if "palm" in link_name:
                mesh_name = link_name
            elif "thumb" in link_name:

                mesh_name = "_".join(link_name.split("_")[1:])
            elif link_name in self.rigid_object_list:
                mesh_name = link_name

            else:
                parts = link_name.split("_")[1:]
                if parts[-1].isdigit():
                    mesh_name = "_".join(
                        parts[:-1])  # Drop the last numeric part
                else:
                    mesh_name = "_".join(parts)

            self.process_mesh(env, mesh_name, link_name)

        # if self.env_cfg["params"]["arm_type"] is not None and self.env_cfg[
        #         "params"]["spawn_robot"]["spawn_arm"]:
        #     if self.env_cfg["params"]["add_right_hand"]:
        #         self.process_mesh(env, "link_base", "right_hand")
        #     if self.env_cfg["params"]["add_left_hand"]:
        #         self.process_mesh(env, "link_base", "left_hand")

        self.synthesize_points = torch.cat(
            list(self.mesh_dict.values()), dim=0).to(device).repeat_interleave(
                num_envs, dim=0).unsqueeze(0).repeat_interleave(num_envs,
                                                                dim=0)

        self.init_mesh = True

    def synthesize_pc(self, env):
        # start_time = time.time()
        if not self.init_mesh:

            self.load_mesh(env)

        state_info_list = []
        for link_name in self.mesh_dict.keys():

            state_info = env.scene[link_name]._data.root_state_w[:, :7]
            state_info_list.append(state_info.unsqueeze(0))

        state_final = torch.cat(state_info_list, dim=1).reshape(-1, 7)
        transformed_batch_point = math_utils.transform_points(
            self.synthesize_points[..., :3].clone().reshape(
                -1, self.num_downsample_points, 3), state_final[:, :3],
            state_final[:, 3:7]).reshape(
                -1, self.num_downsample_points * len(self.mesh_dict), 3)

        # print("synthesize_pc time:", time.time() - start_time)
        pc_list = vis_pc(transformed_batch_point.cpu().numpy()[0])
        o3d.visualization.draw_geometries([pc_list])

        # if self.colorize_pc:

        #     transformed_batch_point = torch.cat([
        #         transformed_batch_point,
        #         self.synthesize_points[..., 3:6].reshape(
        #             -1, self.num_downsample_points * len(self.mesh_dict), 3)
        #     ],
        #                                         dim=2)

        return transformed_batch_point

    def synthesize_pc_offline(self, data):

        if not self.init_mesh:

            self.load_mesh(None)

        obs_data = data["policy"]
        state_info_list = []
        nnew_mesh_dict = {}

        for link_name in obs_data.keys():
            new_link_name = link_name.replace(".*", "right")
            if new_link_name not in self.mesh_dict.keys():
                continue

            state_info = obs_data[link_name][:, :7]
            state_info_list.append(state_info.unsqueeze(0))
            nnew_mesh_dict[new_link_name] = self.mesh_dict[new_link_name]

        synthesize_points = torch.cat(list(
            nnew_mesh_dict.values()), dim=0).to("cuda").repeat_interleave(
                1, dim=0).unsqueeze(0).repeat_interleave(1, dim=0)

        state_final = torch.cat(state_info_list, dim=1).reshape(-1, 7)
        import pdb
        pdb.set_trace()

        transformed_batch_point = math_utils.transform_points(
            synthesize_points[..., :3].clone().reshape(
                -1, self.num_downsample_points, 3), state_final[:, :3],
            state_final[:, 3:7]).reshape(
                -1, self.num_downsample_points * len(nnew_mesh_dict), 3)
        pc_list = vis_pc(transformed_batch_point.cpu().numpy()[0])
        o3d.visualization.draw_geometries([pc_list])
