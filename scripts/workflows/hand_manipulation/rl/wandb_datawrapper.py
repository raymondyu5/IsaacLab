import wandb
import os
import trimesh
import torch
from torch_cluster import fps
import isaaclab.utils.math as math_utils

from tools.visualization_utils import vis_pc, visualize_pcd

import open3d as o3d


class WandbDataWrapper:

    def __init__(self, env, args_cli, env_config):
        self.env = env
        self.args_cli = args_cli
        self.env_config = env_config
        if self.args_cli.add_right_hand:
            self.right_hand = True
            self.hand_side = "right"
        if self.args_cli.add_left_hand:
            self.left_hand = True
            self.hand_side = "left"
        self.object_name = env_config["params"]["target_manipulated_object"]

        self.robot = self.env.scene[f"{self.hand_side}_hand"]

        self.robot_link_names = self.robot._data.body_names
        self.num_robot_links = len(self.robot_link_names)
        self.init_setting()

    def init_setting(self):

        arm_mesh_dir = self.env_config["params"]["arm_mesh_dir"]
        hand_mesh_dir = self.env_config["params"]["hand_mesh_dir"]
        object_mesh_dir = self.env_config["params"]["spawn_rigid_objects"][
            "object_mesh_dir"]

        self.mesh_dir = {}

        for index, link_name in enumerate(self.robot_link_names +
                                          [self.object_name]):

            if "palm" in link_name:
                mesh_name = f"{self.hand_side}_" + link_name

            else:
                parts = link_name.split("_")
                if parts[-1].isdigit():
                    mesh_name = "_".join(
                        parts[:-1])  # Drop the last numeric part
                else:
                    mesh_name = "_".join(parts)

            if "panda" in link_name:
                mesh_path = os.path.join(arm_mesh_dir, mesh_name + ".obj")

            elif link_name == self.object_name:

                mesh_path = os.path.join(object_mesh_dir, mesh_name + ".obj")
            else:
                mesh_path = os.path.join(hand_mesh_dir, mesh_name + ".obj")
            mesh = trimesh.load_mesh(mesh_path)
            mesh = trimesh.util.concatenate(mesh)

            vertices = torch.as_tensor(mesh.vertices).to(
                self.env.device).unsqueeze(0)

            vertices = math_utils.fps_points(vertices, downsample_points=512)
            self.mesh_dir[link_name] = vertices

        all_vertices = torch.cat(list(self.mesh_dir.values()), dim=0)
        reordered_indices = [
            list(self.mesh_dir.keys()).index(name)
            for name in self.robot_link_names + [self.object_name]
        ]

        self.all_vertices = all_vertices[reordered_indices].to(torch.float32)

    def render_pc(self):
        robot_body_state = self.robot._data.body_link_state_w[0,
                                                              ..., :7].clone()
        robot_body_state[:, :3] -= self.env.scene.env_origins[0]

        object_state = self.env.scene[
            self.object_name]._data.root_state_w[:, :7].clone()
        object_state[:, :3] -= self.env.scene.env_origins

        all_state = torch.cat([robot_body_state, object_state[0].unsqueeze(0)],
                              dim=0)

        trasformed_pc = math_utils.transform_points(self.all_vertices,
                                                    all_state[:, :3],
                                                    all_state[:, 3:7])

        # pc_list = vis_pc(trasformed_pc.reshape(-1, 3).cpu().numpy())
        # o3d.visualization.draw_geometries([pc_list])
        return trasformed_pc.reshape(-1, 3).cpu().numpy()
