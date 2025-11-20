from pathlib import Path
import yaml

from dex_retargeting.retargeting_config import RetargetingConfig

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

import torch
import numpy as np

from torch_cluster import fps
import isaaclab.utils.math as math_utils
import trimesh
import copy

link_name_mapping = {
    "panda_link0": "panda_link0.obj",
    "panda_link1": "panda_link1.obj",
    "panda_link2": "panda_link2.obj",
    "panda_link3": "panda_link3.obj",
    "panda_link4": "panda_link4.obj",
    "panda_link5": "panda_link5.obj",
    "panda_link6": "panda_link6.obj",
    "panda_link7": "panda_link7.obj",
    "palm_lower": "palm_lower.glb",
    "mcp_joint": "mcp_joint.glb",
    "pip": "pip.glb",
    "dip": "dip.glb",
    "fingertip": "fingertip.glb",
    "mcp_joint_2": "mcp_joint.glb",
    "dip_2": "dip.glb",
    "fingertip_2": "fingertip.glb",
    "mcp_joint_3": "mcp_joint.glb",
    "pip_3": "pip.glb",
    "dip_3": "dip.glb",
    "fingertip_3": "fingertip.glb",
    "thumb_temp_base": "pip.glb",
    "thumb_pip": "thumb_pip.glb",
    "thumb_dip": "thumb_dip.glb",
    "thumb_fingertip": "thumb_fingertip.glb",
    "pip_2": "pip.glb",
}


class SynthesizeRealRobotPC:

    def __init__(
        self,
        mesh_dir: str,
        target_link_name: list,
    ):

        self.mesh_dir = mesh_dir
        self.target_link_name = target_link_name
        self.load_mesh()

        #     self.prepare_color_list(env, link_name)

    def load_mesh(self, ):

        self.mesh_dict = {}
        for link_name in self.target_link_name:

            # if "pip" in link_name:

            #     mesh = trimesh.load(self.mesh_dir + f"/pip.glb")
            # elif "dip" in link_name:

            #     mesh = trimesh.load(self.mesh_dir + f"/dip.glb")
            # elif "fingertip" in link_name:

            #     mesh = trimesh.load(self.mesh_dir + f"/fingertip.glb")

            # else:
            #     try:
            #         mesh = trimesh.load(self.mesh_dir + f"/{link_name}.obj")
            #     except:
            #         mesh = trimesh.load(self.mesh_dir + f"/{link_name}.glb")
            mesh = trimesh.load(
                self.mesh_dir + f"/{link_name_mapping[link_name]}", )

            vertices = trimesh.util.concatenate(mesh).vertices
            self.mesh_dict[link_name] = torch.as_tensor(vertices).to(
                torch.float32)

    def synthesize_pc(self, all_link_pose):
        all_link_pose = torch.as_tensor(all_link_pose).float()
        bodies_mesh = []

        for index, link_name in enumerate(list(self.mesh_dict.keys())):
            vertices = self.mesh_dict[link_name]

            quat = math_utils.quat_from_matrix(
                all_link_pose[index, :3, :3]).to(torch.float32)
            translate = all_link_pose[index, :3, 3].to(torch.float32)

            transformed_vertices = math_utils.transform_points(
                vertices.unsqueeze(0),
                translate.unsqueeze(0),
                quat.unsqueeze(0),
            )
            bodies_mesh.append(transformed_vertices)

        return torch.cat(bodies_mesh, dim=1)

    def synthesize_whole_robot_pc(self, env, downsample_points):

        if self.mesh_dict is None:

            self.env = env
            self.robot = self.env.scene[f"{self.hand_side}_hand"]
            self.init_settings()

        robot_vertices, robot_faces = self.synthesize_pc()

        # show_robot_mesh(
        #     robot_vertices[-5].cpu().numpy(),
        #     robot_faces[0].cpu().numpy(),
        # )

        downsampled_points = math_utils.fps_points(
            robot_vertices,
            downsample_points,
        )
        # cam_o3d = vis_pc(downsampled_points[-5].cpu().numpy())
        # visualize_pcd([cam_o3d])
        return downsampled_points
