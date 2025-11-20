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
import open3d as o3d
import pyrender
from scipy.spatial.transform import Rotation as R
import kaolin as kal
from collections import defaultdict
import kaolin.ops.mesh as km
import pytorch_volumetric as pv

import isaaclab.utils.math as math_utils
from tools.visualization_utils import vis_pc, visualize_pcd

link_name_mapping = {
    "panda_link0": "panda_link0",
    "panda_link1": "panda_link1",
    "panda_link2": "panda_link2",
    "panda_link3": "panda_link3",
    "panda_link4": "panda_link4",
    "panda_link5": "panda_link5",
    "panda_link6": "panda_link6",
    "panda_link7": "panda_link7",
    "palm_lower": "palm_lower",
    "mcp_joint": "mcp_joint",
    "pip": "pip",
    "dip": "dip",
    "fingertip": "fingertip",
    "mcp_joint_2": "mcp_joint",
    "dip_2": "dip",
    "fingertip_2": "fingertip",
    "mcp_joint_3": "mcp_joint",
    "pip_3": "pip",
    "dip_3": "dip",
    "fingertip_3": "fingertip",
    "thumb_temp_base": "pip",
    "thumb_pip": "thumb_pip",
    "thumb_dip": "thumb_dip",
    "thumb_fingertip": "thumb_fingertip",
    "pip_2": "pip",
    "thumb_right_temp_base": "thumb_right_temp_base"
}

ROBOT_MESH = "/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/franka/raw_mesh"

finger_dict = {
    "thumb": ["thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"],
    "index": ["mcp_joint", "pip", "dip", "fingertip"],
    'middle': ["mcp_joint_2", "pip_2", "dip_2", "fingertip_2"],
    "ring": ["mcp_joint_3", "pip_3", "dip_3", "fingertip_3"],
}


class SDFCompuation:

    def __init__(self, ):

        self.mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/leap_hand_v2/glb_mesh/"
        self.target_link_name = [
            # "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",
            # "panda_link5", "panda_link6", "panda_link7",
            "palm_lower",
            "mcp_joint",
            "pip",
            "dip",
            "fingertip",
            "mcp_joint_2",
            "dip_2",
            "fingertip_2",
            "mcp_joint_3",
            "pip_3",
            "dip_3",
            "fingertip_3",
            "thumb_temp_base",
            "thumb_pip",
            "thumb_dip",
            "thumb_fingertip",
            "pip_2",
            "thumb_right_temp_base"
        ]
        self.load_mesh()
        self.hand_side = "right"

        #     self.prepare_color_list(env, link_name)

    def load_mesh(self, ):

        self.mesh_dict = {}
        for link_name in self.target_link_name:
            if "panda" in link_name:
                mesh = trimesh.load(
                    ROBOT_MESH + f"/{link_name_mapping[link_name]}.obj", )
            else:

                mesh = trimesh.load(
                    self.mesh_dir + f"/{link_name_mapping[link_name]}.glb", )
            # mesh.show()
            if isinstance(mesh, trimesh.Scene):
                # Convert to a single mesh
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

            self.mesh_dict[link_name] = mesh

    def render_pose(
        self,
        link_pose,
        link_names,
    ):

        finger_mesh_dict = defaultdict(list)
        finger_mesh_dict["thumb"] = []
        finger_mesh_dict["index"] = []
        finger_mesh_dict["middle"] = []
        finger_mesh_dict["ring"] = []
        for i, link_name in enumerate(link_names):
            if link_name not in self.target_link_name:
                continue

            mesh = copy.deepcopy(self.mesh_dict[link_name])
            if isinstance(mesh, trimesh.Scene):
                # Convert to a single mesh
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            import pdb
            pdb.set_trace()
            # pdb.set_trace()
            math_utils.transform_points(mesh.vertices, link_pose[:, i])
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(mesh))

            # if link_name in finger_dict["thumb"]:
            #     finger_mesh_dict["thumb"].append(mesh)
            if link_name in finger_dict["index"]:
                finger_mesh_dict["index"].append(mesh)
            if link_name in finger_dict["middle"]:
                finger_mesh_dict["middle"].append(mesh)
            # if link_name in finger_dict["ring"]:
            #     finger_mesh_dict["ring"].append(mesh)

        dist = self.compute_penetration(finger_mesh_dict, "index", "middle")

        color = np.array([1.0, 0.0, 0.0])
        penetrating_vertices = dist["penetrating_vertices"]

        print(dist["penetration_depths"].sum())

    def compute_penetration(self, env, finger_name01, finger_name02):

        body_state_w = env.scene[
            f"{self.hand_side}_hand"]._data.body_state_w.clone()
        body_names = env.scene[f"{self.hand_side}_hand"].body_names
        body_state_w[..., :3] -= env.scene.env_origins[:, None, :3]

        finger01_mesh = []
        finger02_mesh = []
        trimesh_faces = []
        face_offset = 0

        for name in body_names:
            if name in finger_dict[finger_name01]:
                link_pose01 = body_state_w[:, body_names.index(name)]
                mesh01 = copy.deepcopy(self.mesh_dict[name])
                vertices = torch.from_numpy(
                    mesh01.vertices).float().unsqueeze(0).repeat(
                        env.num_envs, 1, 1).to(body_state_w.device)
                transformed_verts = math_utils.transform_points(
                    vertices, link_pose01[:, :3], link_pose01[:, 3:7])
                finger01_mesh.append((transformed_verts))
                trimesh_faces.append(
                    torch.from_numpy(mesh01.faces).long() + face_offset)
                face_offset += mesh01.vertices.shape[0]

            if name in finger_dict[finger_name02]:
                link_pose02 = body_state_w[:, body_names.index(name)]
                mesh02 = copy.deepcopy(self.mesh_dict[name])
                vertices = torch.from_numpy(
                    mesh02.vertices).float().unsqueeze(0).repeat(
                        env.num_envs, 1, 1).to(body_state_w.device)
                transformed_verts = math_utils.transform_points(
                    vertices, link_pose02[:, :3], link_pose02[:, 3:7])
                finger02_mesh.append((transformed_verts))

        finger02_mesh = torch.cat(finger02_mesh, dim=1)
        finger01_mesh = torch.cat(finger01_mesh, dim=1)
        # whole_mesh = torch.cat([finger01_mesh, finger02_mesh], dim=1)
        # pcd_list = visualize_pcd([vis_pc(whole_mesh[0].cpu().numpy())])

        trimesh_faces = torch.cat(trimesh_faces, dim=0).to(body_state_w.device)
        return self.inference_sdf(finger01_mesh, trimesh_faces, finger02_mesh,
                                  trimesh_faces)
        # pcd_list = visualize_pcd([vis_pc(finger01_mesh

    def inference_sdf(self, verts1, faces1, verts2, faces2, symmetric=True):

        B = verts1.shape[0]

        # === verts1 → mesh2 ===
        face_vertices2 = km.index_vertices_by_faces(verts2[0][None],
                                                    faces2)  # (1, F2, 3, 3)
        face_vertices2 = face_vertices2.repeat(B, 1, 1, 1)  # (B, F2, 3, 3)

        dist12, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
            verts1, face_vertices2)

        dist12 = torch.sqrt(dist12)  # (B, N1)

        # Flatten verts1 for sign test
        sign12 = kal.ops.mesh.check_sign(verts2, faces2, verts1)  # (B*N1,)
        sign12 = sign12.view(B, -1)  # (B, N1)

        sdf12 = dist12 * torch.where(sign12, -1.0, 1.0)
        penetration_depths = sdf12[sdf12 < 0]
        if len(penetration_depths) > 0:
            import pdb
            pdb.set_trace()


if __name__ == "__main__":
    mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/leap_hand_v2/glb_mesh/"

    synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)
    state_dir = "logs/data_0705/retarget_visionpro_data/rl_data/relative_pose/state"

    env_info = torch.load(f"{state_dir}/episode_0.npz")

    obs_info = env_info["obs"]
    link_pose = []

    for obs in obs_info:
        link_poses = obs["right_hand_link_pose"].cpu().numpy()

        link_pose.append(link_poses[None])
        link_names = obs["right_body_names"]

    link_pose = np.concatenate(link_pose, axis=0)  #.transpose(1, 0, 2, 3)

    synthesize_pc.synthesize_pc(
        link_pose,
        link_names[0],
    )
