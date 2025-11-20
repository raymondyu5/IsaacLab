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
import argparse
import os
import pymeshlab

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

ROBOT_MESH = "source/robot/franka/raw_mesh"

finger_dict = {
    "thumb": ["thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"],
    "index": ["mcp_joint", "pip", "dip", "fingertip"],
    'middle': ["mcp_joint_2", "pip_2", "dip_2", "fingertip_2"],
    "ring": ["mcp_joint_3", "pip_3", "dip_3", "fingertip_3"],
}

import torch
from torch_cluster import fps
import isaaclab.utils.math as math_utils


def downsample_points(points, ratio=0.05):
    """
    points: (N, 3) numpy array
    ratio: fraction of points to keep (e.g. 0.05 = keep 5%)
    """
    pts = torch.as_tensor(points, dtype=torch.float32).cuda()
    idx = fps(pts, batch=None, ratio=ratio)  # indices of kept points
    return pts[idx].cpu().numpy()


def make_circle(center, radius=0.02, color=(1.0, 0.0, 0.0, 1.0)):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)

    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,
                                                  alphaMode="BLEND")
    return pyrender.Mesh.from_trimesh(sphere, material=material)


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
            num_faces = mesh.faces.shape[0]

            mesh = pymeshlab.Mesh(vertex_matrix=np.array(mesh.vertices),
                                  face_matrix=np.array(mesh.faces,
                                                       dtype=np.int32))

            # Create a MeshSet and add the mesh to it
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh, 'my_mesh')
            ms.meshing_remove_duplicate_faces()
            ms.meshing_repair_non_manifold_edges()

            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(
                num_faces * 0.5),
                                                        preservenormal=True,
                                                        qualitythr=0.3)
            current_mesh = ms.current_mesh()
            vertices = current_mesh.vertex_matrix()
            faces = current_mesh.face_matrix()

            self.mesh_dict[link_name] = trimesh.Trimesh(vertices, faces)

    def render_pose(
        self,
        link_pose,
        link_names,
    ):

        scene = pyrender.Scene()
        finger_mesh_dict = defaultdict(list)
        finger_mesh_dict["thumb"] = []
        finger_mesh_dict["index"] = []
        finger_mesh_dict["middle"] = []
        finger_mesh_dict["ring"] = []
        finger_vertices_count_dict = defaultdict(list)
        finger_vertices_count_dict["thumb"] = 0
        finger_vertices_count_dict["index"] = 0
        finger_vertices_count_dict["middle"] = 0
        finger_vertices_count_dict["ring"] = 0

        finger_faces_dict = copy.deepcopy(finger_mesh_dict)
        for i, link_name in enumerate(link_names):
            if link_name not in self.target_link_name:
                continue

            mesh = copy.deepcopy(self.mesh_dict[link_name])
            if isinstance(mesh, trimesh.Scene):
                # Convert to a single mesh
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

            vertices = torch.as_tensor(
                mesh.vertices).cuda().unsqueeze(0).repeat_interleave(
                    link_pose.shape[0], 0)
            quat = torch.as_tensor(link_pose[:, i, 3:7]).cuda()
            trasnlate = torch.as_tensor(link_pose[:, i, :3]).cuda()
            trasformed_verties = math_utils.transform_points(
                vertices.to(torch.float32), trasnlate.to(torch.float32),
                quat.to(torch.float32))
            transformed_faces = torch.as_tensor(mesh.faces).cuda()

            if link_name in finger_dict["thumb"]:
                finger_mesh_dict["thumb"].append(trasformed_verties)
                finger_faces_dict["thumb"].append(
                    transformed_faces + finger_vertices_count_dict["thumb"])
                finger_vertices_count_dict[
                    "thumb"] += trasformed_verties.shape[-2]
            if link_name in finger_dict["index"]:
                finger_mesh_dict["index"].append(trasformed_verties)
                finger_faces_dict["index"].append(
                    transformed_faces + finger_vertices_count_dict["index"])
                finger_vertices_count_dict[
                    "index"] += trasformed_verties.shape[-2]
            if link_name in finger_dict["middle"]:
                finger_mesh_dict["middle"].append(trasformed_verties)
                finger_faces_dict["middle"].append(
                    transformed_faces + finger_vertices_count_dict["middle"])
                finger_vertices_count_dict[
                    "middle"] += trasformed_verties.shape[-2]
            if link_name in finger_dict["ring"]:
                finger_mesh_dict["ring"].append(trasformed_verties)
                finger_faces_dict["ring"].append(
                    transformed_faces + finger_vertices_count_dict["ring"])
                finger_vertices_count_dict["ring"] += trasformed_verties.shape[
                    -2]

        dist = self.compute_penetration(finger_mesh_dict, finger_faces_dict,
                                        "index", "middle")

        color = np.array([1.0, 0.0, 0.0])

        # print(dist["penetration_depths"].sum())
        # downsampled = downsample_points(penetrating_vertices, ratio=0.005)

        # if len(downsampled) > 0:
        #     center = downsampled.mean(axis=0)
        #     circle = make_circle(center,
        #                          radius=0.02,
        #                          color=(1.0, 0.0, 0.0, 1.0))
        #     scene.add(circle)

        # pyrender.Viewer(scene, use_raymond_lighting=True)
        return abs(dist["failure_ratio"])

    def compute_penetration(self, finger_mesh_dict, finger_faces_dict,
                            finger_name01, finger_name02):

        # Batched verts: (B, V, 3), faces: (F, 3)
        verts1 = torch.cat(finger_mesh_dict[finger_name01],
                           dim=-2)  # already (B, V1, 3)
        faces1 = torch.cat(finger_faces_dict[finger_name01], dim=0)
        verts2 = torch.cat(finger_mesh_dict[finger_name02],
                           dim=-2)  # (B, V2, 3)
        faces2 = torch.cat(finger_faces_dict[finger_name02], dim=0)

        # finger01_mesh = trimesh.Trimesh(verts1[-1].cpu().numpy(),
        #                                 faces1.cpu().numpy())
        # finger02_mesh = trimesh.Trimesh(verts2[-1].cpu().numpy(),
        #                                 faces2.cpu().numpy())
        # all_contacts = trimesh.util.concatenate([finger01_mesh, finger02_mesh])
        # all_contacts.show()

        # Sanity check
        assert verts1.shape[-2] == faces1.max() + 1
        assert verts2.shape[-2] == faces2.max() + 1

        verts1, verts2 = verts1.float(), verts2.float()
        faces1, faces2 = faces1.long(), faces2.long()

        # Face vertices for mesh2 (batched)
        face_vertices2 = km.index_vertices_by_faces(verts2,
                                                    faces2)  # (B, F2, 3, 3)

        # Distances from verts1 → mesh2
        dist12, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
            verts1, face_vertices2)  # (B, V1)
        dist12 = torch.sqrt(dist12)  # (B, V1)

        # Signs wrt mesh2
        sign12 = kal.ops.mesh.check_sign(verts1, faces2, verts2)  # (B, V1)

        sdf12 = dist12 * torch.where(sign12, -1.0, 1.0)

        penetration = sdf12[sdf12 < 0]
        max_pentration = torch.min(sdf12, dim=1).values
        failure = (max_pentration < -0.02).float()
        print(torch.min(max_pentration))
        metrics = {
            "max_depth":
            -penetration.min().item() if penetration.numel() > 0 else 0.0,
            "mean_depth":
            -penetration.mean().item() if penetration.numel() > 0 else 0.0,
            "penetrating_ratio":
            penetration.numel() / (verts1.shape[0] * verts1.shape[1]),
            "all_sdf":
            sdf12,
            "failure":
            failure,
            "failure_ratio":
            failure.mean().item()
        }
        return metrics

    def render_image(self, scene):

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2) / 2

        camera_pose = np.array([
            [0.0, -s, s, 0.15],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, s, 0.30],
            [0.0, 0.0, 0.0, 1.0],
        ])

        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)
        r = pyrender.OffscreenRenderer(2048, 2048)
        color, depth = r.render(scene)

        # Save image
        plt.imsave("logs/rendered_image.png", color)

        # Or preview inside notebook/script
        plt.imshow(color)
        plt.axis("off")
        plt.show()

    def synthesize_pc(
        self,
        link_pose,
        link_names,
        save_dir="logs/",
    ):
        all_link_pose = torch.as_tensor(link_pose).float()
        penetration_list = []

        for demo_idx in range(all_link_pose.shape[0]):

            link_sorted_pose = []
            target_link_name = []
            for _, link_name in enumerate(list(self.mesh_dict.keys())):

                if link_name not in link_names:
                    continue
                target_link_name.append(link_name)

                index = link_names.index(link_name)

                trajectories_pose = all_link_pose[demo_idx, :, index]
                link_sorted_pose.append(trajectories_pose[None])

            link_sorted_pose = np.concatenate(link_sorted_pose,
                                              axis=0).transpose(1, 0, 2)
            penetration_value = 0.0
            penetration_list.append([])

            penetration_value += self.render_pose(
                link_sorted_pose,
                target_link_name,
            )
            print(
                f"demo_idx: {demo_idx}, penetration_value: {penetration_value}"
            )
            penetration_list[-1].append(penetration_value)

            np.savez(
                f"{save_dir}/penetration.npz",
                penetration=np.array(penetration_list),
            )


target_link_name = [
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
if __name__ == "__main__":
    mesh_dir = "source/assets/robot/leap_hand_v2/glb_mesh/"

    synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state_dir",
        type=str,
        default="logs/trash/sdf/episode_0.npz"
        # default=
        # "logs/data_0705/retarget_visionpro_data/rl_data/demo_data/ours/sdf/bunny/episode_0.npz",
    )
    args = parser.parse_args()
    state_dir = args.state_dir
    # load state dict

    env_info = torch.load(state_dir)

    save_dir = "/".join(state_dir.split("/")[:-2])
    os.makedirs(save_dir, exist_ok=True)

    obs_info = env_info["obs"]
    link_pose = []

    for obs in obs_info:
        link_poses = obs["right_hand_link_pose"].cpu().numpy()

        link_pose.append(link_poses[None])
        link_names = obs["right_body_names"]

    link_pose = np.concatenate(link_pose, axis=0).transpose(1, 0, 2, 3)

    synthesize_pc.synthesize_pc(link_pose, link_names[0], save_dir=save_dir)
