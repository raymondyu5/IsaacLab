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
                num_faces * 0.05),
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

        base_pose = link_pose[0]
        r = R.from_quat(base_pose[3:7][[1, 2, 3, 0]])
        base_transform = np.eye(4)
        base_transform[:3, 3] = base_pose[:3]
        base_transform[:3, :3] = r.as_matrix()

        # Convert to rotation matrix
        base_transform[:3, :3] = r.as_matrix()

        scene = pyrender.Scene()
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

            # Create a Rotation object
            r = R.from_quat(link_pose[i][3:7][[1, 2, 3, 0]])
            transform = np.eye(4)
            transform[:3, 3] = link_pose[i][:3]

            # Convert to rotation matrix
            transform[:3, :3] = r.as_matrix()

            mesh.apply_transform(np.linalg.inv(base_transform) @ transform)
            # mesh.apply_transform(transform)

            rotation = trimesh.transformations.rotation_matrix(
                np.radians(-45),
                [0, 1, 0]  # y-axis
            )
            mesh.apply_transform(rotation)

            finger_mesh_dict[link_name] = mesh
            scene.add(pyrender.Mesh.from_trimesh(mesh))

            if link_name in finger_dict["thumb"]:
                finger_mesh_dict["thumb"].append(mesh)
            if link_name in finger_dict["index"]:
                finger_mesh_dict["index"].append(mesh)
            if link_name in finger_dict["middle"]:
                finger_mesh_dict["middle"].append(mesh)
            if link_name in finger_dict["ring"]:
                finger_mesh_dict["ring"].append(mesh)

        dist = self.compute_penetration(finger_mesh_dict, "index", "middle")

        color = np.array([1.0, 0.0, 0.0])
        penetrating_vertices = dist["penetrating_vertices"]

        # print(dist["penetration_depths"].sum())
        # downsampled = downsample_points(penetrating_vertices, ratio=0.005)

        # if len(downsampled) > 0:
        #     center = downsampled.mean(axis=0)
        #     circle = make_circle(center,
        #                          radius=0.02,
        #                          color=(1.0, 0.0, 0.0, 1.0))
        #     scene.add(circle)

        # pyrender.Viewer(scene, use_raymond_lighting=True)
        return abs(dist["penetration_depths"].sum()), 1 - int(dist["failure"])

    def compute_penetration(self, finger_mesh_dict, finger_name01,
                            finger_name02):
        # Load two meshes
        finger_mesh01 = trimesh.util.concatenate(
            finger_mesh_dict[finger_name01])
        finger_mesh02 = trimesh.util.concatenate(
            finger_mesh_dict[finger_name02])

        verts1 = torch.from_numpy(np.array(
            finger_mesh01.vertices)).float().cuda()
        faces1 = torch.from_numpy(np.array(finger_mesh01.faces)).long().cuda()
        verts2 = torch.from_numpy(np.array(
            finger_mesh02.vertices)).float().cuda()
        faces2 = torch.from_numpy(np.array(finger_mesh02.faces)).long().cuda()

        # Ensure correct types
        verts1, verts2 = verts1.float(), verts2.float()
        faces1, faces2 = faces1.long(), faces2.long()

        # Batchify
        verts1_b, verts2_b = verts1.unsqueeze(0), verts2.unsqueeze(0)
        faces2_b = faces2  #.unsqueeze(0)

        # Distances from verts1 to mesh2
        face_vertices2 = km.index_vertices_by_faces(verts2_b, faces2_b)
        dist12, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
            verts1_b, face_vertices2)
        dist12 = torch.sqrt(dist12.squeeze(0))  # (V1,)

        # Signs: inside/outside test
        sign12 = kal.ops.mesh.check_sign(verts2_b, faces2,
                                         verts1_b)  # (1, V1) bool
        sign12 = sign12.squeeze(0)

        # Build signed distance
        sdf12 = dist12 * torch.where(sign12, -1.0, 1.0)

        # Penetration stats
        penetration_depths = sdf12[sdf12 < 0]
        penetrating_vertices = verts1[sdf12 < 0].detach().cpu().numpy()
        # if len(penetrating_vertices) > 1:
        #     print(np.min(penetrating_vertices))
        if len(penetrating_vertices) > 30:

            failure = np.min(penetrating_vertices) < -0.05
        else:
            failure = False

        return {
            "sdf":
            sdf12,
            "penetration_depths":
            penetration_depths,
            "mean_penetration":
            penetration_depths.abs().mean().item()
            if penetration_depths.numel() > 0 else 0.0,
            "num_penetrating_vertices":
            penetration_depths.numel(),
            "penetrating_vertices":
            penetrating_vertices,
            "failure":
            failure
        }

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
        count = 0
        success_count = 0

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
            for id in range(link_sorted_pose.shape[0]):

                penetration_value2, success = self.render_pose(
                    link_sorted_pose[id],
                    target_link_name,
                )
                penetration_value += penetration_value2
                count += 1

                success_count += int(success)
            print(
                f"demo_idx: {demo_idx}, penetration_value: {penetration_value}",
                success_count / count)
            penetration_list[-1].append(penetration_value.cpu().numpy())

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
