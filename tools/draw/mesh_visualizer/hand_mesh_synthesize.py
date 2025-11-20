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
            # scene = pyrender.Scene()
            # scene.add(pyrender.Mesh.from_trimesh([mesh]))
            # pyrender.Viewer(scene, use_raymond_lighting=True)

            # # vertices = trimesh.util.concatenate(mesh).vertices
            self.mesh_dict[link_name] = mesh
            # # o3d.visualization.draw_geometries([mesh])

    def render_pose(self,
                    link_pose,
                    link_names,
                    object_pose=None,
                    render_object=True):
        all_mesh = []
        scene = pyrender.Scene()
        if object_pose is not None:
            object_mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/plush"
            target_object_name = "bunny"
            self.object_mesh = trimesh.load(
                f"{object_mesh_dir}/{target_object_name}/{target_object_name}.glb"
            )

            rotation = trimesh.transformations.rotation_matrix(
                np.radians(-90),  # 90 degrees
                [0, 0, 1]  # y-axis
            )
            self.object_mesh.apply_transform(rotation)

            rotation = trimesh.transformations.rotation_matrix(
                np.radians(-180),  # 90 degrees
                [0, 1, 0]  # y-axis
            )
            self.object_mesh.apply_transform(rotation)

            rotation = trimesh.transformations.rotation_matrix(
                np.radians(90),  # 90 degrees
                [0, 1, 0]  # y-axis
            )
            self.object_mesh.apply_transform(rotation)
            if isinstance(self.object_mesh, trimesh.Scene):
                # Convert to a single mesh
                self.object_mesh = trimesh.util.concatenate(self.object_mesh)

        base_pose = link_pose[0]
        r = R.from_quat(base_pose[3:7][[1, 2, 3, 0]])
        base_transform = np.eye(4)
        base_transform[:3, 3] = base_pose[:3]
        base_transform[:3, :3] = r.as_matrix()

        # Convert to rotation matrix
        base_transform[:3, :3] = r.as_matrix()

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

            # rotation = trimesh.transformations.rotation_matrix(
            #     np.radians(180),
            #     [0, 0, 1]  # y-axis
            # )

            all_mesh.append(mesh)

            scene.add(pyrender.Mesh.from_trimesh(mesh))

        if render_object:
            object_transform = np.eye(4)
            object_mesh = copy.deepcopy(self.object_mesh)
            object_transform[:3, 3] = object_pose[:3]
            r = R.from_quat(object_pose[3:][[1, 2, 3, 0]])
            object_transform[:3, :3] = r.as_matrix()
            if isinstance(object_mesh, trimesh.Scene):
                # Convert to a single mesh
                object_mesh = trimesh.util.concatenate(
                    tuple(object_mesh.geometry.values()))

            object_mesh.apply_transform(
                np.linalg.inv(base_transform) @ object_transform)
            rotation = trimesh.transformations.rotation_matrix(
                np.radians(-45),
                [0, 1, 0]  # y-axis
            )
            object_mesh.apply_transform(rotation)

            scene.add(pyrender.Mesh.from_trimesh(object_mesh))

        # axis = trimesh.creation.axis(origin_size=0.1, axis_length=0.2)
        # scene.add(pyrender.Mesh.from_trimesh(axis, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        return self.render_image(scene)

    def render_image(self, scene):

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5, aspectRatio=1.0)
        s = np.sqrt(2) / 2

        camera_pose = np.array([
            [0.0, -s, s, 0.25],
            [1.0, 0.0, 0.0, 0.05],
            [0.0, s, s, 0.40],
            [0.0, 0.0, 0.0, 1.0],
        ])

        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)
        r = pyrender.OffscreenRenderer(2048, 2048)
        color, depth = r.render(scene)

        # # # # # # Save image
        # plt.imsave("logs/rendered_image.png", color)

        # # Or preview inside notebook/script
        # plt.imshow(color)
        # plt.axis("off")
        # plt.show()
        return color

    def synthesize_pc(self,
                      link_pose,
                      link_names,
                      target_object_name=None,
                      object_mesh_dir=None,
                      object_pose=None,
                      render_object=False):
        all_link_pose = torch.as_tensor(link_pose).float()
        if target_object_name is not None:
            self.object_mesh = trimesh.load(
                f"{object_mesh_dir}/{target_object_name}/{target_object_name}.glb"
            )
            raw_mesh = copy.deepcopy(self.object_mesh)
            R = trimesh.transformations.rotation_matrix(
                np.radians(90),  # 90 degrees
                [1, 0, 0]  # y-axis
            )
            self.object_mesh.apply_transform(R)

            R = trimesh.transformations.rotation_matrix(
                np.radians(-180),  # 90 degrees
                [0, 0, 1]  # y-axis
            )
            self.object_mesh.apply_transform(R)
            if isinstance(self.object_mesh, trimesh.Scene):
                # Convert to a single mesh
                self.object_mesh = trimesh.util.concatenate(self.object_mesh)

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
            id = 20

            if object_pose is not None:

                self.render_pose(link_sorted_pose[id], target_link_name,
                                 object_pose[demo_idx][id], render_object)
            else:
                self.render_pose(link_sorted_pose[id], target_link_name, None,
                                 render_object)


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
    mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/leap_hand_v2/glb_mesh/"

    object_mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/plush"

    synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)
    state_dir = "logs/data_0705/retarget_visionpro_data/rl_data/relative_pose/state"
    object_name = np.loadtxt(f"{state_dir}/object_name_0.txt",
                             dtype=str).tolist()

    env_info = torch.load(f"{state_dir}/episode_0.npz")

    target_object_name = "bunny"
    render_object = False

    indices = [
        i for i, name in enumerate(object_name) if name == target_object_name
    ]

    obs_info = env_info["obs"]
    link_pose = []
    object_pose = []
    for obs in obs_info:
        link_poses = obs["right_hand_link_pose"].cpu().numpy()

        link_pose.append(link_poses[None])
        link_names = obs["right_body_names"]

        object_poses = obs["right_manipulated_object_pose"].cpu().numpy()
        object_pose.append(object_poses[None])

    link_pose = np.concatenate(link_pose, axis=0).transpose(1, 0, 2, 3)
    object_pose = np.concatenate(object_pose, axis=0).transpose(1, 0, 2)

    synthesize_pc.synthesize_pc(link_pose[indices],
                                link_names[0],
                                target_object_name,
                                object_mesh_dir,
                                object_pose[indices],
                                render_object=render_object)
