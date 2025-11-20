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

# from torch_cluster import fps
# import isaaclab.utils.math as math_utils

# from tools.visualization_utils import vis_pc, visualize_pcd
import open3d as o3d
import trimesh
import os
import time
import pymeshlab
import shutil
import copy
import pickle
try:
    from isaaclab.managers import ObservationTermCfg as ObsTerm
except:
    print("ObservationTermCfg not found, using ObsTerm instead.")

COLOR_LIST = {
    "LINK": [0.0, 0.0, 1.0],
    "FINGER": [0.0, 0.0, 1.0],
    "PALM": [0.0, 0.0, 1.0],
    "RIGID_BODIES": ([[1.0, 0.27, 0.27]])
}

# The index to map the OpenXR hand joints to the hand joints used
# in Dex-retargeting.
_HAND_JOINTS_INDEX = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25
]

# The transformation matrices to convert hand pose to canonical view.
_OPERATOR2MANO_RIGHT = np.asarray([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])

_OPERATOR2MANO_LEFT = np.asarray([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])


def configure_sysnthesize_robot_pc(obs_policy,
                                   env_cfg,
                                   sythesize_object_pc=False):

    if env_cfg["params"]["add_right_hand"]:
        setattr(obs_policy, "right_hand_sythesizer", "right_hand_sythesizer")

        right_hand_sythesizer = SynthesizeRobotPC(None, env_cfg, "right")
        obs_policy.right_hand_sythesizer = ObsTerm(
            func=right_hand_sythesizer.synthesize_whole_robot_pc,
            params={"downsample_points": env_cfg["params"]["num_hand_pcd"]},
        )

        setattr(obs_policy, "right_arm_sythesizer", "right_arm_sythesizer")
        right_arm_sythesizer = SynthesizeRobotPC(None,
                                                 env_cfg,
                                                 "right",
                                                 mesh_dir="arm_mesh_dir")
        obs_policy.right_arm_sythesizer = ObsTerm(
            func=right_arm_sythesizer.synthesize_whole_robot_pc,
            params={"downsample_points": env_cfg["params"]["num_arm_pcd"]},
        )
        if sythesize_object_pc:
            setattr(obs_policy, "right_object_sythesizer",
                    "right_object_sythesizer")
            right_object_sythesizer = SynthesizeRobotPC(
                None, env_cfg, "right", mesh_dir="object_mesh_dir")
            obs_policy.right_object_sythesizer = ObsTerm(
                func=right_object_sythesizer.synthesize_whole_robot_pc,
                params={
                    "downsample_points": env_cfg["params"]["num_object_pcd"]
                },
            )

    if env_cfg["params"]["add_left_hand"]:
        setattr(obs_policy, "left_hand_sythesizer", "left_hand_sythesizer")
        left_hand_sythesizer = SynthesizeRobotPC(None, env_cfg, "left")
        obs_policy.right_hand_sythesizer = ObsTerm(
            func=left_hand_sythesizer.synthesize_whole_robot_pc,
            params={"downsample_points": env_cfg["params"]["num_hand_pcd"]},
        )

        setattr(obs_policy, "left_arm_sythesizer", "left_arm_sythesizer")
        left_arm_sythesizer = SynthesizeRobotPC(None,
                                                env_cfg,
                                                "right",
                                                mesh_dir="arm_mesh_dir")
        obs_policy.left_arm_sythesizer = ObsTerm(
            func=left_arm_sythesizer.synthesize_whole_robot_pc,
            params={"downsample_points": env_cfg["params"]["num_arm_pcd"]},
        )


class SynthesizeRobotPC:

    def __init__(
        self,
        env,
        env_cfg,
        hand_side,
        mesh_dir="hand_mesh_dir",
    ):

        self.spawn_mesh_list = []
        self.env = env
        self.env_cfg = env_cfg
        self.hand_side = hand_side
        self.mesh_dir = mesh_dir
        self.mesh_dict = None

        try:
            self.robot = self.env.scene[f"{self.hand_side}_hand"]
            self.init_settings()
        except:
            pass

    def init_settings(self, ):

        self.spawn_mesh_list = self.robot.body_names

        self.mesh_path = self.env_cfg["params"][self.mesh_dir]
        self.load_mesh()

    def save_data(self, source_path, target_path):
        # print(os.listdir(source_path))
        for filename in os.listdir(source_path):
            src_file = os.path.join(source_path, filename)
            dst_file = os.path.join(target_path, filename)
            if os.path.isfile(src_file):  # optionally skip subdirectories
                shutil.copy(src_file, dst_file)

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

            mesh_path = os.path.join(self.mesh_path, mesh_name + ".obj")

            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path)
            mesh = trimesh.util.concatenate(mesh)
            vertices = torch.from_numpy(
                mesh.vertices).to(device).to(dtype=torch.float32)
            faces = torch.from_numpy(mesh.faces).to(device)

            self.mesh_dict[link_name] = [index, vertices, faces]

    def synthesize_pc(self):

        body_state_w = self.robot._data.body_state_w
        bodies_mesh = []
        bodies_faces = []
        num_vertices = 0

        for link_name in self.mesh_dict.keys():
            index, vertices, faces = copy.deepcopy(self.mesh_dict[link_name])

            link_state_w = body_state_w[:, index].clone()
            link_state_w[:, :3] -= self.env.scene.env_origins

            transformed_vertices = math_utils.transform_points(
                vertices.unsqueeze(0), link_state_w[:, :3], link_state_w[:,
                                                                         3:7])
            bodies_mesh.append(transformed_vertices)
            bodies_faces.append(faces.unsqueeze(0) + num_vertices)
            num_vertices += vertices.shape[0]
        # cam_o3d = vis_pc(torch.cat(bodies_mesh, dim=1)[0].cpu().numpy())
        # visualize_pcd([cam_o3d])

        return torch.cat(bodies_mesh, dim=1), torch.cat(bodies_faces, dim=1)

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

        return downsampled_points


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def display_hand(info, ax=None, alpha=0.2, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints, mano_faces = info["hand_info"]['verts'], info["hand_info"][
        'joints'], info["hand_info"]['faces']
    if "color" in info["hand_info"]:
        hand_color = np.asarray(info["hand_info"]['color'])
        valid_mask = ~(np.all(hand_color == [0, 0, 0], axis=1))
        filtered_verts = verts[valid_mask]
        filtered_colors = hand_color[valid_mask]

        ax.scatter(filtered_verts[:, 0],
                   filtered_verts[:, 1],
                   filtered_verts[:, 2],
                   color=filtered_colors,
                   s=30)

    if "obj_info" in info:
        obj_verts = info["obj_info"]['verts']
        obj_faces = info["obj_info"]['faces']
        mano_faces = torch.cat(
            (mano_faces, obj_faces + len(info["hand_info"]['verts'])), dim=0)
        verts = torch.cat((verts, obj_verts), dim=0)

    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.scatter(joints[:, 0],
               joints[:, 1],
               joints[:, 2],
               color=(0.5, 0.0, 0.5),
               s=60)
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()


def get_hand_joint_names(object, hand_side):
    robot = object.env.scene[f"{hand_side}_hand"]
    robot_joint_names = []

    for action_name in object.env.action_manager._terms.keys():
        if hand_side in action_name:
            control_joint_names = object.env.action_manager._terms[
                action_name].cfg.joint_names
            robot_joint_names += robot.find_joints(control_joint_names)[1]

    return robot_joint_names


def init_leap_hand_retarget(
    object,
    kinematics_path="source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_leap.yml",
    add_left_hand=False,
    add_right_hand=False,
    retarget_type=None,
):
    kinematics_path = Path(kinematics_path)
    with kinematics_path.open("r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        if add_left_hand:
            cfg_dict = yaml_config["left"]
            # object.hand_scaling = cfg_dict["retargeting"]["scaling_factor"]
        if add_right_hand:
            cfg_dict = yaml_config["right"]
            # object.hand_scaling = cfg_dict["retargeting"]["scaling_factor"]

    if (not add_left_hand and add_right_hand) or (add_left_hand
                                                  and not add_right_hand):

        if retarget_type is None:
            retargeting_config = RetargetingConfig.from_dict(
                cfg_dict["retargeting"])
            retargeting = retargeting_config.build()
            object.retargeting = [retargeting]

        elif retarget_type == "vectorada":
            from dex_retargeting.seq_retarget import SeqRetargeting
            # Temporarily set type to "vector" for base class parsing
            cfg_dict["retargeting"]["type"] = "vector"
            base_config = RetargetingConfig.from_dict(cfg_dict["retargeting"])
            cfg_dict["retargeting"]["type"] = "vectorada"  # Restore original

            base_tool = base_config.build()
            base_optimizer = base_tool.optimizer

            # Construct VectorAda optimizer
            vector_ada_optimizer = VectorAdaOptimizer(
                robot=base_optimizer.robot,
                target_joint_names=base_optimizer.target_joint_names,
                target_origin_link_names=base_optimizer.origin_link_names,
                target_task_link_names=base_optimizer.task_link_names,
                target_link_human_indices=base_optimizer.
                target_link_human_indices,
                scaling=base_config.scaling_factor,
                norm_delta=base_config.normal_delta,
                huber_delta=base_config.huber_delta,
                num_constrain=cfg_dict.get("num_constrain", 4),
            )

            # Wrap in sequential retargeting container
            retargeting = SeqRetargeting(
                vector_ada_optimizer,
                has_joint_limits=base_tool.has_joint_limits,
                lp_filter=base_tool.filter,
            )
            object.retargeting = [retargeting]

        return retargeting
    else:
        left_retargeting_config = RetargetingConfig.from_dict(
            yaml_config["left"]["retargeting"])
        right_retargeting_config = RetargetingConfig.from_dict(
            yaml_config["right"]["retargeting"])

        left_retargeting = left_retargeting_config.build()
        right_retargeting = right_retargeting_config.build()

        object.retargeting = [left_retargeting, right_retargeting]

        return left_retargeting, right_retargeting


def show_robot_mesh(vertices, faces, tip_vertices=None, sphere_radius=0.003):
    # Convert to Open3D format
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))

    # Optional: Add normals and color
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # light gray

    # Create red spheres at tip vertices
    tip_spheres = []
    if tip_vertices is not None:

        for tip in tip_vertices:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=sphere_radius)
            sphere.translate(tip)  # move to tip location
            sphere.paint_uniform_color([1, 0, 0])  # red
            tip_spheres.append(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1,  # size of the XYZ axes (change it if needed)
        origin=[0, 0, 0]  # center at (0,0,0) or move it somewhere else
    )
    # Visualize mesh and spheres
    o3d.visualization.draw_geometries([mesh] + tip_spheres)


# recon 45-dim pose from 9-dim pca coordinates
def reconstruct_mano_pose45(eigengrasp_vectors, D_mean, D_std, coordinates):

    standardized_pose = torch.matmul(coordinates, eigengrasp_vectors)
    original_pose = standardized_pose * D_std + D_mean

    return original_pose


import numpy as np
import torch
from typing import List
from dex_retargeting.optimizer import Optimizer
from dex_retargeting.robot_wrapper import RobotWrapper


class VectorAdaOptimizer(Optimizer):
    retargeting_type = "VECTORADA"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
        scaling: float = 1.0,
        num_constrain: int = 4,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)

        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names

        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta,
                                                reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling
        self.num_constrain = num_constrain

        # Unique list of all links involved in optimization
        self.computed_link_names = list(
            set(self.origin_link_names + self.task_link_names))
        self.origin_link_indices = torch.tensor([
            self.computed_link_names.index(name)
            for name in self.origin_link_names
        ])
        self.task_link_indices = torch.tensor([
            self.computed_link_names.index(name)
            for name in self.task_link_names
        ])
        self.computed_link_indices = self.get_link_indices(
            self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # Norm thresholds (empirical)
        self.norm_min = np.array([0.0015, 0.0072, 0.0166,
                                  0.0123])[:num_constrain]
        self.norm_max = np.array([0.1621, 0.1963, 0.2094,
                                  0.2239])[:num_constrain]

    def get_objective_function(self, target_vector: np.ndarray,
                               fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        """
        Build the NLopt-compatible objective function.
        """
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        # Compute adaptive weights for final touch joints
        touch_weights = np.ones(len(target_vector)) * 1.2
        touch_weights[-self.num_constrain:] = self._compute_touch_degree(
            target_vector[-self.num_constrain:])
        touch_weights = torch.as_tensor(touch_weights, dtype=torch.float32)
        touch_weights.requires_grad_(False)

        target_vector_tensor = torch.as_tensor(target_vector * self.scaling,
                                               dtype=torch.float32)
        target_vector_tensor.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Prevent scope shadowing
            qpos_to_use = qpos
            if self.adaptor is not None:
                qpos_to_use = self.adaptor.forward_qpos(qpos)

            self.robot.compute_forward_kinematics(qpos_to_use)
            target_poses = [
                self.robot.get_link_pose(idx)
                for idx in self.computed_link_indices
            ]
            body_positions = np.array([pose[:3, 3] for pose in target_poses])
            torch_body_pos = torch.tensor(body_positions, requires_grad=True)

            origin_pos = torch_body_pos[self.origin_link_indices]
            task_pos = torch_body_pos[self.task_link_indices]
            robot_vectors = task_pos - origin_pos

            vector_diff = robot_vectors - target_vector_tensor
            error_norm = torch.norm(vector_diff, dim=1, keepdim=False)
            weighted_error = error_norm * touch_weights
            loss = self.huber_loss(weighted_error,
                                   torch.zeros_like(weighted_error))

            result = loss.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for idx, link_index in enumerate(self.computed_link_indices):
                    local_jac = self.robot.compute_single_link_local_jacobian(
                        qpos_to_use, link_index)[:3]
                    rot = target_poses[idx][:3, :3]
                    jac_world = rot @ local_jac
                    jacobians.append(jac_world)

                jacobians = np.stack(jacobians, axis=0)
                loss.backward()

                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, jacobians).mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                grad[:] = grad_qpos

            return result

        return objective

    def _compute_touch_degree(self, adapt_vectors: np.ndarray) -> np.ndarray:
        """
        Compute per-vector touch weight using norm scaling and sigmoid shaping.
        """
        norms = np.linalg.norm(adapt_vectors, axis=1)

        # Update normalization bounds
        for i in range(self.num_constrain):
            self.norm_min[i] = min(self.norm_min[i], norms[i])
            self.norm_max[i] = max(self.norm_max[i], norms[i])

        normalized = 1.0 - (norms - self.norm_min) / (self.norm_max -
                                                      self.norm_min)
        return fine_tune_touch_degree(normalized)


def fine_tune_touch_degree(norms: np.ndarray,
                           k=10.0,
                           c=0.8,
                           A=1.8) -> np.ndarray:
    """
    Sigmoid-like function to enhance low-norm penalties and smooth sharp transitions.
    """
    return A / (1.0 + np.exp(-k * (norms - c)))
