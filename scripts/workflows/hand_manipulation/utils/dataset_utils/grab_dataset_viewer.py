from typing import Dict, List

import numpy as np

from pytransform3d import rotations
from tqdm import trange
from pytransform3d import transformations as pt

from scripts.workflows.hand_manipulation.utils.dataset_utils.dexycb_dataset import DexYCBDatasetLoader, YCB_CLASSES
import torch
import sapien
import isaaclab.utils.math as math_utils
import yaml
from manopth import demo
# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget, display_hand
import copy
import trimesh
from dex_retargeting.constants import RobotName, HandType
from scripts.workflows.hand_manipulation.utils.dataset_utils.grab_dataset import GrabDatasetLoader
from tqdm import tqdm
import multiprocessing as mp


def update_last_right_retargeted_qpos(joint):
    global global_right_retargeting

    optimizer = global_right_retargeting.optimizer
    retargeting_type = optimizer.retargeting_type
    indices = optimizer.target_link_human_indices

    ref_value = joint[indices, :]
    retargeted_qpos = global_right_retargeting.retarget(ref_value)

    return retargeted_qpos


def worker_init_right_retarget(right_retargeting):
    global global_right_retargeting
    global_right_retargeting = right_retargeting
    print("Worker initialized.")


global_right_retargeting = None
global_right_retargeting = None


class GrabDataset:

    def __init__(self,
                 env_config,
                 args_cli,
                 env,
                 robot: None,
                 data_root=None,
                 ee_name="base"):

        self.robot = robot

        self.env_config = env_config
        self.env = env

        self.data_root = data_root
        self.device = env.unwrapped.device
        self.add_right_hand, self.add_left_hand = args_cli.add_right_hand, args_cli.add_left_hand

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.ee_name = ee_name
        self.args_cli = args_cli

        self.ycb_dataset = GrabDatasetLoader(
            data_root,
            add_translation=args_cli.add_translation,
        )
        self.num_data = len(self.ycb_dataset)

        self.init_retargeting_config()

    def init_retargeting_config(self):

        init_leap_hand_retarget(
            self,
            kinematics_path=
            # "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_free_leap_dexpilot.yml",
            "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_leap_vector_offline.yml",
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
        )
        self.robot_joint_name = get_hand_joint_names(
            self,
            hand_side="right" if self.add_right_hand else "left",
        )[-self.num_hand_joints:]
        if isinstance(self.retargeting, list):
            self.left_retargeting = self.retargeting[0]
            self.right_retargeting = self.retargeting[1]

            self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                self.left_retargeting.optimizer.target_joint_names.index(j)
                for j in self.robot_joint_name
            ])

        else:
            if self.add_left_hand:
                self.left_retargeting = self.retargeting
                self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                    self.left_retargeting.optimizer.target_joint_names.index(j)
                    for j in self.robot_joint_name
                ])

            if self.add_right_hand:
                self.right_retargeting = self.retargeting
                self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                    self.right_retargeting.optimizer.target_joint_names.index(
                        j) for j in self.robot_joint_name
                ])

    def load_data(self, idx):

        sampled_data = self.ycb_dataset[idx]
        return sampled_data

    def viz_hand_object(self, object_name, object_pose, vertex, joint):
        if object_name is None:
            obj_mesh = trimesh.load(self.object_mesh_dir +
                                    f"/{object_name}/textured_simple.obj")
            object_vertices = torch.as_tensor(
                obj_mesh.vertices).to(dtype=torch.float32)

            object_pose = torch.as_tensor(object_pose).unsqueeze(0)

            transformed_obj_points = math_utils.transform_points(
                object_vertices.unsqueeze(0), object_pose[:, :3],
                object_pose[:, 3:7])[0]

        display_hand({
            "hand_info": {
                'verts': torch.as_tensor(vertex),
                'joints': torch.as_tensor(joint),
                "faces": self.mano_layer._mano_layer.th_faces
            },
            "obj_info": {
                "verts": transformed_obj_points,
                "faces": torch.as_tensor(obj_mesh.faces)
            }
        })
        return transformed_obj_points.cpu().numpy(), obj_mesh.faces

    def update_last_retargeted_qpos(self, joint, retargeting):

        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]
            ref_value[:, 1] *= 1.3
            ref_value[[3, 7], 1] *= 1.3
            # ref_value[2, 1] *= 1.3
            # ref_value[3, 1] *= 1.3

        retargeted_qpos = retargeting.retarget(ref_value)
        return retargeted_qpos[self.retarget2sim]

    def retaget(self, idx=None):
        data = self.load_data(idx)

        rhand_joints = data["rhand_joints"].cpu().numpy()

        lhand_joints = data["lhand_joints"].cpu().numpy()

        retarget_right_joints, retarget_left_joints = [], []

        if self.add_right_hand:

            local_orient = data["rhand_pose"][0, 3:7]
            local_orient = math_utils.quat_mul(
                torch.as_tensor([0.707, 0.0, 0.707, 0.0]).unsqueeze(0),
                local_orient.unsqueeze(0))
            local_orient = math_utils.quat_mul(
                torch.as_tensor([0.707, 0.0, 0.0, 0.707]).unsqueeze(0),
                local_orient)[0]
            local_pose = rhand_joints[0, 0] * 0.0

            self.right_retargeting.reset()

            self.right_retargeting.warm_start(
                local_pose,
                # data["rhand_pose"][0, :3].cpu().numpy(),
                local_orient.cpu().numpy(),
                hand_type=HandType.right
                if self.add_right_hand else HandType.left,
                is_mano_convention=True,
            )

            for i in range(50):
                self.update_last_retargeted_qpos(rhand_joints[0],
                                                 self.right_retargeting)

            for joint in rhand_joints:
                retarget_right_joints.append(
                    self.update_last_retargeted_qpos(
                        joint, self.right_retargeting)[None])

            retarget_right_joints = np.concatenate(retarget_right_joints,
                                                   axis=0)
            data["retarget_right_joints"] = torch.from_numpy(
                retarget_right_joints).to(dtype=torch.float32)

        if self.add_left_hand:
            retarget_left_joints = []
            for joint in lhand_joints:
                retarget_left_joints.append(
                    self.update_last_retargeted_qpos(
                        joint, self.left_retargeting)[None])
            retarget_left_joints = np.concatenate(retarget_left_joints, axis=0)
            data["retarget_left_joints"] = torch.from_numpy(
                retarget_left_joints).to(dtype=torch.float32)
        data["num_frame"] = len(data["rhand_joints"])

        return data
