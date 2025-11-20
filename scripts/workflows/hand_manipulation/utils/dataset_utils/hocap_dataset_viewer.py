import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from pytransform3d import rotations
from tqdm import trange
from pytransform3d import transformations as pt
import sys
import os
from manopth import demo
# local_path = os.path.abspath("scripts/workflows/hand_manipulation/utils")
# sys.path.insert(0, local_path)

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from scripts.workflows.hand_manipulation.utils.dataset_utils.hocap_dataset import HOCAPDatasetLoader
import torch
import sapien
import isaaclab.utils.math as math_utils
import yaml
# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

from scripts.workflows.hand_manipulation.utils.dex_retargeting.mano_layer import MANOLayer


class HOCAPDataset:

    def __init__(self,
                 env_config,
                 args_cli,
                 env,
                 robot_name: str,
                 robot: None,
                 data_root=None,
                 hand_type="right",
                 ee_name="base"):

        self.robot_name = robot_name
        self.robot = robot

        self.env_config = env_config
        self.env = env
        self.hand_type = hand_type
        self.data_root = data_root
        self.device = env.unwrapped.device

        self.ee_name = ee_name
        self.args_cli = args_cli

        self.hocap_dataset = HOCAPDatasetLoader(
            data_root,
            hand_type=hand_type,
        )
        self.num_data = len(self.hocap_dataset._data_folders)

        self.init_retargeting_config()

    def init_retargeting_config(self):

        if self.hand_type == "left":
            self.retarget_hand_type = HandType.left
        elif self.hand_type == "right":
            self.retarget_hand_type = HandType.right

        kinematics_path = Path(
            "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_leap.yml"
        )
        with kinematics_path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            left_config = yaml_config["left"]
            right_config = yaml_config["right"]

        retargeting_config = RetargetingConfig.from_dict(
            cfg_dict["retargeting"])
        self.retargeting = retargeting_config.build()

        sim_joint_names = self.robot.joint_names

        # self.env.action_manager._terms[
        #     f"{self.hand_type}_hand_action"].cfg.joint_names  #self.robot.joint_names

        if self.env_config["params"]["arm_type"] is not None:
            sim_joint_names = [
                "dummy_x_translation_joint",
                "dummy_y_translation_joint",
                "dummy_z_translation_joint",
                "dummy_x_rotation_joint",
                "dummy_y_rotation_joint",
                "dummy_z_rotation_joint",
            ] + sim_joint_names[:, -16:]
        else:
            sim_joint_names = [
                "dummy_x_translation_joint",
                "dummy_y_translation_joint",
                "dummy_z_translation_joint",
                "dummy_x_rotation_joint",
                "dummy_y_rotation_joint",
                "dummy_z_rotation_joint",
            ] + sim_joint_names

        self.retarget2sim = np.array([
            self.retargeting.joint_names.index(joint)
            for joint in sim_joint_names
        ]).astype(int)

    def load_data(self, idx):
        idx = np.random.randint(self.num_data) if idx is None else idx
        sampled_data = self.hocap_dataset.lood_data(idx)
        return sampled_data

    def load_object_hand(self, data: Dict):

        hand_shape = data["hand_shape"]
        # extrinsic_mat = data["extrinsics"]

        self.mano_layer = MANOLayer(self.hand_type,
                                    hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        # pose_vec = pt.pq_from_transform(extrinsic_mat)

        # self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):

        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None

        p = torch.from_numpy(hand_pose_frame[:48].astype(
            np.float32)).unsqueeze(0)
        t = torch.from_numpy(hand_pose_frame[48:51].astype(
            np.float32)).unsqueeze(0)
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]

        return vertex, joint

    def retaget(self, idx=None):
        data = self.load_data(idx)
        self.load_object_hand(data)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        object_name = data["object_ids"]

        num_frame = hand_pose.shape[0]

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):

            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0:3])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        # demo.display_hand(
        #     {
        #         'verts': torch.as_tensor(vertex).unsqueeze(0),
        #         'joints': torch.as_tensor(joint).unsqueeze(0),
        #     },
        #     mano_faces=self.mano_layer._mano_layer.th_faces)

        if joint is None:
            return

        self.retargeting.warm_start(
            joint[0, :],
            wrist_quat,
            hand_type=self.retarget_hand_type,
            is_mano_convention=True,
        )
        target_qpos = []

        robot_ee_pose = []
        hocap_object_pose = {}
        for obj_name in object_name:
            hocap_object_pose[obj_name] = []

        for i in trange(start_frame, num_frame):

            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            indices = self.retargeting.optimizer.target_link_human_indices
            if joint is None:
                return None, None, None
            ref_value = joint[indices, :]

            qpos = self.retargeting.retarget(ref_value)[self.retarget2sim]
            target_qpos.append(qpos)

            ee_index = self.retargeting.optimizer.robot.get_link_index(
                self.ee_name)
            ee_pose = torch.as_tensor(
                self.retargeting.optimizer.robot.get_link_pose(ee_index))
            ee_quat = math_utils.quat_from_matrix(ee_pose[:3, :3])
            ee_pos = torch.cat([ee_pose[:3, 3], ee_quat])

            robot_ee_pose.append(ee_pos)
            for object_id, obj_name in enumerate(object_name):

                obj_pose = torch.as_tensor(object_pose[i][object_id]).to(
                    self.device)

                hocap_object_pose[obj_name].append(
                    math_utils.pose_from_transformations(
                        obj_pose.unsqueeze(0)).reshape(7))

        return torch.tensor(target_qpos), hocap_object_pose, robot_ee_pose
