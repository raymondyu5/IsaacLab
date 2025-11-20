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

from scripts.workflows.hand_manipulation.utils.dex_retargeting.mano_layer import MANOLayer

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget, display_hand
import copy
import trimesh


class DexYCBDataset:

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

        if args_cli.add_right_hand:

            self.hand_side = "right"
        elif args_cli.add_left_hand:

            self.hand_side = "left"

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.ee_name = ee_name
        self.args_cli = args_cli

        self.ycb_dataset = DexYCBDatasetLoader(data_root,
                                               hand_side=self.hand_side)
        self.num_data = len(self.ycb_dataset)
        self.object_mesh_dir = self.env_config["params"][
            "spawn_rigid_objects"]["object_mesh_dir"]

        self.init_retargeting_config()

    def init_retargeting_config(self):

        init_leap_hand_retarget(
            self,
            kinematics_path=
            "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_free_leap_dexpilot.yml",
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
        )
        self.robot_joint_name = get_hand_joint_names(
            self,
            self.hand_side,
        )[-self.num_hand_joints:]

        self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
            self.retargeting.optimizer.target_joint_names.index(j)
            for j in self.robot_joint_name
        ])

    def load_data(self, idx):

        sampled_data = self.ycb_dataset[idx]
        return sampled_data

    def load_object_hand(self, data: Dict):

        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]

        self.mano_layer = MANOLayer(self.hand_side,
                                    hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        pose_vec = pt.pq_from_transform(extrinsic_mat)

        self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):

        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(
            np.float32)) * 0.0
        p[:, :3] *= 0
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]

        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()
            # camera_mat = self.camera_pose.numpy()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

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
            # "obj_info": {
            #     "verts": transformed_obj_points,
            #     "faces": torch.as_tensor(obj_mesh.faces)
            # }
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

        retargeted_qpos = retargeting.retarget(ref_value)
        return retargeted_qpos[self.retarget2sim]

    def retaget(self, idx=None):
        data = self.load_data(idx)
        self.load_object_hand(data)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        ycb_ids = data["ycb_ids"]
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
            hand_pose_start[0, 0:3])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        from dex_retargeting.constants import RobotName, HandType
        self.retargeting.warm_start(
            joint[0, :],
            wrist_quat,
            hand_type=HandType.right
            if self.hand_side == "right" else HandType.left,
            is_mano_convention=True,
        )

        if joint is None:
            return None

        target_qpos = []
        num_ycb_objects = len(data["ycb_ids"])

        ycb_objects_pose = {}
        robot_ee_pose = []
        refs_value = []
        init_object_pose = {}
        lift_objects = []
        for i in range(num_ycb_objects):
            ycb_id = ycb_ids[i]
            ycb_name = YCB_CLASSES[ycb_id]
            ycb_objects_pose[ycb_name] = []
            init_object_pose[ycb_name] = []

        data = {
            "rhand_verts": [],
            "rhand_joints": [],
            "rhand_faces": [],
            "lhand_verts": [],
            "lhand_joints": [],
            "retarget_right_joints": [],
            "retarget_left_joints": [],
        }

        for i in range(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            indices = self.retargeting.optimizer.target_link_human_indices
            if joint is None:
                return None, None, None

            qpos = self.update_last_retargeted_qpos(joint, self.retargeting)

            target_qpos.append(qpos)

            ee_index = self.retargeting.optimizer.robot.get_link_index(
                self.ee_name)
            ee_pose = torch.as_tensor(
                self.retargeting.optimizer.robot.get_link_pose(ee_index))
            ee_quat = math_utils.quat_from_matrix(ee_pose[:3, :3])
            ee_pos = torch.cat([ee_pose[:3, 3], ee_quat])
            robot_ee_pose.append(ee_pos)

            if self.add_right_hand:
                data["rhand_verts"].append(vertex)
                data["rhand_joints"].append(
                    torch.as_tensor(joint).to(self.device))
                data["retarget_right_joints"].append(
                    torch.as_tensor(qpos).unsqueeze(0))

            if self.add_left_hand:
                data["lhand_verts"].append(vertex)
                data["lhand_joints"].append(
                    torch.as_tensor(joint).to(self.device))
                data["retarget_left_joints"].append(
                    torch.as_tensor(qpos).unsqueeze(0))

            # display_hand({
            #     "hand_info": {
            #         'verts': torch.as_tensor(vertex),
            #         'joints': torch.as_tensor(joint),
            #         "faces": self.mano_layer._mano_layer.th_faces
            #     },
            # })
            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]
                                                  ]))
                final_object_pose = torch.as_tensor(
                    np.concatenate([pose.p, pose.q]))

                ycb_objects_pose[YCB_CLASSES[ycb_ids[k]]].append(
                    final_object_pose)
                if i == start_frame:
                    init_object_pose[YCB_CLASSES[ycb_ids[k]]].append(
                        torch.as_tensor(copy.deepcopy(final_object_pose)))
                else:

                    init_pose = init_object_pose[YCB_CLASSES[ycb_ids[k]]][0]
                    lift_or_not = (final_object_pose[2] - init_pose[2]) > 0.10
                    if lift_or_not:
                        lift_objects.append(YCB_CLASSES[ycb_ids[k]])

        if self.add_right_hand:
            data["rhand_faces"] = self.mano_face
            data["retarget_right_joints"] = torch.cat(
                data["retarget_right_joints"], dim=0).to(self.device)

        if self.add_left_hand:
            data["lhand_faces"] = self.mano_face
            data["retarget_left_joints"] = torch.cat(
                data["retarget_right_joints"], dim=0).to(self.device)

        data["num_frame"] = num_frame - start_frame

        return data
