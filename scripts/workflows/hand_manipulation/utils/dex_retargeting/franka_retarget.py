# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
import yaml
from scipy.spatial.transform import Rotation as R

import omni.log
from dex_retargeting.retargeting_config import RetargetingConfig

from scipy.spatial.transform import Rotation as R

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import init_leap_hand_retarget, _HAND_JOINTS_INDEX, _OPERATOR2MANO_RIGHT, _OPERATOR2MANO_LEFT


class FrankaRetargeting:
    """A class for hand retargeting with GR1Fourier.

    Handles retargeting of OpenXRhand tracking data to GR1T2 robot hand joint angles.
    """

    def __init__(
        self,
        env=None,
        add_left_hand: bool = False,
        add_right_hand: bool = False,
        num_hand_joints: int = 16,
        kinematics_path="source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_free_leap_dexpilot_openxr.yml",
    ):

        self.add_left_hand = add_left_hand
        self.add_right_hand = add_right_hand
        self.env = env
        self.num_hand_joints = num_hand_joints
        self.kinematics_path = kinematics_path

        self.init_retargeting()

    def load_user_config(self, ):
        """Load user-specific configuration from a YAML file."""

        with open(
                f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/franka_teleop.yml",
                'r') as file:
            user_config = yaml.safe_load(file)

        self.kinematics_path = user_config.get("kinematics_path",
                                               self.kinematics_path)

        omni.log.info("Failed to load user config, using default.")

    def init_retargeting(self):

        self.load_user_config()
        init_leap_hand_retarget(
            self,
            kinematics_path=self.kinematics_path,
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
        )

        if self.add_left_hand:
            self.left_retargeting = self.retargeting[0]

        if self.add_right_hand:
            self.right_retargeting = self.retargeting[-1]

    def convert_hand_joints(self, hand_poses: dict[str, np.ndarray],
                            operator2mano: np.ndarray) -> np.ndarray:
        """Prepares the hand joints data for retargeting.

        Args:
            hand_poses: Dictionary containing hand pose data with joint positions and rotations
            operator2mano: Transformation matrix to convert from operator to MANO frame

        Returns:
            Joint positions with shape (21, 3)
        """

        # joint_position = hand_poses[_HAND_JOINTS_INDEX][:, :3]

        joint_position = np.zeros((21, 3))
        hand_joints = list(hand_poses.values())
        for i in range(len(_HAND_JOINTS_INDEX)):
            joint = hand_joints[_HAND_JOINTS_INDEX[i]]
            joint_position[i] = joint[:3]

        # Convert hand pose to the canonical frame.
        joint_position = joint_position - joint_position[0:1, :]

        xr_wrist_quat = hand_poses.get("wrist")[3:7]
        # OpenXR hand uses w,x,y,z order for quaternions but scipy uses x,y,z,w order
        wrist_rot = R.from_quat([
            xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3],
            xr_wrist_quat[0]
        ]).as_matrix()

        return joint_position @ wrist_rot @ operator2mano

    def compute_ref_value(
        self,
        joint,
        retargeting,
    ) -> np.ndarray:
        """Computes reference value for retargeting.

        Args:
            joint_position: Joint positions array
            indices: Target link indices
            retargeting_type: Type of retargeting ("POSITION" or other)

        Returns:
            Reference value in cartesian space
        """
        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:

            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]

        with torch.enable_grad():
            with torch.inference_mode(False):

                retargeted_qpos = retargeting.retarget(ref_value[..., :3])

                return retargeted_qpos

    def compute_one_hand(self, hand_joints: dict[str, np.ndarray],
                         retargeting: RetargetingConfig,
                         operator2mano: np.ndarray) -> np.ndarray:
        """Computes retargeted joint angles for one hand.

        Args:
            hand_joints: Dictionary containing hand joint data
            retargeting: Retargeting configuration object
            operator2mano: Transformation matrix from operator to MANO frame

        Returns:
            Retargeted joint angles
        """

        human_joint_pos = self.convert_hand_joints(
            hand_joints, operator2mano)  # hunand hand joint positions
        hand_joints = self.compute_ref_value(human_joint_pos, retargeting)

        return hand_joints, human_joint_pos

    def get_joint_names(self) -> list[str]:
        """Returns list of all joint names."""
        return self.robot_joint_name

    def get_left_joint_names(self) -> list[str]:
        """Returns list of left hand joint names."""
        return self.robot_joint_name

    def get_right_joint_names(self) -> list[str]:
        """Returns list of right hand joint names."""
        return self.robot_joint_name

    def get_hand_indices(self, robot) -> np.ndarray:
        """Gets indices of hand joints in robot's DOF array.

        Args:
            robot: Robot object containing DOF information

        Returns:
            Array of joint indices
        """
        return np.array(
            [robot.dof_names.index(name) for name in self.robot_joint_name],
            dtype=np.int64)

    def compute_left(self, left_hand_poses: dict[str,
                                                 np.ndarray]) -> np.ndarray:
        """Computes retargeted joints for left hand.

        Args:
            left_hand_poses: Dictionary of left hand joint poses

        Returns:
            Retargeted joint angles for left hand
        """

        if left_hand_poses is not None:
            left_hand_q, left_human_joint_pos = self.compute_one_hand(
                left_hand_poses, self.retargeting[0], _OPERATOR2MANO_LEFT)
        else:
            left_hand_q = np.zeros(self.num_hand_joints)
        return left_hand_q, left_human_joint_pos

    def compute_right(self, right_hand_poses: dict[str,
                                                   np.ndarray]) -> np.ndarray:
        """Computes retargeted joints for right hand.

        Args:
            right_hand_poses: Dictionary of right hand joint poses

        Returns:
            Retargeted joint angles for right hand
        """
        if right_hand_poses is not None:
            right_hand_q, right_human_joint_pos = self.compute_one_hand(
                right_hand_poses, self.retargeting[-1], _OPERATOR2MANO_RIGHT)
        else:
            right_hand_q = np.zeros(self.num_hand_joints)
        return right_hand_q, right_human_joint_pos
