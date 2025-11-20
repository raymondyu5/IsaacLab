# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import numpy as np
import torch
from dataclasses import dataclass
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# This import exception is suppressed because gr1_t2_dex_retargeting_utils depends on pinocchio which is not available on windows
# with contextlib.suppress(Exception):
from scripts.workflows.hand_manipulation.utils.cloudxr.leap_dex_retargeting_utils import LeapDexRetargeting

from isaaclab.markers.config import FRAME_MARKER_CFG
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Se3AbsRetargeterCfg(RetargeterCfg):
    """Configuration for absolute position retargeter."""

    zero_out_xy_rotation: bool = True
    use_wrist_rotation: bool = False
    use_wrist_position: bool = True
    enable_visualization: bool = False
    bound_hand: OpenXRDevice.TrackingTarget = OpenXRDevice.TrackingTarget.HAND_RIGHT


class LeapAbsRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to GR1T2 hand end-effector commands.

    This retargeter maps hand tracking data from OpenXR to joint commands for the GR1T2 robot's hands.
    It handles both left and right hands, converting poses of the hands in OpenXR format joint angles for the GR1T2 robot's hands.
    """

    def __init__(
        self,
        env,
        add_left_hand=False,
        add_right_hand=False,
        use_arm=False,
        teleop_user="Entong",
        cfg=Se3AbsRetargeterCfg,
    ):
        """Initialize the GR1T2 hand retargeter.

        Args:
            bound_hand: The hand to track (OpenXRDevice.TrackingTarget.HAND_LEFT or OpenXRDevice.TrackingTarget.HAND_RIGHT)
            zero_out_xy_rotation: If True, zero out rotation around x and y axes
            use_wrist_rotation: If True, use wrist rotation instead of finger average
            use_wrist_position: If True, use wrist position instead of pinch position
            enable_visualization: If True, visualize the target pose in the scene
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """

        self.env = env
        self.add_left_hand = add_left_hand
        self.add_right_hand = add_right_hand

        self._hands_controller = LeapDexRetargeting(
            env=self.env,
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
            use_arm=use_arm,
            teleop_user=teleop_user,
        )

        # Initialize visualization if enabled
        super().__init__(cfg)
        if cfg.bound_hand not in [
                OpenXRDevice.TrackingTarget.HAND_LEFT,
                OpenXRDevice.TrackingTarget.HAND_RIGHT
        ]:
            raise ValueError(
                "bound_hand must be either OpenXRDevice.TrackingTarget.HAND_LEFT or"
                " OpenXRDevice.TrackingTarget.HAND_RIGHT")
        self.bound_hand = cfg.bound_hand

        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._use_wrist_rotation = cfg.use_wrist_rotation
        self._use_wrist_position = cfg.use_wrist_position

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([1.0, 0.0, 0.0, 0.0])

    def retarget(
            self, data: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert hand joint poses to robot end-effector commands.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.

        Returns:
            tuple containing:
                Left wrist pose
                Right wrist pose in USD frame
                Retargeted hand joint angles
        """

        # Access the left and right hand data using the enum key

        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        actions = []
        human_actions = []

        if self.add_left_hand:
            left_hand_pose, left_human_joint_pos = self._hands_controller.compute_left(
                left_hand_poses)
            actions.append(left_hand_pose)
            human_actions.append(left_human_joint_pos)
        if self.add_right_hand:
            right_hand_pose, right_human_joint_pos = self._hands_controller.compute_right(
                right_hand_poses)
            actions.append(right_hand_pose)
            human_actions.append(right_human_joint_pos)

        actions = np.concatenate(actions, axis=0)
        human_actions = np.concatenate(human_actions, axis=0)

        hand_data = data[self.bound_hand]
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")
        wrist = hand_data.get("wrist")

        ee_command_np = self._retarget_abs(thumb_tip, index_tip, wrist)

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np,
                                  dtype=torch.float32,
                                  device=self.env.device)

        return left_wrist, right_wrist, human_actions, actions

    def _retarget_abs(self, thumb_tip: np.ndarray, index_tip: np.ndarray,
                      wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            thumb_tip: 7D array containing position (xyz) and orientation (quaternion)
                for the thumb tip
            index_tip: 7D array containing position (xyz) and orientation (quaternion)
                for the index tip
            wrist: 7D array containing position (xyz) and orientation (quaternion)
                for the wrist

        Returns:
            np.ndarray: 7D array containing position (xyz) and orientation (quaternion)
                for the robot end-effector
        """

        # Get position
        if self._use_wrist_position:
            position = wrist[:3]
        else:
            position = (thumb_tip[:3] + index_tip[:3]) / 2

        # Get rotation
        if self._use_wrist_rotation:
            # wrist is w,x,y,z but scipy expects x,y,z,w
            base_rot = Rotation.from_quat([*wrist[4:], wrist[3]])
        else:
            # Average the orientations of thumb and index using SLERP
            # thumb_tip is w,x,y,z but scipy expects x,y,z,w
            r0 = Rotation.from_quat([*thumb_tip[4:], thumb_tip[3]])
            # index_tip is w,x,y,z but scipy expects x,y,z,w
            r1 = Rotation.from_quat([*index_tip[4:], index_tip[3]])
            key_times = [0, 1]
            slerp = Slerp(key_times, Rotation.concatenate([r0, r1]))
            base_rot = slerp([0.5])[0]

        # Apply additional x-axis rotation to align with pinch gesture
        final_rot = base_rot * Rotation.from_euler("x", 90, degrees=True)

        if self._zero_out_xy_rotation:
            z, y, x = final_rot.as_euler("ZYX")
            y = 0.0  # Zero out rotation around y-axis
            x = 0.0  # Zero out rotation around x-axis
            final_rot = Rotation.from_euler("ZYX",
                                            [z, y, x]) * Rotation.from_euler(
                                                "X", np.pi, degrees=False)

        # Convert back to w,x,y,z format
        quat = final_rot.as_quat()
        rotation = np.array([quat[3], quat[0], quat[1],
                             quat[2]])  # Output remains w,x,y,z

        # Update visualization if enabled
        if self._enable_visualization:
            self._visualization_pos = position
            self._visualization_rot = rotation
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose.

        If visualization is enabled, the target end-effector pose is visualized in the scene.
        """
        if self._enable_visualization:
            trans = np.array([self._visualization_pos])
            quat = Rotation.from_matrix(self._visualization_rot).as_quat()
            rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
            self._goal_marker.visualize(translations=trans, orientations=rot)
