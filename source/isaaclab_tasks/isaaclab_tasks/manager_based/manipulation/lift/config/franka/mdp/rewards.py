# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka+Leap reward functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.rewards import (
    action_l2_clamped,
    action_rate_l2_clamped,
    object_ee_distance,
    success_reward,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def leap_hand_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Contact detection for Leap Hand fingertips.

    Args:
        env: The RL environment.
        threshold: Contact force threshold in Newtons.

    Returns:
        Boolean tensor indicating good contact (num_envs,).
    """
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_fingertip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["fingertip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["fingertip_2_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["fingertip_3_object_s"]

    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)

    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )

    return good_contact_cond1


def fingertips_approach_object_when_near(
    env: ManagerBasedRLEnv,
    palm_distance_threshold: float = 0.15,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]),
) -> torch.Tensor:
    """Reward fingertips moving toward object when palm is near, encouraging grasping.

    This creates an intermediate reward signal that helps the agent learn to wrap
    fingers around the object when the palm is already close, bridging the gap
    between reaching (fingers_to_object) and contact (good_finger_contact).

    Args:
        env: The RL environment.
        palm_distance_threshold: Only give reward when palm is within this distance of object (meters).
        std: Standard deviation for the tanh kernel on fingertip-object distance.
        object_cfg: Configuration for the object entity.
        robot_cfg: Configuration for the robot entity (should include palm_lower and fingertips).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import RigidObject, Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get all body positions (palm + fingertips) - shape: (num_envs, num_bodies, 3)
    body_positions = robot.data.body_pos_w[:, robot_cfg.body_ids]

    # First body is palm_lower (index 0), rest are fingertips
    palm_pos = body_positions[:, 0, :]  # (num_envs, 3)
    fingertip_positions = body_positions[:, 1:, :]  # (num_envs, 4, 3) - 4 fingertips

    # Check if palm is close to object
    palm_to_object_dist = torch.norm(palm_pos - object.data.root_pos_w, dim=-1)  # (num_envs,)
    is_palm_near = palm_to_object_dist < palm_distance_threshold

    # Get distances from each fingertip to object
    object_pos = object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    fingertip_to_object_dist = torch.norm(fingertip_positions - object_pos, dim=-1)  # (num_envs, 4)

    # Average reward across all fingertips (encourages all fingers to approach)
    avg_fingertip_proximity = (1 - torch.tanh(fingertip_to_object_dist / std)).mean(dim=-1)  # (num_envs,)

    # Only give reward when palm is near object
    return avg_fingertip_proximity * is_palm_near.float()


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position tracking reward gated by contact.

    Args:
        env: The RL environment.
        std: Standard deviation for the tanh kernel.
        command_name: Name of the pose command.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.utils.math import combine_frame_transforms

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)

    return (1 - torch.tanh(distance / std)) * leap_hand_contacts(env, 1.0).float()


def power_grasp_approach(
    env: ManagerBasedRLEnv,
    z_threshold: float = 0.1,
    approach_distance: float = 0.2,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for approaching low objects with a straight-down power grasp posture.

    When the object is below z_threshold (typically on the table), this reward encourages:
    1. Palm directly above the object (small XY offset)
    2. Palm approaching from above (higher Z than object)
    3. Fingers positioned around the object (ready for power grasp)

    Args:
        env: The RL environment.
        z_threshold: Only activate when object Z position is below this value (meters).
        approach_distance: Only reward when within this distance of object.
        std: Standard deviation for the tanh kernel.
        robot_cfg: Configuration for the robot entity (palm + fingertips).
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import RigidObject, Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get body positions (palm + fingertips)
    body_positions = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (num_envs, 5, 3)
    palm_pos = body_positions[:, 0, :]  # (num_envs, 3)
    fingertip_positions = body_positions[:, 1:, :]  # (num_envs, 4, 3) - 4 fingertips

    object_pos = object.data.root_pos_w  # (num_envs, 3)

    # Check if object is low (on table or falling with gravity)
    is_object_low = object_pos[:, 2] < z_threshold

    # Check if palm is within approach distance
    palm_distance = torch.norm(palm_pos - object_pos, dim=-1)
    is_approaching = palm_distance < approach_distance

    # 1. Palm vertical alignment reward
    xy_offset = torch.norm(palm_pos[:, :2] - object_pos[:, :2], dim=-1)
    vertical_alignment = 1 - torch.tanh(xy_offset / std)

    # 2. Palm height reward (optimal clearance for fingers)
    z_diff = palm_pos[:, 2] - object_pos[:, 2]
    optimal_height = 0.18  # 18cm above object center
    height_reward = torch.exp(-2.0 * (z_diff - optimal_height).pow(2))  # Gaussian around optimal height
    has_clearance = (z_diff > 0.10).float()  # At least 10cm above

    # 3. Finger positioning reward - fingers should surround the object
    # Calculate distances from each fingertip to object
    object_pos_expanded = object_pos.unsqueeze(1)  # (num_envs, 1, 3)
    fingertip_to_object_dist = torch.norm(fingertip_positions - object_pos_expanded, dim=-1)  # (num_envs, 4)

    # Reward fingers being at appropriate distance (not too far, not too close)
    optimal_finger_dist = 0.08  # 8cm from object center (adjust based on object size)
    finger_dist_reward = torch.exp(-5.0 * (fingertip_to_object_dist - optimal_finger_dist).pow(2))  # (num_envs, 4)

    # Average across all fingers
    finger_positioning = finger_dist_reward.mean(dim=-1)  # (num_envs,)

    # 4. Finger spread reward - fingers should be distributed around the object, not clustered
    # Calculate variance of fingertip positions in XY plane
    fingertip_xy = fingertip_positions[:, :, :2]  # (num_envs, 4, 2)
    fingertip_xy_mean = fingertip_xy.mean(dim=1, keepdim=True)  # (num_envs, 1, 2)
    fingertip_variance = ((fingertip_xy - fingertip_xy_mean) ** 2).sum(dim=-1).mean(dim=-1)  # (num_envs,)
    finger_spread = torch.tanh(fingertip_variance * 10.0)  # Normalize to 0-1

    # Combine all components
    power_grasp_reward = (
        vertical_alignment * 0.3 +  # Palm above object
        height_reward * 0.2 +        # Palm at good height
        finger_positioning * 0.3 +    # Fingers at right distance
        finger_spread * 0.2           # Fingers spread around object
    ) * has_clearance * is_approaching.float() * is_object_low.float()

    return power_grasp_reward


def straight_wrist_bonus(
    env: ManagerBasedRLEnv,
    target_angles: dict = None,
    std: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping wrist joints near target angles (typically straight).

    This encourages a neutral wrist posture which is better for power grasps.

    Args:
        env: The RL environment.
        target_angles: Dict of joint names to target angles. If None, uses zeros.
        std: Standard deviation for the tanh kernel.
        robot_cfg: Configuration for the robot entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # Default target angles for straight wrist (adjust based on your robot)
    if target_angles is None:
        target_angles = {
            "panda_joint5": 0.0,  # Wrist 1
            "panda_joint6": 3.037,  # Wrist 2 (default straight position)
            "panda_joint7": 0.741,  # Wrist 3
        }

    # Get current joint positions
    joint_pos = robot.data.joint_pos  # (num_envs, num_joints)

    # Calculate deviation from target for each wrist joint
    total_reward = torch.zeros(env.num_envs, device=env.device)

    for joint_name, target_angle in target_angles.items():
        # Find joint index (this is a bit hacky, might need adjustment)
        joint_names = robot.joint_names
        if joint_name in joint_names:
            joint_idx = joint_names.index(joint_name)
            deviation = torch.abs(joint_pos[:, joint_idx] - target_angle)
            joint_reward = 1 - torch.tanh(deviation / std)
            total_reward += joint_reward

    # Average over all wrist joints
    num_joints = len(target_angles)
    return total_reward / num_joints if num_joints > 0 else total_reward


def pinch_grasp_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Contact detection for pinch grasp using only thumb and index finger.

    Args:
        env: The RL environment.
        threshold: Contact force threshold in Newtons.

    Returns:
        Boolean tensor indicating good pinch contact (num_envs,).
    """
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_fingertip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["fingertip_object_s"]

    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)

    # Both thumb and index must be in contact for a proper pinch
    good_pinch_contact = (thumb_contact_mag > threshold) & (index_contact_mag > threshold)

    return good_pinch_contact


def pinch_fingertips_approach_object_when_near(
    env: ManagerBasedRLEnv,
    palm_distance_threshold: float = 0.15,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip"]),
) -> torch.Tensor:
    """Reward thumb and index fingertips moving toward object when palm is near, encouraging pinch.

    This focuses the approach reward on only the thumb and index finger for pinch grasping.

    Args:
        env: The RL environment.
        palm_distance_threshold: Only give reward when palm is within this distance of object (meters).
        std: Standard deviation for the tanh kernel on fingertip-object distance.
        object_cfg: Configuration for the object entity.
        robot_cfg: Configuration for the robot entity (should include palm_lower, index, and thumb).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import RigidObject, Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get all body positions (palm + 2 fingertips) - shape: (num_envs, num_bodies, 3)
    body_positions = robot.data.body_pos_w[:, robot_cfg.body_ids]

    # First body is palm_lower (index 0), then index finger (1), then thumb (2)
    palm_pos = body_positions[:, 0, :]  # (num_envs, 3)
    fingertip_positions = body_positions[:, 1:, :]  # (num_envs, 2, 3) - index and thumb only

    # Check if palm is close to object
    palm_to_object_dist = torch.norm(palm_pos - object.data.root_pos_w, dim=-1)  # (num_envs,)
    is_palm_near = palm_to_object_dist < palm_distance_threshold

    # Get distances from each fingertip to object
    object_pos = object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    fingertip_to_object_dist = torch.norm(fingertip_positions - object_pos, dim=-1)  # (num_envs, 2)

    # Average reward across index and thumb (encourages both to approach)
    avg_fingertip_proximity = (1 - torch.tanh(fingertip_to_object_dist / std)).mean(dim=-1)  # (num_envs,)

    # Only give reward when palm is near object
    return avg_fingertip_proximity * is_palm_near.float()


def pinch_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position tracking reward gated by pinch contact (thumb + index only).

    Args:
        env: The RL environment.
        std: Standard deviation for the tanh kernel.
        command_name: Name of the pose command.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.utils.math import combine_frame_transforms

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)

    return (1 - torch.tanh(distance / std)) * pinch_grasp_contacts(env, 1.0).float()
