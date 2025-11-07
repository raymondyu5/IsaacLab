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
