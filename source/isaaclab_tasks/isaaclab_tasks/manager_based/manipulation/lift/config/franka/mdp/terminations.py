# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for Franka+Leap manipulation tasks."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_flung_termination(
    env: "ManagerBasedRLEnv",
    velocity_threshold: float = 3.0,
    angular_velocity_threshold: float = 10.0,
    minimum_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate episode if object is being flung (high velocity without proper grasp).

    This prevents the policy from learning to fling/throw objects instead of grasping.
    An object moving too fast or spinning too fast indicates flinging behavior.

    IMPORTANT: Only triggers if object is lifted (above minimum_height) to avoid
    false positives from spawn/settling physics.

    Args:
        env: The RL environment.
        velocity_threshold: Linear velocity threshold (m/s) for flinging detection.
        angular_velocity_threshold: Angular velocity threshold (rad/s) for flinging.
        minimum_height: Minimum object height (m) before flinging check is active.
        object_cfg: Configuration for the object entity.

    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Get object position and velocities
    object_pos = object.data.root_pos_w  # (num_envs, 3)
    linear_vel = object.data.root_lin_vel_w  # (num_envs, 3)
    angular_vel = object.data.root_ang_vel_w  # (num_envs, 3)

    # Compute velocity magnitudes
    linear_speed = torch.norm(linear_vel, dim=-1)  # (num_envs,)
    angular_speed = torch.norm(angular_vel, dim=-1)  # (num_envs,)

    # Check if object is lifted (above minimum height)
    is_lifted = object_pos[:, 2] > minimum_height

    # Only check for flinging if object is actually lifted
    velocity_too_high = (linear_speed > velocity_threshold) | (angular_speed > angular_velocity_threshold)
    flung = is_lifted & velocity_too_high

    return flung
