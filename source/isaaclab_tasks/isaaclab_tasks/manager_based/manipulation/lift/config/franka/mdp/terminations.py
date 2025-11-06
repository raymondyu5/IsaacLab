# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka+Leap termination functions."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def abnormal_robot_state_relaxed(
    env: "ManagerBasedRLEnv",
    velocity_multiplier: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Abnormal robot termination with configurable velocity multiplier.

    Args:
        env: The RL environment.
        velocity_multiplier: Multiplier for joint velocity limits.
        asset_cfg: Configuration for the robot entity.

    Returns:
        Boolean tensor indicating abnormal state (num_envs,).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * velocity_multiplier)).any(dim=1)
