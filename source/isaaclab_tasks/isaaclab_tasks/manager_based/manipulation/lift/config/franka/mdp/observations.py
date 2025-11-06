# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka+Leap observation functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.observations import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingers_contact_force_b_flattened(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Flattened contact forces in base frame.

    Args:
        env: The RL environment.
        contact_sensor_names: Names of contact sensors.
        asset_cfg: Robot configuration.

    Returns:
        Flattened tensor of shape (num_envs, 3*num_sensors).
    """
    from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.observations import fingers_contact_force_b

    forces = fingers_contact_force_b(env, contact_sensor_names, asset_cfg)
    return forces.reshape(env.num_envs, -1)
