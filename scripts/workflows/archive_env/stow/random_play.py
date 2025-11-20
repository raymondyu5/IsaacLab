# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import sys

sys.path.append(".")

from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

import argparse

import time
from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

from isaaclab.utils import Timer
import isaaclab.utils.math as math_utils
from scripts.workflows.tasks.utils.parser_setting import save_params_to_yaml, parser
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from tools.deformable_obs import *

from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles

enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw
from tools.curobo_planner import IKPlanner


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    save_config, config = save_params_to_yaml(args_cli)
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    init_pos = torch.as_tensor([0.5, 0.0, 0.30, 0.0, 1.0, 0.0,
                                0.0]).to(env.unwrapped.device)
    delta_quat = obtain_target_quat_from_multi_angles(
        [2, 1], [np.pi / 2, 0]).to(env.unwrapped.device)
    init_pos[3:] = math_utils.quat_mul(delta_quat, init_pos[3:])
    init_pos = init_pos.unsqueeze(0).repeat_interleave(env.num_envs, 0)

    curobo_ik = IKPlanner()

    init_qpos = curobo_ik.plan_motion(init_pos[:, :3], init_pos[:, 3:])
    robot_target_qpos = env.scene[
        "robot"]._data.default_joint_pos[:, :9].clone()

    robot_target_qpos[:] = init_qpos.squeeze(1)

    indices = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)

    while simulation_app.is_running():

        # sample actions from -1 to 1
        actions = torch.rand(env.action_space.shape,
                             device=env.unwrapped.device) * 0
        env.scene["robot"].root_physx_view.set_dof_positions(
            robot_target_qpos, indices)
        env.scene["robot"].root_physx_view.set_dof_velocities(
            robot_target_qpos * 0, indices)

        next_obs, reward, terminate, time_out, info = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
