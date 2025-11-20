# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import pinocchio as pin
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""
"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)
parser.add_argument(
    "--action_framework",
    default=None,
)

parser.add_argument(
    "--target_object_name",
    type=str,
    default=None,
)

parser.add_argument(
    "--random_camera_pose",
    action="store_true",
)
parser.add_argument(
    "--sythesize_robot_pc",
    action="store_true",
)
# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

import os

from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():

    save_config, config = save_params_to_yaml(args_cli,
                                              args_cli.log_dir,
                                              random_shuffle=False)

    object_name = args_cli.target_object_name
    save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
        "objects_list"] = [object_name]

    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["real_eval_mode"] = True
    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose
    save_config["params"]["sythesize_robot_pc"] = args_cli.sythesize_robot_pc

    env = setup_env(args_cli, save_config)
    env.reset()

    args_cli.load_path = "raw_data/" + object_name

    rl_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        replay_mode=True,
    )

    rl_env.replay()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
