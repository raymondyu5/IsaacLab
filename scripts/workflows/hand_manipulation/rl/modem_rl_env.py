# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import sys

sys.path.append("submodule/stable-baselines3")
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

from scripts.sb3.ppo import PPO

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
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

from scripts.workflows.hand_manipulation.env.rl_env.modem_tdmpc_env import TDMPCDemoWrapper

from scripts.workflows.open_policy.utils.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():

    # =======================================================================
    # =======================================================================
    # =======================================================================

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand

    env = setup_env(args_cli, save_config)
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    # =======================================================================
    # =======================================================================
    # =======================================================================
    # init the environment

    rl_env = TDMPCDemoWrapper(
        env,
        save_config,
        args_cli=args_cli,
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
