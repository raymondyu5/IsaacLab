# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

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
import sys

sys.path.append("submodule/stable-baselines3")
from stable_baselines3 import PPO
from scripts.sb3.sac import SAC
from scripts.sb3.td3 import TD3
from scripts.sb3.rl_algo_wrapper import rl_parser

# launch omniverse app
AppLauncher.add_app_launcher_args(rl_parser)
args_cli, hydra_args = rl_parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.bimanual_rl_wrapper import BimanualRLDatawrapperEnv
from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import imageio
import os
# import os
from scripts.workflows.hand_manipulation.env.rl_env.offline_env_wrapper import OfflineEnvWrapper


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.offline_rl.offline.awac import load_awac_model
from scripts.offline_rl.offline.td3_bc import load_td3_model


def main():
    # parse configuration

    checkpoint_path = args_cli.checkpoint

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli,
                                              args_cli.log_dir,
                                              random_shuffle=False)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["real_eval_mode"] = True
    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    if args_cli.action_range is not None:
        save_config["params"]["Task"][
            "action_range"][:2] = args_cli.action_range

    env = setup_env(args_cli, save_config)
    env.reset()

    if args_cli.rl_type == "awac":
        agent = load_awac_model(checkpoint_path)
    elif args_cli.rl_type == "td3bc":
        agent = load_td3_model(checkpoint_path)

    offine_env = OfflineEnvWrapper(env, agent)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.no_grad():

            offine_env.step()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
