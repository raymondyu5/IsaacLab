# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import sys

sys.path.append("submodule/stable-baselines3")

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser

import os
import numpy as np

rl_parser.add_argument("--ibrl_config",
                       type=str,
                       default=None,
                       help="Path to the ibrl config file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(rl_parser)
# parse the arguments
args_cli = rl_parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None).unwrapped


import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
from scripts.workflows.hand_manipulation.finetunue.ibrl.env.ibrl_env_wrapper import IBRLEnvWrapper
from scripts.sb3.rl_algo_wrapper import initalize_rl_env


def main():
    """Zero actions agent with Isaac Lab environment."""
    # =======================================================================
    # =======================================================================

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs

    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]

    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose

    save_config["params"]["sample_points"] = True
    if args_cli.rl_type == "sac":
        save_config["params"]["reward_scale"] = 50.0

    if args_cli.action_framework in ["pcd_diffusion"]:
        save_config["params"]["Camera"][
            "extract_rgb"] = False  # disable initial camera pose
    if args_cli.action_framework in ["image_diffusion"]:
        save_config["params"]["Camera"]["extract_seg_pc"] = False
        save_config["params"]["Camera"]["extract_rgb"] = True
    # save_config["params"]["residual_rew"] = True

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    # reset environment

    rl_env = IBRLEnvWrapper(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        use_joint_pose=True if "Joint-Rel" in args_cli.task else False,
        use_visal_obs=False,
    )
    rl_env.train()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
