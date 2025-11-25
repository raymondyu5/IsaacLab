# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ISAAC_BASE_DIR = os.path.join(CUR_DIR, "../../../../")

import sys
sys.path.append(ISAAC_BASE_DIR) # isaac lab base dir

from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser

import argparse
import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

sys.path.append(os.path.join(ISAAC_BASE_DIR, "submodule", "stable-baselines3"))
from stable_baselines3 import PPO
from scripts.sb3.sac import SAC

rl_parser.add_argument(
    "--sythesize_robot_pc",
    action="store_true",
)

rl_parser.add_argument(
    "--use_failure",
    action="store_true",
)

rl_parser.add_argument(
    "--clip_real_world_actions",
    action="store_true",
    help="Apply real-world action clipping during data collection (matches hardware constraints)",
)

rl_parser.add_argument(
    "--save_all_data",
    action="store_true",
    help="Save all trajectories regardless of success or failure (temporary flag for testing)",
)

rl_parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Number of initial steps to send zero actions (warmup period for robot settling)",
)

rl_parser.add_argument(
    "--zero_thumb_joints",
    action="store_true",
    help="Zero out abduction/adduction joints for thumb, index, middle fingers (j0, j4, j8 = action dims 6, 10, 14)",
)

# launch omniverse app
AppLauncher.add_app_launcher_args(rl_parser)
args_cli, hydra_args = rl_parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import imageio
import os


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    return env


def main():
    # parse configuration

    checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    save_config, config = save_params_to_yaml(args_cli,
                                              args_cli.log_dir,
                                              random_shuffle=False)
    if args_cli.target_object_name is not None:
        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    else:
        object_name = "all"

    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs

    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose
    save_config["params"]["sythesize_robot_pc"] = args_cli.sythesize_robot_pc

    if args_cli.action_range is not None:
        save_config["params"]["Task"][
            "action_range"][:2] = args_cli.action_range

    save_config["params"]["real_eval_mode"] = args_cli.real_eval_mode

    # if not args_cli.enable_camera:
    # save_config["params"]["Camera"]["initial"] = False
    env = setup_env(args_cli, save_config)
    env.reset()

    if args_cli.save_path is not None:

        args_cli.save_path += "/" + object_name

    rl_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        eval_mode=True,
        collect_mode=True,
    )

    # wrap around environment for stable baselines
    rl_agent_env = Sb3VecEnvWrapper(rl_env, args_cli=args_cli)

    # create agent from stable baselines

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    if "zip" in checkpoint_path:
        if args_cli.rl_type == "ppo":

            agent = PPO.load(checkpoint_path,
                             rl_agent_env,
                             print_system_info=True)
        elif args_cli.rl_type == "sac":
            agent = SAC.load(checkpoint_path,
                             rl_agent_env,
                             print_system_info=True)
    else:
        if args_cli.rl_type == "ppo":
            agent = PPO
        elif args_cli.rl_type == "sac":
            agent = SAC

    total_count = 0
    success_count = 0

    last_obs, _ = rl_env.reset()
    success_count = 0
    with torch.no_grad():
        while rl_env.eval_env.collector_interface.traj_count < args_cli.num_demos:
            succss_or_not = rl_env.eval_env.collect_data(agent)
            success_count += succss_or_not.sum().item()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
