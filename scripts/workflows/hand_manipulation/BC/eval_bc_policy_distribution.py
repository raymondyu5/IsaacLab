# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser
import os
import numpy as np

rl_parser.add_argument(
    "--exploration_type",
    type=str,
    default="evenly",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(rl_parser)
# parse the arguments
args_cli = rl_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

import imageio


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.workflows.hand_manipulation.env.bc_env.bc_distribution_env import BCDistributionEnv


def main():
    """Zero actions agent with Isaac Lab environment."""
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
    save_config["params"]["action_framework"] = args_cli.action_framework
    save_config["params"]["eval_mode"] = args_cli.real_eval_mode

    if args_cli.action_framework == "state_diffusion":
        # save_config["params"]["eval_mode"] = True
        save_config["params"]["Camera"][
            "initial"] = False  # disable initial camera pose

    save_config["params"]["real_eval_mode"] = args_cli.real_eval_mode
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    obs, _ = env.reset()

    policy_env = BCDistributionEnv(env, save_config, args_cli)

    result_path = os.path.join(
        args_cli.log_dir, "eval_results", args_cli.target_object_name
        if args_cli.target_object_name is not None else "all")
    os.makedirs(result_path, exist_ok=True)
    policy_env.eval_policy_distribution(result_path)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
