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
from scripts.sb3.rfs.DSRLTD3 import DSRLTD3
from scripts.sb3.rl_algo_wrapper import rl_parser
from scripts.sb3.rfs.DSRL import DSRL

rl_parser.add_argument(
    "--render_video",
    action="store_true",
)
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


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg,
                    render_mode="rgb_array").unwrapped


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
    save_config["params"]["real_eval_mode"] = args_cli.real_eval_mode
    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    if args_cli.action_range is not None:
        save_config["params"]["Task"][
            "action_range"][:2] = args_cli.action_range

    env = setup_env(args_cli, save_config)
    env.reset()

    rl_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        eval_mode=True,
    )

    # wrap around environment for stable baselines
    rl_agent_env = Sb3VecEnvWrapper(rl_env,
                                    args_cli=args_cli,
                                    gpu_buffer=False)

    # create agent from stable baselines

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")

    if args_cli.rl_type == "ppo":
        agent = PPO
    elif args_cli.rl_type == "sac":
        agent = SAC
    elif args_cli.rl_type == "td3":
        agent = TD3
    elif args_cli.rl_type == "dsrltd3":
        agent = DSRLTD3
    elif args_cli.rl_type == "dsrl":
        agent = DSRL
    if "zip" in checkpoint_path:
        if args_cli.rl_type in ["dsrltd3", "dsrl"]:

            agent = agent.load(
                checkpoint_path,
                rl_agent_env,
                print_system_info=False,
                diffusion_model=rl_agent_env.env.wrapper.diffusion_model,
                diffusion_obs_space=rl_agent_env.env.wrapper.
                diffusion_obs_space,
                diffusion_action_space=rl_agent_env.env.wrapper.
                diffusion_action_space,
            )
        else:
            agent = agent.load(checkpoint_path,
                               rl_agent_env,
                               print_system_info=False)

    total_count = 0
    success_count = 0

    last_obs, _ = rl_env.reset()
    for _ in range(10):

        fname = env.render()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode

        with torch.inference_mode():
            if "zip" in checkpoint_path:
                success_or_not, _ = rl_env.eval_checkpoint(agent)
                # success_or_not = rl_env.eval_symmetry(agent, last_obs)
            else:
                rl_env.init_eval_result_folder()
                success_or_not = rl_env.eval_all_checkpoint(
                    agent, last_obs, rl_agent_env)
                break

            total_count += env.num_envs
            success_count += success_or_not.sum().item()
            print("Success rate: ", success_count / total_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
