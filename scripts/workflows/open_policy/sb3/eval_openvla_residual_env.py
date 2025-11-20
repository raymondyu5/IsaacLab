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

from stable_baselines3 import PPO
from scripts.sb3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
import sys

sys.path.append("submodule/robomimic_openrt")
import robomimic.utils.file_utils as FileUtils

parser.add_argument("--video",
                    action="store_true",
                    default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length",
                    type=int,
                    default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval",
                    type=int,
                    default=2000,
                    help="Interval between video recordings (in steps).")

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")
parser.add_argument("--checkpoint",
                    type=str,
                    default=None,
                    help="Path to model checkpoint.")
parser.add_argument("--rl_type", type=str, default="ppo", help="rl_type")

parser.add_argument("--base_policy", default=None, help="checkpoint path")
# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv

from scripts.workflows.open_policy.utils.sb3_residual_wrapper import Sb3VecEnvWrapper
import imageio
import os


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    # parse configuration

    checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)
    env.reset()

    base_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.base_policy, device=args_cli.device, verbose=True)
    base_policy.start_episode()
    base_policy.policy.set_eval()

    openvla_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    openvla_rl_env = Sb3VecEnvWrapper(openvla_env, base_policy)

    # create agent from stable baselines

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    if args_cli.rl_type == "ppo":
        agent = PPO.load(checkpoint_path,
                         openvla_rl_env,
                         print_system_info=True)
    elif args_cli.rl_type == "sac":
        agent = SAC.load(checkpoint_path,
                         openvla_rl_env,
                         print_system_info=True)
    total_count = 0
    success_count = 0

    last_obs, _ = openvla_env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            success_or_not = openvla_env.eval_residual_checkpoint(
                openvla_rl_env, agent, last_obs)

            total_count += len(success_or_not)
            success_count += success_or_not.sum().item()
            print("Success rate: ", success_count / total_count)

            if openvla_env.collector_interface is not None:
                if openvla_env.collector_interface.is_stopped():
                    break
    if openvla_env is not None:
        openvla_env.collector_interface.close()

    # close the simulator

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
