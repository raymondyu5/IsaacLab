# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--checkpoint", default=None, help="checkpoint path")

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

parser.add_argument("--mode", default="replay", help="mode for robot eval")
parser.add_argument("--model_type", default="bc", help="mode for robot eval")
parser.add_argument("--eval", action="store_true", help="mode for robot eval")
parser.add_argument(
    "--robot_type",
    default="franka",
)

parser.add_argument(
    "--use_time",
    action="store_true",
    default=False,
)

# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym

from scripts.workflows.open_policy.task.bc_openvla_wrapper import BCDatawrapperOpenPolicy
import torch
import sys
from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv

sys.path.append("submodule/robomimic_openrt")
import robomimic.utils.file_utils as FileUtils


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    """Zero actions agent with Isaac Lab environment."""

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    openvla_env = BCDatawrapperOpenPolicy(
        env.env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )
    if args_cli.mode not in ["replay", "replay_normalized"]:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args_cli.checkpoint,
            device=args_cli.device,
            verbose=True)
        policy.start_episode()

        policy.policy.set_eval()
    else:
        policy = None

    if args_cli.eval:
        rl_env = RLDatawrapperEnv(
            env,
            save_config,
            args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
        )
    else:
        rl_env = None

    success_count = 0
    total_count = 0

    while True:
        with torch.no_grad():

            if args_cli.mode in ["replay", "replay_normalized"]:
                success = openvla_env.replay_policy(policy)
            elif args_cli.mode == "open_loop":
                success = openvla_env.open_loop_policy(policy)
            elif args_cli.mode == "close_loop":
                success = openvla_env.close_loop_policy(policy, rl_env)

            success_count += success.sum().item()
            total_count += success.shape[0]
            print("success count: ", success_count / total_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
