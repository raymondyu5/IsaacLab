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

from scripts.sb3.multiagent.multiagent_wrapper import MultiAgentWrapper

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")

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
    "--use_dict_obs",
    default=False,
)

parser.add_argument(
    "--vae_path",
    default="logs/grab_hand",
)

parser.add_argument(
    "--latent_dim",
    default=32,
    type=int,
)
parser.add_argument(
    "--use_relative_finger_pose",
    action="store_true",
    default=True,
)

parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)

parser.add_argument(
    "--rl_type",
    default="ippo",
)

parser.add_argument(
    "--share_policy",
    default=False,
)
parser.add_argument("--checkpoint",
                    type=str,
                    default=None,
                    help="Path to model checkpoint.")
# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.multiagent_sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

import os
import yaml
from box import Box
from scripts.sb3.multiagent.multiagent_wandb_callback import setup_wandb, WandbCallback
from scripts.workflows.hand_manipulation.env.rl_env.bimanual_rl_wrapper import BimanualRLDatawrapperEnv

from scripts.workflows.hand_manipulation.rl.wandb_datawrapper import WandbDataWrapper


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

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    agent_cfg = Box(yaml_data)
    agent_cfg[
        "seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg[
            "seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg[
            "n_steps"] * args_cli.num_envs
    agent_cfg.seed = agent_cfg["seed"]
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # =======================================================================
    # =======================================================================
    # =======================================================================

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs

    env = setup_env(args_cli, save_config)
    env.reset()
    for i in range(10):
        env.step(torch.as_tensor(env.action_space.sample() * 0.0).to("cuda"))
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    # =======================================================================
    # =======================================================================
    # =======================================================================
    # init the environment

    rl_env = BimanualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    rl_agent_env = Sb3VecEnvWrapper(rl_env, args_cli, gpu_buffer=False)

    # create agent from stable baselines

    agent = MultiAgentWrapper(rl_agent_env,
                              rl_agent_env.num_agents,
                              policy_arch,
                              args_cli,
                              eval=True,
                              eval_per_step=False,
                              **agent_cfg)
    agent.load(args_cli.checkpoint)

    while simulation_app.is_running():
        # reset environment
        obs = rl_agent_env.reset()
        agent.prepare_rollout()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
