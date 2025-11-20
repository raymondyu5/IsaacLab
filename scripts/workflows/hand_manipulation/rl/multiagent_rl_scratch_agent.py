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
from scripts.sb3.IPPO import IPPO

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
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
    default=True,
)

parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)
parser.add_argument("--max_iterations",
                    type=int,
                    default=1000,
                    help="RL Policy training iterations.")
# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.multiagent.multiagent_wrapper import MultiAgentEnvWrapper
from scripts.multiagent.marl.runner import Runner

import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
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


def process_MultiAgentRL(env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    # on policy marl
    from scripts.multiagent.marl.runner import Runner
    marl = Runner(vec_env=env, config=config, model_dir=model_dir)

    return marl


def main():
    """Zero actions agent with Isaac Lab environment."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    agent_cfg = yaml_data
    agent_cfg["experiment_name"] = args_cli.task
    agent_cfg["run_dir"] = args_cli.log_dir
    agent_cfg["seed"] = args_cli.seed
    if args_cli.add_right_hand and args_cli.add_left_hand:
        agent_cfg["num_agents"] = 2
    else:
        agent_cfg["num_agents"] = 1

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs

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

    rl_env = BimanualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    rl_agent_env = MultiAgentEnvWrapper(rl_env, args_cli)

    agent = process_MultiAgentRL(rl_agent_env, agent_cfg)

    # train the agent
    agent.run(num_learning_iterations=args_cli.max_iterations)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
