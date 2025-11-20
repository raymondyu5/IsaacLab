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

from scripts.sb3.sb3_rlpd import SACRLPD

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")
parser.add_argument(
    "--use_buffer",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--use_buffer_only",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
)

parser.add_argument(
    "--offline_ratio",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--use_all_data",
    action="store_true",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=10,
)
parser.add_argument("--warmup_steps", default=1000000, type=int)
# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv
from scripts.workflows.open_policy.task.rl_replay_datawrapper import RLReplayDatawrapper

from isaaclab.utils.io import dump_pickle, dump_yaml

from scripts.workflows.open_policy.utils.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import imageio
import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
import sys

sys.path.append("submodule/stable-baselines3")

import sys


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

    # Load YAML file
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    # Convert to Box
    agent_cfg = Box(yaml_data)

    agent_cfg[
        "seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg[
            "seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg[
            "n_steps"] * args_cli.num_envs

    #set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    agent_cfg.seed = agent_cfg["seed"]
    # agent_cfg.sim.device = args_cli.device if args_cli.device is not None else "cpu"

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # dump the configuration into log-directory

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)
    env.reset()

    openvla_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    if args_cli.use_buffer:
        openvla_rl_env = Sb3VecEnvWrapper(env, gpu_buffer=False)

        openvla_replay_env = RLReplayDatawrapper(
            openvla_rl_env,
            save_config,
            raw_args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
            use_all_data=args_cli.use_all_data)
        openvla_replay_env.step()
        openvla_replay_env.test_demo(openvla_env)
        gpu_buffer = openvla_replay_env.rollout_buffer
    else:
        gpu_buffer = None

    # reset environment

    # wrap around environment for stable baselines
    openvla_rl_env = Sb3VecEnvWrapper(openvla_env, gpu_buffer=False)

    # create agent from stable baselines

    wandb_run = setup_wandb(args_cli,
                            "sb3_rlpd",
                            tags=None,
                            project="isaaclab")

    if args_cli.use_buffer_only:
        env.close()

    agent = SACRLPD(policy_arch,
                    openvla_rl_env,
                    verbose=1,
                    gpu_buffer=gpu_buffer,
                    use_buffer_only=args_cli.use_buffer_only,
                    warmup_steps=args_cli.warmup_steps,
                    n_critic=args_cli.n_critic,
                    offline_ratio=args_cli.offline_ratio,
                    **agent_cfg)

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        callback=WandbCallback(
            model_save_freq=3000,
            model_save_path=str(args_cli.log_dir + "/rlpd"),
            eval_env_fn=None,
            eval_freq=10,
            eval_cam_names=None,
        ),
    )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()
    wandb_run.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
