# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting  import save_params_to_yaml, parser

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
parser.add_argument("--algo_name",
                    type=str,
                    default="bc",
                    help="RL Policy training iterations.")
# launch omniverse app
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw

from scripts.workflows.open_policy.task.rl_replay_datawrapper import RLReplayDatawrapper

import imageio
import os
from isaaclab.utils .dict import print_dict
from isaaclab.utils .io import dump_pickle, dump_yaml


from isaaclab_tasks.utils.hydra import hydra_task_config
from scripts.workflows.open_policy.utils.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import imageio
import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback

import random
import datetime

from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv
from scripts.sb3.ppo_bc import PPOBC as PPO

import sys

sys.path.append("submodule/d3rlpy")

import d3rlpy
from d3rlpy.logging.logger import AlgProtocol
from d3rlpy.logging.wandb_adapter import WanDBAdapter
from d3rlpy.logging import WanDBAdapterFactory
from d3rlpy.models.encoders import EncoderFactory, PixelEncoderFactory, VectorEncoderFactory, DefaultEncoderFactory, SimBaEncoderFactory, register_encoder_factory, make_encoder_field

from d3rlpy.algos import SACConfig, BCConfig, IQLConfig


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
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    reset environment
    collect buffer data

    openvla_rl_env = Sb3VecEnvWrapper(env, gpu_buffer=False)
    openvla_replay_env = RLReplayDatawrapper(
        openvla_rl_env,
        save_config,
        raw_args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # openvla_replay_env.step()
    # openvla_replay_env.test_demo()

    # d3rlpy_dataset = d3rlpy.dataset.MDPDataset(
    #     observations=openvla_replay_env.rollout_buffer.observations,
    #     actions=openvla_replay_env.rollout_buffer.actions,
    #     rewards=openvla_replay_env.rollout_buffer.rewards,
    #     terminals=openvla_replay_env.rollout_buffer.terminates,
    # )
    
    bc = BCConfig(
        compile_graph=True,
        batch_size=2048,
        encoder_factory=DefaultEncoderFactory(
            hidden_units=[256, 128, 64], dropout_rate=0.2),
    ).create(device="cuda:0")
    import pdb
    pdb.set_trace()

   
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
