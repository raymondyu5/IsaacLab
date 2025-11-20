# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--seed",
                    type=int,
                    default=100,
                    help="Seed used for the environment")

# Define arguments
parser.add_argument("--offline_ratio",
                    type=float,
                    default=0.001,
                    help="Offline ratio")
parser.add_argument("--log_interval",
                    type=int,
                    default=100,
                    help="Logging interval")
parser.add_argument("--batch_size",
                    type=int,
                    default=6000,
                    help="Mini batch size")
parser.add_argument("--max_steps",
                    type=int,
                    default=int(1e6),
                    help="Number of training steps")
parser.add_argument("--start_training",
                    type=int,
                    default=int(20),
                    help="Training start step")
parser.add_argument("--pretrain_steps",
                    type=int,
                    default=0,
                    help="Number of offline updates")
parser.add_argument("--utd_ratio",
                    type=int,
                    default=20,
                    help="Update-to-data ratio")

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

from scripts.workflows.open_policy.task.rl_replay_datawrapper import RLReplayDatawrapper

from isaaclab_tasks.utils.hydra import hydra_task_config
from scripts.workflows.open_policy.utils.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv
from scripts.sb3.ppo_bc import PPOBC as PPO
from scripts.sb3.rlpd_jax import RLPDWrapper


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

    env.reset()

    openvla_rl_env = Sb3VecEnvWrapper(env, gpu_buffer=False)

    openvla_replay_env = RLReplayDatawrapper(
        openvla_rl_env,
        save_config,
        raw_args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        use_all_data=True)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    # collect buffer data
    openvla_replay_env.step()
    openvla_replay_env.test_demo()

    # reinstantiate the environment

    openvla_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
    wandb_run = setup_wandb(args_cli, "rlpd", tags=None, project="isaaclab")
    openvla_rl_env = Sb3VecEnvWrapper(openvla_env, gpu_buffer=False)

    agent = RLPDWrapper(args_cli, openvla_rl_env,
                        openvla_replay_env.rollout_buffer, args_cli.seed)

    agent.learn(wandb_run)

    # save the final model

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
