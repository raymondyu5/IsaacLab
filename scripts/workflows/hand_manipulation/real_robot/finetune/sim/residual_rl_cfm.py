# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import sys

sys.path.append("submodule/stable-baselines3")

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser

import os
import numpy as np

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(rl_parser)
# parse the arguments
args_cli = rl_parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None).unwrapped


import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg

import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
from scripts.workflows.hand_manipulation.real_robot.finetune.sim.env.residual_rl_wrapper import ResidualRLDatawrapperEnv
from scripts.sb3.rl_algo_wrapper import initalize_rl_env


def main():
    """Zero actions agent with Isaac Lab environment."""
    # =======================================================================
    # =======================================================================

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

    save_config["params"]["real_eval_mode"] = True
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    #### set up the environment for RL training ####
    # =======================================================================
    # =======================================================================
    if args_cli.video:
        video_kwargs = {
            "video_folder":
            os.path.join(args_cli.diffusion_path, "videos", "train"),
            "step_trigger":
            lambda step: step % args_cli.video_interval == 0,
            "video_length":
            args_cli.video_length,
            "disable_logger":
            True,
        }
        print("[INFO] Recording videos during training.")

        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    rl_env = ResidualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        use_residual=args_cli.use_residual_action,
    )

    env.reset()

    for i in range(10):
        action = torch.as_tensor(env.action_space.sample()).to(
            env.unwrapped.device) * 0.0
        env.step(action)
    env.reset()
    rl_agent_env = Sb3VecEnvWrapper(
        rl_env, gpu_buffer=False, concatenate_obs=not args_cli.use_visual_obs)

    initalize_rl_env(args_cli,
                     rl_agent_env,
                     args_cli.diffusion_path,
                     rl_env,
                     wamrup=True,
                     model_save_freq=2)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
