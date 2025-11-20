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

parser.add_argument(
    "--diffusion_path",
    default=None,
)

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

parser.add_argument(
    "--eval_mode",
    type=str,
    default="close_loop",  # Options: open_loop, close_loop, replay
)

parser.add_argument(
    "--diffusion_checkpoint",
    type=str,
    default="latest",  # Options: open_loop, close_loop, replay
)

parser.add_argument(
    "--random_camera_pose",
    action="store_true",
)

parser.add_argument(
    "--action_framework",
    default="pcd_diffusion",  # Options: diffusion, pca, vae, reactive_vae
)

parser.add_argument(
    "--target_object_name",
    type=str,
    default=None,  # Options: tomato_soup_can, banana, cereal_box, etc.
)

##### rl scratch agent args #####

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")

parser.add_argument(
    "--rl_type",
    default="ppo",
)

parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)
parser.add_argument(
    "--use_residual_action",
    action="store_true",
)

parser.add_argument(
    "--use_chunk_action",
    action="store_true",
)
parser.add_argument(
    "--use_interpolate_chunk",
    action="store_true",
)

parser.add_argument(
    "--resume",
    action="store_true",
)

parser.add_argument(
    "--checkpoint",
    default=None,
)

parser.add_argument(
    "--use_base_action",
    action="store_true",
)

parser.add_argument(
    "--use_visual_obs",
    action="store_true",
)

# launch omniverse app
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.bimanual_rl_wrapper import BimanualRLDatawrapperEnv
from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
from scripts.workflows.hand_manipulation.real_robot.finetune.sim.env.residual_rl_wrapper import ResidualRLDatawrapperEnv
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

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli,
                                              args_cli.log_dir,
                                              random_shuffle=False)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["real_eval_mode"] = True
    save_config["params"]["residual_rew"] = True

    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]

    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose

    save_config["params"]["sample_points"] = True

    if args_cli.action_framework in ["pcd_diffusion"]:
        save_config["params"]["Camera"][
            "extract_rgb"] = False  # disable initial camera pose
    if args_cli.action_framework in ["image_diffusion"]:
        save_config["params"]["Camera"]["extract_seg_pc"] = False
        save_config["params"]["Camera"]["extract_rgb"] = True

    env = setup_env(args_cli, save_config)
    env.reset()

    rl_env = ResidualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        share_obs_encoder=False)

    # wrap around environment for stable baselines
    rl_agent_env = Sb3VecEnvWrapper(rl_env)

    # create agent from stable baselines

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    if "zip" in checkpoint_path:
        if args_cli.rl_type == "ppo":

            agent = PPO.load(checkpoint_path,
                             rl_agent_env,
                             print_system_info=True)
        elif args_cli.rl_type == "sac":
            agent = SAC.load(checkpoint_path,
                             rl_agent_env,
                             print_system_info=True)
    else:
        if args_cli.rl_type == "ppo":
            agent = PPO
        elif args_cli.rl_type == "sac":
            agent = SAC
    total_count = 0
    success_count = 0

    last_obs, _ = rl_env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode

        with torch.inference_mode():

            if "zip" in checkpoint_path:
                success_or_not, _ = rl_env.eval_checkpoint(agent)

            else:
                rl_env.warmup(agent, bc_eval=True)
                rl_env.init_eval_result_folder()
                success_or_not = rl_env.eval_all_checkpoint(
                    agent, rl_agent_env)

            success_count += success_or_not.sum().item()
            total_count += env.num_envs
            print("Success rate: ", success_count / total_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
