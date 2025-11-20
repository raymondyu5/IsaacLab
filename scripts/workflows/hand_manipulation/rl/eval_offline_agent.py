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

from scripts.sb3.rl_algo_wrapper import rl_parser

import zarr
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

import sys

sys.path.append("submodule/d3rlpy")

import d3rlpy
from scripts.workflows.utils.multi_datawrapper import list_zarr_files


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

    last_obs, _ = env.reset()
    agent = d3rlpy.load_learnable(
        "logs/d3rlpy_logs/TD3PlusBC_20251112141643/model_50000.d3")
    # simulate environment
    target_obs_key = [
        'right_ee_pose', 'right_hand_joint_pos',
        'right_manipulated_object_pose', 'right_object_in_tip',
        'right_target_object_pose', 'right_contact_obs'
    ]

    data_file = list_zarr_files(args_cli.load_path)

    observations = []
    actions = []
    rewards = []
    terminals = []

    for file in data_file:
        data = zarr.open(file, mode="r")  # to fix zarr file permission issue
        obs_key = list(data["data"].keys())

        per_obs = []

        for obs_key in target_obs_key:
            per_obs.append(data["data"][obs_key][:])
        per_obs = np.concatenate(per_obs, axis=-1)
        observations.append(per_obs)
        actions.append(np.array(data["data"]["actions"][:]))
        rewards.append(np.array(data["data"]["rewards"][:]))
        termin = np.array(data["data"]["dones"][:])
        termin[-1] = 1.0  # ensure last step is terminal
        terminals.append(termin)

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)

    max_action = np.max(actions, axis=0)
    min_action = np.min(actions, axis=0)
    # actions = (actions - min_action) / (max_action - min_action) * 2 - 1

    # actions = agent.predict(observations)
    # actions = (actions + 1) / 2 * (max_action - min_action) + min_action
    max_action = torch.as_tensor(max_action).to(device=env.device)
    min_action = torch.as_tensor(min_action).to(device=env.device)

    counter = 0
    while simulation_app.is_running():
        # run everything in inference mode
        while True:

            for i in range(20):
                last_obs, rewards, terminated, time_outs, extras = env.step(
                    torch.zeros(env.unwrapped.action_space.shape).to(
                        device=env.device))

            for i in range(160):
                per_obs = []

                for obs_key in target_obs_key:

                    per_obs.append(last_obs['policy'][obs_key])

                per_obs = torch.cat(per_obs, dim=-1).cpu().numpy()
                pred_actions = torch.as_tensor(
                    agent.predict(per_obs)).to(device=env.device)
                gt_action = torch.as_tensor(actions[counter]).to(
                    device=env.device).unsqueeze(0).repeat_interleave(
                        env.num_envs, dim=0)
                pred_actions = (pred_actions +
                                1) / 2 * (max_action - min_action) + min_action

                last_obs, rewards, terminated, time_outs, extras = env.step(
                    pred_actions)
                counter += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
