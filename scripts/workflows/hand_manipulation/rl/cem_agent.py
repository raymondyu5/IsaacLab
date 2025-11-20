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
from scripts.sb3.rl_algo_wrapper import initalize_rl_env, rl_parser

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

# launch omniverse app
AppLauncher.add_app_launcher_args(rl_parser)
args_cli, hydra_args = rl_parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    return gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None).unwrapped


from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper
from scripts.workflows.hand_manipulation.env.rl_env.cem_latent import OnlineCEM


def main():
    """Zero actions agent with Isaac Lab environment."""

    # =======================================================================

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["rl_train"] = True
    save_config["params"]["sample_points"] = True
    if args_cli.target_object_name is not None:
        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    if "Joint-Rel" in args_cli.task:
        args_cli.log_dir = os.path.join(args_cli.log_dir, "joint_pose")
    if args_cli.action_range is not None:
        save_config["params"]["Task"][
            "action_range"][:2] = args_cli.action_range
    save_config["params"]["real_eval_mode"] = args_cli.real_eval_mode

    # if args_cli.rl_type == "sac":
    #     save_config["params"]["reward_scale"] = 50.0
    env = setup_env(args_cli, save_config)
    obs, _ = env.reset()

    for i in range(10):
        if "Rel" in args_cli.task:

            env.step(
                torch.as_tensor(env.action_space.sample()).to(env.device) *
                0.0)
        else:
            init_ee_pose = torch.as_tensor([[0.50, 0.0, 0.5, 0, 1, 0, 0]]).to(
                env.device).repeat_interleave(args_cli.num_envs, dim=0)
            final_ee_pose = torch.cat(
                [init_ee_pose,
                 torch.zeros((env.num_envs, 16)).to(env.device)],
                dim=-1)
            env.step(final_ee_pose)
    env.reset()

    # reset environment

    # =======================================================================
    # =======================================================================
    # =======================================================================
    # init the environment

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")

        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    raw_action_space = gym.spaces.Box(
        low=-2 * np.pi,
        high=2 * np.pi,
        shape=(env.action_space.shape[-1], ),
        dtype=env.action_space.dtype,
    )

    rl_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        use_joint_pose=True if "Joint-Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    rl_agent_env = Sb3VecEnvWrapper(rl_env,
                                    gpu_buffer=False,
                                    args_cli=args_cli)

    setattr(rl_agent_env, "raw_action_space", raw_action_space)

    if args_cli.action_framework == "diffusion":
        setattr(rl_agent_env, "diffusion_obs_space",
                rl_agent_env.env.wrapper.diffusion_obs_space)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {rl_agent_env.observation_space}")
    print(f"[INFO]: Gym action space: {rl_agent_env.action_space}")

    initalize_rl_env(args_cli, rl_agent_env, args_cli.log_dir)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
