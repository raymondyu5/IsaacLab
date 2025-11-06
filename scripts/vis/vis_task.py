#!/usr/bin/env python3
"""Simple script to visualize the Franka+Leap environment.

Usage:
    # With GUI (to see it live):
    ./isaaclab.sh -p scripts/visualize_franka_leap.py --task Isaac-Lift-Cube-Franka-Leap-v0

    # Headless with video recording:
    ./isaaclab.sh -p scripts/visualize_franka_leap.py --task Isaac-Lift-Cube-Franka-Leap-v0 --headless --enable_cameras --video --video_length 200

    # YCB objects:
    ./isaaclab.sh -p scripts/visualize_franka_leap.py --task Isaac-Lift-YCB-Franka-Leap-v0
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""
"""./isaaclab.sh -p -m torch.distributed.run --nnodes=1 --nproc_per_node=1 scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 --num_envs 4096 --headless --distributed"""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser = argparse.ArgumentParser(description="Visualize Franka+Leap environment")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Leap-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--video", action="store_true", default=False, help="Record videos")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (steps)")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between videos (steps)")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def main():
    """Visualize the environment with random actions."""
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    # Create environment (with render mode if recording video)
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    print("\n" + "="*80)
    print(f"Visualizing: {args_cli.task}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("="*80 + "\n")

    # Setup video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", args_cli.task),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"Recording videos to: {video_kwargs['video_folder']}")
        print(f"Video length: {args_cli.video_length} steps")
        print(f"Video interval: {args_cli.video_interval} steps\n")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Reset environment
    obs, _ = env.reset()

    # Get the underlying environment's device
    device = env.unwrapped.device

    # Run simulation with random actions
    print("Running visualization with random actions...")
    print("Press Ctrl+C to stop\n")

    step_count = 0
    episode_count = 0

    try:
        while simulation_app.is_running():
            # Sample random action (convert to torch tensor)
            action = torch.from_numpy(env.action_space.sample()).to(device)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Reset if episode ended
            if terminated.any() or truncated.any():
                episode_count += 1
                print(f"Episode {episode_count} finished after {step_count} steps")
                obs, _ = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user")

    # Close environment
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# # Create argparser

# # Append AppLauncher args (this adds --headless and other flags automatically)
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# set_cluster_graphics_vars()

# # Launch app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import os

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


