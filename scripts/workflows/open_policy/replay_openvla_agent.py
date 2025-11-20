# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument(
    "--robot_type",
    type=str,
    default="franka",
)

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

from scripts.workflows.open_policy.task.replay_datawrapper_bridge import ReplayDatawrapperBridge
from scripts.workflows.open_policy.task.replay_datawrapper_droid import ReplayDatawrapperDroid

import imageio
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
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    if args_cli.robot_type == "widowx":
        openvla_replay_env = ReplayDatawrapperBridge(
            env.env,
            save_config,
            args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
        )
    elif args_cli.robot_type == "franka":
        openvla_replay_env = ReplayDatawrapperDroid(
            env.env,
            save_config,
            args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
        )
    count = 0
    success_count = 0

    # simulate environment
    for count in range(openvla_replay_env.num_collected_demo):
        # gs_env.step()
        last_obs = openvla_replay_env.reset()
        if save_config["params"]["Task"]["render_camera"]:
            os.makedirs(f"{args_cli.log_dir}/replay_video", exist_ok=True)
            video_name = f"{args_cli.log_dir}/replay_video/{count}"
        else:
            video_name = None
        success_or_not = openvla_replay_env.step(last_obs,
                                                 video_name=video_name)
        count += 1
        if success_or_not:
            success_count += 1
        print("success count: ", success_count / count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
