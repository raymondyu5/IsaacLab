# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument(
    "--unnorm_key",
    default="bridge_ori",
)
parser.add_argument(
    "--robot_type",
    default="franka",
)
parser.add_argument(
    "--base_policy",
    default="openpi",
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

from scripts.workflows.open_policy.task.openvla_env import OpenVlaEvalEnv

from scripts.workflows.utils.client.OpenPi_client import OpenPiClient

from scripts.workflows.utils.client.openvla_localclient import OpenVLAClient
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
    if args_cli.base_policy == "openvla":
        inference_client = OpenVLAClient(args_cli.client_url)
        prompt = "In: What action should the robot take to pick up the bottle?\nOut:"
    elif args_cli.base_policy == "openpi":
        inference_client = OpenPiClient()
        prompt = "pick up the mug"

    openval_env = OpenVlaEvalEnv(
        env.env,
        save_config,
        args_cli=args_cli,
        prompt=prompt,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        inference_client=inference_client,
        render_camera=save_config["params"]["Task"]["render_camera"])
    count = 0
    success_count = 0

    # simulate environment
    for count in range(10000):
        # gs_env.step()
        last_obs = openval_env.reset()
        if save_config["params"]["Task"]["render_camera"]:
            os.makedirs(f"{args_cli.log_dir}/video", exist_ok=True)
            video_name = f"{args_cli.log_dir}/video/{count}"
        else:
            video_name = None
        success_or_not = openval_env.step(last_obs, video_name=video_name)
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
