# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

parser.add_argument(
    "--action_framework",
    default="vae",
)

parser.add_argument(
    "--vae_path",
    default=None,
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
import pinocchio

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app
"""Rest everything follows."""
from scripts.workflows.hand_manipulation.env.teleop_env.replay_multi_teleoperation_wrapper import ReplayMultiTeleopFrankaLeapWrapper
import gymnasium as gym


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


import torch


def main():
    """Zero actions agent with Isaac Lab environment."""

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    # save_config["params"]["use_teleop"] = True
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()

    replay_wrapper = ReplayMultiTeleopFrankaLeapWrapper(
        env, save_config, args_cli)

    replay_wrapper.step_env()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
