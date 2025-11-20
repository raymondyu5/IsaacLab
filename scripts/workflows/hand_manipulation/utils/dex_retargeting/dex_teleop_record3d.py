# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
from scripts.workflows.hand_manipulation.env.teleop_env.dexrecord3d import DexRRecord3detargeting
from scripts.workflows.hand_manipulation.utils.hand_detector.hand_monitor import Record3DSingleHandMotionControl

parser.add_argument(
    "--robot_name",
    default="leap",
)

parser.add_argument("--dexycb_filename",
                    default=None,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--free_hand",
    action="store_true",
    default=False,
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

    motion_control = Record3DSingleHandMotionControl(hand_mode="right_hand",
                                                     show_hand=True)
    while True:
        success, motion_data = motion_control.step()

        if success:
            break

    dexteleop_env = DexRRecord3detargeting(args_cli, save_config, env,
                                           motion_control)

    while simulation_app.is_running():

        dexteleop_env.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
