# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument(
    "--robot_name",
    default="leap",
)

parser.add_argument("--add_right_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--add_left_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")

parser.add_argument(
    "--save_data",
    action="store_true",
)
parser.add_argument(
    "--collision_checker",
    action="store_true",
)

parser.add_argument(
    "--data_type",
    default="dexycb",
)

parser.add_argument(
    "--evaluate_mode",
    default="close_loop",
)
parser.add_argument(
    "--ckpt_path",
    default=None,
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


from scripts.workflows.hand_manipulation.env.teleop_env.evaluation_env import EvaluationEnv


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    args_cli.ckpt_path = "submodule/robomimic_openrt/bc_trained_models/test/20250421132800/models/model_epoch_500.pth"
    # reset environment

    env.reset()

    DexDataEve_env = EvaluationEnv(
        args_cli,
        save_config,
        env,
    )
    success_count = 0
    total_count = 0

    while simulation_app.is_running():
        success = DexDataEve_env.run()
        total_count += 1
        if success is not None:
            if success:
                success_count += 1
            print("success:", success_count / total_count, total_count)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
