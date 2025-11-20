# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import sys

sys.path.append("submodule/robomimic_openrt")
import robomimic.utils.file_utils as FileUtils

parser.add_argument("--checkpoint", default=None, help="checkpoint path")
parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)
parser.add_argument("--mode", default="replay", help="mode for robot eval")
parser.add_argument("--model_type", default="bc", help="mode for robot eval")
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
from scripts.workflows.hand_manipulation.utils.cloudxr.evaluate_franka_leap_wrapper import EvaluateFrankaLeapWrapper
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

    # create environment

    env = setup_env(args_cli, save_config)

    if args_cli.mode not in ["replay", "replay_normalized"]:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args_cli.checkpoint,
            device=args_cli.device,
            verbose=True)
        policy.start_episode()

        policy.policy.set_eval()
    else:
        policy = None

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()

    replay_wrapper = EvaluateFrankaLeapWrapper(env, save_config, args_cli)

    while simulation_app.is_running():

        if args_cli.mode in ["replay", "replay_normalized"]:
            success = replay_wrapper.replay()
        elif args_cli.mode == "open_loop":
            success = replay_wrapper.open_loop_policy(policy)
        elif args_cli.mode == "close_loop":
            success = replay_wrapper.close_loop_policy(policy)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
