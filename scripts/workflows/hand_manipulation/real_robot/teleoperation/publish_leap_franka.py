# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""
"""Launch Isaac Sim Simulator first."""
import socket
from isaaclab.app import AppLauncher

from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--teleop_device",
                    type=str,
                    default="dualhandtracking_abs",
                    help="Device for interacting with environment")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity",
                    type=float,
                    default=1.0,
                    help="Sensitivity factor.")

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
import pinocchio

app_launcher_args = vars(args_cli)

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app
"""Rest everything follows."""

from scripts.workflows.hand_manipulation.real_robot.teleoperation.real_leap_wrapper import RealLeapWrapper, send_command_to_vision_pro
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
    save_config["params"]["use_teleop"] = True
    save_config["params"]["real_eval_mode"] = True
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()

    se3_wrapper = RealLeapWrapper(env, save_config, args_cli)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            try:

                se3_wrapper.step()
            except:

                send_command_to_vision_pro(
                    ip_address=se3_wrapper.avp_ip,
                    port=se3_wrapper.port,
                    command=
                    "Encountered error and need to debug. Stopping teleoperation."
                )
                import pdb
                pdb.set_trace()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
