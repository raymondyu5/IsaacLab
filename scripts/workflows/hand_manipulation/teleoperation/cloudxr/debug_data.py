# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""
"""Launch Isaac Sim Simulator first."""

import argparse

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
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import omni.log

from isaaclab.devices import OpenXRDevice, Se3Gamepad, Se3Keyboard, Se3SpaceMouse

from isaaclab.devices.openxr import XrCfg

teleop_interface = OpenXRDevice(XrCfg())
reset_stopped_teleoperation = False
teleoperation_active = True


# Callback handlers
def reset_recording_instance():
    """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
    global reset_stopped_teleoperation
    reset_stopped_teleoperation = True


def start_teleoperation():
    """Activate teleoperation control of the robot.

    This callback enables active control of the robot through the input device.
    It's typically triggered by a specific gesture or button press and is used when:
    - Beginning a new teleoperation session
    - Resuming control after temporarily pausing
    - Switching from observation mode to control mode

    While active, all commands from the device will be applied to the robot.
    """
    global teleoperation_active
    teleoperation_active = True


def stop_teleoperation():
    """Deactivate teleoperation control of the robot.

    This callback temporarily suspends control of the robot through the input device.
    It's typically triggered by a specific gesture or button press and is used when:
    - Taking a break from controlling the robot
    - Repositioning the input device without moving the robot
    - Pausing to observe the scene without interference

    While inactive, the simulation continues to render but device commands are ignored.
    """
    global teleoperation_active
    teleoperation_active = False


teleop_interface.add_callback("RESET", reset_recording_instance)
teleop_interface.add_callback("START", start_teleoperation)
teleop_interface.add_callback("STOP", stop_teleoperation)


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

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    teleop_data = []
    import copy

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            data = teleop_interface.advance()
            teleop_data.append(copy.deepcopy(data))

            actions = torch.rand(env.action_space.shape,
                                 device=env.unwrapped.device) * 2 - 1

            obs, rewards, terminated, time_outs, extras = env.step(actions *
                                                                   0.0)
            np.savez("teleop_data", teleop_data)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
