# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""
"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
from isaaclab.utils.math import apply_delta_pose
import isaaclab.utils.math as math_utils

parser.add_argument("--teleop_device",
                    type=str,
                    default="keyboard",
                    help="Device for interacting with environment")

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
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args[
        "experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg


def pre_process_actions(env, delta_pose: torch.Tensor,
                        gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command

        hand_vel = torch.zeros(16, device=delta_pose.device)

        return torch.concat([hand_vel.unsqueeze(0),
                             delta_pose.unsqueeze(0)],
                            dim=1)


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    # create environment

    env = setup_env(args_cli, save_config)

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.01 * args_cli.sensitivity,
            rot_sensitivity=0.01 * args_cli.sensitivity)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.05 * args_cli.sensitivity)
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity,
            rot_sensitivity=0.1 * args_cli.sensitivity)
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(
            OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", env.reset)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5),
                           lookat=(0.6, 0, 0),
                           asset_name="viewer")
        ViewportCameraController(env, viewer)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
        )

    # add teleoperation key for env reset
    reset_stopped_teleoperation = False

    def reset_recording_instance():
        nonlocal reset_stopped_teleoperation
        reset_stopped_teleoperation = True

    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()
    default_ee_pose = torch.as_tensor([
        0.35,
        0.0,
        0.35,
        0.5,
        -0.5,
        0.5,
        -0.5,
    ]).unsqueeze(0)

    # default_ee_pose[:, 3:7] = math_utils.quat_mul(
    #     torch.as_tensor([[0.707, 0.0, -0.707, 0.0]]), default_ee_pose[:, 3:7])
    # default_ee_pose[:, 3:7] = math_utils.quat_mul(
    #     torch.as_tensor([[0.707, 0.707, 0.0, 0.0]]), default_ee_pose[:, 3:7])

    default_actions = torch.cat([torch.zeros((1, 16)), default_ee_pose],
                                dim=1).to(env.device)

    while simulation_app.is_running():
        env.reset()

        for i in range(30):
            env.step(default_actions)

        # env.scene["robot"]._data.joint_pos

        # ee_pose = default_actions[:,
        #                           -7:]  # env.scene["ee_pose"]._data.root_state_w.clone()[:, :7]
        ee_pose = env.scene["left_hand"]._data.root_state_w.clone()[:, :7]
        for i in range(200):
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            delta_pose = torch.as_tensor(delta_pose).to(
                env.device).unsqueeze(0)
            target_pos, target_rot = apply_delta_pose(ee_pose[:, :3],
                                                      ee_pose[:,
                                                              3:7], delta_pose)

            actions = default_actions.clone()
            ee_pose = torch.cat([target_pos, target_rot], dim=1)

            actions[:, -7:] = ee_pose.clone()

            # apply actions
            env.step(actions)

            if reset_stopped_teleoperation:
                env.reset()
                reset_stopped_teleoperation = False

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
