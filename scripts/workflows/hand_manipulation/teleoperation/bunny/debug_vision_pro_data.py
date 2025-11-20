# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import numpy as np

parser.add_argument("--add_right_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--add_left_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
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

from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp

init_jpose = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "j0",
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "j7",
    "j8",
    "j9",
    "j10",
    "j11",
    "j12",
    "j13",
    "j14",
    "j15",
]


def robot_setting(env, robot, hand="left"):

    robot_joint_names = []

    for action_name in env.action_manager._terms.keys():
        if hand in action_name:
            control_joint_names = env.action_manager._terms[
                action_name].cfg.joint_names

            robot_joint_names += robot.find_joints(control_joint_names)[1]

    retarget2isaac = torch.as_tensor(
        [init_jpose.index(joint) for joint in robot_joint_names],
        dtype=torch.int32).to(env.device)

    return retarget2isaac


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
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    index = 0
    robot = env.scene["right_hand"]

    play_mode = False if "Play" not in args_cli.task else True

    data = np.load(
        "/home/ensu/Documents/weird/IsaacLab/submodule/BunnyVisionPro/teleop_data.npy",
        allow_pickle=True).item()

    num_data = len(data["right"])
    left_robot = env.scene["left_hand"]
    right_robot = env.scene["right_hand"]

    env_ids = torch.tensor([0], device=env.unwrapped.device)

    if play_mode:

        retarget2isaac = robot_setting(env, robot, hand="left")
    else:
        init_joint_names = save_config["params"]["grasper"]["joint_names"]
        retarget2isaac = torch.as_tensor(
            [init_joint_names.index(joint) for joint in robot.joint_names],
            dtype=torch.int32).to(env.device)

    while simulation_app.is_running():
        for i in range(num_data):

            left_action = torch.as_tensor(
                data["left"][i], device=env.unwrapped.device)[retarget2isaac]
            right_action = torch.as_tensor(
                data["right"][i], device=env.unwrapped.device)[retarget2isaac]

            # compute zero actions
            actions = torch.cat([left_action, right_action],
                                dim=0).unsqueeze(0)

            # if not play_mode:
            #     actions = torch.zeros(env.action_space.shape,
            #                           device=env.unwrapped.device)

            #     left_robot.root_physx_view.set_dof_positions(
            #         left_action.unsqueeze(0), indices=env_ids)
            #     right_robot.root_physx_view.set_dof_positions(
            #         right_action.unsqueeze(0), indices=env_ids)

            obs, rewards, terminated, time_outs, extras = env.step(actions)

            index += 1

        # env.scene["right_arm_ee"].root_physx_view.set_dof_positions()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
