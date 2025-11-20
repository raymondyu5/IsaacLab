# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import pinocchio as pin
import numpy as np
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import cv2

parser.add_argument("--add_right_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--add_left_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--random_camera_pose",
    action="store_true",
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
import h5py
# from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp

# from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
# import isaaclab.utils.math as math_utils
import matplotlib.pyplot as plt
from tools.visualization_utils import *
import h5py


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
    save_config["params"]["real_eval_mode"] = True
    save_config["params"]["Camera"]["random_pose"] = True

    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()

    # Open the file in read-only mode

    action_buffer = {
        "delta_ee_pose": [],
        "ee_pose": [],
        "joint_pose": [],
        "init_joints_pose": [],
        "init_ee_pose": [],
    }

    with h5py.File('logs/master_chef_can.hdf5', 'r') as f:
        demo_keys = list(f["data"].keys())
        for demo_key in demo_keys:
            delta_ee_pose = np.array(
                f["data"][demo_key]["obs"]["delta_ee_control_action"])
            ee_pose = np.array(f["data"][demo_key]["obs"]["ee_control_action"])
            joint_pose = np.array(
                f["data"][demo_key]["obs"]["joint_control_action"])
            action_buffer["delta_ee_pose"].append(delta_ee_pose)
            action_buffer["ee_pose"].append(ee_pose)
            action_buffer["joint_pose"].append(joint_pose)
            action_buffer["init_joints_pose"].append(
                np.array(
                    f["data"][demo_key]["obs"]["right_hand_joint_pos"][0]))

    num_demos = len(demo_keys)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            action_data = action_buffer["delta_ee_pose"]

            for demo_id in range(num_demos):
                # compute zero actions
                env.reset()

                init_arm_qpos = action_buffer["init_joints_pose"][demo_id][
                    ..., :7].tolist()
                # init_arm_qpos = [
                #     0.04233345, -0.18839844, -0.25485227, -2.74782944,
                #     2.34452176, 2.09500456, -0.42667976
                # ]
                env.scene["right_hand"].root_physx_view.set_dof_positions(
                    torch.tensor(init_arm_qpos + [0] * 16,
                                 device=env.unwrapped.device).unsqueeze(
                                     0).repeat_interleave(env.num_envs, 1),
                    torch.arange(env.num_envs).to(env.device))
                for i in range(10):
                    obs, rewards, terminated, time_outs, extras = env.step(
                        torch.as_tensor(
                            action_buffer["ee_pose"][demo_id][0]).to(
                                env.device).unsqueeze(0))

                for act in action_data[demo_id][1:]:

                    cur_wrist_pose = env.scene[
                        "right_panda_link7"]._data.root_state_w[:, :7]
                    next_wrist_pose = math_utils.apply_delta_pose(
                        cur_wrist_pose[:, :3], cur_wrist_pose[:, 3:7],
                        torch.as_tensor(act).to(env.device).unsqueeze(0))
                    action_wrist = torch.cat([
                        next_wrist_pose[0], next_wrist_pose[1],
                        torch.as_tensor(act[6:]).to(env.device).unsqueeze(0)
                    ],
                                             dim=-1)
                    obs, rewards, terminated, time_outs, extras = env.step(
                        action_wrist)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
