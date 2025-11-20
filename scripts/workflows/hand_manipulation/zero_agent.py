# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

import numpy as np
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import initalize_rfs_env, rl_parser
import cv2

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(rl_parser)
# parse the arguments
args_cli = rl_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

# from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp

# from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
# import isaaclab.utils.math as math_utils
import matplotlib.pyplot as plt
from tools.visualization_utils import *

# from tools.trash.amazon.test_space_mouse import SpaceMouseExpert


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg,
                    render_mode="rgb_array").unwrapped


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    # save_config["params"]["use_teleop"] = True

    save_config["params"]["Camera"]["random_pose"] = True
    if args_cli.target_object_name is not None:
        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    # create environment
    save_config["params"]["real_eval_mode"] = True

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()

    for i in range(10):
        env.step(
            torch.as_tensor(env.action_space.sample()).to(env.device) * 0.0)
    env.reset()
    # sm = SpaceMouseExpert()
    # env.scene["right_articulated_object"].write_joint_position_limit_to_sim(
    #     torch.as_tensor([[-1, 0]]).to(env.device))

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # for act in actions:
            env.reset()

            # place_state = env.scene[
            #     f"right_hand_place_object"]._data.root_state_w[:, :7]
            # grasp_state = env.scene[
            #     f"right_hand_object"]._data.root_state_w[:, :7]
            # grasp_state[:, :3] = place_state[:, :3]
            # grasp_state[:, 2] = 0.03
            # env.scene["right_hand_object"].write_root_link_pose_to_sim(
            #     grasp_state)

            for i in range(150):

                actions = (torch.rand(env.action_space.shape,
                                      device=env.unwrapped.device) * 2 - 1) * 0
                # env.scene["right_hand_object"].write_root_link_pose_to_sim(
                #     grasp_state)

                obs, rewards, terminated, time_outs, extras = env.step(actions)

                # Save RGB image every 10 steps
                if i % 10 == 0:
                    import os
                    os.makedirs(f"{args_cli.log_dir}/images", exist_ok=True)
                    if "rgb" in obs["policy"]:
                        rgb_img = obs["policy"]["rgb"][0, 0].cpu().numpy()
                        cv2.imwrite(
                            f"{args_cli.log_dir}/images/rgb_step_{i:04d}.png",
                            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                    print(f"Saved image at step {i}")

            # Save point cloud visualization at the end of episode
            if "seg_pc" in obs["policy"]:
                pcd = vis_pc(obs["policy"]["seg_pc"][0][0].cpu().numpy())
                render_pcd = visualize_pcd([pcd],
                                           rotation_axis=[0, 1],
                                           rotation_angles=[1.57, 2.3],
                                           translation=[0.2, 0.4, 0.5],
                                           render=True)
                cv2.imwrite(f"{args_cli.log_dir}/images/pcd_final.png",
                            cv2.cvtColor(render_pcd, cv2.COLOR_RGB2BGR))
                print("Saved point cloud visualization")

            # env.scene[
            #     "right_articulated_object"].root_physx_view.set_dof_positions(
            #         env.scene["right_articulated_object"]._data.reset_joint_pos
            #         * 0.0 - 0.1,
            #         indices=torch.arange(env.num_envs, device=env.device))

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
