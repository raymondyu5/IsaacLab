# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

sys.path.append(".")

from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

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

from isaaclab_tasks.utils import parse_env_cfg
from tools.deformable_obs import *

from scripts.workflows.automatic_articulation.utils.data_wrapper import Datawrapper
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.automatic_articulation.utils.replay_datawrapper import ReplayDatawrapper

from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from tools.visualization_utils import vis_pc, visualize_pcd
import imageio
from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    """Random actions agent with Isaac Lab environment."""
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment configuration
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    init_grasp = True
    init_open = False
    init_placement = False
    init_close = False
    collect_data = False

    env.reset()
    stats = np.load(args_cli.log_dir + "/grasp_stats.npy", allow_pickle=True)
    total_count = 0
    success_count = 0
    demo_index = 0
    import h5py
    normalized_grasp = h5py.File(f"{args_cli.log_dir}/grasp_normalized.hdf5",
                                 'r+')
    kitchen = env.scene["kitchen"]

    def reset_scene():
        demo = normalized_grasp["data"][f"demo_{demo_index}"]
        num_demos = len(normalized_grasp["data"])
        robot = env.scene["robot"]
        grasp_object = env.scene["mug"]
        env_ids = torch.arange(0, env.num_envs).to(env.device)

        obs = demo["obs"]
        init_joint_pos = torch.as_tensor(obs["joint_pos"][0]).to(
            env.device).unsqueeze(0)
        robot_base = torch.as_tensor(obs["robot_base"][0]).to(
            env.device).unsqueeze(0)
        object_root_pose = torch.as_tensor(obs["mug_root_pose"][0]).to(
            env.device).unsqueeze(0)

        robot.write_root_pose_to_sim(robot_base, env_ids=env.env_ids)

        preset_object_root_states = grasp_object.data.default_root_state[
            env.env_ids].clone()

        grasp_object.write_root_pose_to_sim(object_root_pose, env_ids=env_ids)
        grasp_object.write_root_velocity_to_sim(
            preset_object_root_states[:, 7:] * 0, env_ids=env_ids)

        default_jpos = kitchen._data.default_joint_pos.clone()
        kitchen._data.reset_joint_pos = default_jpos
        kitchen.root_physx_view.set_dof_positions(default_jpos, env_ids)

        for i in range(50):  # reset for stable initial status
            robot.root_physx_view.set_dof_positions(init_joint_pos, env_ids)
            robot_pos, robot_quat = get_robottip_pose(robot, env.device)

            observation, reward, terminate, time_out, info = env.step(
                torch.cat([
                    robot_pos, robot_quat,
                    torch.ones((1, len(robot_pos))).to(env.device)
                ],
                          dim=1))

    sys.path.append("../robomimic")
    import robomimic.utils.file_utils as FileUtils
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=
        "/media/lme/data4/weird/robomimic/bc_trained_models/diffusion/20241031094443/models/model_epoch_585_best_validation_0.0011904271435923875.pth",
        device=env.device,
        verbose=True)

    policy.start_episode()

    while simulation_app.is_running():

        obs = normalized_grasp["data"][f"demo_{demo_index}"]["obs"]
        obs_buffer = {}
        gt_actions = torch.as_tensor(
            np.array(
                normalized_grasp["data"][f"demo_{demo_index}"]["actions"])).to(
                    env.device)

        for key in obs.keys():
            obs_buffer[key] = torch.cat([
                torch.as_tensor(obs[key][0]).unsqueeze(0),
                torch.as_tensor(obs[key][0]).unsqueeze(0)
            ],
                                        dim=0).to(env.device)
        for actions_id in range(len(gt_actions)):
            for key in obs.keys():
                obs_buffer[key][0] = obs_buffer[key][-1].clone()

                obs_buffer[key][-1] = torch.as_tensor(obs[key][actions_id]).to(
                    self.device).unsqueeze(0)
            import copy
            target_obs = copy.deepcopy(obs_buffer)
            target_obs["seg_pc"] = target_obs["seg_pc"].permute(0, 2,
                                                                1)[:, :3, :]
            predicted_action = policy(obs_buffer)
            import pdb
            pdb.set_trace()
            # predicted_action[:3] = unnormalize(predicted_action[:3][None],
            #                                    stats.item()["action"])[0]
            # predicted_action = torch.as_tensor(predicted_action).unsqueeze(
            #     0).to(env.device)

        env.reset()
        # total_count += 1
        # if success:
        #     success_count += 1
        # print(
        #     f"Success rate: {success_count}/{total_count} = {success_count/total_count}"
        # )

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
