# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import logging

sys.path.append(".")
import time

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import gc
import imageio
from isaaclab.utils import Timer
import wandb
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--disable_fabric",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",
                    type=int,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--env_config",
                    type=str,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--random_params",
                    type=str,
                    default=None,
                    help="randomness_params.")

parser.add_argument("--Date", type=str, default=None, help="date.")
parser.add_argument("--beign_traing_id",
                    type=int,
                    default=None,
                    help="begin training id ")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_gripper",
                    action="store_true",
                    default=False,
                    help="")
parser.add_argument("--parmas_range",
                    type=list,
                    default=None,
                    help="parmas_range.")
parser.add_argument("--num_explore_actions", type=str, default=[2], help="")
parser.add_argument("--name", type=str, default=None, help="")
parser.add_argument("--fix_params", type=str, default=None, help="")
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

from scripts.workflows.sysID.utils.attachment_utils import initialize_gripper
from scripts.workflows.sysID.utils.data_utilis import initialize_buffers
from scripts.workflows.sysID.utils.file_utils import create_log_dirs
from scripts.workflows.sysID.ASID.tool.utilis import save_target_video
from scripts.workflows.sysID.utils.wandb_utils import setup_wandb, log_media, log_paramas
from scripts.workflows.sysID.ASID.cem.cem import CEM
from scripts.workflows.sysID.ASID.tool.utilis import *
from scripts.workflows.sysID.utils.file_utils import save_params_to_yaml
from tools.visualization_utils import *


def create_real_params(loop_num, num_values, num_parmas, randomness=False):

    if isinstance(num_values, int):
        param_ranges = [
            torch.linspace(0, 1, num_values) for _ in range(num_parmas)
        ]
    elif isinstance(num_values, list):
        param_ranges = [
            torch.linspace(0, 1, num_values[i]) for i in range(num_parmas)
        ]

    # Create the combination using torch's meshgrid and reshape the result
    combinations = torch.stack(torch.meshgrid(*param_ranges),
                               dim=-1).reshape(-1, num_parmas)

    if randomness:
        combinations = torch.rand(loop_num, num_parmas)

    return combinations


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():

    if args_cli.Date is None:
        import datetime
        current_datetime = datetime.datetime.now()
        args_cli.Date = current_datetime.strftime("%m%d")

    object_name = args_cli.env_config.split('/')[2].split("_")[0]
    log_path = create_log_dirs(f"logs/{args_cli.Date}/cem/{object_name}/",
                               args_cli)

    save_config, config = save_params_to_yaml(args_cli,
                                              save_config_path=log_path)
    random_params = args_cli.random_params.strip('[]').split(',')
    setup_wandb(parser_config=config,
                exp_name=''.join(f"_{param}" for param in random_params),
                tags="cem",
                project="cem")

    env = setup_env(args_cli, save_config)
    import warnings

    # Suppress all UserWarnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs = env.reset()

    loop_num = 25

    rollout_num = 12

    save_interval = env.max_episode_length - 1
    cem_num = 3

    gripper_manager = initialize_gripper(
        env,
        args_cli,
        buffer=None,
        use_wandb=True,
        require_segmentation=True,
        reset_camera_obs_list=["whole_pc", "whole_rgb", "seg_rgb", "seg_pc"],
        render_all=False,
    )
    buffer = initialize_buffers(
        env,
        obs,
        device="cuda:0",
        exclude_obs_keys=["whole_pc", "seg_rgb", "rgb", "segmentation"],
        target_object_seg_id=gripper_manager.target_object_seg_id,
        require_segmentation=True)
    gripper_manager.buffer = buffer

    num_optimized_params = len(
        env.scene["deform_object"].cfg.deform_cfg['physical_params']
        ["params_values"].items())
    params_nam = [
        *env.scene["deform_object"].cfg.deform_cfg['physical_params']
        ["params_values"].keys()
    ]

    cem = CEM(
        dynamics=buffer,
        running_cost=None,
        nx=1024,  #num of pc
        nu=num_optimized_params,  #num of the phyiscal properties
        horizon=1,
        u_min=torch.ones(num_optimized_params) * 1e-6,
        u_max=torch.ones(num_optimized_params),
        num_elite=10,
        init_cov_diag=0.3,
        device="cuda:0")

    real_sample_parms = create_real_params(
        loop_num,
        int(np.sqrt(loop_num)) if args_cli.parmas_range is None else [
            int(i)
            for i in ''.join(args_cli.parmas_range).replace('[', '').replace(
                ']', '').split(',')
        ],
        num_optimized_params,
        randomness=False)
    print("training params name",
          env.scene["deform_object"].parames_generator.parames_name)
    pre_training_num = len(os.listdir(log_path + "/result"))
    if args_cli.beign_traing_id is not None:
        pre_training_num = args_cli.beign_traing_id

    while simulation_app.is_running():

        target_propoerties = []
        rollout_properties = []
        rollout_properties_std = []

        for num_loop in range(len(real_sample_parms)):

            if num_loop < pre_training_num:

                target_propoerties.append([])
                rollout_properties.append([])
                rollout_properties_std.append([])
                continue

            print("=========================")
            cem.reset()

            print("Collecting target data")

            gripper_manager.step_gripper_manager(
                sample_parms=real_sample_parms[num_loop]
                [None].repeat_interleave(env.num_envs, 0).cpu().numpy(),
                explore_type="target",
                log_path=log_path,
                num_loop=num_loop,
                save_interval=save_interval)
            for cem_index in range(cem_num):

                buffer._clear_buffer(buffer_type="train")

                if np.all(np.diag(cem.cov.cpu().numpy()) < 0.00005):
                    continue
                print(f"Begin sampling,itr {cem_index},loop {num_loop}")

                sample_train_deformable_params = cem.sample_parameter(
                    env.num_envs * rollout_num).cpu().numpy()
                cur_time = time.time()

                gripper_manager.step_gripper_manager(
                    sample_parms=sample_train_deformable_params,
                    explore_type="train",
                    log_path=log_path,
                    num_loop=num_loop,
                    rollout_num=rollout_num,
                    save_interval=save_interval)
                print("Rollout time", time.time() - cur_time)

                print("finish collecting training data")
                cem.command()
                real_data = buffer.target_buffer["deform_physical_params"][0,
                                                                           0,
                                                                           0]

                print(f"Real: num_cem: {cem_index}, mean:{real_data}, std:0.0")
                print(
                    f"Rollout: num_cem:{cem_index}, mean:{cem.mean}, std:{cem.cov}"
                )
                gc.collect()

            target_propoerties.append(real_data.squeeze().cpu().numpy())
            rollout_properties.append(cem.mean.squeeze().cpu().numpy())

            rollout_properties_std.append(np.diag(cem.cov.cpu().numpy()))

            log_paramas(
                params_nam, target_propoerties[num_loop] if isinstance(
                    target_propoerties[num_loop], list) else
                [target_propoerties[num_loop]], rollout_properties[num_loop]
                if isinstance(rollout_properties[num_loop],
                              list) else [rollout_properties[num_loop]],
                rollout_properties_std[num_loop] if isinstance(
                    rollout_properties_std[num_loop],
                    list) else [rollout_properties_std[num_loop]], num_loop)

            if num_optimized_params == 1:

                np.savez(f"{log_path}/result/{num_loop}.npz",
                         param_names=env.scene["deform_object"].
                         parames_generator.parames_name,
                         result=np.concatenate([
                             real_data.cpu().numpy(),
                             cem.mean.cpu().numpy(),
                             np.diag(cem.cov.cpu().numpy())
                         ]))

            else:
                np.savez(f"{log_path}/result/{num_loop}.npz",
                         param_names=env.scene["deform_object"].
                         parames_generator.parames_name,
                         result=np.concatenate([
                             real_data.squeeze(0).cpu().numpy(),
                             cem.mean.cpu().numpy(),
                             np.diag(cem.cov.cpu().numpy())
                         ]))

            eval_sample_parms = cem.sample_parameter(
                env.num_envs).cpu().numpy()

            gripper_manager.step_gripper_manager(
                sample_parms=eval_sample_parms,
                explore_type="eval",
                log_path=log_path,
                num_loop=num_loop,
                save_interval=save_interval)

            buffer._clear_buffer(buffer_type="train")
            buffer._clear_buffer(buffer_type="target")
            buffer._clear_buffer(buffer_type="eval")

        # plot_result(params_nam,
        #             target_propoerties,
        #             rollout_properties,
        #             rollout_properties_std,
        #             log_path=log_path + "/result")

        break

    # close the simulator
    env.close()


if __name__ == "__main__":

    # run the main function
    main()
    # close sim app
    simulation_app.close()

# pcd = o3d.geometry.PointCloud()
# seg_pc = next_obs["policy"]["seg_pc"].cpu().numpy()
# pcd.points = o3d.utility.Vector3dVector(seg_pc[0, :, :3])
# pcd.colors = o3d.utility.Vector3dVector(seg_pc[0, :, 3:6] /
#                                         255)
# o3d.visualization.draw_geometries([pcd])
