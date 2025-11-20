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
from tools.visualization_utils import *

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


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def explore(env,
            gripper_manager,
            buffer,
            sequeece_attachment_points,
            sample_parms,
            explore_type,
            num_explore_actions=None,
            log_path=None,
            num_loop=None):
    """
    Unified function to handle target, train, and eval exploration.

    :param env: Simulation environment.
    :param gripper_manager: Manager for gripper actions.
    :param buffer: Buffer to store transitions.
    :param sequeece_attachment_points: Attachment points for training/exploration.
    :param sample_parms: Parameters for sample exploration.
    :param explore_type: Type of exploration - 'target', 'train', or 'eval'.
    :param num_explore_actions: Number of target actions (required for target and eval).
    :param log_path: Path to store logs.
    :param num_loop: Loop number for log naming.
    """

    gripper_manager.randomize_deformable_properties(random_method="customize",
                                                    sample_parms=sample_parms)

    images_buffer = []

    if explore_type == "train":
        num_interactions = min(num_explore_actions,
                               len(sequeece_attachment_points))
    else:
        num_interactions = 1

    for k in range(num_interactions):

        for _ in range(env.max_episode_length):

            if env.episode_length_buf[0] == 0:
                gripper_manager.reset_deformable_visual_obs(
                    names=["seg_pc", "whole_pc", "seg_rgb", "seg_pc"],
                    bool=True)

                if explore_type == "train":

                    sample_attachment_points = sequeece_attachment_points[
                        k].unsqueeze(0).repeat_interleave(env.num_envs, 0)
                    gripper_manager.reset_deformable_visual_obs(
                        names=["seg_pc", "whole_pc", "seg_rgb", "seg_pc"],
                        bool=False)
                elif explore_type == "target":
                    sample_attachment_points = gripper_manager.sample_attachment_xyz(
                    )
                    target_env_attachment_points = gripper_manager.sample_attachment_xyz(
                        num_explore_actions)

                    sample_attachment_points[:
                                             num_explore_actions] = target_env_attachment_points[:
                                                                                                 num_explore_actions]

                else:
                    sample_attachment_points = gripper_manager.sample_attachment_xyz(
                    )
                    sample_attachment_points[:
                                             num_explore_actions] = sequeece_attachment_points[:
                                                                                               num_explore_actions]

                # reset the environment
                gripper_manager.explore_type = explore_type
                gripper_manager.explore_action_index = k
                next_obs = gripper_manager.on_training_start(
                    sample_attachment_points)

            # generate next actions
            actions = gripper_manager.on_step()

            if explore_type == "target" or explore_type == "eval":

                images_buffer.append(next_obs["policy"]["seg_rgb"].cpu())

            if env.episode_length_buf[0] > env.max_episode_length - 3:
                gripper_manager.reset_deformable_visual_obs(
                    names=["seg_pc", "whole_pc"], bool=True)

            next_obs, reward, terminate, time_out, info = env.step(actions)
        curr_time = time.time()
        buffer.create_transitions(
            next_obs,
            reward,
            actions,
            target_count=num_explore_actions
            if explore_type == "target" or explore_type == "eval" else None,
            cache_type=explore_type)
        print(time.time() - curr_time)
    curr_time = time.time()
    buffer._store_transition(cache_type=explore_type)
    buffer._clear_cache(cache_type=explore_type)
    print("save_time", time.time() - curr_time)
    if explore_type == "target" or explore_type == "eval":

        save_target_video(images_buffer,
                          log_path,
                          num_loop,
                          folder_name=f"{explore_type}/video",
                          num_explore_actions=num_explore_actions)

        log_media(num_explore_actions,
                  buffer,
                  log_path,
                  num_loop,
                  log_type=explore_type)

    del images_buffer
    gc.collect()
    torch.cuda.empty_cache()
    if explore_type == "target":
        return sample_attachment_points[:num_explore_actions]


def create_real_params(loop_num, num_values, num_parmas, randomness=False):
    param_ranges = [
        torch.linspace(0, 1, num_values) for _ in range(num_parmas)
    ]

    # Create the combination using torch's meshgrid and reshape the result
    combinations = torch.stack(torch.meshgrid(*param_ranges),
                               dim=-1).reshape(-1, num_parmas)

    if randomness:
        combinations = torch.rand(loop_num, num_parmas)

    return combinations


def main():

    if args_cli.Date is None:
        import datetime
        current_datetime = datetime.datetime.now()
        args_cli.Date = current_datetime.strftime("%m%d")
    log_path = create_log_dirs(f"logs/{args_cli.Date}/cem/rabbit/",
                               args_cli.random_params)

    save_config, config = save_params_to_yaml(args_cli.env_config,
                                              args_cli.random_params,
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

    gripper_manager = initialize_gripper(
        env,
        args_cli,
    )

    buffer = initialize_buffers(
        env,
        obs,
        device="cuda:0",
        exclude_obs_keys=["whole_pc", "seg_rgb", "rgb", "segmentation"],
        target_object_seg_id=gripper_manager.target_object_seg_id)

    num_optimized_params = len(
        env.scene["deform_object"].cfg.deform_cfg["params"].items())
    params_nam = [*env.scene["deform_object"].cfg.deform_cfg["params"].keys()]

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

    loop_num = 25

    rollout_num = 10

    num_explore_actions = 3
    cem_num = 3

    # real_sample_parms = (torch.arange(loop_num) /
    #                      (loop_num - 1)).unsqueeze(1).repeat_interleave(
    #                          num_optimized_params, -1)
    real_sample_parms = create_real_params(
        loop_num,
        int(np.sqrt(loop_num)) if num_optimized_params == 2 else int(
            np.cbrt(loop_num)),
        num_optimized_params,
        randomness=False)

    pre_training_num = len(os.listdir(log_path + "/result"))
    if args_cli.beign_traing_id is not None:
        pre_training_num = args_cli.beign_traing_id

    while simulation_app.is_running():

        target_propoerties = []
        rollout_properties = []
        rollout_properties_std = []

        for num_loop in range(loop_num):

            if num_loop < pre_training_num:

                target_propoerties.append([])
                rollout_properties.append([])
                rollout_properties_std.append([])
                continue

            print("=========================")
            cem.reset()

            print("Collecting target data")

            sequeece_attachment_points = explore(
                env,
                gripper_manager,
                buffer,
                sequeece_attachment_points=None,
                sample_parms=real_sample_parms[num_loop]
                [None].repeat_interleave(env.num_envs, 0).cpu().numpy(),
                explore_type="target",
                num_explore_actions=num_explore_actions,
                log_path=log_path,
                num_loop=num_loop)
            for cem_index in range(cem_num):

                buffer._clear_buffer(buffer_type="train")
                print(f"Begin sampling,itr {cem_index},loop {num_loop}")

                if np.all(np.diag(cem.cov.cpu().numpy()) < 0.001):
                    continue

                sample_train_deformable_parmsr = cem.sample_parameter(
                    env.num_envs * rollout_num).cpu().numpy()

                rollout_count = 0

                while rollout_count < rollout_num:
                    print("Collecting training data for rollout",
                          rollout_count)
                    train_sample_parms = sample_train_deformable_parmsr[
                        rollout_count * env.num_envs:(rollout_count + 1) *
                        env.num_envs]

                    explore(
                        env,
                        gripper_manager,
                        buffer,
                        sequeece_attachment_points=sequeece_attachment_points,
                        sample_parms=train_sample_parms,
                        explore_type="train",
                        num_explore_actions=num_explore_actions,
                        log_path=log_path,
                        num_loop=num_loop)

                    rollout_count += 1
                print("finish collecting training data")
                cem.command()
                real_data = buffer.target_buffer["deform_physical_params"][0]

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

            np.savez(f"{log_path}/result/{num_loop}.npz",
                     param_names=env.scene["deform_object"].parames_generator.
                     parames_name,
                     result=np.concatenate([
                         real_data.squeeze(0).cpu().numpy(),
                         cem.mean.cpu().numpy(),
                         np.diag(cem.cov.cpu().numpy())
                     ]))

            eval_sample_parms = cem.sample_parameter(
                env.num_envs).cpu().numpy()

            explore(env,
                    gripper_manager,
                    buffer,
                    sequeece_attachment_points=sequeece_attachment_points,
                    sample_parms=eval_sample_parms,
                    explore_type="eval",
                    num_explore_actions=num_explore_actions,
                    log_path=log_path,
                    num_loop=num_loop)

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
