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
import time
from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import gc
import imageio
from isaaclab.utils import Timer
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
from scripts.workflows.sysID.utils.attachment_utils import GripperActionManager, hide_prim
from scripts.workflows.sysID.utils.data_utilis import DataBuffer
from scripts.workflows.sysID.ASID.tool.utilis import save_target_video
from tools.data_collection import load_dataset_saver
from scripts.workflows.sysID.ASID.cem.cem import CEM
from scripts.workflows.sysID.ASID.tool.utilis import *


def setup_env(args_cli):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def initialize_gripper(env,
                       args_cli,
                       sample_id=None,
                       static_frames=3,
                       gripper_offset=0.0,
                       gripper_offset_xyz=None):

    gripper_manager = GripperActionManager(
        env,
        args_cli.task,
        ee_height_compensate=0.0486,
        static_frames=static_frames,
        sample_seg_id=sample_id,
        gripper_offset=gripper_offset,
        gripper_offset_xyz=gripper_offset_xyz)

    if "Abs" in args_cli.task or "Rel" in args_cli.task:
        for i in range(env.num_envs):
            hide_prim(env.scene.stage, f"/World/envs/env_{i}/gripper")
    return gripper_manager


def initialize_buffers(env, obs, device, exclude_obs_keys=[], sample_id=None):
    obs_keys = [*obs[0]["policy"].keys()]

    # remove the whole point cloud and segmentation point cloud
    for name in exclude_obs_keys:
        if name in obs_keys:
            obs_keys.remove(name)

    return DataBuffer(env, device, obs_keys, sample_id=sample_id)


def create_log_dirs(base_dir="logs/rabbit/static_gs/"):
    os.makedirs(base_dir + "/result", exist_ok=True)
    os.makedirs(base_dir + "/video", exist_ok=True)
    return base_dir


def reset_obs_cam(env, names, bool=False):
    for name in names:
        env.scene["deform_object"].cfg.deform_cfg["camera_obs"][name] = bool


def trigger_reset(
    gripper_manager,
    target_gripper_traj,
    gripper_actions,
):

    if "Abs" in args_cli.task or "Rel" in args_cli.task:

        next_obs = gripper_manager.technical_reset(
            target_gripper_traj,
            gripper_actions,
        )
    else:
        next_obs = gripper_manager.technical_reset()
    return next_obs


def on_step(gripper_manager, target_gripper_traj, gripper_actions, next_obs):

    if "Abs" in args_cli.task or "Rel" in args_cli.task:
        actions = gripper_manager.assemble_actions(
            target_gripper_traj, gripper_actions,
            next_obs["policy"]["ee_pose"])
    else:
        actions = gripper_manager.assemble_actions(target_gripper_traj,
                                                   gripper_actions, None)
    return actions


def random_sample_deformable_parms(env, num_env):

    # #sample random physical properties for deformable object
    sample_deformable_parms = env.scene[
        "deform_object"].parames_generator.uniform_random(num_env)

    return sample_deformable_parms


def sample_grasp_index(gripper_manager, num_grasp=100):

    sample_explore_grasp_indices = gripper_manager.sample_grasp_index_from_pc(
        num_grasp)

    # Randomly select 10 values from the tensor
    random_indices = torch.randperm(sample_explore_grasp_indices.size(0))
    sequeece_grasp_index = sample_explore_grasp_indices[random_indices]
    return sequeece_grasp_index


def generate_target_explores(env, gripper_manager, buffer,
                             target_explore_actions, random_orientation_range,
                             proposed_position_dir, gripper_actions,
                             real_sample_parms, log_path, num_loop):

    env.scene["deform_object"].parames_generator.random_method = "customize"
    env.scene[
        "deform_object"].parames_generator.params_range = real_sample_parms[
            None].repeat_interleave(env.num_envs, 0).cpu().numpy()
    sequeece_grasp_index = sample_grasp_index(gripper_manager, num_grasp=100)

    images_buffer = []

    for _ in range(env.max_episode_length):

        if env.episode_length_buf[0] == 0:
            proposed_oreintation = gripper_manager.random_orientation(
                random_orientation_range) * 0

            reset_obs_cam(env,
                          names=["seg_pc", "whole_pc", "whole_rgb"],
                          bool=True)

            # generate  target gripper trajectory
            target_gripper_traj = gripper_manager.on_training_start(
                proposed_oreintation,
                proposed_position_dir=proposed_position_dir * 2,
                sample_grasp_indices=sequeece_grasp_index[:env.num_envs])

            reset_obs_cam(env,
                          names=["seg_pc", "whole_pc", "whole_rgb", "seg_rgb"],
                          bool=False)

            # trigger reset for the gripper
            next_obs = trigger_reset(gripper_manager, target_gripper_traj,
                                     gripper_actions)

        # generate next actions
        actions = on_step(gripper_manager, target_gripper_traj,
                          gripper_actions, next_obs)

        next_obs, reward, terminate, time_out, info = env.step(actions)
        # images_buffer.append(next_obs["policy"]["seg_rgb"].cpu())
        # prepare for loading data
        if env.episode_length_buf[0] > env.max_episode_length - 3:
            reset_obs_cam(env,
                          names=["seg_pc", "whole_pc", "seg_rgb"],
                          bool=True)

    transition = {}

    transition["obs"] = {}
    for key in buffer.obs_key:
        transition["obs"][key] = next_obs["policy"][
            key][:target_explore_actions]
    transition["reward"] = reward[
        :target_explore_actions,
    ] * 0
    transition["action"] = actions[:target_explore_actions, :]

    buffer.cache_traj(transition, cache_type="target")

    # save_target_video(images_buffer,
    #                   log_path,
    #                   num_loop,
    #                   folder_name="target/video")

    buffer.store_transition(cache_type="target")
    buffer.clear_cache(cache_type="target")

    del images_buffer
    gc.collect()
    torch.cuda.empty_cache()

    return sequeece_grasp_index[:target_explore_actions]


def generate_train_explores(env, gripper_manager, buffer, sequeece_grasp_index,
                            random_orientation_range, proposed_position_dir,
                            gripper_actions, log_path, num_loop):

    for grasp_position_index in sequeece_grasp_index:
        for k in range(env.max_episode_length):

            if env.episode_length_buf[0] == 0:

                proposed_oreintation = gripper_manager.random_orientation(
                    random_orientation_range) * 0

                reset_obs_cam(env,
                              names=["seg_pc", "whole_pc", "whole_rgb"],
                              bool=True)

                # generate  target gripper trajectory
                target_gripper_traj = gripper_manager.on_training_start(
                    proposed_oreintation,
                    proposed_position_dir=proposed_position_dir * 2,
                    sample_grasp_indices=grasp_position_index.unsqueeze(
                        0).repeat_interleave(env.num_envs, 0))

                reset_obs_cam(
                    env,
                    names=["seg_pc", "whole_pc", "whole_rgb", "seg_rgb"],
                    bool=False)

                # trigger reset for the gripper
                next_obs = trigger_reset(gripper_manager, target_gripper_traj,
                                         gripper_actions)

            # generate next actions
            actions = on_step(gripper_manager, target_gripper_traj,
                              gripper_actions, next_obs)

            next_obs, reward, terminate, time_out, info = env.step(actions)
            # images_buffer.append(next_obs["policy"]["seg_rgb"].cpu())
            # prepare for loading data
            if env.episode_length_buf[0] > env.max_episode_length - 3:
                reset_obs_cam(env,
                              names=["seg_pc", "whole_pc", "seg_rgb"],
                              bool=True)

        transition = {}

        transition["obs"] = {}
        for key in buffer.obs_key:
            transition["obs"][key] = next_obs["policy"][key]
        transition["reward"] = reward * 0
        transition["action"] = actions

        buffer.cache_traj(transition, cache_type="train")

        # save_target_video(images_buffer,
        #                   log_path,
        #                   num_loop,
        #                   folder_name="train/video")

    buffer.store_transition(cache_type="train")
    buffer.clear_cache(cache_type="train")
    gc.collect()
    torch.cuda.empty_cache()


def main():
    env = setup_env(args_cli)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    log_path = create_log_dirs(base_dir="logs/rabbit/cem/")

    obs = env.reset()

    # robot setting
    random_orientation_range = [0, 3.14]

    random_orientation_range = [0, 3.14]
    proposed_position_dir = torch.as_tensor([0.0, 0.0, 0.01]).to(env.device)
    gripper_actions = torch.as_tensor([-1]).to(env.device)
    gripper_offset_xyz = torch.as_tensor(
        env.scene["deform_object"].cfg.deform_cfg["gripper_offset_xyz"]).to(
            env.device).repeat(env.num_envs, 1)
    sample_id = (torch.arange(env.num_envs) + 1).to(env.device) * env.scene[
        "deform_object"].cfg.deform_cfg["camera_obs"]["segmentation_id"]

    gripper_manager = initialize_gripper(
        env,
        args_cli,
        sample_id,
        gripper_offset=env.scene["deform_object"].cfg.
        deform_cfg["gripper_offset"] + 0.01,
        gripper_offset_xyz=gripper_offset_xyz)
    buffer = initialize_buffers(
        env,
        obs,
        device="cuda:0",
        exclude_obs_keys=["whole_pc", "seg_rgb", "rgb", "segmentation"],
        sample_id=sample_id)

    cem = CEM(
        dynamics=buffer,
        running_cost=None,
        nx=1024,  #num of pc
        nu=1,  #num of the phyiscal properties
        horizon=1,
        u_min=torch.as_tensor([1e-6]),
        u_max=torch.as_tensor([1]),
        num_elite=10,
        init_cov_diag=0.3)

    loop_num = 3

    rollout_num = 8

    target_explore_actions = 5
    cem_num = 3

    real_sample_parms = (torch.arange(loop_num) / (loop_num - 1))[:, None]

    while simulation_app.is_running():

        target_propoerties = []
        rollout_properties = []
        rollout_properties_std = []

        for num_loop in range(loop_num):
            print("=========================")
            cem.reset()

            buffer.clear_buffer(buffer_type="target")

            print("Collecting target data")

            sequeece_grasp_index = generate_target_explores(
                env, gripper_manager, buffer, target_explore_actions,
                random_orientation_range, proposed_position_dir,
                gripper_actions, real_sample_parms[num_loop], log_path,
                num_loop)
            for cem_index in range(cem_num):
                print(f"Begin sampling,itr {cem_index},loop {num_loop}")

                if cem.cov.squeeze().cpu().numpy() < 0.0001:
                    continue
                buffer.clear_buffer(buffer_type="train")

                sample_train_deformable_parmsr = cem.sample_parameter(
                    env.num_envs * rollout_num).cpu().numpy()

                rollout_count = 0

                while rollout_count < rollout_num:
                    print("Collecting training data for rollout",
                          rollout_count)

                    # using customize random method
                    env.scene[
                        "deform_object"].parames_generator.random_method = "customize"

                    env.scene[
                        "deform_object"].parames_generator.params_range = sample_train_deformable_parmsr[
                            rollout_count * env.num_envs:(rollout_count + 1) *
                            env.num_envs]

                    generate_train_explores(
                        env, gripper_manager, buffer, sequeece_grasp_index,
                        random_orientation_range, proposed_position_dir,
                        gripper_actions, log_path,
                        num_loop * rollout_num + rollout_count)

                    rollout_count += 1
                    time.sleep(1)

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
            rollout_properties_std.append(cem.cov.squeeze().cpu().numpy())
            np.save(
                f"{log_path}/result/{num_loop}.npy",
                np.concatenate([
                    real_data.squeeze(0).cpu().numpy(),
                    cem.mean.cpu().numpy(), cem.cov[0].cpu().numpy()
                ]))

        plot_result(target_propoerties,
                    rollout_properties,
                    rollout_properties_std,
                    name="cem")

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
