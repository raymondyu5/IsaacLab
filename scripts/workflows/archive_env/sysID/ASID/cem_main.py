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
from scripts.workflows.sysID.utils.data_utilis import DataBuffer
from scripts.workflows.sysID.ASID.cem.cem import CEM
import imageio
# from pytorch3d.loss import chamfer_distance
import carb
import matplotlib.pyplot as plt
import torchvision
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

from scripts.workflows.sysID.ASID.tool.utilis import *


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    logdir = "logs/cem/"

    os.makedirs(logdir, exist_ok=True)
    # reset environment

    delta_pose = torch.as_tensor([0.0, 0.0, 0.01, 1., 0., 0., 0., 1],
                                 dtype=torch.float64).to(
                                     env.device)[None].repeat_interleave(
                                         env.num_envs, 0)
    gripper_offset_xyz = torch.as_tensor([0.5, -0.0, 0.09]).to(
        env.device)[None].repeat_interleave(env.num_envs, 0)

    buffer = DataBuffer(env, "cpu", [*env.observation_space["policy"].keys()])

    target_num = 1
    target_count = 0

    rollout_num = 10
    rollout_count = 0

    num_cem = 3

    loop_num = 20

    cem = CEM(
        dynamics=buffer,
        running_cost=None,
        nx=1024,  #num of pc
        nu=1,  #num of the phyiscal properties
        horizon=1,
        u_min=torch.as_tensor([1e-6]),
        u_max=torch.as_tensor([1]),
        num_elite=10,
        init_cov_diag=0.2)

    real_sample_parms = (torch.arange(loop_num) / (loop_num - 1))[:, None]

    while simulation_app.is_running():
        target_propoerties = []
        rollout_properties = []
        rollout_properties_std = []

        for num_loop in range(loop_num):
            print("=========================")
            target_count = 0
            cem.reset()
            buffer.clear_buffer(buffer_type="target")

            for cem_index in range(num_cem):
                if cem.cov.squeeze().cpu().numpy() < 0.0001:
                    continue
                rollout_count = 0
                buffer.clear_buffer(buffer_type="train")

                print(f"Begin sampling,itr {cem_index},loop {num_loop}")
                #sample new data at each cem iteraction
                sample_parms = cem.sample_parameter(args_cli.num_envs *
                                                    rollout_num)

                ### begin collect data and sample
                while True:
                    if target_num > target_count:

                        env.scene[
                            "deform_object"].parames_generator.random_method = "customize"

                        env.scene[
                            "deform_object"].parames_generator.params_range = real_sample_parms[
                                num_loop][None].repeat_interleave(
                                    args_cli.num_envs, 0).cpu().numpy()
                        collect_env_trajectories(env,
                                                 buffer,
                                                 delta_pose,
                                                 gripper_offset_xyz,
                                                 cache_type="target",
                                                 log_path=logdir,
                                                 id=num_loop)
                        target_count += 1

                    elif rollout_num > rollout_count:
                        print("collect", rollout_count)
                        env.scene[
                            "deform_object"].parames_generator.random_method = "customize"

                        env.scene[
                            "deform_object"].parames_generator.params_range = sample_parms[
                                rollout_count *
                                args_cli.num_envs:(rollout_count + 1) *
                                args_cli.num_envs].cpu().numpy()

                        collect_env_trajectories(env,
                                                 buffer,
                                                 delta_pose,
                                                 gripper_offset_xyz,
                                                 cache_type="train")
                        rollout_count += 1

                    else:
                        print("finish collection")
                        cem.command()

                        real_data = buffer.target_buffer[
                            "deform_physical_params"][0][0]
                        print(
                            f"Real: num_cem: {cem_index}, mean:{real_data}, std:0.0"
                        )
                        print(
                            f"Rollout: num_cem:{cem_index}, mean:{cem.mean}, std:{cem.cov}"
                        )

                        break  #exit after the optimization

            target_propoerties.append(real_data.squeeze().cpu().numpy())
            rollout_properties.append(cem.mean.squeeze().cpu().numpy())
            rollout_properties_std.append(cem.cov.squeeze().cpu().numpy())
        plot_result(target_propoerties,
                    rollout_properties,
                    rollout_properties_std,
                    name="cem.png")

        # close the simulator
        break
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
