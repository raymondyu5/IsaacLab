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
import sys

from scripts.workflows.sysID.utils.data_utilis import DataBuffer

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
parser.add_argument("--logdir",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--filename",
                    type=str,
                    default="hdf_dataset",
                    help="Basename of output file.")
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
import gc
from tools.deformable_obs import object_3d_observation, object_3d_seg_rgb
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector

from tools.collect_data import load_dataset_saver

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
    # reset environment

    # specify directory for logging experiments
    log_dir = f"./logs/{args_cli.filename}/property/"
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    delta_pose = torch.as_tensor([0.0, 0.0, 0.01, 1., 0., 0., 0., 1],
                                 dtype=torch.float64).to(
                                     env.device)[None].repeat_interleave(
                                         env.num_envs, 0)
    gripper_offset_xyz = torch.as_tensor([0.5, -0.0, 0.09]).to(
        env.device)[None].repeat_interleave(env.num_envs, 0)

    buffer = DataBuffer(env, "cpu", [*env.observation_space["policy"].keys()])

    while simulation_app.is_running():

        for k in range(10):
            collector_interface = load_dataset_saver(args_cli, log_dir, k)

            for num_loop in range(30):

                buffer.clear_buffer(buffer_type="target")

                collect_env_trajectories(env,
                                         buffer,
                                         delta_pose,
                                         gripper_offset_xyz,
                                         cache_type="target",
                                         log_path=log_dir,
                                         id=num_loop + k * 100,
                                         image_key="seg_rgb")

                for key in buffer.target_buffer.keys():
                    if key != "actions":
                        collector_interface.add(f"obs/{key}",
                                                buffer.target_buffer[key])

                collector_interface.add(
                    f"actions", buffer.target_buffer["actions"].cpu().numpy())

                collector_interface.flush((np.arange(
                    len(buffer.target_buffer["deform_physical_params"]))
                                           ).astype(np.int16))

                torch.cuda.empty_cache()
        collector_interface.close()
        break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
