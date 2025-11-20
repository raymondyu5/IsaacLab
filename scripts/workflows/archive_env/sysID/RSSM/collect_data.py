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

from scripts.workflows.sysID.bayes_sim.utils.data_utils import collect_trajectories
from scripts.workflows.sysID.bayes_sim.utils.plot import plot_trajectories

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


def load_dataset_saver(log_dir, iter):

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=f"{args_cli.filename}_{iter}",
        num_demos=1000,
        flush_freq=args_cli.num_envs,
        env_config={"device": args_cli.device},
    )
    collector_interface.reset()
    return collector_interface


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

    state_name = "deform_pos_w"
    # specify directory for logging experiments
    log_dir = f"./logs/{args_cli.filename}/property/"
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    start_index = 8
    end_index = 28
    interval = 2

    # sim_prms_list = []
    # sim_params_smpls_classifer_list = []
    # sim_traj_states_list = []
    # sim_traj_rgb_list = []
    # sim_traj_acts_list = []
    env.reset()

    while simulation_app.is_running():
        for iter in range(10):

            collector_interface = load_dataset_saver(log_dir, iter)
            for k in range(10):

                sim_prms, sim_traj_states, sim_traj_acts, sim_traj_rgb, sim_params_smpls_classifer = collect_trajectories(
                    30,
                    env,
                    device='cuda:0',
                    state_name=state_name,
                    camera_obs=True,
                    vis_camera=False,
                    camera_function=object_3d_seg_rgb,
                    resize=None)

                # sim_prms_list.append(sim_prms.cpu().numpy())
                # sim_params_smpls_classifer_list.append(
                #     sim_params_smpls_classifer.cpu().numpy())
                # sim_traj_states_list.append(sim_traj_states.cpu().numpy())
                # sim_traj_rgb_list.append(sim_traj_rgb.cpu().numpy())
                # sim_traj_acts_list.append(sim_traj_acts.cpu().numpy())

                # sim_prms = np.concatenate(sim_prms_list)
                # sim_traj_rgb = np.concatenate(sim_traj_rgb_list)
                # sim_traj_states = np.concatenate(sim_traj_states_list)
                # sim_params_smpls_classifer = np.concatenate(
                #     sim_params_smpls_classifer_list)
                # sim_traj_acts = np.concatenate(sim_traj_acts_list)

                collector_interface.add(f"obs/sim_traj_rgb",
                                        sim_traj_rgb.cpu().numpy())
                collector_interface.add(f"obs/sim_traj_states",
                                        sim_traj_states.cpu().numpy())
                collector_interface.add(f"obs/sim_prms",
                                        sim_prms.cpu().numpy())
                collector_interface.add(
                    f"obs/sim_params_smpls_classifer",
                    sim_params_smpls_classifer.cpu().numpy())
                collector_interface.add(f"actions",
                                        sim_traj_acts.cpu().numpy())

                collector_interface.flush(
                    (np.arange(len(sim_traj_rgb))).astype(np.int16))

                save_classifier_video(sim_traj_rgb.cpu().numpy()[..., ::-1],
                                      sim_params_smpls_classifer.cpu().numpy(),
                                      "logs/video/real")
                # plot_trajectories(sim_traj_states.cpu().numpy()[:4])

                del sim_prms
                del sim_traj_states
                del sim_traj_acts
                del sim_traj_rgb
                del sim_params_smpls_classifer
                torch.cuda.empty_cache()
            collector_interface.close()
        break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
