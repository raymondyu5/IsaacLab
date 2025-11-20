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

from scripts.workflows.sysID.bayes_sim.bayes_sim import BayesSim
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

    env.scene["deform_object"].cfg.deform_cfg['bayessim'].update(
        {"trainTrajLen": env.max_episode_length})
    state_name = "deform_pos_w"

    bsim = BayesSim(
        model_cfg=env.scene["deform_object"].cfg.deform_cfg['bayessim'],
        obs_dim=env.observation_space["policy"][state_name].shape[1],
        act_dim=1,
        params_dim=env.scene["deform_object"].data.physical_params.shape[1],
        params_lows=env.scene["deform_object"].parames_generator.lows(),
        params_highs=env.scene["deform_object"].parames_generator.highs(),
        prior=None,
        proposal=None,
        device=env.device)

    n_train_trajs = env.scene["deform_object"].cfg.deform_cfg['bayessim'][
        'trainTrajs']
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args_cli.logdir, 'bsim'),
                           flush_secs=10)
    start_index = 8
    end_index = 28
    interval = 2

    while simulation_app.is_running():
        # collect real data
        # env.scene["deform_object"].parames_generator.high_range = [0.0, 0.0]
        # env.scene["deform_object"].parames_generator.low_range = [0.6, 0.6]

        real_prms, real_traj_states, real_traj_acts, real_traj_rgb, real_params_smpls_classifer = collect_trajectories(
            20,
            env,
            device='cuda:0',
            state_name=state_name,
            camera_obs=True,
            vis_camera=False,
            camera_function=object_3d_seg_rgb,
            resize=None)

        # writer.images_to_video(real_traj_rgb[0].cpu().numpy()[..., ::-1],
        #                        os.path.join(args_cli.logdir, "real_video.mp4"))

        save_classifier_video(real_traj_rgb[:, start_index:end_index:interval],
                              real_params_smpls_classifer, "logs/video/real")
        plot_trajectories(real_traj_states.cpu().numpy()[:4])

        for real_iter_id in range(
                env.scene["deform_object"].cfg.deform_cfg['bayessim']
            ['realIters']):

            if real_iter_id < 3:  # sample first
                env.scene["deform_object"].parames_generator.high_range = [
                    1.0, 1.0
                ]
                env.scene["deform_object"].parames_generator.low_range = [
                    0.0, 0.0
                ]
            else:  # narrow down to the real range
                env.scene[
                    "deform_object"].parames_generator.random_method = "sim_distri"
                env.scene[
                    "deform_object"].parames_generator.sim_distri = sim_params_distr
                # collect eval video
                eval_prms, eval_traj_states, eval_traj_acts, eval_traj_rgb, eval_params_smpls_classifer = collect_trajectories(
                    5,
                    env,
                    device='cuda:0',
                    state_name=state_name,
                    camera_obs=True,
                    vis_camera=False,
                    camera_function=object_3d_seg_rgb)
                save_classifier_video(
                    eval_traj_rgb[:, start_index:end_index:interval],
                    eval_params_smpls_classifer, "logs/video/eval")

                writer.images_to_video(
                    eval_traj_rgb[0].cpu().numpy()[..., ::-1],
                    os.path.join(args_cli.logdir,
                                 f"eval_{real_iter_id}_video.mp4"))
            env.reset()
            n_trajs_done = 0
            index = 0
            while n_trajs_done < n_train_trajs:
                n_trajs_per_batch = BayesSim.get_n_trajs_per_batch(
                    n_train_trajs, n_trajs_done)
                print('Collect', n_trajs_per_batch, 'trajs')

                sim_prms, sim_traj_states, sim_traj_acts, sim_traj_rgb, sim_params_smpls_classifer = collect_trajectories(
                    env.scene["deform_object"].cfg.deform_cfg['bayessim']
                    ["trainTrajs"],
                    env,
                    device='cuda:0',
                    state_name=state_name,
                    vis_camera=False)

                # for index in range(sim_traj_rgb.shape[1]):
                #     rgb_images = sim_traj_rgb[:,index]
                #     image = rgb_images.permute(1, 0, 2, 3).reshape(rgb_images.shape[1],rgb_images.shape[2] * 4, 3).numpy()
                #     cv2.imshow("image", image[...,::-1])
                #     cv2.waitKey(20)
                print('Train BayesSim...')

                log_bsim = bsim.run_training(
                    sim_prms, sim_traj_states[:,
                                              start_index:end_index:interval],
                    sim_traj_acts[:, start_index:end_index:interval])
                n_trajs_done += n_trajs_per_batch
                print(f'n_trajs_done {n_trajs_done:d} (of {n_train_trajs:d})')

                sim_params_distr = bsim.predict(
                    real_traj_states[0, start_index:end_index:interval][None],
                    real_traj_acts[0, start_index:end_index:interval][None])

                print(sim_params_distr.gen(), real_prms[0])

                del sim_prms
                del sim_traj_states
                del sim_traj_acts
                gc.collect()
                torch.cuda.empty_cache()

                writer.add_scalar('BayesSim/train_loss',
                                  log_bsim['train_loss'][-1], real_iter_id)
                writer.add_scalar('BayesSim/test_loss',
                                  log_bsim['test_loss'][-1], real_iter_id)
                writer.flush()
                sys.stdout.flush()
                os.makedirs(os.path.join(args_cli.logdir, 'bsim',
                                         "checkpoints"),
                            exist_ok=True)
                bsim.save_weight(
                    os.path.join(args_cli.logdir, 'bsim', "checkpoints",
                                 f"{real_iter_id}_{index}.pth"))
                index += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
