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
from torchvision.utils import save_image
import torchvision
import torch.optim as optim
from scripts.workflows.sysID.bayes_sim.utils.data_utils import collect_trajectories
from scripts.workflows.sysID.vid2param.models.vrnn import *
from scripts.workflows.sysID.vid2param.utils.vid2param_dataset import BallDataset
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

parser.add_argument('--back_frames', type=int, default=10)
# Number of prediction frames
parser.add_argument('--forw_frames', type=int, default=10)
# Beta for the KL divergence
parser.add_argument('--beta', type=int, default=1)
# Alpha for the Phys
parser.add_argument('--alpha', type=int, default=10)
# Gamma for the reconstruction
parser.add_argument('--gamma', type=int, default=1)
# Theta for the same z encoding
parser.add_argument('--theta', type=int, default=1)
# Size of the latent space
parser.add_argument('--z_dim', type=int, default=2)
# Size of the hidden space
parser.add_argument('--h_dim', type=int, default=400)
# Original paper - feeding the image into the recurrent bit?
parser.add_argument('--original_paper', type=int, default=0)
parser.add_argument('--dataset_type', type=str, default=None)
# Decoder or not
parser.add_argument('--decoder', type=int, default=0)
# append AppLauncher cli args
parser.add_argument('--niters', type=int, default=2000)

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

    env.reset()
    model = VRNN(x_dim=50 * 100 * 1,
                 h_dim=args_cli.h_dim,
                 z_dim=args_cli.z_dim,
                 n_layers=2,
                 alpha=args_cli.alpha,
                 beta=args_cli.beta,
                 gamma=args_cli.gamma,
                 device=args_cli.device,
                 original_paper=args_cli.original_paper,
                 dataset=args_cli.dataset_type,
                 decoder=args_cli.decoder).to(args_cli.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    state_name = "deform_pos_w"
    # note: the number of steps might vary depending on how complicated the scene is.
    for _ in range(12):

        env.sim.step()

    sim_prms_list = []
    sim_traj_rgb_list = []

    while simulation_app.is_running():

        for i in range(5):

            sim_prms, sim_traj_states, sim_traj_acts, sim_traj_rgb, sim_params_smpls_classifer = collect_trajectories(
                5,
                env,
                device='cuda:0',
                state_name=state_name,
                camera_obs=True,
                vis_camera=False,
                camera_function=object_3d_seg_rgb)
            sim_prms_list.append(sim_prms.cpu().numpy())
            sim_traj_rgb_list.append(sim_traj_rgb.cpu().numpy())

            print("Done", (i + 1) * 100)

            save_classifier_video(sim_traj_rgb, sim_params_smpls_classifer,
                                  "logs/video")

            del sim_prms
            del sim_traj_states
            del sim_traj_acts
            del sim_traj_rgb
            gc.collect()
            torch.cuda.empty_cache()

        break

    env.close()

    sim_prms = np.concatenate(sim_prms_list)
    sim_traj_rgb = np.concatenate(sim_traj_rgb_list)[:, 6:26]

    train_dset = BallDataset(args_cli.back_frames, args_cli.forw_frames, 32,
                             32, 3, args_cli.device, [sim_traj_rgb, sim_prms])
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)
    from tensorboardX import SummaryWriter
    sw = SummaryWriter('runs/test')
    # batch_images_to_video(
    #     sim_traj_rgb[4][:20].cpu().numpy()[..., ::-1], "test.mp4", 10)
    # print(aa)
    itr = 0

    for epoch in range(1, args_cli.niters):
        for batch_idx, (input, output, time,
                        params) in enumerate(train_loader):

            model.train()
            model.count = epoch
            data = torch.cat((input, output), 1)
            data = data.to(args_cli.device).float()
            if (batch_idx == 0 and epoch == 1):
                if (args_cli.dataset_type == "mnist"):
                    sw.add_video('logs/vid2param/data',
                                 data[:min(16, 8)],
                                 epoch,
                                 fps=30)
                else:
                    sw.add_video('logs/vid2paramdata',
                                 data[:min(16, 8)],
                                 epoch,
                                 fps=30)

            optimizer.zero_grad()
            kld_loss, nll_loss, phys_loss, _, dec, h, _, _, _, _ = model(
                data, params=params)
            if (args_cli.decoder):
                loss = kld_loss + phys_loss + nll_loss
            else:
                loss = kld_loss + phys_loss
            current_batch_size = input.shape[0]
            sw.add_scalar('loss', loss / current_batch_size, itr)
            sw.add_scalar('kld_loss',
                          kld_loss / args_cli.beta / current_batch_size, itr)
            sw.add_scalar('nll_loss',
                          nll_loss / args_cli.gamma / current_batch_size, itr)
            sw.add_scalar('phys_loss',
                          phys_loss / args_cli.alpha / current_batch_size, itr)
            print(loss / current_batch_size,
                  phys_loss / args_cli.alpha / current_batch_size)
            itr += 1
            loss.backward()
            optimizer.step()

            # if epoch % 5 == 0 and batch_idx == 0:
            #     from utils.tools import *
            #     acc_latent_space(test_loader, 'Test', model, args_cli.device,
            #                      params_intervals, args_cli, sw, epoch,
            #                      args.back_frames + args_cli.forw_frames)
            #     test_pred_model(test_loader, 'Test', model, args_cli.device,
            #                     params_intervals, args_cli, sw, epoch,
            #                     args_cli.decoder)
            #     torch.save(model.state_dict(),
            #                'logs/vid2param/weight' + str(epoch))
            #     if (not args_cli.train):
            #         sw.close()
            #         sys.exit()
        sw.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
