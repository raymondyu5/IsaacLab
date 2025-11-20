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
from scripts.workflows.sysID.ASID.data_utilis import DataBuffer

import imageio
from pytorch3d.loss import chamfer_distance
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
from PIL import Image, ImageDraw, ImageFont


def add_text_to_frame(frame, text):
    # Convert tensor to PIL Image

    frame_img = Image.fromarray((frame).astype(np.uint8))

    # Initialize ImageDraw
    draw = ImageDraw.Draw(frame_img)

    # Define the font size
    font = ImageFont.load_default()

    # Add text to the image
    draw.text((20, 20), text, font=font, font_scale=3, fill=(255, 255, 255))

    # Convert back to tensor
    frame_with_text = torch.from_numpy(np.array(frame_img))

    return frame_with_text


def collect_env_trajectories(env,
                             buffer,
                             delta_pose,
                             gripper_offset_xyz,
                             cache_type,
                             log_path=None,
                             id=None):

    if env.episode_length_buf[0] == 0:
        last_obs = env.reset()[0]

    if cache_type == "target":
        images_buffer = []

    for _ in range(env.max_episode_length):
        transition = {}
        abs_pos = delta_pose[:, :3] * (
            env.episode_length_buf[:, None].repeat_interleave(3, 1)) * 1.5
        abs_quat = delta_pose[:, 3:7]
        actions = torch.cat([abs_pos, abs_quat, delta_pose[:, -1][..., None]],
                            dim=1)
        actions[:, :3] += gripper_offset_xyz

        next_obs, reward, terminate, time_out, info = env.step(actions)

        transition["next_obs"] = next_obs["policy"]
        transition["obs"] = last_obs["policy"]
        transition["reward"] = reward * 0
        transition["action"] = actions
        if cache_type == "target":
            images_buffer.append(object_3d_observation(env, image_name="rgb"))

        buffer.cache_traj(transition, cache_type=cache_type)

        last_obs = next_obs

        del next_obs

    buffer.store_transition(cache_type)
    buffer.clear_cache(cache_type)
    if cache_type == "target":
        return images_buffer


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

    loop_num = 20
    grid_w = 4
    grid_h = 5

    real_sample_parms = (torch.arange(loop_num) / (loop_num - 1))[:, None]

    videos_buffer = []

    while simulation_app.is_running():

        for num_loop in range(loop_num):
            print("=========================")
            print(f"num loop:{num_loop}")

            buffer.clear_buffer(buffer_type="target")

            env.scene[
                "deform_object"].parames_generator.random_method = "customize"

            env.scene[
                "deform_object"].parames_generator.params_range = real_sample_parms[
                    num_loop][None].repeat_interleave(args_cli.num_envs,
                                                      0).cpu().numpy()
            images_buffer = collect_env_trajectories(env,
                                                     buffer,
                                                     delta_pose,
                                                     gripper_offset_xyz,
                                                     cache_type="target",
                                                     log_path=logdir,
                                                     id=num_loop)
            images_buffer = torch.stack(images_buffer)[..., :3]
            images_buffer = images_buffer.permute(0, 1, 3, 4, 2, 5)

            concatenated_images = images_buffer[:, :, :, :, 0, :]
            videos_buffer.append(concatenated_images[:, 0])

        videos_buffer = torch.stack(videos_buffer).permute(1, 0, 2, 3, 4)
        seq_length, batch_size, w, h, dim = videos_buffer.shape

        target_video_buffer = torch.zeros(
            (seq_length, grid_w * w, grid_h * h, dim))

        for i in range(seq_length):
            for j in range(grid_w):
                for k in range(grid_h):
                    frame = videos_buffer[i, j * grid_h + k, :]
                    text = f"softness {j * grid_h + k}"

                    target_video_buffer[i, j * w:(j + 1) * w,
                                        k * h:(k + 1) * h] = torch.as_tensor(
                                            add_text_to_frame(
                                                frame.cpu().numpy(), text))

        torchvision.io.write_video("softness.mp4", target_video_buffer, fps=10)

        # close the simulator
        break
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
