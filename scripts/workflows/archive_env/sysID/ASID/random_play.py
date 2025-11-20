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

import imageio
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
import isaaclab.utils.math as math_utils

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

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--parmas_range",
                    type=list,
                    default=None,
                    help="parmas_range.")
parser.add_argument("--debug_vis",
                    action="store_true",
                    help="debug visualization.")

parser.add_argument("--use_gripper",
                    action="store_true",
                    default=False,
                    help="")
parser.add_argument("--num_explore_actions", type=int, default=2, help="")

parser.add_argument("--random_params",
                    type=str,
                    default=None,
                    help="randomness_params.")
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

from scripts.workflows.sysID.ASID.cem.cem import CEM
from scripts.workflows.sysID.ASID.tool.utilis import *
from scripts.workflows.sysID.utils.file_utils import save_params_to_yaml

from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw


def save_fem_mesh(env):

    indices = env.scene["object"].root_physx_view.get_sim_element_indices(
    )[0].cpu().numpy()
    vertices = env.scene["object"].root_physx_view.get_sim_nodal_positions(
    )[0].cpu().numpy()
    simulation_hexahedral_resolution = env.scene["object"].cfg.deform_cfg[
        "simulation_hexahedral_resolution"]
    write_fem_to_obj(
        f"logs/rabbit/deform_fem_{simulation_hexahedral_resolution}/{0}/",
        f"{env.episode_length_buf.cpu().numpy()[0]}.obj", vertices, indices)


def vis_noda_pc(observation):

    noda_position = observation["policy"]["object_node"][0][:, :, :3].view(
        -1, 3)
    noda_position = vis_pc(noda_position.cpu().numpy())
    pc_image = visualize_pcd([noda_position], render=False)
    return pc_image


def extract_rgbd_obs(observation):

    rgb_images = observation["policy"]["camera_observation"].cpu()[..., :3][
        0]  # Shape: (3, 480, 640, 3)
    depth_images = observation["policy"]["camera_observation"].cpu()[..., -1][
        0]  # Shape: (3, 480, 640)

    depth_combined = depth_images.permute(1, 0, 2).reshape(
        depth_images.shape[1], depth_images.shape[2] * 3).numpy()
    rgbd_image = rgb_images.permute(1, 0, 2,
                                    3).reshape(rgb_images.shape[1],
                                               rgb_images.shap2[2] * 3,
                                               3).numpy()

    return rgbd_image, depth_combined


def extract_color_pc(observation):

    # rgb pc visualization
    color_pc = observation["policy"]['color_pc'].cpu().numpy()

    o3d_pc = vis_pc(color_pc[0, :, :3], color_pc[0, :, 3:])
    # pc_image = visualize_pcd([o3d_pc], render=False)

    # # # Combine multiple RGB images into one
    # rgb_images = torch.cat(
    #     [rgb_images, torch.as_tensor(pc_image[None])])
    # # Arrange the images in a 2x2 grid
    # rgb_top = torch.cat((rgb_images[0], rgb_images[1]),
    #                     dim=1)  # Top row
    # rgb_bottom = torch.cat((rgb_images[2], rgb_images[3]),
    #                        dim=1)  # Bottom row
    # rgbd_image = torch.cat((rgb_top, rgb_bottom), dim=0).numpy()
    return o3d_pc


def setup_env(args_cli, save_config):
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


import numpy as np

import os
# import open3d as o3d


def transform_pc(pc, extrinsic_matrix):

    T_A_to_B = extrinsic_matrix
    R = T_A_to_B[:3, :3]
    t = T_A_to_B[:3, 3]

    # Compute inverse rotation andd translation
    R_inv = R.T
    t_inv = -R_inv @ t

    # Assemble the inverse transformation matrix
    T_B_to_A = np.eye(4)
    T_B_to_A[:3, :3] = R_inv
    T_B_to_A[:3, 3] = t_inv
    pc_homogeneous = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))

    # Apply the transformation
    transformed_pc_homogeneous = np.dot(pc_homogeneous, T_B_to_A.T)

    # Convert back to 3D coordinates (discard the homogeneous coordinate)
    transformed_pc = transformed_pc_homogeneous[:, :3]
    pc[:, 3:] /= 255

    pc_color = np.concatenate([transformed_pc, pc[:, 3:]], axis=1)
    transformed_pc = pc_color[:, :3]

    transformed_pc_color = np.array([
        (x, y, z, r, g, b) for index, (x, y, z, r, g, b) in enumerate(pc_color)
        if -0.4 <= x <= 0.4 and -0.4 <= y <= 0.5 and -1 <= z <= -0.05
    ])

    camera_o3d = o3d.geometry.PointCloud()
    camera_o3d.points = o3d.utility.Vector3dVector(transformed_pc_color[:, :3])
    colors = np.zeros_like(camera_o3d.points)

    colors[:] = transformed_pc_color[:, 3:]
    camera_o3d.colors = o3d.utility.Vector3dVector(colors)
    # camera_o3d.remove_statistical_outlier(nb_neighbors=20,
    #                                             std_ratio=2.0)
    transformed_pc_homogeneous = np.asanyarray(camera_o3d.points)
    #ax_plot.scatter(transformed_pc_homogeneous[:,0].reshape(-1), transformed_pc_homogeneous[:,1].reshape(-1), transformed_pc_homogeneous[:,2].reshape(-1), c=color_name)

    return camera_o3d  #.transform(T_A_to_B)


caliberation_files = os.listdir(
    '/media/lme/data4/weird/droid/trash/caliberation')
pc_list = []
transformed_pc = []
for cali_file in caliberation_files:

    cali_data = np.load(
        f'/media/lme/data4/weird/droid/trash/caliberation/{cali_file}',
        allow_pickle=True)

    intrinsic_matrix = cali_data["intrinsic_matrix"]
    dist_coeffs = cali_data["dist_coeffs"]
    extrinsic_matrix = cali_data["extrinsic_matrix"]

    rgbd_data = np.load(f'/media/lme/data4/weird/droid/trash/pc/{cali_file}',
                        allow_pickle=True)

    rgb_image = rgbd_data["rgb_image"]
    depth_image = rgbd_data["depth_image"]

    color_pc = transform_pc(depth_image, extrinsic_matrix)

    pc_list.append(color_pc)

    transformed_pc.append(
        np.concatenate(
            [np.asarray(color_pc.points),
             np.asarray(color_pc.colors)], axis=1))

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[-0.0, 0.0, 0.0])
transformed_pc = np.concatenate(transformed_pc, axis=0)
indices = np.random.choice(transformed_pc.shape[0], 2048 * 10, replace=False)
transformed_pc_downsampled = transformed_pc  #[indices, :]
o3d.visualization.draw_geometries(pc_list + [coordinate_frame])

transformed_pc_downsampled[:, 2] *= -1


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    save_config, config = save_params_to_yaml(args_cli)
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    # simulate environment
    # init_pos = torch.as_tensor([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                             0.0]).to(env.unwrapped.device)
    init_pos = torch.as_tensor([0.4, 0.0, 0.50, 0.0, 1.0, 0.0,
                                0.0]).to(env.unwrapped.device)

    for i in range(10):
        env.sim.step()
    index = 0
    draw = _debug_draw.acquire_debug_draw_interface()
    while simulation_app.is_running():

        # sample actions from -1 to 1
        actions = torch.rand(env.action_space.shape,
                             device=env.unwrapped.device) * 0

        # # if env.episode_length_buf[0] < 10:
        # actions[:, :-2] = init_pos
        # actions[:, -2:] = 1

        next_obs, reward, terminate, time_out, info = env.step(actions)

        # draw.draw_points(
        #     transformed_pc_downsampled[:, :3],
        #     np.concatenate([
        #         transformed_pc_downsampled[:, 3:],
        #         np.ones((len(transformed_pc_downsampled), 1))
        #     ],
        #                    axis=1), [10] * len(transformed_pc_downsampled))
        # apply actions
        index += 1

        # image = next_obs["policy"]["joint_qpos"][0, 0].cpu().numpy()

        # # Convert from RGB to BGR for OpenCV
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("test.png", (image_bgr).astype(np.uint8))

        pcd = o3d.geometry.PointCloud()

        seg_pc = next_obs["policy"]["seg_pc"][0].cpu().numpy()
        seg_pc = seg_pc.reshape(-1, seg_pc.shape[-1])
        np.save("cat_pc.npy", seg_pc)
        # # seg_pc[:, :3] -= env.scene.env_origins[3].cpu().numpy()

        pcd.points = o3d.utility.Vector3dVector(seg_pc[:, :3])

        pcd.colors = o3d.utility.Vector3dVector(seg_pc[:, 3:6] / 255)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd] + [coordinate_frame])

        # # Show the image
        # cv2.imshow("image", image_bgr)

        # # Add a short delay to avoid blocking
        # cv2.waitKey(1)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
