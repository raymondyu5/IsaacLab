# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import sys

sys.path.append(".")

from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

import argparse

import time
from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

from isaaclab.utils import Timer
import isaaclab.utils.math as math_utils
from scripts.workflows.bimanual.args_setting import parser
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

from scripts.workflows.bimanual.utils.attachment_utils import initialize_gripper
from scripts.workflows.bimanual.utils.data_utilis import initialize_buffers
from scripts.workflows.bimanual.utils.file_utils import create_log_dirs

from scripts.workflows.bimanual.utils.file_utils import save_params_to_yaml

from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles

enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw
from tools.curobo_planner import IKPlanner


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

    init_pos = torch.as_tensor([0.5, 0.0, 0.30, 0.0, 1.0, 0.0,
                                0.0]).to(env.unwrapped.device)
    delta_quat = obtain_target_quat_from_multi_angles(
        [2, 1], [np.pi / 2, 0]).to(env.unwrapped.device)
    init_pos[3:] = math_utils.quat_mul(delta_quat, init_pos[3:])
    init_pos = init_pos.unsqueeze(0).repeat_interleave(env.num_envs, 0)

    curobo_ik = IKPlanner()

    init_qpos = curobo_ik.plan_motion(init_pos[:, :3], init_pos[:, 3:])
    robot_target_qpos = env.scene[
        "robot"]._data.default_joint_pos[:, :9].clone()

    robot_target_qpos[:] = init_qpos.squeeze(1)

    indices = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)

    while simulation_app.is_running():

        # sample actions from -1 to 1
        actions = torch.rand(env.action_space.shape,
                             device=env.unwrapped.device) * 0
        env.scene["robot"].root_physx_view.set_dof_positions(
            robot_target_qpos, indices)
        env.scene["robot"].root_physx_view.set_dof_velocities(
            robot_target_qpos * 0, indices)

        next_obs, reward, terminate, time_out, info = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
