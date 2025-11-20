import numpy as np
from tools.visualization_utils import vis_pc, visualize_pcd, crop_points, sample_fps
import imageio

import h5py
import os
import torch
import omni
import isaaclab.utils.math as math_utils

from scipy.spatial.transform import Rotation as R

video_writer = imageio.get_writer(f"logs/video/test.mp4", fps=30)
data = np.load("/home/lme/Downloads/ee_act_replay.npy",
               allow_pickle=True).item()
video_writer_image = imageio.get_writer(f"logs/video/test_image.mp4", fps=30)
for i in range(len(data["obs"])):
    print(i)
    pc = data["obs"][i]["point_cloud"]
    xyz = np.array(data["obs"][i]["point_cloud"])

    quaternion = math_utils.obtain_target_quat_from_multi_angles(
        [0, 1], [np.pi / 2, np.pi]).numpy()
    rotation_matrix = R.from_quat(
        [quaternion[1], quaternion[2], quaternion[3],
         quaternion[0]])  # Note the order [x, y, z, w]
    rotation_matrix = rotation_matrix.as_matrix()

    # Rotate the point cloud
    xyz = xyz @ rotation_matrix.T

    # Define the rotation matrix for a 45-degree rotation around the y-axis

    # pc_rotated[:, 2] -= 0.4
    xyz[:, 0] += 0.3
    xyz[:, 1] += 0.4

    o3d = vis_pc(xyz)

    visualize_pcd([o3d], video_writer=video_writer)

    video_writer_image.append_data(data["obs"][i]["213522250963_rgb"])
