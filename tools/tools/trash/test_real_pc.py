import numpy as np
from tools.visualization_utils import vis_pc, visualize_pcd, crop_points, sample_fps
import imageio

import h5py
import os
import torch
import omni
import isaaclab.utils.math as math_utils

from scipy.spatial.transform import Rotation as R

os.makedirs("logs/video/", exist_ok=True)

normalized_grasp = h5py.File(
    f"logs/1118_cabine_y/cabinet_normalized_noise_aug.hdf5", 'r+')
video_writer = imageio.get_writer(f"logs/video/test.mp4", fps=30)
for i in range(2, len(normalized_grasp["data"])):
    print(i)
    # video_writer = imageio.get_writer(f"logs/video/pc_{i}.mp4", fps=30)
    data = normalized_grasp["data"][f"demo_{i}"]

    pcd = torch.as_tensor(data["obs"]["seg_pc"])[..., :3]

    # pcd = sample_fps(pcd)
    if "imagin_robot" in data["obs"].keys():

        pcd = torch.cat(
            [pcd, torch.as_tensor(data["obs"]["imagin_robot"])], dim=1)

    for k in range(34, 44):

        xyz = np.array(pcd[k, :, :3].numpy())

        # quaternion = math_utils.obtain_target_quat_from_multi_angles(
        #     [0, 1, 0], [np.pi / 2, -np.pi * 0.4, np.pi * 0.1]).numpy()
        # rotation_matrix = R.from_quat(
        #     [quaternion[1], quaternion[2], quaternion[3],
        #      quaternion[0]])  # Note the order [x, y, z, w]
        # rotation_matrix = rotation_matrix.as_matrix()

        # # Rotate the point cloud
        # xyz = xyz @ rotation_matrix.T

        # # Define the rotation matrix for a 45-degree rotation around the y-axis

        # # pc_rotated[:, 2] -= 0.4
        # xyz[:, 0] -= 0.4
        # xyz[:, 1] += 0.4

        o3d = vis_pc(xyz)

        visualize_pcd([o3d], video_writer=None)
    # video_writer.close()
