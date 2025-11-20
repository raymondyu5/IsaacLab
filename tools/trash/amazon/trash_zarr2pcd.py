import zarr

import zarr
import time
import cv2
import numpy as np
import imageio
from tools.visualization_utils import *

import open3d as o3d


def list_zarr_files(root_dir):
    zarr_files = []
    for root, dirs, files in os.walk(root_dir):
        # Check all directories (since .zarr is usually a folder)
        for d in dirs:
            if d.lower().endswith(".zarr"):
                zarr_files.append(os.path.join(root, d))
        # (Optional) also check for rare cases where .zarr is a file
        for f in files:
            if f.lower().endswith(".zarr"):
                zarr_files.append(os.path.join(root, f))
    return zarr_files


import os

source_dir = "logs/trash/video/image_pcd"
all_zarr_files = list_zarr_files(
    "logs/data_1007/teleop_data/grasp/abs_grasp_image/ycb")
os.makedirs(source_dir, exist_ok=True)
import re

# Sort by the integer number in each filename
all_zarr_files.sort(
    key=lambda x: int(re.search(r'episode_(\d+)\.zarr', x).group(1)))

for file in all_zarr_files[::20]:
    print(f"Processing {file}...")
    zarr_data = zarr.open(file, mode='r')

    # Assuming the images are stored under a group named 'images'
    point_cloud = np.array(zarr_data['data/seg_pc'])

    video_filename = os.path.join(
        source_dir,
        os.path.basename(file).replace('.zarr', '_pcd.mp4'))
    pcd_writer = imageio.get_writer(video_filename,
                                    fps=30)  # Adjust fps as needed

    # Create a video writer

    for i in range(len(point_cloud)):
        pcd = vis_pc(point_cloud[i])
        visualize_pcd(
            [pcd],
            rotation_axis=[0, 1],
            rotation_angles=[1.57, 2.3],
            translation=[0.2, 0.4, 0.5],
            video_writer=pcd_writer,
        )

    pcd_writer.close()
    print(f"Saved video to {video_filename}")

    images_group = np.array(zarr_data['data/rgb_0'])

    # Create a video writer
    video_filename = os.path.join(
        source_dir,
        os.path.basename(file).replace('.zarr', '_image.mp4'))
    image_writer = imageio.get_writer(video_filename,
                                      fps=30)  # Adjust fps as needed

    for i in range(len(images_group)):
        img_array = images_group[i][:]
        image_writer.append_data(img_array)

    image_writer.close()
    print(f"Saved video to {video_filename}")
