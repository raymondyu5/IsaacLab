import os
import re
import cv2
import numpy as np
import imageio
import open3d as o3d
import zarr
import time

from tools.visualization_utils import visualize_pcd, vis_pc  # assuming your function is here

# -----------------------------
# Configuration
# -----------------------------
source_dir = "logs/trash/video/real_rfs_image_pcd"
episode_root = "logs/trash/residual_data/bunny"
target_cam_id = ["CL838420160", "CL8384200N1"]

os.makedirs(source_dir, exist_ok=True)
episode_files = sorted(os.listdir(episode_root))

# -----------------------------
# Main loop
# -----------------------------
for episode in episode_files[20:]:
    episode_path = os.path.join(episode_root, episode)
    if not os.path.isdir(episode_path):
        continue  # skip non-directory files

    print(f"\nProcessing episode: {episode}")

    # Store all found PNGs for this episode
    png_files = []
    for cam_id in target_cam_id:
        matched_pngs = [
            os.path.join(episode_path, f) for f in os.listdir(episode_path)
            if f.lower().endswith(".png") and cam_id in f
        ]

        matched_pngs.sort()
        pcd_writer = imageio.get_writer(source_dir +
                                        f"/{episode}_{cam_id}_pc_video.mp4",
                                        fps=30)
        rgb_writer = imageio.get_writer(source_dir +
                                        f"/{episode}_{cam_id}_rgb_video.mp4",
                                        fps=30)

        for png_file in matched_pngs:

            png_image = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
            png_image = cv2.resize(png_image, (480, 480))
            pcd_file = png_file.replace('.png', '.npy')
            pcd_data = np.load(pcd_file)
            pcd = vis_pc(pcd_data)

            visualize_pcd(
                [pcd],
                rotation_axis=[0, 1],
                rotation_angles=[1.57, 2.2],
                translation=[0.2, 0.4, 0.5],
                video_writer=pcd_writer,
            )
            rgb_writer.append_data(
                cv2.cvtColor(png_image[:, :, :3], cv2.COLOR_BGR2RGB))
        pcd_writer.close()
        rgb_writer.close()
