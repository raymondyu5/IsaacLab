import zarr

import zarr
import time
import cv2
import numpy as np
import imageio


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

all_zarr_files = list_zarr_files("logs/trash/eval_image_bc/image")
source_dir = "logs/trash/video/eval_image_bc"
os.makedirs(source_dir, exist_ok=True)
import re

# Sort by the integer number in each filename
all_zarr_files.sort(
    key=lambda x: int(re.search(r'episode_(\d+)\.zarr', x).group(1)))

for file in all_zarr_files:
    print(f"Processing {file}...")
    zarr_data = zarr.open(file, mode='r')

    # Assuming the images are stored under a group named 'images'
    images_group = np.array(zarr_data['data/rgb_0'])

    # Create a video writer
    video_filename = os.path.join(
        source_dir,
        os.path.basename(file).replace('.zarr', '.mp4'))
    writer = imageio.get_writer(video_filename, fps=30)  # Adjust fps as needed

    for i in range(len(images_group)):
        img_array = images_group[i][:]
        writer.append_data(img_array)

    writer.close()
    print(f"Saved video to {video_filename}")
