import numpy as np
from tools.visualization_utils import vis_pc, visualize_pcd, crop_points
import imageio
import isaaclab.utils.math as math_utils
import h5py
import os
import torch
import cv2

dir_path = "logs/1215_cabinet01_top_cabinet/"
hdf5_path = "failure/x_offset/failure_data_x_offset.hdf5"
video_path = dir_path + "/video/" + hdf5_path.split(".")[-2]

os.makedirs(video_path, exist_ok=True)
lang_buffer = []
# os.makedirs(f"{dir_path}/images", exist_ok=True)

normalized_grasp = h5py.File(f"{dir_path}/{hdf5_path}", 'r+')

for i in range(len(normalized_grasp["data"])):

    pick_video_writer = imageio.get_writer(f"{video_path}/video_{i}.mp4",
                                           fps=30)

    data = normalized_grasp["data"][f"demo_{i}"]

    rgb = np.array(data["obs"]["rgb"])
    if "language_intruction" in data["obs"]:
        lang_buffer.append(
            f"video_{i}:" +
            data["obs"]["language_intruction"][0].decode("utf-8"))

    for image_id, images in enumerate(rgb):

        pick_video_writer.append_data(np.concatenate(images, axis=1))

    # cv2.imwrite(f"{dir_path}/images/{i}/rgb_{image_id}.png",
    #             cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
    # # pick_video_writer.close()
    # # place_video_writer.close()
# with open(f"{video_path}/language_texts.txt", "w", encoding="utf-8") as file:
#     for text in lang_buffer:
#         file.write(text + "\n")  # Add a newline after each text
