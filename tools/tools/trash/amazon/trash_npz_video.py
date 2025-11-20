import numpy as np

import torch
import imageio
import pickle
import zarr

data = zarr.open("logs/result_trash/banana/episode_0.zarr", mode='r')
# data = torch.load("logs/trash/banana/episode_0.npz", pickle_module=pickle)

video = imageio.get_writer("logs/trash/banana/test.mp4", fps=20)
for index in range(len(data["data/rgb"])):

    image = np.array(data["data/rgb"][index][0])
    video.append_data(image.astype(np.uint8))
