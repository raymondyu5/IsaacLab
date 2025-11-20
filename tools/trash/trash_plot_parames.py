import sys

sys.path.append("/media/aurmr/data1/weird/IsaacLab")

from scripts.workflows.sysID.ASID.tool.utilis import plot_2d_result, plot_3d_result, plot_1d_result, make_2d_video, make_1d_video

import numpy as np

import os

data_dir = "logs/0930/cem/sysID/"
combinations = os.listdir(data_dir)

for name in combinations:
    num_params = 1
    num_params = np.floor(len(name.split("_")) / 2)
    print(name, num_params)
    if int(num_params) == 1:
        plot_1d_result(log_path=f"{data_dir}/{name}")
        make_1d_video(log_path=f"{data_dir}/{name}")
    if int(num_params) == 2:
        plot_2d_result(log_path=f"{data_dir}/{name}", min_batch=15)
        make_2d_video(log_path=f"{data_dir}/{name}")
    if int(num_params) == 3:
        plot_3d_result(log_path=f"{data_dir}/{name}")
