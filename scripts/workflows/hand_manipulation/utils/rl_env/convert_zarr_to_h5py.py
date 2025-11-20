from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import pickle
import json
import zarr

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# # # launch omniverse app

from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test
import json
import h5py
import os
import shutil
import numpy as np
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

args_cli = parser.parse_args()
import torch


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy floats to Python floats
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)  # Convert NumPy ints to Python ints
    return obj


class Converter:

    def __init__(
        self,
        args_cli,
        filter_keys,
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path

        self.filter_keys = filter_keys

        self.init_collector_interface()

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(
            self.args_cli.task, os.path.dirname(self.save_path), filename,
            num_demos)
        collector_interface.reset()

        return collector_interface

    def init_collector_interface(self):

        self.collector_interface = self.init_collectors(
            self.args_cli.num_demos, filename=self.save_path.split("/")[-1])

    def convert(self):

        for file in os.listdir(f"{self.load_path}"):
            if file.endswith(".zarr"):

                data = zarr.open(os.path.join(f"{self.load_path}", file),
                                 mode='r')

                obs_keys = []
                data.visit(lambda k: obs_keys.append(k)
                           if k.startswith("data/") and k != "data" else None)

                actions_buffer = np.array(data["data/actions"])
                print(f"actions_buffer shape: {actions_buffer.shape}")
                for i in range(actions_buffer.shape[0]):
                    self.collector_interface.add("actions",
                                                 actions_buffer[i][None])

                    for key in obs_keys:

                        obs_key = key.replace("data/", "")

                        if "actions" in key or obs_key in self.filter_keys:

                            continue

                        value = np.array(data[key])

                        self.collector_interface.add(f"obs/{obs_key}",
                                                     value[i][None])
                    dones = np.zeros(1, dtype=bool)
                    if i == actions_buffer.shape[0] - 1:

                        dones[:] = True

                    reset_env_ids = np.nonzero(dones)[0]
                    self.collector_interface.add("dones", dones)

                    self.collector_interface.flush(reset_env_ids)


Converter(
    args_cli,
    filter_keys=["segmentation", "seg_rgb", "id2lables", "seg_pc", "rgb_0"],
).convert()
