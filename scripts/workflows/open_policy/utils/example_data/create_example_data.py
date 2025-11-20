from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import pickle
import json
from scripts.workflows.utils.client.openvla_client import resize_image

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
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path + "/train", exist_ok=True)
        os.makedirs(self.save_path + "/val", exist_ok=True)

    def save_data(self, data, save_path):

        obs = data['obs']
        actions = data['actions']

        episode = []
        for step in range(len(obs)):
            episode.append({
                'image':
                resize_image(
                    obs[step]["policy"]["gs_image"][0][0].cpu().numpy(),
                    (224, 224)),
                'action':
                actions[step].cpu().numpy(),
                'language_instruction':
                'put eggplant into yellow basket',
            })

        np.save(f'{save_path}', episode)

    def convert(self):
        # npz_files = [
        #     file for file in os.listdir(self.load_path)
        #     if file.endswith('.npz')
        # ]
        num_traj = args_cli.num_demos
        num_train = int(num_traj * 0.8)
        num_test = num_traj - num_train

        for index in range(num_train):

            data = torch.load(os.path.join(f"{self.load_path}",
                                           f"episode_{index}.npz"),
                              pickle_module=pickle)

            self.save_data(data, f"{self.save_path}/train/episode_{index}.npy")
        for index in range(num_train, num_traj):
            data = torch.load(os.path.join(f"{self.load_path}",
                                           f"episode_{index}.npz"),
                              pickle_module=pickle)
            self.save_data(data, f"{self.save_path}/val/episode_{index}.npy")

        return False


Converter(args_cli, ).convert()
