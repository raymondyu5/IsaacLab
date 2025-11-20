from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import pickle
import json

parser.add_argument("--data_type", default=None)

parser.add_argument("--synthesize_pc", action="store_true")
parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

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

args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.synthesize_pc import SynthesizePC
import yaml


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy floats to Python floats
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)  # Convert NumPy ints to Python ints
    return obj


import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Converter:

    def __init__(
        self,
        args_cli,
        filter_keys,
        noramilize_action=False,
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir
        self.save_dir = args_cli.log_dir
        self.end_name = args_cli.end_name
        self.add_noise = args_cli.noise_pc
        self.filter_keys = filter_keys

        if self.add_noise:
            self.noise_level = 2.0
            self.noise_name = "_noise"
        else:
            self.noise_level = 0
            self.noise_name = ""

        self.noramilize_action = noramilize_action

        self.category_name = list(os.listdir(args_cli.log_dir))

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(
            self.args_cli.task, self.save_dir + "/raw_data", filename,
            num_demos)
        collector_interface.reset()

        return collector_interface

    def init_collector_interface(self, object_name):

        self.collector_interface = self.init_collectors(
            self.args_cli.num_demos, filename=object_name)

    def filter_obs_buffer(self, obs_buffer, index, indices, obs_name="obs"):

        if "policy" in obs_buffer[index]:
            obs = obs_buffer[index]["policy"]
        else:
            obs = obs_buffer[index]

        for key, value in obs.items():
            if key in self.filter_keys:
                continue
            if not isinstance(value, torch.Tensor):
                continue

            else:

                if len(value.size()) == 1:
                    value = value.unsqueeze(0)

            self.collector_interface.add(f"{obs_name}/{key}", value[indices])

    def convert(self):

        try:

            self.env_config = np.load(f"{self.load_path}/env_setting.npy",
                                      allow_pickle=True)

            config = self.env_config.item()

            # Use the `default` parameter of json.dumps to handle non-serializable objects
            json_string = json.dumps(config,
                                     default=convert_to_serializable,
                                     indent=4)
            self.collector_interface._h5_data_group.attrs[
                "env_setting"] = json_string
        except:
            pass
        files = os.listdir(f"{self.load_path}{self.end_name}")
        files.sort()
        for object_name in self.category_name:
            self.init_collector_interface(object_name)
            count_num = 0

            for file in files:
                if not file.endswith(".npz"):
                    continue
                count_num += 1
                if count_num > self.args_cli.num_demos:
                    break
                if self.collector_interface._is_stop:
                    break

                index = int(file.split(".")[0].split("_")[1])
                object_list = list(
                    np.loadtxt(self.load_path + f"/object_name_{index}.txt",
                               dtype=str))

                data = torch.load(os.path.join(f"{self.load_path}", file),
                                  pickle_module=pickle)

                obs_buffer = data["obs"]

                actions_buffer = data["actions"]

                rewards_buffer = data["rewards"]

                does_buffer = data["dones"]

                print("trajectories length: ", len(obs_buffer))
                indices = [
                    i for i, name in enumerate(object_list)
                    if name == object_name
                ]

                for index in range(len(obs_buffer)):

                    self.filter_obs_buffer(obs_buffer,
                                           index,
                                           indices,
                                           obs_name="obs")

                    rewards = rewards_buffer[index][indices]

                    if index == len(obs_buffer) - 1:

                        if len(actions_buffer) == 2:

                            dones = torch.tensor([True], device='cuda:0')
                        else:
                            dones = torch.tensor(
                                [True] * actions_buffer[index].shape[0],
                                device='cuda:0')
                    else:

                        if len(actions_buffer) == 2:
                            dones = torch.tensor([False], device='cuda:0')
                        else:
                            dones = torch.tensor(
                                [False] * actions_buffer[index].shape[0],
                                device='cuda:0')

                    self.collector_interface.add(
                        "actions", actions_buffer[index][indices])

                    self.collector_interface.add("rewards", rewards)

                    self.collector_interface.add("dones", dones[indices])

                    reset_env_ids = dones[indices].nonzero(
                        as_tuple=False).squeeze(-1)

                    self.collector_interface.flush(reset_env_ids)

                    torch.cuda.empty_cache()

            # sample_train_test(
            #     h5py.File(self.save_dir + f"/raw_data/{object_name}.hdf5",
            #               'r+'))
            print(
                "Data for object category '{}' has been converted and saved.".
                format(object_name))
            self.collector_interface.close()

        return False


Converter(
    args_cli,
    filter_keys=[
        "segmentation", "seg_rgb", "id2lables", "seg_pc", "object_verts",
        "lhand_faces", "rhand_faces", "lhand_verts", "rhand_verts",
        "object_transformation"
    ],
).convert()
