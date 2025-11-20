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
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path

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

        self.init_collector_interface()
        if self.args_cli.synthesize_pc:

            save_config, config = save_params_to_yaml(args_cli,
                                                      args_cli.log_dir)
            # create environment

            save_config["params"]["add_right_hand"] = args_cli.add_right_hand
            save_config["params"]["add_left_hand"] = args_cli.add_left_hand

            self.synthesize_pc_function = SynthesizePC(save_config)

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(
            self.args_cli.task, os.path.dirname(self.save_path), filename,
            num_demos)
        collector_interface.reset()

        return collector_interface

    def init_collector_interface(self):

        self.collector_interface = self.init_collectors(
            self.args_cli.num_demos,
            filename=self.save_path.split("/")[-1] + self.end_name +
            self.noise_name)

    def filter_obs_buffer(self, obs_buffer, index, obs_name="obs"):

        if "policy" in obs_buffer[index]:
            obs = obs_buffer[index]["policy"]
        else:
            obs = obs_buffer[index]

        for key, value in obs.items():
            if key in self.filter_keys:
                continue
            if not isinstance(value, torch.Tensor):
                continue
                # if isinstance(value[0], str):
                value = np.array(value,
                                 dtype=h5py.string_dtype(encoding='utf-8'))
            else:

                # if "joint_pos" in key:
                #     value += (torch.rand(value.shape).to(value.device) * 2 -
                #               1) * 0.10

                if len(value.size()) == 1:
                    value = value.unsqueeze(0)

            self.collector_interface.add(f"{obs_name}/{key}", value)

    def continue_or_not(self, obs_buffer, data_type):

        manipulated_object_name = obs_buffer[0]["policy"][
            "manipulate_object_name"]
        if len(manipulated_object_name[0]) == 0:
            return False
        lift_height = obs_buffer[-1]["policy"][manipulated_object_name[0]
                                               [0]][0][2]

        return (lift_height > 0.10).cpu().tolist() & data_type

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

        for file in files:
            if file.endswith(".npz"):

                data = torch.load(os.path.join(
                    f"{self.load_path}{self.end_name}", file),
                                  pickle_module=pickle)

                move_on = True

                obs_buffer = data["obs"]

                if self.args_cli.data_type == "success":
                    move_on = self.continue_or_not(obs_buffer, True)
                elif self.args_cli.data_type == "failure":
                    move_on = self.continue_or_not(obs_buffer, False)
                else:
                    move_on = True
                if not move_on:
                    continue
                actions_buffer = data["actions"]

                rewards_buffer = data["rewards"]

                # if obs_buffer[-1]["policy"]["mustard_bottle"][:, 2] < 0.15:
                #     continue
                does_buffer = data["dones"]

                print("trajectories length: ", len(obs_buffer))

                for index in range(len(obs_buffer)):

                    if self.args_cli.synthesize_pc:
                        self.synthesize_pc_function.synthesize_pc_offline(
                            obs_buffer[index])

                    self.filter_obs_buffer(obs_buffer, index, obs_name="obs")

                    rewards = rewards_buffer[index]
                    # if not isinstance(obs_buffer, list):

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
                        "actions", actions_buffer[index].reshape(
                            -1, actions_buffer[index].shape[-1]))

                    self.collector_interface.add("rewards", rewards)

                    self.collector_interface.add("dones", dones)

                    reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

                    self.collector_interface.flush(reset_env_ids)

                    torch.cuda.empty_cache()
        file_name = self.save_path.split("/")[-1]

        if self.args_cli.normalize_action:
            self.normalize_h5py(
                h5py.File(
                    os.path.dirname(self.save_path) + f"/{file_name}" +
                    self.end_name + self.noise_name + ".hdf5", 'r+'))

        sample_train_test(
            h5py.File(
                os.path.dirname(self.save_path) + f"/{file_name}" +
                self.end_name + self.noise_name + ".hdf5", 'r+'))

        return False

    def normalize_h5py(self, raw_data):

        self.normalize_ations(raw_data)

    def normalize(self, arr, stats):
        min_val, max_val = stats["min"], stats["max"]
        return 2 * (arr - min_val) / (max_val - min_val) - 1

    def normalize_ations(self, data):

        actions_buffer = []

        for demo_id in range(len(data["data"].keys())):

            actions = np.array(data["data"][f"demo_{demo_id}"]["actions"])

            actions_buffer.append(actions)

        all_actions = np.concatenate(actions_buffer, axis=0)

        stats = {
            "action": {
                "min": all_actions.min(axis=0),
                "max": all_actions.max(axis=0),
            }
        }
        print(stats)

        for demo_id in range(len(data["data"].keys())):

            actions = data["data"][f"demo_{demo_id}"]["actions"]
            del data["data"][f"demo_{demo_id}"]["actions"]

            normalized_actions = self.normalize(actions, stats["action"])
            data["data"][f"demo_{demo_id}"]["actions"] = normalized_actions
            # if max(abs(np.array(
            #         data["data"][f"demo_{demo_id}"]["actions"]))) > 1.0:
            #     import pdb
            #     pdb.set_trace()
        # Save stats to a separate file
        np.save(self.args_cli.log_dir + f"/stats.npy", stats)


Converter(
    args_cli,
    filter_keys=[
        "segmentation", "seg_rgb", "id2lables", "seg_pc", "object_verts",
        "lhand_faces", "rhand_faces", "lhand_verts", "rhand_verts",
        "object_transformation"
    ],
    noramilize_action=False,
).convert()
