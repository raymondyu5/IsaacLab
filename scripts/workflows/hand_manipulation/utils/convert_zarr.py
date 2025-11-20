import zarr
import numpy as np

import h5py

import torch
from tools.visualization_utils import *

import shutil
import pickle
import json
import argparse

parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--log_dir", default=None)

parser.add_argument("--save_path", default=None)
parser.add_argument("--load_list",
                    nargs='+',
                    default=None,
                    help="List of load paths")


class ZarrConverter:

    def __init__(
        self,
        args_cli,
        filter_keys,
    ):
        self.args_cli = args_cli
        self.load_list = args_cli.load_list
        self.save_dir = args_cli.log_dir + "/" + args_cli.save_path

        self.filter_keys = filter_keys

    def filter_obs_buffer(self, obs_buffer, index, obs_dict):

        if "policy" in obs_buffer[index]:
            obs = obs_buffer[index]["policy"]
        else:
            obs = obs_buffer[index]

        for key, value in obs.items():
            if key in self.filter_keys:
                continue
            if not isinstance(value, torch.Tensor):
                continue

            value = value

            obs_dict[key].append(value.cpu().numpy(
            ) if isinstance(value, torch.Tensor) else value)

    def convert(self):
        if "all" in self.load_list:
            self.load_list = os.listdir(self.args_cli.log_dir)

        for load_path in self.load_list:
            load_dir = self.args_cli.log_dir + "/" + load_path

            files = os.listdir(f"{load_dir}")
            files.sort()

            for file in files:
                if not file.endswith(".npz"):
                    continue
                file_name = file.split(".")[0]

                os.makedirs(self.save_dir + f"/{load_path}", exist_ok=True)

                zarr_root = zarr.group(
                    self.save_dir + f"/{load_path}/{file_name}.zarr", )
                zarr_data = zarr_root.create_group('data')
                zarr_meta = zarr_root.create_group('meta')

                data = torch.load(os.path.join(f"{load_dir}", file),
                                  pickle_module=pickle)

                obs_buffer = data["obs"]

                actions_buffer = data["actions"]

                print("trajectories length: ", len(obs_buffer))

                for index in range(len(obs_buffer)):
                    if index == 0:
                        obs_dict = {}

                        for key in obs_buffer[0].keys():
                            if key not in self.filter_keys:
                                obs_dict[key] = []

                    self.filter_obs_buffer(obs_buffer, index, obs_dict)

                compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

                actions_buffer = torch.cat(actions_buffer, dim=0)

                zarr_data.create_dataset('actions',
                                         data=actions_buffer.cpu().numpy(),
                                         chunks=actions_buffer.shape,
                                         dtype='float32',
                                         overwrite=True,
                                         compressor=compressor)
                for key, value in obs_dict.items():
                    value = np.concatenate(value, axis=0)
                    if key == "rgb":
                        image_shape = value.shape[-3:]
                        value = value.reshape(-1, *image_shape)

                    zarr_data.create_dataset(key,
                                             data=value,
                                             chunks=value.shape,
                                             dtype='float32',
                                             overwrite=True,
                                             compressor=compressor)


args_cli = parser.parse_args()
ZarrConverter(
    args_cli,
    filter_keys=[
        "segmentation", "seg_rgb", "id2lables", "object_verts", "lhand_faces",
        "rhand_faces", "lhand_verts", "rhand_verts", "object_transformation"
    ],
).convert()
