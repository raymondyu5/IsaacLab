import torch

from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test, fps_points
import json
import h5py
import os
import shutil
import numpy as np

# from dgl.geometry import farthest_point_sampler
import isaaclab.utils.math as math_utils
import copy
import imageio
import gzip

import zarr
import time
import cv2
import sys


from tools.visualization_utils import *


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


class MultiDatawrapper:

    def __init__(self,
                 args_cli,
                 env_config,
                 filter_keys=[],
                 use_fps=False,
                 load_path=None,
                 save_path=None,
                 train_percentate=0.9,
                 normalize_action=False,
                 use_joint_pos=False,
                 save_npz=True,
                 save_zip=False,
                 save_zarr=False):

        self.args_cli = args_cli

        self.env_config = env_config
        self.filter_keys = filter_keys

        self.train_percentate = train_percentate
        self.save_zarr = save_zarr
        # self.save_zarr = False

        self.use_fps = use_fps

        self.save_npz = save_npz
        self.traj_count = 0
        self.use_joint_pos = use_joint_pos
        self.save_zip = save_zip
        self.collector_interface_h5py = None

        if save_path is not None:
            self.save_path = self.args_cli.log_dir + "/" + save_path
            os.makedirs(self.save_path, exist_ok=True)

            filenames = os.listdir(self.save_path)
            counts = []

            for f in filenames:
                name = os.path.splitext(f)[0]  # Remove extension
                parts = name.split(".")[0].split("_")
                try:
                    count = int(
                        parts[-1])  # Try to get the last part as integer
                    counts.append(count)
                except ValueError:
                    continue  # Skip files that don't end in a number

            self.traj_count = max(counts) + 1 if counts else 0  # s

        if load_path is not None:
            self.load_path = self.args_cli.log_dir + f"/{load_path}.hdf5"
            self.load_h5py()
            if save_path is not None and normalize_action:
                self.normalize_h5py()

        if normalize_action:
            self.load_normalization_stats()

    def init_collectors(self, num_demos, filename):

        self.collector_interface_h5py = RobomimicDataCollector(
            self.args_cli.task, self.save_path, filename, num_demos)
        self.collector_interface_h5py.reset()

        save_config_json = json.dumps(self.env_config)
        self.collector_interface_h5py._h5_data_group.attrs[
            "env_setting"] = save_config_json
        return self.collector_interface_h5py

    def init_collector_interface(self, save_path=None):
        if save_path is None:
            save_path = "demo"

        if self.args_cli.save_path is not None:

            self.collector_interface = self.init_collectors(
                self.args_cli.num_demos, filename=save_path)

    def h5py_group_to_dict(self, h5_group):
        result = {}
        for key in h5_group:
            item = h5_group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[(
                )]  # Load dataset into memory (as NumPy array or scalar)
            elif isinstance(item, h5py.Group):
                result[key] = self.h5py_group_to_dict(item)  # Recursive call
        return result

    def load_h5py(self):
        self.raw_data = {}
        with h5py.File(f"{self.load_path}", 'r') as file:
            self.raw_data = self.h5py_group_to_dict(file)

    def add_demonstraions_to_buffer(self,
                                    obs_buffer,
                                    actions_buffer,
                                    rewards_buffer,
                                    does_buffer,
                                    next_obs_buffer=None,
                                    external_filename=None):
        stop = False
        start = time.time()

        if not self.save_npz:
            stop = self.save_to_h5py(obs_buffer, actions_buffer,
                                     rewards_buffer, does_buffer,
                                     next_obs_buffer)
            self.traj_count += 1
        elif self.save_zarr:
            self.save_to_zarr(obs_buffer, actions_buffer, rewards_buffer,
                              does_buffer, next_obs_buffer)

        else:
            self.save_to_npz(obs_buffer,
                             actions_buffer,
                             rewards_buffer,
                             does_buffer,
                             next_obs_buffer,
                             external_filename=external_filename)

            if self.traj_count == (self.args_cli.num_demos - 1):
                stop = True
            if self.traj_count == 0:
                np.save(f"{self.save_path}/env_setting.npy", self.env_config)
            self.traj_count += 1

        print("time taken to save:", time.time() - start)
        return stop

    def downsample_points(self, obs_buffer):
        handle_points_buffer = []
        pc_buffer = []

        if "seg_pc" in obs_buffer[0].keys():
            for index in range(len(obs_buffer)):
                obs = obs_buffer[index]
                if "seg_pc" in obs.keys():
                    points = obs["seg_pc"]

                    pc_buffer.append(points)
                if "handle_points" in obs.keys():
                    handle_points = obs["handle_points"]
                    handle_points_buffer.append(handle_points)
            point_clouds = torch.cat(pc_buffer, dim=0)
            sample_points = fps_points(point_clouds)
            if "handle_points" in obs.keys():
                handle_points_buffer = torch.cat(handle_points_buffer, dim=0)
                sample_points = torch.cat(
                    [sample_points, handle_points_buffer], dim=1)
            print(sample_points.size())
        return sample_points

    def filter_obs_buffer(self, obs_buffer):

        save_obs_buffer = []

        for index, obs in enumerate(obs_buffer):

            per_obs = {}

            for keys in obs.keys():
                if keys in self.filter_keys:
                    continue

                per_obs[keys] = obs[keys]

            save_obs_buffer.append(per_obs)
        return save_obs_buffer

    def save_to_npz(self,
                    obs_buffer,
                    actions_buffer,
                    rewards_buffer,
                    does_buffer,
                    next_obs_buffer,
                    external_filename=None):
        if external_filename is not None:
            filename = os.path.join(
                self.save_path,
                f"episode_{external_filename}_{self.traj_count}.npz")
        else:
            filename = os.path.join(self.save_path,
                                    f"episode_{self.traj_count}.npz")

        save_obs_buffer = self.filter_obs_buffer(obs_buffer)
        if next_obs_buffer is not None:
            save_next_obs_buffer = self.filter_obs_buffer(next_obs_buffer)
        else:
            save_next_obs_buffer = None

        data = {
            'obs': save_obs_buffer,
            'actions': actions_buffer,
            'rewards': rewards_buffer,
            'dones': does_buffer,
            "next_obs": save_next_obs_buffer
        }
        if self.save_zip:
            with gzip.open(f"{filename}.gz", "wb") as f:
                torch.save(data, f, pickle_protocol=4)
        else:
            torch.save(data, filename)

        print(f"Saved episode {self.traj_count} to {filename}", "length",
              len(actions_buffer))

    def save_to_zarr(self, obs_buffer, actions_buffer, rewards_buffer,
                     does_buffer, next_obs_buffer):

        num_target_env = rewards_buffer[0].shape[0]
        save_obs_buffer = self.filter_obs_buffer(obs_buffer)
        actions_buffer = torch.stack(actions_buffer, dim=0).cpu().numpy()
        rewards_buffer = torch.stack(rewards_buffer, dim=0).cpu().numpy()
        does_buffer = torch.stack(does_buffer, dim=0).cpu().numpy()
        for target_env_id in range(num_target_env):
            # try:
            filename = os.path.join(self.save_path,
                                    f"episode_{self.traj_count}.zarr")

            if os.path.exists(filename):
                shutil.rmtree(filename)

            zarr_root = zarr.group(filename, )
            zarr_data = zarr_root.create_group('data')

            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

            zarr_data.create_dataset(
                'actions',
                data=actions_buffer[:, target_env_id],
                chunks=actions_buffer[:, target_env_id].shape,
                dtype='float32',
                overwrite=True,
                compressor=compressor)

            zarr_data.create_dataset(
                'rewards',
                data=rewards_buffer[:, target_env_id],
                chunks=rewards_buffer[:, target_env_id].shape,
                dtype='float32',
                overwrite=True,
                compressor=compressor)

            zarr_data.create_dataset('dones',
                                     data=does_buffer[:, target_env_id],
                                     chunks=does_buffer[:,
                                                        target_env_id].shape,
                                     dtype='bool',
                                     overwrite=True,
                                     compressor=compressor)

            for index in range(len(save_obs_buffer)):
                if index == 0:
                    obs_dict = {}

                    for key in obs_buffer[0].keys():
                        if key not in self.filter_keys: obs_dict[key] = []

                obs = save_obs_buffer[index]

                for key, value in obs.items():
                    if key not in self.filter_keys:

                        obs_dict[key].append(
                            value[target_env_id].unsqueeze(0).cpu().numpy(
                            ) if isinstance(value, torch.Tensor
                                            ) else value[target_env_id][None])
                # import pdb
                # pdb.set_trace()

                # visualize_pcd(
                #     [vis_pc(obs["seg_pc"][0].reshape(-1, 3).cpu().numpy())])

                # plt.imshow(obs["rgb"][0][0].cpu().numpy())
                # plt.show()

            for key, value in obs_dict.items():

                if key in ["rgb"]:

                    value = np.concatenate(value, axis=0)

                    for num_channel in range(value.shape[1]):

                        zarr_data.create_dataset(
                            key + f"_{num_channel}",
                            data=value[:, num_channel],
                            chunks=value[:, num_channel].shape,
                            dtype='uint8',
                            overwrite=True,
                            compressor=compressor)

                else:
                    if key == "seg_pc":

                        max_points = max([v.shape[-2] for v in value])
                        pcd_list = []
                        for pcd in value:
                            if isinstance(pcd, torch.Tensor):
                                pcd = pcd.cpu().numpy().reshape(1, -1, 3)
                            else:
                                pcd = pcd.reshape(1, -1, 3)
                            if pcd.shape[-2] < max_points:
                                padding_num = max_points - pcd.shape[-2]
                                pcd = np.concatenate(
                                    [pcd, pcd[:, :padding_num, :]], axis=1)
                            pcd_list.append(pcd)

                        value = np.concatenate(pcd_list, axis=0)

                        zarr_data.create_dataset(key,
                                                 data=value,
                                                 chunks=value.shape,
                                                 dtype='float32',
                                                 overwrite=True,
                                                 compressor=compressor)

                    else:

                        value = np.concatenate(value, axis=0)

                        zarr_data.create_dataset(key,
                                                 data=value,
                                                 chunks=value.shape,
                                                 dtype='float32',
                                                 overwrite=True,
                                                 compressor=compressor)

        # except Exception as e:
        #     print(f"Error saving to zarr, skipping this episode: {e}")
        #     continue

            self.traj_count += 1

        print(f"Saved episode {self.traj_count} to {filename}", "length",
              len(save_obs_buffer))

    def save_to_h5py(self, obs_buffer, actions_buffer, rewards_buffer,
                     does_buffer):

        obs = obs_buffer
        rewards = rewards_buffer
        dones = does_buffer
        for key, value in obs["policy"].items():
            if key in self.filter_keys or key in ["id2lables"]:
                continue

            try:

                self.collector_interface.add(f"obs/{key}", value)
            except:
                import pdb
                pdb.set_trace()
        self.collector_interface.add("actions", actions_buffer)
        self.collector_interface.add("rewards", rewards)

        self.collector_interface.add("dones", dones)
        if dones[0]:
            self.traj_count += 1

        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        self.collector_interface.flush(reset_env_ids)
        torch.cuda.empty_cache()
        return False

    def load_normalization_stats(self):
        if not self.use_joint_pos:
            self.action_stats = np.load(self.args_cli.log_dir + f"/stats.npy",
                                        allow_pickle=True).item()
        else:
            self.action_stats = np.load(self.args_cli.log_dir +
                                        f"/stats_joint_pos.npy",
                                        allow_pickle=True).item()

    def normalize_h5py(self):

        self.action_stats, self.raw_data = self.normalize_ations(self.raw_data)

    def normalize_ations(self, data):

        actions_buffer = []

        if not self.use_joint_pos:
            for demo_id in range(len(data["data"].keys())):

                actions = data["data"][f"demo_{demo_id}"]["actions"]
                actions_buffer.append(actions)

            all_actions = np.concatenate(actions_buffer, axis=0)  #[..., :3]

            stats = {
                "action": {
                    "min": all_actions.min(axis=0),
                    "max": all_actions.max(axis=0),
                }
            }
            # Save stats to a separate file
            np.save(self.args_cli.log_dir + f"/stats.npy", stats)
        else:
            for demo_id in range(len(data["data"].keys())):

                actions = data["data"][f"demo_{demo_id}"]["obs"][
                    "control_joint_action"]
                actions_buffer.append(actions)
            all_actions = np.concatenate(actions_buffer, axis=0)[..., :8]

            all_actions[:, -1] = np.sign(all_actions[:, -1] - 0.01)

            stats = {
                "action": {
                    "min": all_actions.min(axis=0),
                    "max": all_actions.max(axis=0),
                }
            }
            np.save(self.args_cli.log_dir + f"/stats_joint_pos.npy", stats)

        # # Normalize actions for each demo and save them to the copied HDF5 file
        # for demo_id in range(len(data["data"].keys())):
        #     actions = data["data"][f"demo_{demo_id}"]["actions"]

        #     # Normalize the actions using the calculated stats
        #     actions_buffer = self.normalize(actions, stats["action"])

        #     data["data"][f"demo_{demo_id}"].create_dataset("actions",
        #                                                    data=actions_buffer)

        return stats, data

    def normalize(self, arr, stats):
        min_val, max_val = stats["min"], stats["max"]
        return 2 * (arr - min_val) / (max_val - min_val) - 1

    def unnormalize(self, arr, stats):

        min_val, max_val = stats["min"], stats["max"]

        if isinstance(arr, torch.Tensor):
            max_val = torch.tensor(max_val, device=arr.device)
            min_val = torch.tensor(min_val, device=arr.device)

            result = (0.5 * (arr + 1) * (max_val - min_val) + min_val)
        else:
            result = 0.5 * (arr + 1) * (max_val - min_val) + min_val

        return result

    def split_set(self):

        h5_file = h5py.File(f"{self.save_path}", 'a')
        sample_train_test(h5_file)

    def save_video(self, video_name, image_buffer):
        video_writer = imageio.get_writer(f"{video_name}.mp4", fps=10)

        for image in image_buffer:
            video_writer.append_data(image)
        video_writer.close()
