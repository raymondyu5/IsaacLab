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
                 save_zip=False):

        self.args_cli = args_cli

        self.env_config = env_config
        self.filter_keys = filter_keys

        self.train_percentate = train_percentate

        self.use_fps = use_fps

        self.save_npz = save_npz
        self.traj_count = 0
        self.use_joint_pos = use_joint_pos
        self.save_zip = save_zip

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

    def init_collectors(self, num_demos, filename):

        collector_interface = RobomimicDataCollector(self.args_cli.task,
                                                     self.save_path, filename,
                                                     num_demos)
        collector_interface.reset()

        save_config_json = json.dumps(self.env_config)
        collector_interface._h5_data_group.attrs[
            "env_setting"] = save_config_json
        return collector_interface

    def init_collector_interface(self):

        if self.args_cli.save_path is not None:

            # self.collector_interface = self.init_collectors(
            #     2000, filename=f"demo_{self.traj_count}")
            os.makedirs(self.save_path + f"/demo_{self.traj_count}",
                        exist_ok=True)
        self.step_count = 0

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

    def add_demonstraions_to_buffer(self,
                                    obs_buffer,
                                    actions_buffer,
                                    rewards_buffer,
                                    does_buffer,
                                    next_obs_buffer=None):

        # self.save_to_h5py(obs_buffer, actions_buffer, rewards_buffer,
        #                   does_buffer)

        data = {
            'obs': obs_buffer,
            'actions': actions_buffer,
            'rewards': rewards_buffer,
            'dones': does_buffer,
        }
        torch.save(
            data,
            self.save_path + f"/demo_{self.traj_count}/{self.step_count}")

    def save_to_h5py(self, obs_buffer, actions_buffer, rewards_buffer,
                     does_buffer):

        for key, value in obs_buffer.items():
            if key in self.filter_keys:
                continue

            self.collector_interface.add(f"obs/{key}", value)
        self.collector_interface.add("actions", actions_buffer)
        self.collector_interface.add("rewards", rewards_buffer)

        self.collector_interface.add("dones", does_buffer)

        reset_env_ids = does_buffer.nonzero(as_tuple=False).squeeze(-1)

        self.collector_interface.flush(reset_env_ids)
        torch.cuda.empty_cache()
        return False

    def filter_out_data(self, index):
        import pdb
        pdb.set_trace()
