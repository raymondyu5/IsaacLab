import os

from collections import defaultdict

from tqdm import tqdm

import numpy as np
import copy

from functools import cached_property
import cv2
import torch

import shutil


class RealImageDatasetLoader:

    def __init__(
        self,
        data_path,
        load_list,
        obs_key,
        image_tf=None,
        num_demo=0,
        crop_region=None,
        camera_id=None,
    ):
        self.data_path = data_path
        self.load_list = load_list
        self.obs_key = obs_key

        self.image_tf = image_tf
        self.num_demo = num_demo
        self.crop_region = crop_region
        self.camera_id = camera_id
        self.load_data()

    def load_data(self):
        # Implement the logic to load data from the specified path

        self.root = {'meta': {}, 'data': {}}
        self.action_dim = 0
        if "all" in self.load_list:
            self.load_list = os.listdir(self.data_path)
            self.load_list.sort()

        episode_ends = []
        episode_count = 0
        action_buffer = []

        obs_buffer = defaultdict(list)
        print("Load List", self.load_list)
        demo_count = 0
        for demo_id, path_key in enumerate(self.load_list):

            if not os.path.isdir(self.data_path + "/" + path_key):
                continue
            npy_pathes = os.listdir(self.data_path + "/" + path_key)
            npy_pathes.sort()
            obs_buffer = defaultdict(list)
            obs_buffer["image"] = []
            for _, cam_id in enumerate(self.camera_id):
                for file_id in tqdm(range(min([self.num_demo,
                                               len(npy_pathes)])),
                                    desc="Loading Zarr demos"):

                    npy_path = os.path.join(
                        self.data_path + "/" + path_key,
                        f"episode_{file_id}/episode_{file_id}.npy")
                    if not os.path.exists(npy_path):
                        continue

                    # Load the numpy file
                    env_info = np.load(npy_path, allow_pickle=True).item()
                    state_info = env_info['obs']
                    action_info = np.array(env_info['actions'])
                    obs_buffer["action"].append(copy.deepcopy(action_info))

                    # low level obs
                    num_horiozon = len(state_info)

                    for index in range(num_horiozon):
                        step_state_info = state_info[index]
                        self.low_obs_dim = 0

                        for obs_name in self.obs_key:

                            value = np.array(step_state_info[obs_name])

                            obs_buffer[obs_name].append(value)
                            self.low_obs_dim += value.shape[-1]
                    image_list = sorted([
                        self.data_path + "/" + path_key +
                        f"/episode_{file_id}/" + f
                        for f in os.listdir(self.data_path + "/" + path_key +
                                            f"/episode_{file_id}/")
                        if f.endswith(".png") and cam_id in f
                    ])
                    obs_buffer["image"] += image_list

                    num_steps = len(action_info)
                    episode_ends.append(
                        copy.deepcopy(num_steps + episode_count))
                    episode_count += num_steps

                    demo_count += 1

        self.action_dim = action_info.shape[-1]

        self.root['meta']['episode_ends'] = np.array(episode_ends,
                                                     dtype=np.int64)
        assert episode_ends[-1] == episode_count

        self.meta = self.root['meta']
        self.data = self.root['data']

        for key in obs_buffer.keys():

            if key in ["action"]:

                action_buffer = np.concatenate(obs_buffer[key], axis=0)
                self.root['data']['action'] = action_buffer
                assert action_buffer.shape[0] == episode_count
            else:

                if key == "image_file":
                    obs_data = obs_buffer[key]

                else:

                    obs_data = np.array(obs_buffer[key])

                assert len(obs_data) == episode_count

                self.root['data'][key] = obs_data

        print("Total number of episodes: ", demo_count)

        del obs_buffer
        del action_buffer
        del obs_data

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']

    @cached_property
    def meta(self):
        return self.root['meta']

    @property
    def episode_ends(self):
        return self.meta['episode_ends']

    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1], ), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
