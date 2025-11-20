import os

from collections import defaultdict

from tqdm import tqdm

import numpy as np
import copy

from functools import cached_property
import cv2
import torch

import json
from tools.visualization_utils import vis_pc, visualize_pcd
import isaaclab.utils.math as math_utils


def load_calibration_info(calib_info_filepath, keep_time=False):
    if not os.path.isfile(calib_info_filepath):
        return {}
    with open(calib_info_filepath, "r") as jsonFile:
        calibration_info = json.load(jsonFile)

    if not keep_time:
        calibration_info = {
            key: data["pose"]
            for key, data in calibration_info.items()
        }
    return calibration_info


def crop_point_cloud(points_xyz_rgb, bbox):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    bbox_mask = (points_xyz_rgb[..., 0] >= x_min) & (points_xyz_rgb[
        ..., 0] <= x_max) & (points_xyz_rgb[..., 1]
                             >= y_min) & (points_xyz_rgb[..., 1] <= y_max) & (
                                 points_xyz_rgb[..., 2]
                                 >= z_min) & (points_xyz_rgb[..., 2] <= z_max)

    crop_pcd = points_xyz_rgb[bbox_mask]
    return crop_pcd, bbox_mask


class ResidualRealPCDDatasetLoader:

    def __init__(self,
                 data_path,
                 load_list,
                 obs_key,
                 num_demo=0,
                 downsample_points=2048,
                 camera_id=["CL838420160"],
                 crop_region=None):
        self.data_path = data_path
        self.load_list = load_list
        self.obs_key = obs_key
        self.crop_region = crop_region

        self.downsample_points = downsample_points
        self.num_demo = num_demo

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
        obs_buffer["seg_pc"] = []
        print("Load List", self.load_list)
        demo_count = 0

        num_pcd_data = 0

        for demo_id, path_key in enumerate(self.load_list):

            if not os.path.isdir(self.data_path + "/" + path_key):
                continue
            npy_pathes = os.listdir(self.data_path + "/" + path_key)
            npy_pathes.sort()

            for _, cam_id in enumerate(self.camera_id):
                print("loading camer id ", cam_id)
                for file_id in tqdm(range(min([self.num_demo,
                                               len(npy_pathes)])),
                                    desc="Loading demos"):

                    npy_path = os.path.join(
                        self.data_path + "/" + path_key,
                        f"episode_{file_id}/episode_{file_id}.npy")

                    if not os.path.exists(npy_path):
                        continue
                    print(f"episode_{file_id}")

                    # Load the numpy file
                    env_info = np.load(npy_path, allow_pickle=True).item()
                    state_info = env_info['obs']

                    # low level obs
                    num_horiozon = len(state_info)

                    for index in range(num_horiozon):
                        step_state_info = state_info[index]
                        self.low_obs_dim = 0

                        for obs_name in self.obs_key:
                            if obs_name == "base_action":
                                continue

                            value = np.array(step_state_info[obs_name])

                            obs_buffer[obs_name].append(value)
                            self.low_obs_dim += value.shape[-1]
                    pcd_list = sorted([
                        self.data_path + "/" + path_key +
                        f"/episode_{file_id}/" + f
                        for f in os.listdir(self.data_path + "/" + path_key +
                                            f"/episode_{file_id}/")
                        if f.endswith(".npy") and "episode" not in f
                        and cam_id in f
                    ])

                    residual_action_dict = np.load(
                        self.data_path + "/" + path_key +
                        f"/delta_action/delta_action_{file_id}.npy",
                        allow_pickle=True).item()
                    obs_buffer["action"].append(
                        copy.deepcopy(
                            np.array(residual_action_dict["delta_action"])))
                    if "base_action" in self.obs_key:
                        obs_buffer["base_action"].append(
                            copy.deepcopy(
                                np.array(
                                    residual_action_dict["predict_action"])))

                    num_steps = len(residual_action_dict["delta_action"])
                    print(len(pcd_list), num_steps)

                    assert len(pcd_list) == num_steps

                    for pcd_path in pcd_list:

                        proccessed_pcd = np.array(np.load(pcd_path)).astype(
                            np.float32).reshape(-1, 3)

                        shuffled_indices = np.arange(proccessed_pcd.shape[0])
                        np.random.shuffle(shuffled_indices)

                        shuffle_pcd_value = (proccessed_pcd[
                            shuffled_indices[:self.downsample_points * 10], :])

                        # shuffle_pcd_value = math_utils.fps_points(
                        #     torch.as_tensor(proccessed_pcd)[None],
                        #     self.downsample_points * 3,
                        # )[0].numpy()

                        # o3d = vis_pc(shuffle_pcd_value[:, :3])
                        # visualize_pcd([o3d])
                        # import pdb
                        # pdb.set_trace()

                        obs_buffer["seg_pc"].append(shuffle_pcd_value.tolist())

                    episode_ends.append(
                        copy.deepcopy(num_steps + episode_count))
                    episode_count += num_steps
                    num_pcd_data += len(pcd_list)
                    if not (episode_count == num_pcd_data):
                        import pdb
                        pdb.set_trace()
                        raise ValueError("Episode count not match")

                    demo_count += 1
        self.action_dim = 22  #action_info.shape[-1]

        self.root['meta']['episode_ends'] = np.array(episode_ends,
                                                     dtype=np.int64)
        assert episode_ends[-1] == episode_count

        self.meta = self.root['meta']
        self.data = self.root['data']
        if "base_action" in self.obs_key:

            obs_buffer["base_action"] = np.concatenate( obs_buffer["base_action"])

        for key in obs_buffer.keys():

            if key in ["action"]:

                action_buffer = np.concatenate(obs_buffer[key], axis=0)
                self.root['data']['action'] = action_buffer
                assert action_buffer.shape[0] == episode_count
            else:

                obs_data = np.array(obs_buffer[key])

                assert len(obs_data) == episode_count

                self.root['data'][key] = obs_data

        print("Total number of episodes: ", demo_count)

        self.root['data']["base_action"] = np.concatenate(obs_buffer["action"],
                                                          axis=0)

        if obs_name == "base_action":
            self.low_obs_dim += self.root['data']["base_action"].shape[-1]

        del obs_buffer
        del action_buffer
        del obs_data
        return self.action_dim, self.low_obs_dim

    def process_pcd(self, extrisic_matrix, pcd_path):
        points_3d = np.array(np.load(pcd_path)).astype(np.float32).reshape(
            -1, 3) - 2000

        points_3d[:, :3] /= 1000

        # Convert to homogeneous coordinates: (H*W, 4)
        ones = np.ones((points_3d[..., :3].shape[0], 1))

        points_homo = np.concatenate([points_3d[:, :3], ones], axis=1)
        # Apply the 4x4 transform
        points_transformed_homo = (
            extrisic_matrix @ points_homo.T).T  # shape: (H*W, 4)

        # Extract 3D part
        points_transformed = points_transformed_homo[:, :3]

        R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        # 180° about Z
        R_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        R = R_z @ R_y

        points_transformed = points_transformed @ R.T
        if self.crop_region is not None:

            points_transformed, bbox_mask = crop_point_cloud(
                points_transformed, self.crop_region)
        return points_transformed

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
