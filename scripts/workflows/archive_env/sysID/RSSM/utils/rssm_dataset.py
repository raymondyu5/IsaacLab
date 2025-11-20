import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms, datasets
import numpy as np
import cv2
import os
import sys
import h5py
import pdb

from collections import deque
from numpy.random import choice


class RssmDataset:

    def __init__(self,
                 data_dir,
                 device=None,
                 start_index=8,
                 end_index=28,
                 interval=2,
                 resize=None):
        self.device = device

        h5py_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        self.sim_rgb = []
        self.sim_prms = []

        for file in h5py_files:
            with h5py.File(f"{data_dir}/{file}", 'r') as file:
                num_demos = len(file["data"].keys())
                for num_demo in range(num_demos):
                    rgb_images = torch.as_tensor(
                        file["data"][f"demo_{num_demo}"]["obs"]["sim_traj_rgb"]
                        [0])[start_index:end_index + interval:interval]

                    if resize is not None:
                        resize_transform = transforms.Resize(resize)

                        rgb = resize_transform(rgb_images.permute(
                            0, 3, 1, 2)).permute(0, 2, 3, 1)
                    self.sim_rgb.append(rgb.numpy())
                    self.sim_prms.append(
                        (file["data"][f"demo_{num_demo}"]["obs"]["sim_prms"][0]
                         ))
            del file

        self.episodes_num = len(self.sim_rgb)
        self.sim_rgb = np.array(self.sim_rgb)
        self.sim_prms = np.array(self.sim_prms)

        # Generate the evaluation set and training set indices
        self.eval_indices = self._generate_eval_indices()
        self.train_indices = np.setdiff1d(np.arange(self.episodes_num),
                                          self.eval_indices)

    def _generate_eval_indices(self):
        # Calculate 10% of the episodes
        num_eval_episodes = max(1, int(self.episodes_num * 0.1))

        # Randomly select 10% of episodes
        eval_indices = choice(self.episodes_num,
                              num_eval_episodes,
                              replace=False)

        return eval_indices

    def sample(self, batch_size, time_first=True, for_eval=False):

        if for_eval:
            episode_idx = choice(self.eval_indices, batch_size)
        else:
            episode_idx = choice(self.train_indices, batch_size)

        sample_rbg = torch.as_tensor(self.sim_rgb[episode_idx]).to(
            self.device) / 255 - 0.5
        sample_prms = torch.as_tensor(self.sim_prms[episode_idx]).to(
            self.device)

        sample_u = (
            1 / sample_rbg.shape[1] *
            torch.arange(sample_rbg.shape[1] - 1))[None].repeat_interleave(
                len(sample_rbg), 0).to(self.device)

        if time_first:
            return sample_rbg.permute(1, 0, 4,
                                      2, 3), sample_prms, sample_u.permute(
                                          1, 0)[..., None]
        return sample_rbg, sample_prms, sample_u[..., None]

    def eval_set(self):
        eval_rgb = torch.as_tensor(self.sim_rgb[self.eval_indices]).to(
            self.device) / 255 - 0.5
        eval_prms = torch.as_tensor(self.sim_prms[self.eval_indices]).to(
            self.device)

        return eval_rgb, eval_prms
