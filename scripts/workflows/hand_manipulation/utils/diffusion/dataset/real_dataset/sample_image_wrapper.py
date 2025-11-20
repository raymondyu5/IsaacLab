from typing import Optional
import numpy as np
import numba
import zarr
import os
import torch
import random
from torchvision.transforms import Lambda, Compose
import random
import torchvision.transforms.functional as TVF

from PIL import Image
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.utils import ColorRandomizer, GaussianNoiseRandomizer
import cv2
import torchvision.transforms.functional as TF


@numba.jit(nopython=True)
def create_indices(episode_ends: np.ndarray,
                   sequence_length: int,
                   episode_mask: np.ndarray,
                   pad_before: int = 0,
                   pad_after: int = 0,
                   debug: bool = True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length,
                                 episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx -
                        sample_start_idx) == (buffer_end_idx -
                                              buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, sample_start_idx,
                sample_end_idx
            ])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs),
                                    size=n_train,
                                    replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class RealImageSequenceSampler:

    def __init__(
        self,
        replay_buffer,
        image_tf: tuple,
        image_dim: tuple,
        path: str,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        load_list: Optional[list] = None,
        add_randomizer: bool = False,
        crop_region: Optional[tuple] = None,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert (sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())

        self.path = path
        self.image_dim = image_dim
        self.load_list = load_list
        self.crop_region = crop_region
        self.add_randomizer = add_randomizer
        if self.add_randomizer:
            self.color_randomizer = ColorRandomizer(self.image_dim)
            self.gaussian_randomizer = GaussianNoiseRandomizer(
                self.image_dim, )

        self.image_tf = image_tf

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends,
                                     sequence_length=sequence_length,
                                     pad_before=pad_before,
                                     pad_after=pad_after,
                                     episode_mask=episode_mask)
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def load_image(self, key, value):

        img = cv2.imread(value[0], cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB

        img_tensor = torch.as_tensor(np.array(img)).permute(
            2, 0, 1)  # (H,W,C) -> (C,H,W)
        img = (self.image_tf(img_tensor).permute(1, 2, 0).numpy()[..., ::-1] *
               255).astype(np.uint8)
        if self.add_randomizer:
            img = (self.gaussian_randomizer.forward(img / 255) * 255).astype(
                np.uint8)

            pil_img = Image.fromarray(img)
            color_jitter = self.color_randomizer.get_transform()
            img_aug = color_jitter(pil_img)

            # Convert back to numpy, then to CHW
            img = np.asarray(img_aug, dtype=np.uint8)

        return (img / 255).transpose(2, 0, 1)[None]  # (H,W,C) -> (C,H,W)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]

        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible

            if key not in self.key_first_k:
                if key == "image":

                    sample = self.load_image(
                        key, input_arr[buffer_start_idx:buffer_end_idx])

                else:

                    sample = input_arr[buffer_start_idx:buffer_end_idx]

            else:
                # TODO: need to debug this case
                import pdb
                pdb.set_trace()

                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data, ) + input_arr.shape[1:],
                                 fill_value=np.nan,
                                 dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[
                        buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb
                    pdb.set_trace()
            data = sample

            if (sample_start_idx > 0) or (sample_end_idx
                                          < self.sequence_length):
                data = np.zeros(shape=(self.sequence_length, ) +
                                input_arr.shape[1:],
                                dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
