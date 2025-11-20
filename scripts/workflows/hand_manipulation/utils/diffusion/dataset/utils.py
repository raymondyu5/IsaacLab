import torch
import random
from torchvision.transforms import Lambda, Compose
import random
import torchvision.transforms.functional as TVF
from typing import Optional
import numpy as np
import numba


class ColorRandomizer:
    """
    Randomly sample color jitter at input, and then average across color jtters at output.
    """

    def __init__(
        self,
        input_shape,
        brightness=0.7,
        contrast=0.1,
        saturation=0.1,
        hue=0.5,
        num_samples=1,
    ):

        super(ColorRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)

        self.input_shape = input_shape
        self.brightness = [
            max(0, 1 - brightness), 1 + brightness
        ] if type(brightness) in {float, int} else brightness
        self.contrast = [max(0, 1 - contrast), 1 + contrast
                         ] if type(contrast) in {float, int} else contrast
        self.saturation = [
            max(0, 1 - saturation), 1 + saturation
        ] if type(saturation) in {float, int} else saturation
        self.hue = [-hue, hue] if type(hue) in {float, int} else hue
        self.num_samples = num_samples

    @torch.jit.unused
    def get_transform(self):

        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0],
                                               self.brightness[1])
            transforms.append(
                Lambda(
                    lambda img: TVF.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0],
                                             self.contrast[1])
            transforms.append(
                Lambda(lambda img: TVF.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0],
                                               self.saturation[1])
            transforms.append(
                Lambda(
                    lambda img: TVF.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])

            transforms.append(
                Lambda(lambda img: TVF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def get_batch_transform(self, N):

        return Lambda(lambda x: torch.stack(
            [self.get_transform()(x_) for x_ in x for _ in range(N)]))


class GaussianNoiseRandomizer:

    def __init__(
        self,
        input_shape,
        noise_mean=0.0,
        noise_std=0.05,
        limits=None,
    ):

        super(GaussianNoiseRandomizer, self).__init__()

        self.input_shape = input_shape
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.limits = limits

    def forward(self, inputs):
        noise_inputs = np.random.randn(
            *inputs.shape) * self.noise_std + self.noise_mean + inputs

        return np.clip(noise_inputs, 0, 1)
