import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms, datasets
import numpy as np
import cv2
import os
import sys


class BallDataset(Dataset):

    def __init__(self,
                 frames_input,
                 frames_output,
                 image_size_x,
                 image_size_y,
                 color_channels,
                 gpu,
                 dataset,
                 changing_physics=0):
        self.to_tensor = transforms.ToTensor()
        self.device = gpu
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.color_channels = color_channels
        #self.data = torch.from_numpy(np.load)

        self.images = dataset[0]
        self.params = dataset[1]

        self.multiplier = 1
        self.changing_physics = changing_physics

    def __getitem__(self, index):
        images = self.images[index]
        images = torch.as_tensor(images)
        # MNIST
        if len(images.shape) == 3:
            input = images[:self.frames_input *
                           self.multiplier:self.multiplier].view(
                               self.frames_input, self.color_channels, self.
                               image_size_x, self.image_size_y) / 255 - 0.5
            output = images[self.frames_input * self.multiplier:
                            (self.frames_input + self.frames_output) *
                            self.multiplier:self.multiplier].view(
                                self.frames_output, self.color_channels, self.
                                image_size_x, self.image_size_y) / 255 - 0.5
        # REAL
        else:
            input = images[:self.frames_input * self.multiplier:self.
                           multiplier].permute(0, 3, 1, 2) / 255 - 0.5
            output = images[self.frames_input * self.multiplier:
                            (self.frames_input + self.frames_output) *
                            self.multiplier:self.multiplier].permute(
                                0, 3, 1, 2) / 255 - 0.5

        if (self.changing_physics):
            params = torch.as_tensor(
                self.params[index * self.changing_physics:(index + 1) *
                            self.changing_physics, :])
        else:
            params = torch.as_tensor(self.params[index, :])
        time = torch.from_numpy(
            np.linspace(0, 10, self.frames_input + self.frames_output))
        return input.to(self.device), output.to(self.device), time.to(
            self.device), params.to(self.device)

    def __len__(self):
        return self.images.shape[0]
