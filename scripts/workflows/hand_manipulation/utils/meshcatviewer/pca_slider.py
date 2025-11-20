import tkinter as tk
from tkinter import ttk
import numpy as np
import socket
import threading
import json
import argparse
import torch
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action


class PCASliderApp:

    def __init__(self, pca_path=None, hand_side="right", device="cpu"):

        # Define joint names and dimensions
        self.raw_joint_names = [
            'j1', 'j0', 'j2', 'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4',
            'j6', 'j7', 'j9', 'j8', 'j10', 'j11'
        ]
        self.isaac_joint_names = [
            'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14',
            'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
        ]
        self.retarget2pin = [
            self.isaac_joint_names.index(name) for name in self.raw_joint_names
        ]
        self.num_hand_joints = len(self.raw_joint_names)
        self.pca_path = pca_path
        self.hand_side = hand_side
        self.device = device
        self.load_pca_data()

    def reconstruct_hand_pose_from_normalized_action(
        self,
        a_scaled,
    ):
        reconstructed_actions = reconstruct_hand_pose_from_normalized_action(
            a_scaled, self.eigen_vectors, self.min_values, self.max_values,
            self.D_mean, self.D_std)

        return reconstructed_actions[self.retarget2pin]

    def load_pca_data(self):
        print(f"Loading PCA from: {self.pca_path}")
        data = np.load(f"{self.pca_path}/{self.hand_side}_pca.npy",
                       allow_pickle=True).item()

        self.eigen_vectors = np.array(data["eigen_vectors"])  # shape: (K, D)
        self.min_values = np.array(data["min_values"])  # shape: (K,)
        self.max_values = np.array(data["max_values"])  # shape: (K,)
        self.D_mean = np.array(data["D_mean"])  # shape: (D,)
        self.D_std = np.array(data["D_std"])  # shape: (D,)

        # Project min and max values through PCA basis
        min_proj = np.dot(self.min_values, self.eigen_vectors)  # (D,)
        max_proj = np.dot(self.max_values, self.eigen_vectors)  # (D,)

        # Compute elementwise min and max, then denormalize
        # self.min_orig = np.minimum(min_proj,
        #                            max_proj) * self.D_std + self.D_mean
        # self.max_orig = np.maximum(min_proj,
        #                            max_proj) * self.D_std + self.D_mean
        self.min_orig = self.min_values
        self.max_orig = self.max_values
