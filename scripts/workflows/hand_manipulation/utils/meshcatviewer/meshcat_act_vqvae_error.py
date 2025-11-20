import pinocchio as pin
import numpy as np
import socket
import json
import time
from os.path import join
from pinocchio.visualize import MeshcatVisualizer
import copy
import numpy as np
import torch

import sys

sys.path.append("submodule/benchmark_VAE/src")
import yaml
from pythae.models import AutoModel
import os
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    init_chunk_data)
import h5py

import math


class HandVisualizerClient:

    def __init__(self,
                 urdf_path,
                 model_path,
                 args_cli=None,
                 server_host='localhost',
                 server_port=9999,
                 num_hand_joints=16,
                 num_vae=8,
                 spacing=0.3,
                 device="cuda"):
        # Load model
        self.num_hand_joints = num_hand_joints
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, model_path, pin.JointModelFreeFlyer())
        self.q0 = pin.neutral(self.model)
        self.server_host = server_host
        self.server_port = server_port
        self.num_vae = num_vae
        self.spacing = spacing
        self.vae_path = args_cli.vae_path
        self.args_cli = args_cli
        self.device = device

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

        self.joint_limits = np.array(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=np.float32)

        self.load_vae()
        self.init_raw_vizualizer()

        self.init_vae_vizualizer()

    def extract_finger_joints(self, joints):

        raw_joints = (np.array(joints) + 1) / 2 * (
            self.joint_limits[:, 1] -
            self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        return raw_joints

    def load_vae(self, ):

        all_dirs = [
            d for d in os.listdir(self.vae_path)
            if os.path.isdir(os.path.join(self.vae_path, d))
        ]
        last_training = sorted(all_dirs)[-1]

        vae_model = AutoModel.load_from_folder(os.path.join(
            self.vae_path, last_training, 'final_model'),
                                               device=self.device)
        vae_model.eval()
        vae_model.to(self.device)

        with open(f"{self.vae_path}/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)

            action_mean = torch.as_tensor(model_config["action_mean"]).to(
                self.device)
            action_std = torch.as_tensor(model_config["action_std"]).to(
                self.device)
            data_normalizer = model_config["data_normalizer"]
            max_latent_value = np.array(model_config["max_latent_value"])
            min_latent_value = np.array(model_config["min_latent_value"])
            latent_dim = model_config["latent_dim"]
            self.chunk_size = model_config["chunk_size"]

            vae_model_setting = [
                min_latent_value, max_latent_value, data_normalizer,
                action_mean, action_std, latent_dim
            ]
        self.init_data()
        reconstructed_hand_pose = self.recontruct_vae_actions(
            vae_model, vae_model_setting)

        diff = self.extract_finger_joints(
            reconstructed_hand_pose.reshape(
                -1, 16).cpu().numpy()) - self.extract_finger_joints(
                    self.train_dataset[0].clone().reshape(-1,
                                                          16).cpu().numpy())
        diff = torch.as_tensor(diff).to(self.device)

        sum_error = torch.sum(torch.abs(diff), dim=-1).reshape(-1)
        sorted_vals, sorted_indices = sum_error.sort(descending=True)
        top_indices = sorted_indices[:2000:20]

        diff = self.extract_finger_joints(reconstructed_hand_pose.cpu().numpy(
        )) - self.extract_finger_joints(
            self.train_dataset[0].clone().cpu().numpy())

        # self.corrected_hand_pose = self.extract_finger_joints(
        #     self.train_dataset[0].reshape(
        #         -1, 16)[top_indices].cpu().numpy())  #.cpu().numpy()

        # self.error_hand_pose = self.extract_finger_joints(
        #     reconstructed_hand_pose.reshape(-1, 16)[top_indices].cpu().numpy())

        self.corrected_hand_pose = self.extract_finger_joints(
            self.train_dataset[0][:2000:10, -1].cpu().numpy())
        self.error_hand_pose = self.extract_finger_joints(
            reconstructed_hand_pose[:2000:10, -1].cpu().numpy())

    def init_data(self, ):

        self.train_dataset, self.val_dataset, self.action_mean, self.action_std, self.action_min, self.action_max, self.input_dim = init_chunk_data(
            self.args_cli.data_dir, self.num_hand_joints, self.chunk_size, 1.0,
            self.device)

    def recontruct_vae_actions(self, vae_model, vae_config, batch_size=1024):
        reconstructed_list = []

        with torch.no_grad():
            action_chunk = self.train_dataset[0]
            state = self.train_dataset[1]

            for start in range(0, len(action_chunk), batch_size):
                end = start + batch_size
                batch_action = action_chunk[start:end].to(self.device)
                batch_state = state[start:end].to(self.device)

                # Encode
                result = vae_model({
                    "data": {
                        "action_chunk": batch_action,
                        "state": batch_state
                    }
                })

                latent = result.z

                # Decode
                decode_data = {
                    "action_chunk": batch_action,
                    "state": batch_state,
                    "z": latent
                }

                reconstructed = vae_model.decode_action(decode_data)
                reconstructed_list.append(reconstructed)

        # Combine all batches
        reconstructed_hand_pose = torch.cat(reconstructed_list, dim=0)
        return reconstructed_hand_pose

    def init_raw_vizualizer(self):

        self.viz_main = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_main.initViewer(open=True)

    def init_vae_vizualizer(self):
        self.viz_vae = []
        self.q0_vae = []

        num_samples = len(self.corrected_hand_pose)

        # Automatically determine the number of columns based on sqrt
        cols = int(math.ceil(math.sqrt(num_samples)))
        rows = int(math.ceil(
            num_samples / cols)) * 2  # 2 rows per sample (error + GT)

        for i in range(num_samples):
            row_idx = (i //
                       cols) * 2  # Starting row index for this sample pair
            col_idx = i % cols  # Column index

            for j in range(2):  # j = 0 (error), j = 1 (GT)
                row = row_idx + j
                col = col_idx

                offset_x = (col - (cols - 1) / 2) * self.spacing
                offset_y = (
                    (rows - 1) / 2 - row) * self.spacing  # Y top to bottom

                viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                            self.visual_model)
                viz_vae.initViewer(self.viz_main.viewer)
                viz_vae.loadViewerModel(rootNodeName=f"vae_hand_{i}_{j}")

                q0_vae = self._display_offset(viz_vae,
                                              offset=[0.0, offset_x, offset_y])
                self.q0_vae.append(copy.deepcopy(q0_vae))
                self.viz_vae.append(viz_vae)

                self.set_robot_pose(viz_vae, q0_vae, i, j)

    def set_robot_pose(self, viz_vae, q0_vae, sample_idx, row_type):
        # row_type: 0 = error, 1 = ground truth
        q_raw = q0_vae.copy()

        if row_type == 1:
            pose = self.error_hand_pose[sample_idx]
        else:
            pose = self.corrected_hand_pose[sample_idx]

        q_raw[-16:] = pose[self.retarget2pin]
        viz_vae.display(q_raw)

    def _display_offset(self, viz, offset):
        q = self.q0.copy()
        q[0] += offset[0]
        q[1] += offset[1]
        q[2] += offset[2]
        viz.display(q)
        return q

    def run_until_closed(self):

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("🛑 Visualization stopped by user.")


if __name__ == "__main__":
    model_path = "source/assets/robot/leap_hand_v2/archive/leap_hand"
    urdf_filename = "leap_hand_right_glb.urdf"
    urdf_path = join(model_path, urdf_filename)
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae_path",
                        type=str,
                        default=None,
                        help="List of VAE path strings")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="logs/teleop_0531/raw_right_data.hdf5",
    )
    args_cli = parser.parse_args()

    visualizer = HandVisualizerClient(urdf_path=urdf_path,
                                      model_path=model_path,
                                      args_cli=args_cli,
                                      server_host="localhost",
                                      num_vae=3,
                                      server_port=54113)
    visualizer.run_until_closed()
