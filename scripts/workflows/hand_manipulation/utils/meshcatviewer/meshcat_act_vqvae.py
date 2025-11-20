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
    dataset_denrormalizer, dataset_normalizer)


class HandVisualizerClient:

    def __init__(self,
                 urdf_path,
                 model_path,
                 args_cli=None,
                 server_host='localhost',
                 server_port=9999,
                 num_joints=16,
                 num_vae=8,
                 spacing=0.3,
                 device="cuda"):
        # Load model
        self.num_joints = num_joints
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, model_path, pin.JointModelFreeFlyer())
        self.q0 = pin.neutral(self.model)
        self.server_host = server_host
        self.server_port = server_port
        self.num_vae = num_vae
        self.spacing = spacing
        self.vae_path = args_cli.vae_path
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

    def load_vae(self, ):

        self.vae_models = []
        self.vae_model_setting = []

        for index, vae_path in enumerate(self.vae_path):

            all_dirs = [
                d for d in os.listdir(vae_path)
                if os.path.isdir(os.path.join(vae_path, d))
            ]
            last_training = sorted(all_dirs)[-1]

            vae_model = AutoModel.load_from_folder(os.path.join(
                vae_path, last_training, 'final_model'),
                                                   device=self.device)
            vae_model.eval()

            self.vae_models.append(vae_model.to(self.device))

            with open(f"{vae_path}/model_config.yaml", "r") as f:
                model_config = yaml.safe_load(f)

                action_mean = torch.as_tensor(model_config["action_mean"]).to(
                    self.device)
                action_std = torch.as_tensor(model_config["action_std"]).to(
                    self.device)
                data_normalizer = model_config["data_normalizer"]
                max_latent_value = np.array(model_config["max_latent_value"])
                min_latent_value = np.array(model_config["min_latent_value"])
                latent_dim = model_config["latent_dim"]

                self.vae_model_setting.append([
                    min_latent_value, max_latent_value, data_normalizer,
                    action_mean, action_std, latent_dim
                ])
        self.num_hand = min(
            [vae_model.quantizer.num_embeddings**latent_dim, 100])
        import itertools
        all_codebook_combinations = list(
            itertools.product(range(vae_model.quantizer.num_embeddings),
                              repeat=latent_dim))
        with torch.no_grad():
            if data_normalizer is not None:
                self.reconstructed_hand = dataset_denrormalizer(
                    vae_model.decode_action_index(all_codebook_combinations),
                    action_mean, action_std)
            else:

                self.reconstructed_hand = vae_model.decode_action_index(
                    all_codebook_combinations)

    def init_raw_vizualizer(self):

        self.viz_main = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_main.initViewer(open=True)
        self.viz_main.initViewer(open=True)

    def init_vae_vizualizer(self):
        # VAE visualizer (static or not used)
        self.viz_vae = []
        self.q0_vae = []

        grid_size = int(np.ceil(np.sqrt(
            self.num_hand)))  # Square grid dimension

        for i in range(self.num_hand):
            row = i // grid_size
            col = i % grid_size

            offset_x = (col - (grid_size - 1) / 2) * self.spacing
            offset_y = (row - (grid_size - 1) / 2) * self.spacing

            viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                        self.visual_model)
            viz_vae.initViewer(self.viz_main.viewer)
            viz_vae.loadViewerModel(rootNodeName=f"vae_hand_{i}")

            q0_vae = self._display_offset(viz_vae,
                                          offset=[0.0, offset_x, offset_y])
            self.q0_vae.append(copy.deepcopy(q0_vae))
            self.viz_vae.append(viz_vae)

            self.set_robot_pose(viz_vae, q0_vae, i)

    def extract_finger_joints(self, joints):

        raw_joints = (np.array(joints) + 1) / 2 * (
            self.joint_limits[:, 1] -
            self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        return raw_joints

    def set_robot_pose(self, viz_vae, q0_vae, i):

        q_raw = q0_vae.copy()
        q_raw[-16:] = self.extract_finger_joints(
            self.reconstructed_hand[i].cpu().numpy())[self.retarget2pin]
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

    parser.add_argument(
        "--vae_path",
        type=str,
        nargs='+',  # Accept one or more values
        default=None,
        help="List of VAE path strings")
    args_cli = parser.parse_args()

    visualizer = HandVisualizerClient(urdf_path=urdf_path,
                                      model_path=model_path,
                                      args_cli=args_cli,
                                      server_host="localhost",
                                      num_vae=3,
                                      server_port=54113)
    visualizer.run_until_closed()
