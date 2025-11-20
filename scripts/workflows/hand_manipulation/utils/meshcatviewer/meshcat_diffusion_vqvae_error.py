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
import h5py
import time
import sys
import os

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation)
import time

import re
import math

import math
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import sliding_chunks, init_chunk_data


class HandVisualizerClient:

    def __init__(self,
                 urdf_path,
                 model_path,
                 args_cli=None,
                 server_host='localhost',
                 server_port=9999,
                 num_hand_joints=16,
                 num_diffusion=8,
                 spacing=0.3,
                 device="cuda"):
        # Load model
        self.num_hand_joints = num_hand_joints
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, model_path, pin.JointModelFreeFlyer())
        self.q0 = pin.neutral(self.model)
        self.server_host = server_host
        self.server_port = server_port
        self.num_diffusion = num_diffusion
        self.spacing = spacing
        self.diffusion_path = args_cli.diffusion_path
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

        self.load_diffusion()
        self.init_raw_vizualizer()

        self.init_diffusion_vizualizer()

    def extract_finger_joints(self, joints):

        raw_joints = (np.array(joints) + 1) / 2 * (
            self.joint_limits[:, 1] -
            self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        return raw_joints

    def load_diffusion(self, batch_size=2048):

        checkpoint = os.path.join(self.args_cli.diffusion_path, "checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, args_cli=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        all_reconstructed = []
        all_action, all_states = self.init_data()
        for i in range(0, all_action.shape[0], batch_size):
            start = time.time()
            batch_obs = all_action[i:i + batch_size]  # shape (B, 30, 16)

            obs_dict = {"obs": torch.tensor(batch_obs, device=device).float()}
            gt_action = all_action[i:i + batch_size]

            with torch.no_grad():
                action_pred = policy.predict_action(obs_dict)[
                    "action_pred"]  # shape (B, 30, 16) or similar
                reconstructed = extract_finger_joints(
                    action_pred.cpu().numpy(), self.joint_limits)

            all_reconstructed.append(reconstructed)
            print(
                f"Reconstruction time for batch {i // batch_size}: {time.time() - start:.2f} seconds"
            )
        all_reconstructed = np.concatenate(all_reconstructed, axis=0)
        diff = np.abs(all_reconstructed - extract_finger_joints(
            all_action, self.joint_limits)[:, :cfg.horizon])

        sum_error = np.sum(np.abs(diff), axis=-1).reshape(-1)

        sorted_indices = np.argsort(-sum_error)  # Descending
        sorted_vals = sum_error[sorted_indices]
        top_indices = sorted_indices[:2000:20]

        # self.corrected_hand_pose = self.extract_finger_joints(
        #     self.train_dataset[0].reshape(
        #         -1, 16)[top_indices].cpu().numpy())  #.cpu().numpy()

        # self.error_hand_pose = self.extract_finger_joints(
        #     reconstructed_hand_pose.reshape(-1, 16)[top_indices].cpu().numpy())

        self.corrected_hand_pose = self.extract_finger_joints(
            all_action[:2000:10, -1])
        self.error_hand_pose = all_reconstructed[:2000:10, -1]

    def init_data(self, chunk_size=30):

        data = h5py.File(self.args_cli.data_dir, "r")["data"]

        all_actions = []
        all_states = []

        for index in range(len(data)):

            raw_actions = data[f"demo_{index}"]["actions"][
                ..., -self.num_hand_joints:]
            chunks_actions, state = sliding_chunks(raw_actions, chunk_size)

            all_actions.append(chunks_actions)
            all_states.append(state)

        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)

        return all_actions, all_states

    def init_raw_vizualizer(self):

        self.viz_main = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_main.initViewer(open=True)

    def init_diffusion_vizualizer(self):
        self.viz_diffusion = []
        self.q0_diffusion = []

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

                viz_diffusion = MeshcatVisualizer(self.model,
                                                  self.collision_model,
                                                  self.visual_model)
                viz_diffusion.initViewer(self.viz_main.viewer)
                viz_diffusion.loadViewerModel(
                    rootNodeName=f"diffusion_hand_{i}_{j}")

                q0_diffusion = self._display_offset(
                    viz_diffusion, offset=[0.0, offset_x, offset_y])
                self.q0_diffusion.append(copy.deepcopy(q0_diffusion))
                self.viz_diffusion.append(viz_diffusion)

                self.set_robot_pose(viz_diffusion, q0_diffusion, i, j)

    def set_robot_pose(self, viz_diffusion, q0_diffusion, sample_idx,
                       row_type):
        # row_type: 0 = error, 1 = ground truth
        q_raw = q0_diffusion.copy()

        if row_type == 1:
            pose = self.error_hand_pose[sample_idx]
        else:
            pose = self.corrected_hand_pose[sample_idx]

        q_raw[-16:] = pose[self.retarget2pin]
        viz_diffusion.display(q_raw)

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

    parser.add_argument("--diffusion_path",
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
                                      num_diffusion=3,
                                      server_port=54113)
    visualizer.run_until_closed()
