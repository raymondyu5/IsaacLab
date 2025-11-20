import numpy as np

import torch
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import sys

sys.path.append("submodule/benchmark_VAE/src")
from pythae.models import AutoModel
import os
import yaml

from scripts.workflows.hand_manipulation.utils.vae.vae_plot import plot_action
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation, temporal_ensemble_finger_joints)

import time


class ACTVAEVerifyWrapper:

    def __init__(self, env, args_cli, num_hand_joints=16):
        self.args_cli = args_cli
        self.device = args_cli.device
        self.env = env
        self.num_hand_joints = num_hand_joints
        self.hand_side = "left" if args_cli.add_left_hand else "right"

        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)
        self.load_vae_model()
        self.actions_buffer = None
        self.raw_data = None
        self.demo_count = 1
        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=2,
            horizon_K=self.chunk_size,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=1,
            obs_dim=self.num_hand_joints,
        )

    def load_vae_model(self):

        all_dirs = [
            d for d in os.listdir(self.args_cli.vae_path)
            if os.path.isdir(os.path.join(self.args_cli.vae_path, d))
        ]
        last_training = sorted(all_dirs)[-1]

        self.model = AutoModel.load_from_folder(os.path.join(
            self.args_cli.vae_path, last_training, 'final_model'),
                                                device=self.device)
        self.model.eval()

        self.model.to(self.device)
        self.model.eval()

        with open(f"{self.args_cli.vae_path}/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)

        self.max_latent_value = torch.as_tensor(
            model_config["max_latent_value"]).to(self.device)
        self.min_latent_value = torch.as_tensor(
            np.array(model_config["min_latent_value"])).to(self.device)

        self.action_max = torch.as_tensor(model_config['action_max']).to(
            self.device)
        self.action_min = torch.as_tensor(model_config['action_min']).to(
            self.device)
        self.action_mean = torch.as_tensor(model_config['action_mean']).to(
            self.device)
        self.action_std = torch.as_tensor(model_config['action_std']).to(
            self.device)
        self.chunk_size = model_config["chunk_size"]
        self.data_normalizer = model_config["data_normalizer"]

        self.num_embeddings = 1 if model_config[
            "model_type"] == "VQVAE" else model_config["embedding_dim"]
        self.max_latent_value = torch.as_tensor(
            model_config["max_latent_value"]).to(
                self.device)[:self.num_embeddings]
        self.min_latent_value = torch.as_tensor(
            np.array(model_config["min_latent_value"])).to(
                self.device)[:self.num_embeddings]
        if model_config["model_type"] == "VQVAE":
            self.max_latent_value = torch.ones(1).to(self.device)
            self.min_latent_value = torch.ones(1).to(self.device) * -1

    def verify_vae(self, ):

        self.env.reset()
        print("new demo")

        demo_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device).to(torch.float32)

        raw_hand_pose = demo_actions[..., -self.num_hand_joints:].clone()
        self.temporal_obs_buffer.reset(200 * self.chunk_size, 2)
        self.temporal_action_buffer.reset(200 * self.chunk_size, 2)
        with torch.no_grad():
            num_chunks = min(raw_hand_pose.shape[0] - self.chunk_size + 1, 180)

            for index in range(1, num_chunks):

                obs_list = torch.as_tensor(
                    raw_hand_pose[index - 1:index - 1 + self.chunk_size]).to(
                        self.device).to(torch.float32).reshape(
                            1, -1, self.num_hand_joints)
                actions_chunks = torch.as_tensor(
                    raw_hand_pose[index:index + self.chunk_size]).to(
                        self.device).to(torch.float32).reshape(
                            1, -1, self.num_hand_joints)
                result = self.model({
                    "data": {
                        "state": obs_list,
                        "action_chunk": actions_chunks
                    }
                })

                self.add_obs_to_buffer(index - 1)

                normalized_joints = self.temporal_obs_buffer.compute_obs(
                ).clone()

                recontructed_hand_pose = extract_finger_joints(
                    self.model.decode_action({
                        "z":
                        result.z,
                        "state":
                        normalized_joints[-1].reshape(1, -1,
                                                      self.num_hand_joints),
                    }), self.joint_limits).reshape(1, -1, self.num_hand_joints)
                # recontructed_hand_pose = result.recon_x

                actions_chunks = extract_finger_joints(actions_chunks,
                                                       self.joint_limits)

                reconstructed_actions = torch.cat(
                    [actions_chunks, recontructed_hand_pose], dim=0)
                # self.temporal_action_buffer.add_prediction(
                #     index - 1, reconstructed_actions)
                # hand_action = self.temporal_action_buffer.compute_action()

                action = torch.zeros(self.env.action_space.shape).to(
                    self.device)

                action[:reconstructed_actions.shape[0],
                       6:] = reconstructed_actions[:, 0]

                action[-1, 6:] = actions_chunks[0, 0]

                self.env.step(action)
        self.demo_count += 2

    def add_obs_to_buffer(self, index):
        state = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        joint_min = self.joint_limits[:, 0]  # shape [16]
        joint_max = self.joint_limits[:, 1]  # shape [16]
        normalized_joints = ((state - joint_min[None, :]) /
                             (joint_max - joint_min)[None, :]) * 2 - 1

        self.temporal_obs_buffer.add_obs(index, normalized_joints)

    def random_play_vae(self):
        self.env.reset()

        self.temporal_obs_buffer.reset(200 * self.chunk_size,
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(200 * self.chunk_size,
                                          self.env.num_envs)

        samples = torch.rand(
            (self.env.num_envs, self.num_hand_joints),
            device=self.env.device) * 0.2 * self.joint_limits[:, 1]
        rest_joint_pose = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos.clone()
        rest_joint_pose[:, -self.num_hand_joints:] = samples
        self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                rest_joint_pose,
                torch.arange(self.env.num_envs, device=self.env.device))

        # for i in range(20):
        #     self.env.step(rest_joint_pose)

        demo_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device).to(torch.float32)

        raw_hand_pose = demo_actions[..., -self.num_hand_joints:].clone()
        step_count = 0

        for index in range(180):

            raw_actions = torch.rand(
                (self.env.num_envs, self.num_embeddings)).to(
                    self.device) * 2 - 1

            raw_actions = (raw_actions + 1) / 2 * torch.as_tensor(
                self.max_latent_value - self.min_latent_value).to(
                    self.device).to(torch.float32) + torch.as_tensor(
                        self.min_latent_value).to(self.device).to(
                            torch.float32)
            with torch.no_grad():

                self.add_obs_to_buffer(step_count)
                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                recontructed_hand_pose = extract_finger_joints(
                    self.model.decode_rl_action({
                        "z":
                        raw_actions,
                        "state":
                        normalized_joint_des.reshape(self.env.num_envs, -1,
                                                     self.num_hand_joints),
                    }), self.joint_limits)

            action = torch.zeros(self.env.action_space.shape).to(self.device)

            # for i in range(int(recontructed_hand_pose.shape[1])):
            action[:, 6:] = recontructed_hand_pose[:, 0]
            # action[:, 6:] = hand_action

            self.env.step(action)
            step_count += 1

            self.add_obs_to_buffer(step_count)

        # self.env.reset()
        self.demo_count += 2
