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
    dataset_denrormalizer, dataset_normalizer, extract_finger_joints)
import time


class VAEVerifyWrapper:

    def __init__(self, env, args_cli, num_hand_joints=16):
        self.args_cli = args_cli
        self.device = args_cli.device
        self.env = env
        self.num_hand_joints = num_hand_joints
        self.hand_side = "left" if args_cli.add_left_hand else "right"
        # self.load_vae_model()
        self.actions_buffer = None
        self.raw_data = None
        self.demo_count = 0

        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)

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

        with torch.no_grad():
            mu_action = self.model.encoder(
                self.actions_buffer.to(torch.float32)).embedding

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
        self.data_normalizer = model_config["data_normalizer"]
        self.max_latent_value = torch.as_tensor(
            model_config["max_latent_value"]).to(self.device)
        self.min_latent_value = torch.as_tensor(
            np.array(model_config["min_latent_value"])).to(self.device)
        self.embedding_dim = model_config["embedding_dim"]

        if model_config["model_type"] in ["VQVAE"]:
            self.embedding_dim = 1
            self.max_latent_value = torch.ones(self.embedding_dim).to(
                self.device)
            self.min_latent_value = torch.zeros(self.embedding_dim).to(
                self.device) - 1

        plot_action(
            mu_action.cpu().numpy(),
            self.args_cli.vae_path,
            model_config["model_type"],
        )

    def verify_vae(self, ):

        self.env.reset()
        print("new demo")
        demo_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device).to(torch.float32)

        raw_hand_pose = demo_actions[..., -self.num_hand_joints:].clone()

        if self.args_cli.use_relative_pose:
            raw_hand_pose = raw_hand_pose[1:] - raw_hand_pose[:-1]

        with torch.no_grad():

            recontructed_hand_pose = dataset_denrormalizer(
                self.model({
                    "data": raw_hand_pose
                }).recon_x, self.action_mean, self.action_std)

        reconstructed_actions = torch.cat([
            demo_actions[:, -self.num_hand_joints:].unsqueeze(1),
            recontructed_hand_pose.unsqueeze(1)
        ],
                                          dim=1)
        actions = torch.zeros(self.env.action_space.shape,
                              dtype=torch.float32).to(self.device)
        for i in range(reconstructed_actions.shape[0]):
            actions[..., 6:] = extract_finger_joints(reconstructed_actions[i],
                                                     self.joint_limits)
            self.env.step(actions)

        self.demo_count += 1

    def random_play_vae(self):
        self.env.reset()

        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()
        unqiue_actions = []

        for i in range(200):

            raw_actions = (torch.rand(
                (self.env.num_envs, self.embedding_dim)).to(self.device) * 2 -
                           1) * 1.0

            raw_actions = (raw_actions + 1) / 2 * torch.as_tensor(
                self.max_latent_value - self.min_latent_value).to(
                    self.device).to(torch.float32) + torch.as_tensor(
                        self.min_latent_value).to(self.device).to(
                            torch.float32)

            with torch.no_grad():

                reconstructed_hand_actions = self.model.decode_rl_action(
                    raw_actions)
                # unqiue_actions.append(
                #     self.model.decode_quantized_indices.cpu().numpy())

            action = torch.zeros(self.env.action_space.shape).to(self.device)

            action[..., 6:] = extract_finger_joints(reconstructed_hand_actions,
                                                    self.joint_limits)
            # action[..., 2] = 0.15
            self.env.step(action)
            self.pre_finger_action = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                         num_hand_joints:].clone(
                                                         )
