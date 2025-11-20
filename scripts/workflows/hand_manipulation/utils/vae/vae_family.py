import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import yaml
import argparse
import gym
import os
import torch.nn as nn
import torch.nn.functional as F
from scripts.workflows.hand_manipulation.utils.vae.vae_plot import plot_action_distribution, visualize_latent_space
import copy
# add argparse arguments

from torch.utils.data import TensorDataset, DataLoader, random_split
from scripts.workflows.hand_manipulation.utils.vae.vae_vanillaVAE import VanillaVAE
from scripts.workflows.hand_manipulation.utils.vae.vae_cVae import cVAE
from scripts.workflows.hand_manipulation.utils.vae.vae_ae import NonLinearAE
from scripts.workflows.hand_manipulation.utils.vae.vae_vqvae import VQVAE
from scripts.workflows.hand_manipulation.utils.vae.vae_fvae import FactorVAE

VAE_DICT = {
    "VqVAE": VQVAE,
    "cVAE": cVAE,
    "AE": NonLinearAE,
    "vallinaVAE": VanillaVAE,
    "fvae": FactorVAE,
}


class VAEFAMILY:

    def __init__(self,
                 args_cli,
                 eval=False,
                 device="cuda",
                 timstep=1 / 1,
                 hand_side="right",
                 vae_path=None):

        self.args_cli = args_cli
        self.device = device

        self.timstep = timstep
        self.hand_side = hand_side
        self.eval = eval

        if not eval:
            with open(args_cli.config_file, "r") as f:
                yaml_args = yaml.safe_load(f)["params"]
            self.vae_lr = yaml_args["vae_lr"]

            self.layer_dims = yaml_args["layer_dims"]
            self.latent_dim = yaml_args["latent_dim"]
            self.eval_percent = yaml_args["eval_percent"]
            self.data_normalizer = yaml_args["data_normalizer"]
            self.init_data()
            self.vae_type = yaml_args["vae_type"]
            self.batch_size = yaml_args["batch_size"]
            self.epoches = yaml_args["epoches"]

            self.init_vae_model(self.vae_type)
            self.save_path = self.args_cli.log_dir + f"/{self.hand_side}_vae/{self.vae_type}_latent_{self.latent_dim}"
            if self.data_normalizer is not None:
                self.save_path += f"_{self.data_normalizer}/"
            os.makedirs(self.save_path, exist_ok=True)

        else:
            self.load_vae_model(vae_path=vae_path)

    def load_vae_model(self, vae_path=None):

        if vae_path is None:
            vae_path = self.args_cli.vae_path
        with open(f"{vae_path}/{self.hand_side}_decoder.pth", "rb") as f:
            checkpoint = torch.load(f, weights_only=True)

            model_config = checkpoint["model_config"]
            self.input_dim = model_config["input_dim"]
            self.layer_dims = model_config["hidden_dims"]
            self.state_dim = model_config["input_dim"]
            self.latent_dim = model_config["latent_dim"]

            self.data_normalizer = model_config["data_normalizer"]
            self.action_mean = np.array(model_config["action_mean"])
            self.action_std = np.array(model_config["action_std"])

            self.timstep = model_config["timstep"]
            self.model_type = model_config["model_type"]
            self.max_latent_value = np.array(model_config["max_latent_value"])
            self.min_latent_value = np.array(model_config["min_latent_value"])

            self.action_max = torch.as_tensor(model_config['action_max']).to(
                self.device)
            self.action_min = torch.as_tensor(model_config['action_min']).to(
                self.device)
            self.init_vae_model(model_config["model_type"])
            self.model.eval()
            self.model.decoder.load_state_dict(checkpoint["model"])

        with open(f"{vae_path}/{self.hand_side}_encoder.pth", "rb") as f:
            checkpoint = torch.load(f, weights_only=True)
            self.model.encoder.load_state_dict(checkpoint["model"])

    def init_vae_model(self, vae_type):

        self.model = VAE_DICT[vae_type](self.input_dim, self.latent_dim,
                                        self.layer_dims, self.state_dim)
        if not self.eval:

            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.vae_lr)

    def dataset_minmax_normalizer(self, actions, quantile=1):

        if quantile == 1:
            min_val = np.min(actions, axis=0)
            max_val = np.max(actions, axis=0)
        else:
            min_val = np.quantile(actions, 1 - quantile, axis=0)
            max_val = np.quantile(actions, quantile, axis=0)

        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 2
        return mean, std, max_val, max_val

    def init_data(self, ):
        data = h5py.File(self.args_cli.data_dir, "r")["data"]

        all_actions = []
        all_raw_actions = []

        for index in range(len(data)):

            raw_actions = data[f"demo_{index}"]["actions"][..., 6:]
            # raw_actions = (raw_actions + np.pi) % (2 * np.pi) - np.pi
            all_actions.append(raw_actions)

            all_raw_actions.append(data[f"demo_{index}"]["actions"][:-1, 6:])

        all_actions = np.concatenate(all_actions, axis=0)
        all_raw_actions = np.concatenate(all_raw_actions, axis=0)
        all_actions = ((all_actions + np.pi) %
                       (2 * np.pi) - np.pi) / self.timstep

        print("num actions:", all_actions.shape[0])
        if self.data_normalizer == "minmax":
            self.action_mean, self.action_std, self.action_min, self.action_max = self.dataset_minmax_normalizer(
                all_actions)
            all_actions = (all_actions - self.action_mean) / self.action_std
        else:
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0)

            self.action_min = np.min(all_actions, axis=0)
            self.action_max = np.max(all_actions, axis=0)

        num_actions = all_actions.shape[0]
        self.expert_actions = all_actions[:int(num_actions *
                                               self.eval_percent)]
        self.episode_actions = all_actions[int(num_actions *
                                               self.eval_percent):]
        self.raw_state_expert_actions = all_raw_actions[:int(num_actions *
                                                             self.eval_percent
                                                             )]
        self.raw_state_episode_actions = all_raw_actions[int(num_actions *
                                                             self.eval_percent
                                                             ):]

        self.input_dim = self.expert_actions.shape[1]
        self.layer_dims = self.layer_dims
        self.state_dim = self.expert_actions.shape[1]

        # plot_action_distribution(self.expert_actions, )

    def normalize_action(self, action):

        normalized_action = (action - torch.as_tensor(self.action_mean).to(
            action.device)) / torch.as_tensor(self.action_std).to(
                action.device)
        return normalized_action.to(torch.float32)

    def decoder(self, state, z):

        decode_action = self.model.decoder(state, z)
        if isinstance(decode_action, tuple):
            decode_action = decode_action[0]

        if self.data_normalizer == "minmax":
            decode_action = decode_action * torch.as_tensor(
                self.action_std).to(decode_action.device) + torch.as_tensor(
                    self.action_mean).to(decode_action.device)

        return decode_action

    def run_backprop(
        self,
        train_loader,
        val_loader,
    ):
        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(self.epoches):
            train_loss_sum = 0
            val_loss_sum = 0

            for state, action in train_loader:
                vae_loss = self.model.loss(
                    state.to(self.device),
                    action.to(self.device),
                    #  beta=1 / (self.args_cli.epoches) * epoch *
                    #  0.3
                )
                self.optimizer.zero_grad()
                vae_loss.backward()
                self.optimizer.step()
                train_loss_sum += vae_loss.item()

            epoch_train_losses.append(train_loss_sum / len(train_loader))

            with torch.no_grad():
                self.model.eval()

                for state, action in val_loader:
                    vae_loss = self.model.loss(state.to(self.device),
                                               action.to(self.device))
                    val_loss_sum += vae_loss.item()
                epoch_val_losses.append(val_loss_sum / len(val_loader))

            print(
                f"Epoch {epoch}: Train Loss: {epoch_train_losses[-1]}, Val Loss: {epoch_val_losses[-1]}"
            )

    def run_vae(self):

        # Convert data to tensors
        expert_actions_tensor = torch.tensor(self.expert_actions,
                                             dtype=torch.float32)
        raw_state_expert_actions_tensor = torch.tensor(
            self.raw_state_expert_actions, dtype=torch.float32)
        full_dataset = TensorDataset(raw_state_expert_actions_tensor,
                                     expert_actions_tensor)

        # Create TensorDataset and DataLoader
        val_ratio = 0.1
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset,
                                                  [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training loop
        if self.vae_type not in ["fvae"]:

            self.run_backprop(train_loader, val_loader)
        else:
            self.model.train_epoches(train_loader,
                                     val_loader,
                                     epoches=self.epoches)

        max_latent_value, min_latent_value = visualize_latent_space(
            self.model,
            train_loader,
            self.device,
            self.save_path,
            name="train")
        # save the encoder and decoder. The decoder is later used as a prior in DRL.
        os.makedirs(self.save_path, exist_ok=True)

        model_config = {
            "model_type": self.vae_type,
            "input_dim": self.expert_actions.shape[1],
            "latent_dim": self.latent_dim,
            "hidden_dims": self.layer_dims,
            'action_min': self.action_min.tolist(),  # convert NumPy to list
            'action_max': self.action_max.tolist(),
            "timstep": self.timstep,
            "max_latent_value": max_latent_value.tolist(),
            "min_latent_value": min_latent_value.tolist(),
            "data_normalizer": str(self.data_normalizer),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }
        torch.save(
            {
                "model": self.model.decoder.state_dict(),
                "model_config": model_config,
            },
            f"{self.save_path}/{self.hand_side}_decoder.pth",
        )
        torch.save(
            {
                "model": self.model.encoder.state_dict(),
                "model_config": model_config,
            },
            f"{self.save_path}/{self.hand_side}_encoder.pth",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
    )
    parser.add_argument(
        "--config_file",
        type=str,
    )
    parser.add_argument(
        "--hand_side",
        type=str,
        default="right",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/ensu/data/datasets/grab/raw_right_data.hdf5",
    )

    # parser.add_argument(
    #     "--vae_type",
    #     default="vallinaVAE",
    #     choices=["VqVAE", "cVAE", "AE", "vallinaVAE", "fvae"],
    # )
    args_cli, hydra_args = parser.parse_known_args()

    vae = VAEFAMILY(args_cli, timstep=1 / 1, hand_side=args_cli.hand_side)
    vae.run_vae()
