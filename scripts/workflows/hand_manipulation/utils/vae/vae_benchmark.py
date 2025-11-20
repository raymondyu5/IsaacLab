import os
from copy import deepcopy

import torch
import sys

sys.path.append("submodule/benchmark_VAE/src")

from pythae.models import AE, AEConfig, VAE, VAEConfig, VQVAE, VQVAEConfig, VAMPConfig, VAMP, HRQVAE, HRQVAEConfig, INFOVAE_MMD, INFOVAE_MMD_Config

from pythae.pipelines import TrainingPipeline

import argparse
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from scripts.workflows.hand_manipulation.utils.vae.vae_plot import plot_action_distribution, visualize_latent_space, visualize_single_action_latent_space, visualize_vq_latent_space
import numpy as np
from scripts.workflows.hand_manipulation.utils.vae.conder_wrapper import AEEncoder, AEDecoder, VAEEncoder
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_minmax_normalizer, resample_data)

PATH = os.path.dirname(os.path.abspath(__file__))
import h5py
import yaml

conder_dict = {
    "AE": {
        "model": AE,
        "model_config": AEConfig,
        "encoder": AEEncoder,
        "decoder": AEDecoder,
    },
    "VAE": {
        "model": VAE,
        "model_config": VAEConfig,
        "encoder": VAEEncoder,
        "decoder": AEDecoder,
    },
    "InfoVAE": {
        "model": INFOVAE_MMD,
        "model_config": INFOVAE_MMD_Config,
        "encoder": VAEEncoder,
        "decoder": AEDecoder,
    },
    "VQVAE": {
        "model": VQVAE,
        "model_config": VQVAEConfig,
        "encoder": AEEncoder,
        "decoder": AEDecoder,
    },
    "VAMP": {
        "model": VAMP,
        "model_config": VAMPConfig,
        "encoder": VAEEncoder,
        "decoder": AEDecoder,
    },
    "HRQVAE": {
        "model": HRQVAE,
        "model_config": HRQVAEConfig,
        "encoder": AEEncoder,
        "decoder": AEDecoder,
    },
}


class VAEFAMILY:

    def __init__(self,
                 args_cli,
                 eval=False,
                 device="cuda",
                 timstep=1 / 1,
                 hand_side="right",
                 vae_path=None,
                 num_hand_joints=16,
                 data_normalizer=None):

        self.args_cli = args_cli
        self.num_hand_joints = num_hand_joints
        self.device = device

        self.timstep = timstep
        self.hand_side = hand_side
        self.eval = eval
        self.data_normalizer = data_normalizer

        if not eval:

            with open(args_cli.config_file, "r") as f:
                yaml_args = yaml.safe_load(f)["params"]
            if args_cli.embedding_dim > 0:
                yaml_args["embedding_dim"] = args_cli.embedding_dim
                if yaml_args.get("model_config", None) is not None:
                    yaml_args["model_config"][
                        "embedding_dim"] = args_cli.embedding_dim
                else:
                    yaml_args["embedding_dim"] = args_cli.embedding_dim

            if args_cli.num_embeddings > 0:
                if yaml_args.get("model_config", None) is not None:
                    yaml_args["model_config"][
                        "num_embeddings"] = args_cli.num_embeddings

            self.yaml_args = yaml_args
            self.vae_type = yaml_args["vae_type"]
            self.vae_lr = yaml_args["vae_lr"]

            self.layer_dims = yaml_args["layer_dims"]
            if self.vae_type in ["VQVAE"]:
                self.embedding_dim = yaml_args["model_config"]["embedding_dim"]

            else:
                self.embedding_dim = yaml_args["embedding_dim"]

            self.eval_percent = yaml_args["eval_percent"]

            self.data_normalizer = yaml_args["data_normalizer"]
            self.batch_size = yaml_args["batch_size"]

            self.epoches = yaml_args["epoches"]
            self.model_config = yaml_args.get("model_config", {})
            self.save_path = self.args_cli.log_dir + f"/{self.hand_side}_vae/{self.vae_type}/{self.vae_type}_latent_{self.embedding_dim}"
            if self.data_normalizer is not None:
                self.save_path += f"_{self.data_normalizer}"

            if self.model_config.get("num_embeddings", None) is not None:
                num_embeddings = self.model_config["num_embeddings"]
                self.save_path += f"_codebook{num_embeddings}"

            os.makedirs(self.save_path, exist_ok=True)
            self.init_data()

            self.init_model()

    def init_model(self, ):
        conder_list = conder_dict[self.vae_type]

        model_config = conder_list["model_config"](**self.model_config)

        self.model = conder_list["model"](
            model_config=model_config,
            encoder=conder_list["encoder"](self.input_dim, self.embedding_dim,
                                           self.layer_dims, self.device),
            decoder=conder_list["decoder"](self.embedding_dim, self.input_dim,
                                           self.layer_dims, self.device))
        config = BaseTrainerConfig(
            output_dir=self.save_path,
            learning_rate=self.vae_lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_epochs=self.
            epoches,  # Change this to train the model a bit more
        )

        self.train_pipeline = TrainingPipeline(training_config=config,
                                               model=self.model)

        self.train_pipeline(
            train_data=self.train_dataset,
            eval_data=self.eval_dataset,
        )
        # if self.vae_type == "VQVAE":
        #     visualize_vq_latent_space(
        #         self.model,
        #         self.train_dataset.to("cuda"),
        #         self.device,
        #         self.save_path,
        #     )

        max_latent_value, min_latent_value = visualize_single_action_latent_space(
            self.model.encoder,
            self.train_dataset.to("cuda"),
            self.device,
            self.save_path,
        )

        model_config = {
            "model_type": self.vae_type,
            'action_min': self.action_min.tolist(),
            'action_max': self.action_max.tolist(),
            "max_latent_value": max_latent_value.tolist(),
            "min_latent_value": min_latent_value.tolist(),
            "data_normalizer": str(self.data_normalizer),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        } | self.model_config | self.yaml_args
        with open(f"{self.save_path}/model_config.yaml", "w") as f:
            yaml.dump(model_config, f, default_flow_style=False)

    def init_data(self, ):
        data = h5py.File(self.args_cli.data_dir, "r")["data"]

        all_actions = []
        all_raw_actions = []

        for index in range(len(data)):

            raw_actions = data[f"demo_{index}"]["actions"][
                ..., -self.num_hand_joints:]
            # raw_actions = (raw_actions + np.pi) % (2 * np.pi) - np.pi
            all_actions.append(raw_actions)

            all_raw_actions.append(
                data[f"demo_{index}"]["actions"][:-1, -self.num_hand_joints:])

        all_actions = np.concatenate(all_actions, axis=0)
        all_raw_actions = np.concatenate(all_raw_actions, axis=0)

        all_actions = ((all_actions + np.pi) %
                       (2 * np.pi) - np.pi) / self.timstep

        print("num actions:", all_actions.shape[0])
        if self.data_normalizer == "minmax":
            self.action_mean, self.action_std, self.action_min, self.action_max = dataset_minmax_normalizer(
                all_actions)
            all_actions = (all_actions - self.action_mean) / self.action_std
        else:
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0)

            self.action_min = np.min(all_actions, axis=0)
            self.action_max = np.max(all_actions, axis=0)

        all_actions = resample_data(all_actions, bin_size=100)
        num_actions = all_actions.shape[0]

        self.train_dataset = torch.as_tensor(
            all_actions[:int(num_actions * self.eval_percent)]).to(
                self.device).to(torch.float32)
        self.eval_dataset = torch.as_tensor(
            all_actions[int(num_actions * self.eval_percent):]).to(
                self.device).to(torch.float32)
        self.input_dim = self.train_dataset.shape[1]

        plot_action_distribution(all_actions,
                                 self.save_path,
                                 name="raw_actions")
        plot_action_distribution(self.train_dataset.cpu().numpy(),
                                 self.save_path,
                                 name="normalized_actions")

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
        default="logs/data_0604/raw_right_data.hdf5",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=0,
    )
    args_cli, hydra_args = parser.parse_known_args()

    vae = VAEFAMILY(args_cli, timstep=1 / 1, hand_side=args_cli.hand_side)
