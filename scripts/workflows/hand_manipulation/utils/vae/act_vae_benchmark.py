import os
from copy import deepcopy

import torch
import sys

sys.path.append("submodule/benchmark_VAE/src")
sys.path.append("submodule/diffusion_policy")
from diffusion_policy.model.common.normalizer import LinearNormalizer
from pythae.models import AE, AEConfig, VAE, VAEConfig, VQVAE, VQVAEConfig, VAMPConfig, VAMP, HRQVAE, HRQVAEConfig

from pythae.pipelines import TrainingPipeline

import argparse
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from scripts.workflows.hand_manipulation.utils.vae.vae_plot import plot_action_distribution, visualize_latent_space, visualize_benchmark_latent_space, visualize_vq_latent_space
import numpy as np
from scripts.workflows.hand_manipulation.utils.vae.conder_wrapper import DETRVAEEncoder, DETRVAEDecoder
from scripts.workflows.hand_manipulation.utils.vae.ldcp_conder import LDCPGRUEncoder, LDCPTransformEncoder, AutoregressiveLowLevelPolicy, SimpleLowLevelPolicy, SimpleActionStateEncoder, SimpleStateActionDecoder, ReactiveDiffEncoderCNN, ReactiveDiffDecoderRNN
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import sliding_chunks, init_chunk_data
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))
import h5py
import yaml

encoder_dict = {
    "dert": DETRVAEEncoder,
    "ldcp_gru": LDCPGRUEncoder,
    "ldcp_transformer": LDCPTransformEncoder,
    "simple_encoder": SimpleActionStateEncoder,
    "reactive_diff_encoder_cnn": ReactiveDiffEncoderCNN,
}
deconder_dict = {
    "dert": DETRVAEDecoder,
    "autoregressive_low_level_policy": AutoregressiveLowLevelPolicy,
    "simple_low_level_policy": SimpleLowLevelPolicy,
    "simple_decoder": SimpleStateActionDecoder,
    "reactive_diff_decoder_rnn": ReactiveDiffDecoderRNN,
}

vae_dict = {
    "AE": {
        "model": AE,
        "model_config": AEConfig,
    },
    "VAE": {
        "model": VAE,
        "model_config": VAEConfig,
    },
    "VQVAE": {
        "model": VQVAE,
        "model_config": VQVAEConfig,
    },
    "VAMP": {
        "model": VAMP,
        "model_config": VAMPConfig,
    },
    "HRQVAE": {
        "model": HRQVAE,
        "model_config": HRQVAEConfig,
    },
}


class ACTVAEFAMILY:

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
            if args_cli.num_embeddings > 0 and "model_config" in yaml_args:

                yaml_args["model_config"][
                    "num_embeddings"] = args_cli.num_embeddings

            if args_cli.chunk_size > 0:
                yaml_args["chunk_size"] = args_cli.chunk_size
            if args_cli.embedding_dim > 0 and "model_config" in yaml_args:

                self.num_embeddings = yaml_args["model_config"][
                    "num_embeddings"]
            else:
                self.num_embeddings = yaml_args["embedding_dim"]

            self.yaml_args = yaml_args
            self.vae_lr = yaml_args["vae_lr"]
            self.use_state = args_cli.use_state  #yaml_args.get("use_state", False)

            self.layer_dims = yaml_args["layer_dims"][0]

            self.eval_percent = yaml_args["eval_percent"]

            self.data_normalizer = yaml_args["data_normalizer"]
            self.batch_size = yaml_args["batch_size"]
            self.embedding_dim = yaml_args["embedding_dim"]

            self.vae_type = yaml_args["vae_type"]
            self.epoches = yaml_args["epoches"]
            self.model_config = yaml_args.get("model_config", {})
            self.encoder_name = yaml_args.get("encoder_name",
                                              self.args_cli.encoder_name)
            self.decoder_name = yaml_args.get("decoder_name",
                                              self.args_cli.decoder_name)
            self.chunk_size = yaml_args.get("chunk_size", 1)

            if not self.use_state:
                self.save_path = self.args_cli.log_dir + f"/{self.hand_side}_vae_act/{self.vae_type}/act_{self.chunk_size}_{self.vae_type}_{self.encoder_name}_{self.decoder_name}_latent_{self.num_embeddings}"
            else:
                self.save_path = self.args_cli.log_dir + f"/{self.hand_side}_vae_state_act/{self.vae_type}/act_{self.chunk_size}_{self.vae_type}_{self.encoder_name}_{self.decoder_name}_latent_{self.num_embeddings}"
            if self.data_normalizer is not None:
                self.save_path += f"_{self.data_normalizer}"

            if self.model_config.get("num_embeddings", None) is not None:
                num_embeddings = self.model_config["num_embeddings"]
                self.save_path += f"_codebook{num_embeddings}"

            os.makedirs(self.save_path, exist_ok=True)
            self.init_data()

            self.init_model()

    def init_model(self, ):

        encoder = encoder_dict[self.encoder_name](
            self.input_dim,
            self.input_dim,
            self.embedding_dim,
            self.layer_dims,
            self.chunk_size,
            self.device,
            use_vae=True if self.vae_type in ["VAE"] else False,
            use_state=self.use_state)
        decoder = deconder_dict[self.decoder_name](self.input_dim,
                                                   self.input_dim,
                                                   self.embedding_dim,
                                                   self.layer_dims,
                                                   self.chunk_size,
                                                   self.device,
                                                   use_state=self.use_state)

        self.model_config["input_dim"] = list(self.train_dataset[0].shape[1:])

        model_config = vae_dict[self.vae_type]["model_config"](
            **self.model_config)

        self.model = vae_dict[self.vae_type]["model"](
            model_config=model_config, encoder=encoder, decoder=decoder)
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

        loss_dict = self.train_pipeline(
            train_data=self.train_dataset,
            eval_data=self.eval_dataset,
        )

        plt.figure(figsize=(10, 6))
        plt.plot(loss_dict["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(loss_dict["eval_loss"], label="Eval Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/loss_plot.png")
        torch.cuda.empty_cache()

        if self.vae_type == "VQVAE":
            visualize_vq_latent_space(
                self.model,
                {
                    "data": {
                        "action_chunk": self.train_dataset[0],
                        "state": self.train_dataset[1]
                    }
                },
                self.device,
                self.save_path,
            )

        max_latent_value, min_latent_value = visualize_benchmark_latent_space(
            self.model.encoder,
            {
                "action_chunk": self.train_dataset[0],
                "state": self.train_dataset[1]
            },
            self.device,
            self.save_path,
        )
        if self.vae_type == "VQVAE":
            max_latent_value = max_latent_value * 0 + 1
            min_latent_value = min_latent_value * 0 - 1

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
        torch.cuda.empty_cache()

    def init_data(self):
        self.train_dataset, self.eval_dataset, self.action_mean, self.action_std, self.action_min, self.action_max, self.input_dim = init_chunk_data(
            self.args_cli.data_dir, self.num_hand_joints, self.chunk_size,
            self.eval_percent, self.device)

    def decoder(self, state, z):

        decode_action = self.model.decoder(state, z)
        if isinstance(decode_action, tuple):
            decode_action = decode_action[0]

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
        default="logs/data_0616/raw_right_data.hdf5",
    )

    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        default="ldcp_transformer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decoder_name",
        type=str,
        default="ldcp_transformer",
    )
    parser.add_argument(
        "--use_state",
        action="store_true",
    )
    args_cli, hydra_args = parser.parse_known_args()

    vae = ACTVAEFAMILY(args_cli, timstep=1 / 1, hand_side=args_cli.hand_side)
