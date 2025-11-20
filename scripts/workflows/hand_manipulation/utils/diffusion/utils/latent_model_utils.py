import sys

sys.path.append("submodule/benchmark_VAE/src")
from pythae.models import AutoModel
import yaml

import os
import torch
from einops import rearrange


def load_latent_model(latent_model_path: str, device='cuda'):
    all_dirs = [
        d for d in os.listdir(latent_model_path)
        if os.path.isdir(os.path.join(latent_model_path, d))
    ]
    last_training = sorted(all_dirs)[-1]

    model = AutoModel.load_from_folder(os.path.join(latent_model_path,
                                                    last_training,
                                                    'final_model'),
                                       device=device)
    model.eval()

    with open(f"{latent_model_path}/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    chunk_size = model_config["chunk_size"]
    latent_dim = model_config["latent_dim"]
    vae_type = model_config["model_type"]
    if vae_type == "VQVAE":
        latent_dim = 1
    num_embedding = model_config["model_config"].get("num_embeddings", 1)
    return model, chunk_size, latent_dim, num_embedding


class LatentModelWrapper:

    def __init__(self, latent_model_path: str, device='cuda'):
        self.model, self.chunk_size, self.latent_dim, self.num_embedding = load_latent_model(
            latent_model_path, device)
        setattr(self.model, 'embed_table',
                self.model.quantizer.embeddings.to(device))

    def encode(self, data_dict):

        actions = data_dict['action']
        observations = data_dict['obs']
        num_seq = actions.shape[1]

        assert num_seq % self.chunk_size == 0
        batch = int(num_seq / self.chunk_size)
        latent_action = torch.zeros((actions.shape[0], batch, self.latent_dim),
                                    device=actions.device)
        with torch.no_grad():

            for batch_idx in range(batch):

                start_idx = batch_idx * self.chunk_size
                end_idx = (batch_idx + 1) * self.chunk_size
                per_action = actions[:, start_idx:end_idx, :]

                per_observation = observations[:, start_idx:end_idx, :]
                latent_action[:, batch_idx, :] = self.model.forward_encoder({
                    "action_chunk":
                    per_action.to(torch.float32),
                    "state":
                    per_observation.to(torch.float32)
                })

        return latent_action

    def decode(self, data_dict):

        latent_action = data_dict['action']
        observations = data_dict['obs']
        action = []
        for i in range(latent_action.shape[1]):

            start_index = i * self.chunk_size
            end_index = (i + 1) * self.chunk_size
            if observations.shape[1] > (end_index - start_index):
                per_observation = observations[:, start_index:end_index, :]
            else:
                per_observation = observations.clone()

            result = self.model.decode_action({
                "z":
                latent_action[:, i, :].unsqueeze(1).to(torch.float32),
                "state":
                per_observation.to(torch.float32)
            })

            action.append(result)

        return torch.cat(action, dim=1)

    def decode_diffusion(self, data_dict):

        latent_action = data_dict['action']
        observations = data_dict['obs']
        action = []

        for i in range(latent_action.shape[1]):

            start_index = i * self.chunk_size
            end_index = (i + 1) * self.chunk_size
            # per_observation = observations[:, i:i + 1, :]

            result = self.model.decoder({
                "z":
                latent_action[:, i, :].unsqueeze(1).to(torch.float32),
                "state":
                observations.to(torch.float32)
            }).reconstruction
            action.append(result)

        return torch.cat(action, dim=1)
