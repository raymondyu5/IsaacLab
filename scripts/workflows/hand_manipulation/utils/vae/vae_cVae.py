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


def load_cvae(model_path, num_hand_joints, device="cuda", latent_dim=32):

    vae_enconder_dict = torch.load(model_path + "/encoder.pth",
                                   map_location=device,
                                   weights_only=True)
    vae_deconder_dict = torch.load(model_path + "/decoder.pth",
                                   map_location=device,
                                   weights_only=True)

    model = cVAE(num_hand_joints, num_hand_joints, latent_dim).to(device)
    vae_encoder = model.encoder
    vae_decoder = model.decoder
    vae_encoder.load_state_dict(vae_enconder_dict)
    vae_decoder.load_state_dict(vae_deconder_dict)
    vae_encoder.eval()
    vae_decoder.eval()
    return model, vae_encoder, vae_decoder


# Encoder for cVAE
class Encoder(nn.Module):

    def __init__(self, state_dim, action_dim, latent_dim, layer_dims):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []
        dims = [input_dim] + layer_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.mean = nn.Linear(layer_dims[-1], latent_dim)
        self.log_std = nn.Linear(layer_dims[-1], latent_dim)

    def forward(self, state, action):
        x = self.hidden_layers(torch.cat([state, action], dim=1))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-4, 15)
        return mean, log_std


# Decoder for cVAE
class Decoder(nn.Module):

    def __init__(self, state_dim, action_dim, latent_dim, layer_dims):
        super().__init__()
        input_dim = state_dim + latent_dim
        layers = []
        dims = [input_dim] + layer_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dims[-1], action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state, z):
        x = torch.cat([state, z], dim=1)
        return self.model(x)


# Conditional VAE model
class cVAE(nn.Module):

    def __init__(self,
                 action_dim,
                 latent_dim,
                 layer_dims=[128, 128],
                 state_dim=None,
                 max_action=1.0,
                 device="cuda"):
        super().__init__()
        self.encoder = Encoder(state_dim, action_dim, latent_dim,
                               layer_dims).to(device)
        self.decoder = Decoder(state_dim, action_dim, latent_dim,
                               layer_dims).to(device)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state, action):
        mean, log_std = self.encoder(state, action)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        z = torch.tanh(z)  # Bound z to (-1, 1)
        a = self.decoder(state, z)
        return a, mean, std

    def decode(self, state, z=None, clip=None, raw=True):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)
            z = torch.tanh(z)
        elif clip is not None:
            z = z.clamp(-clip, clip)
        a = self.decoder(state, z)
        return a if raw else self.max_action * torch.tanh(a)

    def loss(self, state, action, beta=0.2):

        recon, mean, std = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) -
                          std.pow(2)).mean()
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss
