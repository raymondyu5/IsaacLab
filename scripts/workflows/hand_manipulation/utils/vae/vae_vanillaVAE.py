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
# add argparse arguments
parser = argparse.ArgumentParser()
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_vanillaVAE(model_path,
                    num_hand_joints,
                    device="cuda",
                    latent_dim=32,
                    hidden_size=64):

    vae_enconder_dict = torch.load(model_path + "/encoder.pth",
                                   map_location=device,
                                   weights_only=True)
    vae_deconder_dict = torch.load(model_path + "/decoder.pth",
                                   map_location=device,
                                   weights_only=True)

    encoder = Encoder(num_hand_joints, hidden_size, latent_dim)
    decoder = Decoder(latent_dim, hidden_size, num_hand_joints)
    model = VanillaVAE(encoder, decoder).to(device)

    vae_encoder = model.encoder
    vae_decoder = model.decoder
    vae_encoder.load_state_dict(vae_enconder_dict)
    vae_decoder.load_state_dict(vae_deconder_dict)
    vae_encoder.eval()
    vae_decoder.eval()
    return model, vae_encoder, vae_decoder


class Encoder(nn.Module):

    def __init__(self, input_dim, layer_dims, latent_size, device="cuda"):
        super(Encoder, self).__init__()
        layers = []

        dims = [input_dim] + layer_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers).to(device)
        self.enc_mu = nn.Linear(layer_dims[-1], latent_size).to(device)
        self.enc_log_sigma = nn.Linear(layer_dims[-1], latent_size).to(device)

    def forward(self, state, action):
        x = self.hidden_layers(action)
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)

        return mu, log_sigma


class Decoder(nn.Module):

    def __init__(self, latent_dim, layer_dims, output_dim, device="cuda"):
        super(Decoder, self).__init__()
        layers = []

        dims = [latent_dim] + layer_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(layer_dims[-1],
                                output_dim))  # Output layer: no activation
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, state, z):
        return self.decoder(z), None


class VanillaVAE(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder_dims,
                 state_dim=None,
                 decoder_dims=None,
                 device="cuda"):
        super(VanillaVAE, self).__init__()

        self.encoder = Encoder(input_dim, encoder_dims, latent_dim, device)
        self.decoder = Decoder(latent_dim, encoder_dims[::-1], input_dim,
                               device)  # reverse for symmetry

    def forward(self, state, action):
        mu, log_sigma = self.encoder(state, action)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        z = mu + eps * sigma  # reparameterization trick
        recon, _ = self.decoder(state, z)

        return recon, mu, log_sigma

    def loss(self, state, action, beta=0.1):
        # plot_action_distribution(action)
        recon, mu, log_sigma = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action, reduction="mean")
        kl_div = -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - torch.exp(2 * log_sigma),
            dim=1).mean()
        return recon_loss + beta * kl_div
