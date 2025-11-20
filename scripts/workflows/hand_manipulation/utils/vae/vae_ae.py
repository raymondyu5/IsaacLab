import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_nonlinear_vae(model_path,
                       num_hand_joints,
                       device="cuda",
                       latent_dim=32):

    vae_enconder_dict = torch.load(model_path + "/encoder.pth",
                                   map_location=device,
                                   weights_only=True)
    vae_deconder_dict = torch.load(model_path + "/decoder.pth",
                                   map_location=device,
                                   weights_only=True)

    model = NonLinearAE(num_hand_joints, latent_dim).to(device)
    vae_encoder = model.encoder
    vae_decoder = model.decoder
    vae_encoder.load_state_dict(vae_enconder_dict)
    vae_decoder.load_state_dict(vae_deconder_dict)
    vae_encoder.eval()
    vae_decoder.eval()
    return model, vae_encoder, vae_decoder


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 hidden_dims,
                 state_dim=None,
                 device="cuda"):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, state, action):
        return self.model(action), None


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim, hidden_dims, device="cuda"):
        super().__init__()
        layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, state, z):
        return self.model(z), None


class NonLinearAE(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder_dims,
                 state_dim=None,
                 decoder_dims=None,
                 device="cuda"):
        super().__init__()
        self.device = device
        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]

        self.encoder = Encoder(input_dim, latent_dim, encoder_dims, device)
        self.decoder = Decoder(latent_dim, input_dim, decoder_dims, device)

    def forward(self, state, action):
        z, _ = self.encoder(state, action)
        recon, _ = self.decoder(state, z)
        return recon, z

    def loss(self, state, action, beta=0.2):
        recon, z = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)

        loss = recon_loss

        # Soft latent bounding loss
        if z is not None:
            z_penalized = torch.where((z < -0.8) | (z > 0.8), z,
                                      torch.zeros_like(z))
            latent_loss = torch.mean(torch.exp((z_penalized / 1.2)**10) - 1.0)
            loss += beta * latent_loss

        return loss
