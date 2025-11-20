# adpat from https://github.com/ldcq/ldcq/blob/7ec96f3682e04e89385fe18e17a20f5315f0048d/models/skill_model.py
import numpy as np
import torch
import torch.nn as nn

import torch.distributions.kl as KL
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

# adapted from reactive difffusion policy : https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/main/reactive_diffusion_policy/model/vae/model.py
import einops


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class ReactiveDiffEncoderCNN(BaseEncoder):

    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        n_conv_layers=4,
        hidden_dim=128,
        device='cuda',
        normalize_latent=False,
        use_vae=False,
        use_state=True,
    ):
        super(ReactiveDiffEncoderCNN, self).__init__()

        self.action_dim = action_dim
        self.device = device

        # Build the convolutional encoder
        layers = []
        in_channels = action_dim
        for _ in range(n_conv_layers):
            layers.append(
                nn.Conv1d(in_channels,
                          hidden_dim,
                          kernel_size=5,
                          stride=2,
                          padding=2))
            layers.append(nn.ReLU())
            in_channels = hidden_dim  # update for next layer

        # Final projection to latent space
        layers.append(
            nn.Conv1d(in_channels,
                      latent_dim,
                      kernel_size=5,
                      stride=2,
                      padding=2))

        self.encoder = nn.Sequential(*layers)
        self.apply(weights_init_encoder)

    def forward(self, data, flatten=False):
        """
        Args:
            data: Dict with keys:
                - action_chunk: (N, T, A)
                - state: (N, ...)
            flatten: If True, flatten the output to (N, T*C)

        Returns:
            Tensor of shape (N, T', C) or (N, T'*C) if flatten=True
        """
        actions = data["action_chunk"]  # shape (N, T, A)
        x = einops.rearrange(actions,
                             "N T A -> N A T")  # for Conv1d: (N, C, L)

        h = self.encoder(x)  # shape: (N, latent_dim, T')

        h = einops.rearrange(h, "N C T -> N T C")

        if flatten:
            h = einops.rearrange(h, "N T C -> N (T C)")

        return ModelOutput(embedding=h.reshape(
            actions.shape[0],
            -1,
        ))


class ReactiveDiffDecoderRNN(BaseDecoder):

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 action_chunk_len=10,
                 device='cuda',
                 hidden_dim=128,
                 use_state=True):
        super(ReactiveDiffDecoderRNN, self).__init__()
        self.rnn = nn.GRU(latent_dim + state_dim,
                          hidden_dim,
                          layer_dims,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim * action_chunk_len)
        self.action_dim = action_dim
        self.apply(weights_init_encoder)

    def forward(self, data):

        x = torch.cat([data["z"][:, None], data["state"][:, 0][:, None]],
                      dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)

        x = einops.rearrange(x, "N T A -> N (T A)").reshape(
            x.shape[0], -1, self.action_dim)

        return ModelOutput(reconstruction=x)
