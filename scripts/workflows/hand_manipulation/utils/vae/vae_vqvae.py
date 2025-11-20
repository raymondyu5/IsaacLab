import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dims, device="cuda"):
        super().__init__()
        self.device = device
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, state, action):

        return self.model(action), None


class QuantizedDecoder(nn.Module):

    def __init__(self,
                 latent_dim,
                 output_dim,
                 hidden_dims,
                 num_embeddings=128,
                 commitment_cost=0.25,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.embedding_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Embedding table
        self.embeddings = nn.Embedding(num_embeddings, latent_dim).to(device)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings,
                                             1.0 / num_embeddings)

        # Decoder MLP
        layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, state, z):
        # Flatten
        flat_z = z.view(-1, self.embedding_dim)

        # L2 distance
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) -
                     2 * torch.matmul(flat_z, self.embeddings.weight.t()) +
                     torch.sum(self.embeddings.weight**2, dim=1))

        # Find closest embedding
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        one_hot = torch.zeros(indices.shape[0],
                              self.num_embeddings,
                              device=z.device)
        one_hot.scatter_(1, indices, 1)

        # Quantize
        quantized = torch.matmul(one_hot, self.embeddings.weight).view(z.shape)

        # Losses
        commitment_loss = self.commitment_cost * F.mse_loss(
            z.detach(), quantized)
        embedding_loss = F.mse_loss(z, quantized.detach())
        vq_loss = commitment_loss + embedding_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Decode
        recon = self.decoder(quantized)
        return recon, quantized, vq_loss, indices.view(z.shape[0], -1)


class VQVAE(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder_dims,
                 state_dim=None,
                 decoder_dims=None,
                 num_embeddings=128,
                 commitment_cost=0.25,
                 device="cuda"):
        super().__init__()
        self.device = device
        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]

        self.encoder = Encoder(input_dim, latent_dim, encoder_dims, device)
        self.decoder = QuantizedDecoder(latent_dim, input_dim, decoder_dims,
                                        num_embeddings, commitment_cost,
                                        device)

    def forward(self, state, action):
        z, _ = self.encoder(state, action)
        recon, z_q, vq_loss, _ = self.decoder(state, z)
        return recon, z, z_q, vq_loss

    def loss(self, state, action):
        recon, z, z_q, vq_loss = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)
        total_loss = recon_loss + vq_loss
        return total_loss
