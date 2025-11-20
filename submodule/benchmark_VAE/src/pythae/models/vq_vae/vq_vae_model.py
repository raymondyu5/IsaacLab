import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .vq_vae_config import VQVAEConfig
from .vq_vae_utils import Quantizer, QuantizerEMA


class VQVAE(AE):
    r"""
    Vector Quantized-VAE model.

    Args:
        model_config (VQVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: VQVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        AE.__init__(self,
                    model_config=model_config,
                    encoder=encoder,
                    decoder=decoder)
        self.device = self.encoder.device

        self._set_quantizer(model_config)

        self.model_name = "VQVAE"

    def _set_quantizer(self, model_config):
        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of VQVAEConfig instance must be set to 'data_shape' where "
                "the shape of the data is (C, H, W ..). Unable to set quantizer."
            )

        x = torch.randn((2, ) + self.model_config.input_dim).to(self.device)

        if len(self.model_config.input_dim) > 1:

            z = self.encoder({"action_chunk": x, "state": x}).embedding

        else:
            z = self.encoder(x).embedding
        self.model_config.embedding_dim = z.shape[-1]

        if len(z.shape) < 4:
            z = z.reshape(z.shape[0], 1, 1, -1)

        z = z.permute(0, 2, 3, 1)

        if model_config.use_ema:
            self.quantizer = QuantizerEMA(model_config=model_config)

        else:
            self.quantizer = Quantizer(model_config=model_config)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        uses_ddp = kwargs.pop("uses_ddp", False)

        encoder_output = self.encoder(x)

        embeddings = encoder_output.embedding

        reshape_for_decoding = False

        if len(embeddings.shape) < 4:
            embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
            reshape_for_decoding = True

        embeddings = embeddings.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings, uses_ddp=uses_ddp)

        quantized_embed = quantizer_output.quantized_vector
        quantized_indices = quantizer_output.quantized_indices

        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

        if isinstance(x, dict):
            deconder_dict = {"z": quantized_embed} | x

            recon_x = self.decoder(deconder_dict)[
                "reconstruction"]  # action chunk / conditioning vae
            loss, recon_loss, vq_loss = self.loss_function(
                recon_x, x["action_chunk"], quantizer_output)
        else:
            recon_x = self.decoder(quantized_embed).reconstruction

            loss, recon_loss, vq_loss = self.loss_function(
                recon_x, x, quantizer_output)

        output = ModelOutput(
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            loss=loss,
            recon_x=recon_x,
            z=quantized_embed,
            quantized_indices=quantized_indices,
        )

        return output

    def loss_function(self, recon_x, x, quantizer_output):

        recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1),
                                x.reshape(x.shape[0], -1),
                                reduction="none").sum(dim=-1)

        vq_loss = quantizer_output.loss

        return (
            (recon_loss + vq_loss).mean(dim=0),
            recon_loss.mean(dim=0),
            vq_loss.mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def decode_action(self, z):

        reshape_for_decoding = False
        if isinstance(z, dict):
            embeddings = z["z"]
            state = z.get("state", None)

            recon_x = self.decoder({
                "z": embeddings,
                "state": state
            }).reconstruction

        else:

            if len(z.shape) == 2:
                embeddings = z.reshape(z.shape[0], 1, 1, -1)
                reshape_for_decoding = True
            else:
                embeddings = z

            embeddings = embeddings.permute(0, 2, 3, 1)

            quantizer_output = self.quantizer(embeddings, uses_ddp=False)

            quantized_embed = quantizer_output.quantized_vector

            self.quantizer.embeddings
            self.decode_quantized_indices = quantizer_output.quantized_indices

            if reshape_for_decoding:
                quantized_embed = quantized_embed.reshape(
                    embeddings.shape[0], -1)
            recon_x = self.decoder(quantized_embed).reconstruction

        return recon_x

    def indices_to_embeddings(self, index_combinations, embeddings):
        """
        Args:
            index_combinations: list of tuples, each of length latent_dim
            embeddings: (num_embeddings, embedding_dim)
        Returns:
            (num_combinations, latent_dim, embedding_dim)
        """
        embedding_dim = embeddings.shape[1]
        all_vectors = []
        for idx_combo in index_combinations:
            vectors = [embeddings[i] for i in idx_combo]  # list of tensors
            all_vectors.append(
                torch.stack(vectors))  # shape (latent_dim, embedding_dim)
        return torch.stack(
            all_vectors)  # shape (N^D, latent_dim, embedding_dim)

    def decode_action_index(self, z):

        z_latents = self.quantizer.embeddings[z]

        recon = self.decoder(z_latents)

        return recon.reconstruction

    def decode_rl_action(self,
                         z: torch.Tensor,
                         min_index: float = -1.0,
                         max_index: float = 1.0):
        """
        Input: z ∈ [-1, 1] (continuous), shape [B] or [B, 1]
        Output: reconstructed action from the closest embedding in the VQ codebook
        """
        use_dict = False

        if isinstance(z, dict):

            state = z.get("state", None)
            z = z["z"].reshape(z["z"].shape[0], -1)
            use_dict = True
        z = torch.clamp(z, min_index, max_index)
        # Flatten if needed
        z = z.view(-1)  # shape: [B]

        # Normalize z from [-1, 1] → [0, 1]
        z_normalized = (z - min_index) / (max_index - min_index)

        # Scale to codebook indices: [0, num_embeddings)
        indices = (z_normalized * self.quantizer.num_embeddings).long().clamp(
            0, self.quantizer.num_embeddings + -1)  # shape: [B]

        # Get embeddings from codebook
        quantized_embed = self.quantizer.embeddings[
            indices]  # shape: [B, embedding_dim]

        if use_dict:

            recon_x = self.decoder({
                "z": quantized_embed,
                "state": state
            }).reconstruction

        else:
            # Decode using MLP
            recon_x = self.decoder(quantized_embed).reconstruction

        return recon_x

    def encode_rl_action(self,
                         action: torch.Tensor,
                         min_index: float = -1.0,
                         max_index: float = 1.0):
        """
        Input: action ∈ [-1, 1] (continuous), shape [B] or [B, 1]
        Output: encoded action in the VQ codebook
        """
        # Flatten if needed

        encoder_output = self.encoder(action)

        embeddings = encoder_output.embedding

        if len(embeddings.shape) < 4:
            embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)

        embeddings = embeddings.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings)

        quantized_indices = quantizer_output.quantized_indices.float(
        )  # shape: [180, 1]
        num_embeddings = self.quantizer.num_embeddings  # = 64
        normalized = (quantized_indices /
                      (num_embeddings - 1)) * 2 - 1  # → [-1, 1]
        return normalized.squeeze(1)
