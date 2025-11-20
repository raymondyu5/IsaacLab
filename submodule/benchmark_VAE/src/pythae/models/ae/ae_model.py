import os
from typing import Optional

import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_AE_MLP
from .ae_config import AEConfig


class AE(BaseAE):
    """Vanilla Autoencoder model.

    Args:
        model_config (AEConfig): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically")

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x).embedding

        if isinstance(x, dict):
            deconder_dict = {"z": z} | x

            recon_x = self.decoder(deconder_dict)[
                "reconstruction"]  # action chunk / conditioning vae
            loss = self.loss_function(recon_x, x["action_chunk"])
        else:
            recon_x = self.decoder(z)["reconstruction"]

            loss = self.loss_function(recon_x, x)

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

    def loss_function(self, recon_x, x):

        MSE = F.mse_loss(recon_x.reshape(x.shape[0], -1),
                         x.reshape(x.shape[0], -1),
                         reduction="none").sum(dim=-1)

        return MSE.mean(dim=0)

    def decode_action(self, z):

        recon_x = self.decoder(z)["reconstruction"]
        return recon_x

    def decode_rl_action(self,
                         z,
                         min_index: float = -1.0,
                         max_index: float = 1.0):
        """
        Input: z ∈ [-1, 1] (continuous), shape [B] or [B, 1]
        Output: reconstructed action from the closest embedding in the VQ codebook
        """
        recon_x = self.decoder(z)["reconstruction"]
        return recon_x
