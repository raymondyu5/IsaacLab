import torch
import torch.nn as nn
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
import numpy as np
from scripts.workflows.hand_manipulation.utils.vae.act_transformer import Transformer, TransformerEncoderLayer, TransformerEncoder


class AEEncoder(BaseEncoder):

    def __init__(self, input_dim, latent_dim, hidden_dims, device="cuda"):
        super().__init__()
        self.device = device
        layers = []
        self.latent_dim = latent_dim
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, action):
        self.embedding = self.model(action)

        return ModelOutput(embedding=self.embedding)


class AEDecoder(BaseDecoder):

    def __init__(self, latent_dim, output_dim, hidden_dims, device="cuda"):
        super().__init__()
        # Decoder MLP
        layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, z):

        return ModelOutput(reconstruction=self.decoder(z))


class VAEEncoder(BaseEncoder):

    def __init__(self, input_dim, latent_dim, hidden_dims, device="cuda"):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)
        self.embedding = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)

        self.latent_dim = latent_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> ModelOutput:
        x = x.view(x.size(0), -1)  # flatten input if needed
        out = self.mlp(x)

        mu = self.embedding(out)
        log_var = self.log_var(out)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)  # optiona
        return ModelOutput(embedding=mu, log_covariance=log_var)


class DETRVAEEncoder(BaseEncoder):

    def __init__(self,
                 action_dim,
                 latent_dim,
                 hidden_dim,
                 device="cuda",
                 nhead=2,
                 dropout=0.1,
                 dim_feedforward=2048,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead,
                                                dim_feedforward, dropout,
                                                activation,
                                                normalize_before).to(device)
        encoder_norm = nn.LayerNorm(latent_dim).to(
            device) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, hidden_dim,
                                          encoder_norm).to(device)
        self.qpos_proj = nn.Linear(action_dim, hidden_dim).to(device)
        self.action_proj = nn.Linear(action_dim, hidden_dim).to(device)
        self.cls_embed = nn.Embedding(1, hidden_dim).to(device)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim).to(device)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.register_buffer('pos_table',
                             get_sinusoid_encoding_table(512, hidden_dim))
        self.device = device

    def forward(self, x, is_pad=None):
        """
        x should be a dict with keys:
            'qpos': (bs, qpos_dim)
            'actions': (bs, seq, action_dim)
            'is_pad': (bs, seq)
        """
        if isinstance(x, dict):
            qpos = x["qpos"]
            actions = x["actions"]
            is_pad = x.get("is_pad", None)
            bs = actions.shape[0]

            qpos_embed = self.qpos_proj(qpos).unsqueeze(1)  # (bs, 1, hidden)
            action_embed = self.action_proj(actions)  # (bs, seq, hidden)
            cls_token = self.cls_embed.weight.unsqueeze(0).repeat(
                bs, 1, 1)  # (bs, 1, hidden)

            encoder_input = torch.cat([cls_token, qpos_embed, action_embed],
                                      dim=1)  # (bs, seq+2, hidden)

        else:
            bs = x.shape[0]

            action_embed = self.action_proj(x)  # (bs, seq, hidden)

            cls_token = self.cls_embed.weight.unsqueeze(0).repeat(
                bs, 1, 1)  # (bs, 1, hidden)
            encoder_input = torch.cat([cls_token, action_embed],
                                      dim=1)  # (bs, seq+1, hidden)

        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden)
        cls_joint_is_pad = torch.full(
            (bs, 2), False).to(encoder_input.device)  # False: not a padding
        # If original is_pad is not provided, assume no padding for action tokens
        seq_len = encoder_input.shape[0]
        num_prefix = 2
        if is_pad is None:
            # Create full 'False' mask for action tokens
            num_action_tokens = seq_len - num_prefix
            is_pad = torch.zeros((bs, num_action_tokens),
                                 dtype=torch.bool,
                                 device=encoder_input.device)

        # Create False mask for CLS + qpos
        cls_joint_is_pad = torch.zeros((bs, num_prefix),
                                       dtype=torch.bool,
                                       device=encoder_input.device)

        # Concatenate
        is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)

        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(
            1, 0, 2)[encoder_input.shape[0] - 1, :, :].to(
                encoder_input.device)  # (seq+1, hidden)

        output = self.encoder(encoder_input,
                              pos=pos_embed,
                              src_key_padding_mask=is_pad)[0]  # (bs, hidden)

        self.embedding = self.latent_proj(output)

        return ModelOutput(embedding=self.embedding)


# adapt from act action chunk
def get_sinusoid_encoding_table(n_position, d_hid):

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAEDecoder(BaseDecoder):

    def __init__(self,
                 latent_dim,
                 input_dim=None,
                 hidden_dim=512,
                 device="cuda",
                 num_queries=10,
                 dropout=0.1,
                 dim_feedforward=2048,
                 enc_layers=4,
                 dec_layers=6,
                 pre_norm=False,
                 nhead=2):
        super().__init__()
        self.transformer = Transformer(
            d_model=latent_dim,
            dropout=dropout,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=True,
        )

        self.latent_proj = nn.Linear(latent_dim, self.transformer.d_model)

        self.query_embed = nn.Embedding(num_queries, self.transformer.d_model)
        self.action_head = nn.Linear(self.transformer.d_model, 16)
        self.is_pad_head = nn.Linear(self.transformer.d_model, 1)
        self.input_proj_robot_state = nn.Linear(16, self.transformer.d_model)
        self.additional_pos_embed = nn.Embedding(2, self.transformer.d_model)
        self.pos = torch.nn.Embedding(1, latent_dim)

    def forward(self, z, cond=None):
        """
        Inputs:
            z: (B, latent_dim)
            cond: Optional[dict], may contain 'qpos': (B, qpos_dim)
        Returns:
            reconstruction: predicted action sequence
            is_pad_pred: predicted padding mask
        """
        bs = z.size(0)

        latent_input = self.latent_proj(z)  # (B, d_model)

        if cond is not None and "qpos" in cond:
            proprio_input = self.input_proj_robot_state(
                cond["qpos"])  # (B, d_model)
            transformer_input = proprio_input.unsqueeze(1)  # (B, 1, d_model)

        else:
            # If no qpos, use zeros as dummy input
            d_model = latent_input.shape[-1]
            transformer_input = torch.zeros((bs, 1, d_model),
                                            device=z.device)  # (B, 1, d_model)

        hs = self.transformer(
            src=transformer_input,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=self.pos.weight,
            latent_input=latent_input,
            proprio_input=proprio_input
            if cond is not None and "qpos" in cond else None,
            additional_pos_embed=self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)

        is_pad_hat = self.is_pad_head(hs)
        return ModelOutput(reconstruction=a_hat, is_pad_pred=is_pad_hat)
