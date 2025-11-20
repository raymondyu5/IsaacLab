import torch.nn as nn
from .util import init, get_clones
"""MLP modules."""


class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal,
                 use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        # active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        active_func = [nn.ELU(), nn.ELU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m,
                        init_method,
                        lambda x: nn.init.constant_(x, 0),
                        gain=gain)

        dims = [input_dim] + hidden_size  # e.g., [input_dim, 256, 128, 64]

        self.layers = nn.Sequential(*[
            nn.Sequential(init_(nn.Linear(dims[i], dims[i + 1])), active_func,
                          nn.LayerNorm(dims[i + 1]))
            for i in range(len(hidden_size))
        ])

    def forward(self, x):

        return self.layers(x)


class MLPBase(nn.Module):

    def __init__(self, config, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = config["use_feature_normalization"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_ReLU = config["use_ReLU"]
        self._stacked_frames = config["stacked_frames"]
        self._layer_N = config["layer_N"]
        self.hidden_size = config["hidden_size"]

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size, self._layer_N,
                            self._use_orthogonal, self._use_ReLU)
        # self.mlp_middle_layer = MLPLayer(self.hidden_size, self.hidden_size,
        #                       self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        # x = self.mlp_middle_layer(x)

        return x
