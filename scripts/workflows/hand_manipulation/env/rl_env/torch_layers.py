import gymnasium as gym
import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import sys

sys.path.append("submodule/stable-baselines3")
from stable_baselines3.common.type_aliases import TensorDict

# Code source from Jiayuan Gu: https://github.com/Jiayuan-Gu/torkit3d
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional
from scripts.workflows.hand_manipulation.env.rl_env.common.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu
# from scripts.workflows.hand_manipulation.env.rl_env.pointnet_modules.pointnet import PointNet
import copy
import torchvision
# __all__ = ["PointNet"]

from torchvision import models


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


from scripts.workflows.hand_manipulation.utils.diffusion.backbone.pcd.pointnet import PointNet

# class PointNetExtractor(BaseFeaturesExtractor):
#     """
#     :param observation_space:
#     """

#     def __init__(self,
#                  observation_space,
#                  pc_key: str,
#                  feat_key: Optional[str] = None,
#                  use_bn=True,
#                  local_channels=(64, 128, 256),
#                  global_channels=(256, ),
#                  num_points=512,
#                  one_hot_dim=0):
#         self.pc_key = pc_key
#         if feat_key is not None:
#             if feat_key not in list(observation_space.keys()):
#                 raise RuntimeError(
#                     f"Feature key {feat_key} not in observation space.")

#         flattened_pcd_space_dim = 0

#         for pcd_key in self.pc_key:

#             if pcd_key not in observation_space.spaces.keys():
#                 raise RuntimeError(
#                     f"State key {pcd_key} not in observation space: {observation_space}"
#                 )
#             flattened_pcd_space_dim += observation_space.spaces[pcd_key].shape[
#                 -1]
#             observation_space.spaces.pop(pcd_key)
#         flattened_space_dim = 0
#         for key in observation_space.spaces.keys():
#             flattened_space_dim += observation_space.spaces[key].shape[-1]

#         state_observation_space = gym.spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(flattened_space_dim, ),  # Ensure shape is a tuple
#             dtype=np.float32)
#         pcd_observation_space = gym.spaces.Box(-np.inf,
#                                                np.inf,
#                                                shape=(3, num_points),
#                                                dtype=np.float32)

#         self.state_space = state_observation_space

#         # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
#         self.pc_key = pc_key
#         self.has_feat = feat_key is not None
#         self.feat_key = feat_key

#         pc_dim = pcd_observation_space.shape[0]
#         if self.has_feat:
#             feat_spec = observation_space[feat_key]
#             feat_dim = feat_spec.shape[1]
#         else:
#             feat_dim = 0
#         features_dim = global_channels[-1]

#         super().__init__(observation_space, features_dim)

#         self.point_net = PointNet(
#             use_bn=use_bn,
#             local_channels=local_channels,
#             global_channels=global_channels,
#         )

#         self.n_output_channels = self.point_net.out_channels

#     def forward(self, observations: TensorDict) -> th.Tensor:

#         # if self.has_feat:
#         #     feats = torch.transpose(observations[self.feat_key], 1, 2)
#         # else:
#         #     feats = None

#         return self.point_net(observations)


class PointNetExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    """

    def __init__(
        self,
        observation_space,
        pc_key: str,
        feat_key: Optional[str] = None,
        use_bn=True,
        local_channels=(64, 128, 256),
        global_channels=(256, ),
        use_layernorm: bool = False,
        final_norm: str = 'none',
    ):
        self.pc_key = pc_key

        features_dim = global_channels[-1]

        super().__init__(observation_space, features_dim)

        in_channels = 3
        self.mlp_out_channels = 256
        block_channel = [64, 128, 256, 512]

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], self.mlp_out_channels),
                nn.LayerNorm(self.mlp_out_channels))
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1],
                                              self.mlp_out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_pcd(self, observations: TensorDict) -> th.Tensor:

        x = self.mlp(observations.transpose(1, 2))
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


def pointcloud_extractor(observation, pc_key, num_points):

    points = []

    for key in pc_key:
        pcd_points = observation[key]
        pcd_points = pcd_points
        points.append(pcd_points)
        observation.pop(key)

    state_info = torch.cat([*observation.values()], dim=-1)
    points = torch.cat(points, dim=-1)
    B, N, D = points.shape  # [20, 512, 3]

    # Random indices
    indices = torch.randperm(N)[:num_points]  # shape: [256]

    if num_points > N:
        import pdb
        pdb.set_trace()

    # Apply same indices to all batches
    sampled_points = points[:, indices, :]  #

    return sampled_points.permute(0, 2, 1), state_info


class PointNetStateExtractor(PointNetExtractor):

    def __init__(self,
                 observation_space,
                 pc_key: str,
                 use_bn=True,
                 local_channels=(64, 128, 256),
                 global_channels=(256, ),
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 num_points=1024):
        super().__init__(
            copy.deepcopy(observation_space),
            pc_key,
            None,
            use_bn=use_bn,
            local_channels=local_channels,
            global_channels=global_channels,
        )
        self.pc_key = pc_key
        raw_observation_space = copy.deepcopy(observation_space)

        for pcd_key in self.pc_key:

            if pcd_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {pcd_key} not in observation space: {observation_space}"
                )
            raw_observation_space.spaces.pop(pcd_key)
        flattened_space_dim = 0
        for key in raw_observation_space.spaces.keys():
            flattened_space_dim += raw_observation_space.spaces[key].shape[-1]
        self.num_points = num_points

        state_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(flattened_space_dim, ),  # Ensure shape is a tuple
            dtype=np.float32)

        self.state_space = state_observation_space

        self.state_dim = self.state_space.shape[0]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = self.mlp_out_channels + output_dim
        # + self.state_space.shape[
        #     0]
        self._features_dim = self.n_output_channels
        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:

        shuffled_pcd_points, state_info = pointcloud_extractor(
            copy.deepcopy(observations), self.pc_key, self.num_points)

        pn_feat = self.forward_pcd(shuffled_pcd_points)
        state_feat = self.state_mlp(state_info)

        # if th.max(abs(state_info)) > 5 and state_info.shape[0] > 10:
        #     print(aa)

        return torch.cat([pn_feat, state_feat], dim=-1)


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """

    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


class ImageEncoder(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = True,
        encoder_type: str = 'resnet18',
    ) -> None:

        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        if encoder_type == 'NatureCNN':

            n_input_channels = observation_space.shape[-1]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif encoder_type.startswith('resnet'):

            # func = getattr(torchvision.models, "resnet18")
            # resnet = func()
            # resnet.fc = torch.nn.Identity()

            # self.cnn = resnet

            resnet = models.resnet18(pretrained=True).eval()

            self.cnn = resnet

        # Compute shape by doing one forward pass
        with th.no_grad():

            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()).float().permute(
                    0, 3, 1, 2)).shape[-1]
        self.normalized_image = normalized_image

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())
        self.image_out_channels = features_dim

    def forward_image(self, observations: th.Tensor) -> th.Tensor:

        if self.normalized_image:
            # If the image is normalized, we assume it is already in the right format
            observations = observations / 255.0

        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))


def image_extractor(observation, image_key, exclude_keys):

    image = observation[image_key]

    observation.pop(image_key)

    for obs_key in exclude_keys:
        if obs_key in observation:
            observation.pop(obs_key)

    state_info = torch.cat([*observation.values()], dim=-1)

    return image, state_info


class ImageStateExtractor(ImageEncoder):

    def __init__(self,
                 observation_space,
                 image_key: str,
                 image_features_dim=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 encoder_type='resnet18',
                 num_points=512):

        super().__init__(copy.deepcopy(observation_space[image_key]),
                         features_dim=image_features_dim,
                         encoder_type=encoder_type)
        self.image_key = image_key
        raw_observation_space = copy.deepcopy(observation_space)

        raw_observation_space.spaces.pop(image_key)
        flattened_space_dim = 0
        self.exclude_keys = []
        for key in raw_observation_space.spaces.keys():
            dim = raw_observation_space.spaces[key].shape
            if len(dim) > 1:
                self.exclude_keys.append(key)
                continue

            flattened_space_dim += raw_observation_space.spaces[key].shape[-1]
        self.num_points = num_points

        state_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(flattened_space_dim, ),  # Ensure shape is a tuple
            dtype=np.float32)

        self.state_space = state_observation_space

        self.state_dim = self.state_space.shape[0]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = self.image_out_channels + output_dim
        # + self.state_space.shape[
        #     0]
        self._features_dim = self.n_output_channels
        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:

        image, state_info = image_extractor(observations, self.image_key,
                                            self.exclude_keys)

        state_feat = self.state_mlp(state_info)

        image_feat = self.forward_image(image[:, 0])

        return torch.cat([image_feat, state_feat], dim=-1)


def load_dinov3(mode_name='dinov3_vits16'):
    """
    Load dinov3 model
    """

    REPO_DIR = "submodule/dinov3"  # your local repo (with hubconf.py)
    if mode_name == 'dinov3_vits16':

        CKPT_PATH = "submodule/dinov3/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(repo_or_dir=REPO_DIR,
                           model="dinov3_vits16",
                           source="local",
                           weights=CKPT_PATH).cuda().eval()
    return model


if __name__ == "__main__":
    model = load_dinov3()
