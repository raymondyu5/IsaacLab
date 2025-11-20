from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
import sys

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class MultiImageObsEncoder(ModuleAttrMixin):

    def __init__(
        self,
        shape_meta: dict,
        image_model: Union[nn.Module, Dict[str, nn.Module]],
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():

            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = copy.deepcopy(image_model)
                key_model_map[key] = this_model.to(self.device)

            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map

        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.rgb_keys = rgb_keys

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input

        # pass all rgb obs to rgb model
        images = list()
        for key in self.rgb_keys:
            image = obs_dict[key]
            if batch_size is None:
                batch_size = image.shape[0]
            else:
                assert batch_size == image.shape[0]
            assert image.shape[1:] == self.key_shape_map[key]

            images.append(image)

        # (N*B,C)
        images = torch.cat(images, dim=0)

        # (N*B,D)
        feature = self.key_model_map['seg_pc'](pcds)
        # (N,B,D)
        feature = feature.reshape(batch_size, feature.shape[-1])

        features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros((batch_size, ) + shape,
                                   dtype=self.dtype,
                                   device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]

        return output_shape
