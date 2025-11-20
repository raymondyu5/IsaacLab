import torch
import numpy as np
import torch.nn as nn

from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn


def get_module_device(m: nn.Module):
    device = torch.device('cpu')
    try:
        param = next(iter(m.parameters()))
        device = param.device
    except StopIteration:
        pass
    return device


@torch.no_grad()
def get_output_shape(input_shape: Tuple[int], net: Callable[[torch.Tensor],
                                                            torch.Tensor]):
    device = get_module_device(net)
    test_input = torch.zeros((1, ) + tuple(input_shape), device=device)
    test_output = net(test_input)
    output_shape = tuple(test_output.shape[1:])
    return output_shape


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


def get_tensor(z, device):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return torch.FloatTensor(z.copy()).to(device).unsqueeze(0)
    else:
        return torch.FloatTensor(z.copy()).to(device)
