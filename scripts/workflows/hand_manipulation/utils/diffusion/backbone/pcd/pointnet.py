import torch
import torch.nn as nn

from scripts.workflows.hand_manipulation.utils.diffusion.backbone.pcd.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu


class PointNet(nn.Module):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
            self,
            in_channels=3,
            local_channels=(64, 64, 64, 128, 1024),
            global_channels=(512, 256),
            use_bn=False,
    ):
        super(PointNet, self).__init__()
        self.output_feature = global_channels[-1]

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn

        net_list = []

        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            net_list.append(self.mlp_local)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
            net_list.append(self.mlp_global)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self.reset_parameters()
        self.nets = nn.Sequential(*net_list)

    def forward(self, points) -> dict:
        # points: [B, 3, N]; points_feature: [B, C, N], points_mask: [B, N]

        local_feature = self.mlp_local(points)

        global_feature, max_indices = torch.max(local_feature, 2)
        output_feature = self.mlp_global(global_feature)

        return output_feature

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01

    def output_shape(self, inputs):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return self.output_feature
