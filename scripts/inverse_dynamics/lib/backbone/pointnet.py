"""PointNet architecture for processing point cloud observations.

Based on PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
https://arxiv.org/abs/1612.00593

This is a wrapper around the existing PointNet implementation at:
/gscratch/weirdlab/raymond/Pointnet_Pointnet2_pytorch
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Optional

# Add the PointNet repo to path
POINTNET_REPO = "/gscratch/weirdlab/raymond/Pointnet_Pointnet2_pytorch"
if POINTNET_REPO not in sys.path:
    sys.path.insert(0, POINTNET_REPO)

# Import from the existing implementation
from models.pointnet_utils import PointNetEncoder


class PointNet(nn.Module):
    """PointNet encoder wrapper for extracting global features from point clouds.

    This wraps the existing PointNetEncoder implementation and adds an optional
    output projection to match the desired feature dimension.

    Args:
        in_channels: Number of input channels per point (default: 3 for XYZ)
        output_dim: Desired output feature dimension (default: 256)
        use_transform: Whether to use T-Net spatial transformations (default: False)

    Example:
        >>> pointnet = PointNet(in_channels=3, output_dim=256)
        >>> points = torch.randn(32, 3, 2048)  # (batch, channels, num_points)
        >>> features = pointnet(points)  # (batch, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 256,
        use_transform: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim

        # Use the existing PointNetEncoder
        # Returns 1024D features by default
        self.encoder = PointNetEncoder(
            global_feat=True,
            feature_transform=use_transform,
            channel=in_channels
        )

        # Project from 1024D to desired output_dim
        if output_dim != 1024:
            self.projection = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Extract global features from point cloud.

        Args:
            points: Point cloud tensor of shape (batch, channels, num_points)
                   Typically channels=3 for XYZ coordinates

        Returns:
            Global feature vector of shape (batch, output_dim)
        """
        # PointNetEncoder returns (features, transform, transform_feat)
        features, _, _ = self.encoder(points)  # (batch, 1024)

        # Project to desired dimension if needed
        if self.projection is not None:
            features = self.projection(features)  # (batch, output_dim)

        return features


if __name__ == "__main__":
    # Test PointNet
    print("Testing PointNet...")
    pointnet = PointNet(
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256)
    )

    # Create dummy point cloud: (batch=4, channels=3, points=2048)
    dummy_input = torch.randn(4, 3, 2048)
    output = pointnet(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 256)")

    # Count parameters
    total_params = sum(p.numel() for p in pointnet.parameters())
    print(f"Total parameters: {total_params:,}")