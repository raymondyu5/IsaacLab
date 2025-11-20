"""Observation encoder for combining multiple observation modalities.

This module provides an encoder that combines pointcloud observations (via PointNet)
with low-dimensional proprioceptive observations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .pointnet import PointNet


class ObsEncoder(nn.Module):
    """Multi-modal observation encoder.

    Combines pointcloud observations with low-dimensional observations by:
    1. Processing pointclouds through PointNet to extract features
    2. Concatenating PointNet features with low-dim observations

    Args:
        low_dim_size: Dimension of low-dimensional observations
        pcd_feature_dim: Dimension of PointNet output features
        pcd_in_channels: Number of input channels for PointNet (3 for XYZ)
        pcd_local_channels: Channel dimensions for PointNet local feature extraction
        pcd_global_channels: Feature dimensions for PointNet global feature refinement
        use_pointcloud: Whether to use pointcloud observations (if False, only use low-dim)

    Example:
        >>> encoder = ObsEncoder(low_dim_size=23, pcd_feature_dim=256)
        >>> obs = {
        ...     'low_dim': torch.randn(32, 23),
        ...     'seg_pc': torch.randn(32, 3, 2048)
        ... }
        >>> features = encoder(obs)  # (32, 279)
    """

    def __init__(
        self,
        low_dim_size: int,
        pcd_feature_dim: int = 256,
        pcd_in_channels: int = 3,
        use_pointcloud: bool = True,
        use_transform: bool = False,
        **kwargs  # Ignore legacy parameters
    ):
        super().__init__()

        self.low_dim_size = low_dim_size
        self.pcd_feature_dim = pcd_feature_dim
        self.use_pointcloud = use_pointcloud

        # PointNet encoder for pointcloud observations
        if use_pointcloud:
            self.pointnet = PointNet(
                in_channels=pcd_in_channels,
                output_dim=pcd_feature_dim,
                use_transform=use_transform
            )

            # Output dimension: PointNet features + low-dim observations
            self.output_dim = pcd_feature_dim + low_dim_size
        else:
            self.pointnet = None
            self.output_dim = low_dim_size

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-modal observations.

        Args:
            obs: Dictionary containing:
                - 'low_dim': Low-dimensional observations (batch, low_dim_size)
                - 'seg_pc': Pointcloud observations (batch, 3, num_points) [optional]

        Returns:
            Encoded observation features (batch, output_dim)
        """
        # Extract low-dimensional observations
        low_dim_obs = obs['low_dim']  # (batch, low_dim_size)

        if self.use_pointcloud:
            # Extract and encode pointcloud observations
            if 'seg_pc' not in obs:
                raise ValueError("Expected 'seg_pc' in observation dict when use_pointcloud=True")

            pcd = obs['seg_pc']  # (batch, channels, num_points)

            # Extract PointNet features
            pcd_features = self.pointnet(pcd)  # (batch, pcd_feature_dim)

            # Concatenate features
            combined = torch.cat([pcd_features, low_dim_obs], dim=-1)
            return combined  # (batch, pcd_feature_dim + low_dim_size)
        else:
            # Only use low-dimensional observations
            return low_dim_obs  # (batch, low_dim_size)

    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.output_dim


class DualObsEncoder(nn.Module):
    """Encoder that processes both current and next observations.

    This is a convenience wrapper that uses the same ObsEncoder for both
    current and next observations, then concatenates their features.

    Useful for inverse dynamics models that need (obs, next_obs) → action.

    Args:
        low_dim_size: Dimension of low-dimensional observations
        pcd_feature_dim: Dimension of PointNet output features
        **kwargs: Additional arguments passed to ObsEncoder

    Example:
        >>> encoder = DualObsEncoder(low_dim_size=23, pcd_feature_dim=256)
        >>> obs = {
        ...     'low_dim': torch.randn(32, 23),
        ...     'seg_pc': torch.randn(32, 3, 2048)
        ... }
        >>> next_obs = {
        ...     'low_dim': torch.randn(32, 23),
        ...     'seg_pc': torch.randn(32, 3, 2048)
        ... }
        >>> features = encoder(obs, next_obs)  # (32, 558) = 2 * 279
    """

    def __init__(
        self,
        low_dim_size: int,
        pcd_feature_dim: int = 256,
        **kwargs
    ):
        super().__init__()

        # Shared encoder for both observations
        self.obs_encoder = ObsEncoder(
            low_dim_size=low_dim_size,
            pcd_feature_dim=pcd_feature_dim,
            **kwargs
        )

        self.output_dim = 2 * self.obs_encoder.output_dim

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode both current and next observations.

        Args:
            obs: Current observation dictionary
            next_obs: Next observation dictionary

        Returns:
            Concatenated features (batch, 2 * obs_encoder.output_dim)
        """
        obs_features = self.obs_encoder(obs)
        next_obs_features = self.obs_encoder(next_obs)

        # Concatenate current and next observation features
        combined = torch.cat([obs_features, next_obs_features], dim=-1)
        return combined

    def get_output_dim(self) -> int:
        """Get the output dimension of the dual encoder."""
        return self.output_dim


if __name__ == "__main__":
    print("Testing ObsEncoder...")

    # Test with pointcloud
    encoder = ObsEncoder(low_dim_size=23, pcd_feature_dim=256)
    obs = {
        'low_dim': torch.randn(4, 23),
        'seg_pc': torch.randn(4, 3, 2048)
    }
    output = encoder(obs)
    print(f"Input: low_dim {obs['low_dim'].shape}, seg_pc {obs['seg_pc'].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 279) = 4 × (256 + 23)")
    print(f"Output dim: {encoder.get_output_dim()}")

    print("\n" + "=" * 80)
    print("Testing DualObsEncoder...")

    # Test dual encoder
    dual_encoder = DualObsEncoder(low_dim_size=23, pcd_feature_dim=256)
    obs = {
        'low_dim': torch.randn(4, 23),
        'seg_pc': torch.randn(4, 3, 2048)
    }
    next_obs = {
        'low_dim': torch.randn(4, 23),
        'seg_pc': torch.randn(4, 3, 2048)
    }
    output = dual_encoder(obs, next_obs)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 558) = 4 × 2 × (256 + 23)")
    print(f"Output dim: {dual_encoder.get_output_dim()}")

    print("\n" + "=" * 80)
    print("Testing ObsEncoder without pointcloud...")

    # Test without pointcloud
    encoder_no_pcd = ObsEncoder(low_dim_size=23, use_pointcloud=False)
    obs_no_pcd = {'low_dim': torch.randn(4, 23)}
    output_no_pcd = encoder_no_pcd(obs_no_pcd)
    print(f"Output shape: {output_no_pcd.shape}")
    print(f"Expected: (4, 23)")
    print(f"Output dim: {encoder_no_pcd.get_output_dim()}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters (with PCD): {total_params:,}")