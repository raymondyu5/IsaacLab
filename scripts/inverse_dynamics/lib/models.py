"""Inverse dynamics model architectures.

This module provides inverse dynamics models compatible with the training infrastructure.
Models predict actions given current and next observations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict


class InverseDynamicsMLP(nn.Module):
    """MLP-based inverse dynamics model.

    Predicts action from concatenated (obs, next_obs) using a simple feedforward network.

    Args:
        obs_dim: Dimension of observation
        action_dim: Dimension of action
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout: Dropout probability (0 = no dropout)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = 'relu',
        dropout: float = 0.0,
        name: str = 'id_mlp',
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.name = name

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        # Input: concatenated (obs, next_obs)
        input_dim = obs_dim * 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predicted and target actions.

        Args:
            obs: Current observation (B, obs_dim)
            next_obs: Next observation (B, obs_dim)
            action: Target action (B, action_dim)

        Returns:
            MSE loss (scalar)
        """
        # Concatenate observations
        cond = torch.cat([obs, next_obs], dim=-1)  # (B, obs_dim * 2)

        # Predict action
        pred_action = self.network(cond)  # (B, action_dim)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(pred_action, action)

        return loss

    def predict(self, obs: torch.Tensor, next_obs: torch.Tensor) -> np.ndarray:
        """Predict action given observations.

        Args:
            obs: Current observation (B, obs_dim) or (obs_dim,)
            next_obs: Next observation (B, obs_dim) or (obs_dim,)

        Returns:
            Predicted action as numpy array (B, action_dim) or (action_dim,)
        """
        with torch.no_grad():
            # Handle single sample (no batch dimension)
            squeeze = False
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
                next_obs = next_obs.unsqueeze(0)
                squeeze = True

            # Concatenate observations
            cond = torch.cat([obs, next_obs], dim=-1)

            # Predict
            action = self.network(cond)

            # Remove batch dimension if needed
            if squeeze:
                action = action.squeeze(0)

            return action.cpu().numpy()


class InverseDynamicsResidualMLP(nn.Module):
    """Residual MLP-based inverse dynamics model.

    Uses residual connections for deeper networks.

    Args:
        obs_dim: Dimension of observation
        action_dim: Dimension of action
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout: Dropout probability
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256, 256],
        activation: str = 'relu',
        dropout: float = 0.1,
        name: str = 'id_resmlp',
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.name = name

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input projection
        input_dim = obs_dim * 2
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(
                hidden_dims[i],
                hidden_dims[i + 1],
                activation=self.activation,
                dropout=dropout
            )
            self.residual_blocks.append(block)

        # Output layer
        self.output_proj = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        # Concatenate and project
        cond = torch.cat([obs, next_obs], dim=-1)
        x = self.activation(self.input_proj(cond))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output
        pred_action = self.output_proj(x)

        # MSE loss
        loss = torch.nn.functional.mse_loss(pred_action, action)
        return loss

    def predict(self, obs: torch.Tensor, next_obs: torch.Tensor) -> np.ndarray:
        """Predict action."""
        with torch.no_grad():
            squeeze = False
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
                next_obs = next_obs.unsqueeze(0)
                squeeze = True

            cond = torch.cat([obs, next_obs], dim=-1)
            x = self.activation(self.input_proj(cond))

            for block in self.residual_blocks:
                x = block(x)

            action = self.output_proj(x)

            if squeeze:
                action = action.squeeze(0)

            return action.cpu().numpy()


class ResidualBlock(nn.Module):
    """Residual block with projection if dimensions don't match."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projection if dimensions don't match
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)

        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out = out + residual
        out = self.activation(out)

        return out


class InverseDynamicsPointNet(nn.Module):
    """PointNet-based inverse dynamics model.

    This model combines pointcloud observations (processed via PointNet) with
    low-dimensional proprioceptive observations to predict actions from
    (observation, next_observation) pairs.

    Architecture:
        1. Encode obs and next_obs using shared observation encoder (PointNet + concat)
        2. Concatenate encoded observations
        3. MLP to predict action

    Args:
        obs_dim: Dimension of low-dimensional observations
        action_dim: Dimension of action
        pcd_feature_dim: Dimension of PointNet output features (default: 256)
        pcd_in_channels: Number of input channels for PointNet (default: 3 for XYZ)
        pcd_local_channels: Channel dimensions for PointNet local feature extraction
        pcd_global_channels: Feature dimensions for PointNet global feature refinement
        hidden_dims: List of hidden layer dimensions for action prediction MLP
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout: Dropout probability
        name: Model name for logging
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        pcd_feature_dim: int = 256,
        pcd_in_channels: int = 3,
        use_transform: bool = False,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        dropout: float = 0.1,
        name: str = 'id_pointnet',
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from scripts.inverse_dynamics.lib.backbone.obs_encoder import DualObsEncoder

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pcd_feature_dim = pcd_feature_dim
        self.name = name

        # Default values
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dual observation encoder (processes both obs and next_obs)
        self.dual_obs_encoder = DualObsEncoder(
            low_dim_size=obs_dim,
            pcd_feature_dim=pcd_feature_dim,
            pcd_in_channels=pcd_in_channels,
            use_transform=use_transform,
            use_pointcloud=True,
        )

        # MLP for action prediction
        # Input: concatenated (obs_features, next_obs_features)
        input_dim = self.dual_obs_encoder.get_output_dim()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target actions.

        Args:
            obs: Current observation dict with 'low_dim' and 'seg_pc' keys
                - 'low_dim': (B, obs_dim)
                - 'seg_pc': (B, 3, num_points)
            next_obs: Next observation dict with same structure
            action: Target action (B, action_dim)

        Returns:
            MSE loss (scalar)
        """
        # Encode observations using dual encoder
        encoded = self.dual_obs_encoder(obs, next_obs)  # (B, 2 * (pcd_feature_dim + obs_dim))

        # Predict action
        pred_action = self.mlp(encoded)  # (B, action_dim)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(pred_action, action)

        return loss

    def predict(
        self,
        obs: Dict[str, torch.Tensor],
        next_obs: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Predict action given observations.

        Args:
            obs: Current observation dict
            next_obs: Next observation dict

        Returns:
            Predicted action as numpy array (B, action_dim) or (action_dim,)
        """
        with torch.no_grad():
            # Handle single sample (no batch dimension)
            squeeze = False
            if len(obs['low_dim'].shape) == 1:
                obs = {k: v.unsqueeze(0) for k, v in obs.items()}
                next_obs = {k: v.unsqueeze(0) for k, v in next_obs.items()}
                squeeze = True

            # Encode observations
            encoded = self.dual_obs_encoder(obs, next_obs)

            # Predict action
            action = self.mlp(encoded)

            # Remove batch dimension if needed
            if squeeze:
                action = action.squeeze(0)

            return action.cpu().numpy()
