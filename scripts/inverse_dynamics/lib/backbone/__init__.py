"""Backbone neural network architectures for inverse dynamics models.

This module provides encoder networks for processing different observation modalities.
"""

from .pointnet import PointNet
from .obs_encoder import ObsEncoder

__all__ = ['PointNet', 'ObsEncoder']