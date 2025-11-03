#!/usr/bin/env python3
"""
small script to skip using ID, and directly process data for bc

This script can create two types of datasets:
1. BC (Behavior Cloning): (state, action) pairs from real trajectory actions
2. AT (Action Translation): (state_target, action_source_predicted, action_target) using ID model

Example usage:

# Create BC dataset from trajectories (uses real actions)
./isaaclab.sh -p scripts/data_collection/create_dataset_from_trajectories.py \
    --trajectory_file trajectory_data/slippery/train.hdf5 \
    --output_dir datasets/bc_slippery \
    --dataset_type bc \
    --min_reward 10

# Create Action Translation dataset (uses ID model to predict source actions)
./isaaclab.sh -p scripts/data_collection/create_dataset_from_trajectories.py \
    --trajectory_file trajectory_data/slippery/train.hdf5 \
    --output_dir datasets/action_translation/normal_to_slippery \
    --dataset_type action_translation \
    --source_id_model trained_models/id_normal/model.pth \
    --source_id_config configs/inverse_dynamics/mlp.yaml \
    --min_reward 10
"""

import argparse
import h5py
import numpy as np
import os
import sys
import torch
import yaml
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lib.trajectory_dataset import load_trajectory_dataset
from scripts.lib.training import build_model_from_config


def load_id_model(model_path: str, config_path: str, device: str = 'cuda'):
    """Load inverse dynamics model for predicting source actions.

    Args:
        model_path: Path to ID model checkpoint
        config_path: Path to ID model config YAML
        device: Device to load model on

    Returns:
        model: Loaded ID model
        checkpoint: Checkpoint dict with normalization stats
    """
    print(f"\n[INFO] Loading ID model from {model_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        # Extract dimensions from checkpoint layers
        first_layer_weight = None
        last_layer_weight = None

        state_dict_keys = list(checkpoint['model_state_dict'].keys())

        # Find first layer (input layer)
        for key in state_dict_keys:
            if 'network.0.weight' in key:
                first_layer_weight = checkpoint['model_state_dict'][key]
                break

        # Find last layer (output layer)
        for key in reversed(state_dict_keys):
            if 'weight' in key and 'network' in key:
                last_layer_weight = checkpoint['model_state_dict'][key]
                break

        if first_layer_weight is not None:
            # First layer input is concatenated (state + next_state), so divide by 2
            concat_dim = first_layer_weight.shape[1]
            obs_dim = concat_dim // 2
        else:
            obs_dim = 134  # Default to 134D states

        if last_layer_weight is not None:
            action_dim = last_layer_weight.shape[0]
        else:
            action_dim = 22  # Default to 22D actions for Kuka-Allegro IK
    else:
        obs_dim = 134
        action_dim = 22

    # Build model
    model = build_model_from_config(model_config, obs_dim, action_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    has_normalization = checkpoint.get('normalize_data', False)
    print(f"[INFO] ID model loaded (obs_dim={obs_dim}, action_dim={action_dim}, normalized={has_normalization})")

    return model, checkpoint


def predict_actions_with_id_model(
    model,
    checkpoint: dict,
    states: np.ndarray,
    next_states: np.ndarray,
    device: str = 'cuda',
    batch_size: int = 1024
) -> np.ndarray:
    """Predict actions using ID model with normalization.

    Args:
        model: ID model
        checkpoint: Checkpoint with normalization stats
        states: State array (N, state_dim)
        next_states: Next state array (N, state_dim)
        device: Device for inference
        batch_size: Batch size for prediction

    Returns:
        predicted_actions: (N, action_dim) array
    """
    print(f"\n[INFO] Predicting actions using ID model...")

    # Check if model uses normalization
    normalize_data = checkpoint.get('normalize_data', False)
    if normalize_data and 'states' in checkpoint:
        # Normalization stats stored as tuples (mean, std)
        state_mean, state_std = checkpoint['states']
        state_mean = state_mean.cpu().numpy() if isinstance(state_mean, torch.Tensor) else state_mean
        state_std = state_std.cpu().numpy() if isinstance(state_std, torch.Tensor) else state_std
        # Also get action normalization stats for denormalization
        action_mean, action_std = checkpoint['actions']
        action_mean = action_mean.cpu().numpy() if isinstance(action_mean, torch.Tensor) else action_mean
        action_std = action_std.cpu().numpy() if isinstance(action_std, torch.Tensor) else action_std
    else:
        state_mean = None
        state_std = None
        action_mean = None
        action_std = None

    num_samples = len(states)
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_predictions = []

    for i in tqdm(range(num_batches), desc="Predicting with ID model"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_states = states[start_idx:end_idx]
        batch_next_states = next_states[start_idx:end_idx]

        # Convert to tensors
        states_tensor = torch.FloatTensor(batch_states).to(device)
        next_states_tensor = torch.FloatTensor(batch_next_states).to(device)

        # Normalize if needed
        if normalize_data and state_mean is not None:
            state_mean_t = torch.FloatTensor(state_mean).to(device)
            state_std_t = torch.FloatTensor(state_std).to(device)
            states_tensor = (states_tensor - state_mean_t) / state_std_t
            next_states_tensor = (next_states_tensor - state_mean_t) / state_std_t

        # Predict
        with torch.no_grad():
            predictions = model.predict(states_tensor, next_states_tensor)

            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, device=device)

            # Denormalize predictions if model was trained with normalization
            if normalize_data and action_mean is not None:
                # breakpoint()
                action_mean_t = torch.FloatTensor(action_mean).to(device)
                action_std_t = torch.FloatTensor(action_std).to(device)
                predictions = predictions * action_std_t + action_mean_t

        all_predictions.append(predictions.cpu().numpy())

    predicted_actions = np.concatenate(all_predictions, axis=0)

    print(f"[INFO] Predicted {len(predicted_actions):,} actions")
    print(f"  Action range: [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
    print(f"  Action mean: {predicted_actions.mean():.4f}, std: {predicted_actions.std():.4f}")

    return predicted_actions

# add to utils? slightly different than others
def create_dataset_from_trajectories(
    trajectory_path: str,
    output_path: str,
    dataset_type: str = 'bc',
    source_id_model: Optional[str] = None,
    source_id_config: Optional[str] = None,
    max_demos: Optional[int] = None,
    min_reward: Optional[float] = None,
    max_samples: Optional[int] = None,
    use_90d_states: bool = True  # Keep for backward compat, but always True now
):
    """
    Create dataset from trajectory data using proper state extraction.

    Always uses the same state representation that the ID model was trained on
    (extracted from full observations, not raw 165D obs with action history).

    Args:
        trajectory_path: Path to trajectory HDF5 file
        output_path: Path to save output dataset
        dataset_type: 'bc' or 'action_translation'
        source_id_model: Path to source ID model (for AT only)
        source_id_config: Path to source ID config (for AT only)
        max_demos: Maximum number of demos to use
        min_reward: Minimum reward threshold for filtering
        max_samples: Maximum number of samples to include
        use_90d_states: (Deprecated, kept for compatibility) Always uses proper state extraction
    """
    print("=" * 80)
    print(f"CREATING {dataset_type.upper()} DATASET FROM TRAJECTORIES")
    print("=" * 80)
    print(f"Input: {trajectory_path}")
    print(f"Output: {output_path}")
    print(f"Dataset type: {dataset_type}")
    if use_90d_states:
        print(f"State representation: 90D Markovian states")
    else:
        print(f"State representation: 165D full observations")
    print()

    # Validate inputs
    if dataset_type == 'action_translation':
        if not source_id_model or not source_id_config:
            raise ValueError("For action_translation dataset, must provide --source_id_model and --source_id_config")
        if not os.path.exists(source_id_model):
            raise FileNotFoundError(f"ID model not found: {source_id_model}")
        if not os.path.exists(source_id_config):
            raise FileNotFoundError(f"ID config not found: {source_id_config}")

    # Load trajectory data using unified loader
    print(f"[INFO] Loading trajectory data...")
    states, actions_target, next_states = load_trajectory_dataset(
        trajectory_path,
        filter_success_only=False,
        min_reward_threshold=min_reward,
        num_episodes=max_demos,
        use_observations=not use_90d_states  # If use_90d_states=True, use states; else use observations
    )

    # Limit samples if requested
    if max_samples and max_samples < len(states):
        print(f"[INFO] Limiting to {max_samples:,} samples (from {len(states):,})")
        states = states[:max_samples]
        actions_target = actions_target[:max_samples]
        next_states = next_states[:max_samples]

    # Get dimensions
    state_dim = states.shape[1]
    action_dim = actions_target.shape[1] if len(actions_target.shape) > 1 else 1

    print(f"\n[INFO] Loaded {len(states):,} transitions")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")

    # Create source actions based on dataset type, fix this in later?
    if dataset_type == 'bc':
        # BC: source actions = target actions
        actions_src = actions_target
        print(f"\n[INFO] BC dataset: using real trajectory actions")

    elif dataset_type == 'action_translation':
        # AT: predict source actions using ID model
        print(f"\n[INFO] Action Translation dataset: predicting source actions with ID model")

        # Load ID model
        id_model, id_checkpoint = load_id_model(source_id_model, source_id_config)

        # Predict source actions
        actions_src = predict_actions_with_id_model(
            id_model, id_checkpoint, states, next_states
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'bc' or 'action_translation'")

    # Print statistics
    print(f"\n[INFO] Dataset statistics:")
    print(f"  Total transitions: {len(states):,}")
    print(f"  State shape: {states.shape}")
    print(f"  Actions (source) shape: {actions_src.shape}")
    print(f"  Actions (target) shape: {actions_target.shape}")
    print(f"\n  Source actions - mean: {actions_src.mean():.4f}, std: {actions_src.std():.4f}")
    print(f"  Target actions - mean: {actions_target.mean():.4f}, std: {actions_target.std():.4f}")

    if dataset_type == 'action_translation':
        action_diff = actions_target - actions_src
        print(f"  Action difference - mean: {action_diff.mean():.4f}, std: {action_diff.std():.4f}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save dataset
    print(f"\n[INFO] Saving dataset to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Create data group
        data_group = f.create_group('data')
        data_group.create_dataset('states', data=states.astype(np.float32), compression='gzip')
        data_group.create_dataset('next_states', data=next_states.astype(np.float32), compression='gzip')
        data_group.create_dataset('actions_src', data=actions_src.astype(np.float32), compression='gzip')
        data_group.create_dataset('actions_target', data=actions_target.astype(np.float32), compression='gzip')

        # Create metadata
        meta_group = f.create_group('meta')
        meta_group.attrs['num_samples'] = len(states)
        meta_group.attrs['state_dim'] = state_dim
        meta_group.attrs['action_dim'] = action_dim
        meta_group.attrs['dataset_type'] = dataset_type
        meta_group.attrs['created_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_group.attrs['source_trajectory_file'] = trajectory_path

        if min_reward is not None:
            meta_group.attrs['min_reward_threshold'] = min_reward

        if dataset_type == 'action_translation':
            meta_group.attrs['source_id_model'] = source_id_model
            meta_group.attrs['source_id_config'] = source_id_config

    print(f"\n[INFO] Dataset saved successfully!")
    print("=" * 80)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create BC or Action Translation dataset from trajectory data"
    )

    # Required arguments
    parser.add_argument(
        "--trajectory_file",
        type=str,
        required=True,
        help="Path to trajectory HDF5 file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save dataset"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=['bc', 'action_translation'],
        help="Type of dataset to create"
    )

    # Action translation specific
    parser.add_argument(
        "--source_id_model",
        type=str,
        default=None,
        help="Path to source domain ID model checkpoint (required for action_translation)"
    )
    parser.add_argument(
        "--source_id_config",
        type=str,
        default=None,
        help="Path to source domain ID model config (required for action_translation)"
    )

    # Filtering options
    parser.add_argument(
        "--max_demos",
        type=int,
        default=None,
        help="Maximum number of demos to use (default: all)"
    )
    parser.add_argument(
        "--min_reward",
        type=float,
        default=None,
        help="Minimum reward threshold for filtering demos (default: no filtering)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (default: all)"
    )

    args = parser.parse_args()

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.dataset_type}_dataset_{timestamp}.hdf5"
    output_path = os.path.join(args.output_dir, output_filename)

    create_dataset_from_trajectories(
        trajectory_path=args.trajectory_file,
        output_path=output_path,
        dataset_type=args.dataset_type,
        source_id_model=args.source_id_model,
        source_id_config=args.source_id_config,
        max_demos=args.max_demos,
        min_reward=args.min_reward,
        max_samples=args.max_samples,
        use_90d_states=True  # Always use proper state extraction (actually 134D)
    )
