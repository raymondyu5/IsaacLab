"""Script to create action translation dataset by relabeling transitions with inverse dynamics models.

This script takes trajectory data from a source domain and relabels the transitions using
both source and target domain inverse dynamics models to create paired action data for
training an action translator.

Example usage:
    # Create action translation dataset from normal to slippery domain
    ./isaaclab.sh -p scripts/inverse_dynamics/create_action_translation_dataset.py \
        --source_dataset trajectory_data/full_pipeline/normal/trajectories_Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0_20251020_124234.hdf5 \
        --source_model trained_models/id_normal/kuka_allegro_mlp_id_20251020_172505.pth \
        --target_model trained_models/id_slippery/kuka_allegro_mlp_id_20251020_173915.pth \
        --model_config configs/inverse_dynamics/kuka_allegro_train_mlp.yaml \
        --output_dir trajectory_data/action_translation/normal_to_slippery
"""

import argparse
import os
import yaml
import torch
import h5py
import numpy as np
from tqdm import tqdm
import datetime
import sys

# Import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.inverse_dynamics.utils.data_utils import load_trajectory_dataset


def load_inverse_dynamics_model(model_path, model_config_path, device='cuda'):
    """
    Load inverse dynamics model from checkpoint with normalization statistics.

    Args:
        model_path: Path to the .pth checkpoint file
        model_config_path: Path to the model config YAML file
        device: Device to load the model on

    Returns:
        model: Loaded inverse dynamics model
        checkpoint: Full checkpoint dict containing normalization stats
    """
    print(f"Loading inverse dynamics model from {model_path}")

    # Load config to get model architecture
    with open(model_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})

    # Build model from config
    import inspect
    target_path = model_config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Get only the parameters that the model __init__ accepts
    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    model_params = {k: v for k, v in model_config.items() if k in init_params}

    # Override device
    model_params['device'] = device

    # Create model
    model = model_class(**model_params)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Normalization enabled: {checkpoint.get('normalize_data', False)}")

    return model, checkpoint


def predict_with_normalization(model, checkpoint, states, next_states, device='cuda'):
    """
    Predict actions using inverse dynamics model with proper normalization.

    Args:
        model: Inverse dynamics model
        checkpoint: Checkpoint dict containing normalization stats
        states: State observations (numpy array)
        next_states: Next state observations (numpy array)
        device: Device to run predictions on

    Returns:
        actions: Predicted actions (numpy array)
    """
    normalize_data = checkpoint.get('normalize_data', False)

    # Convert to tensors
    states_tensor = torch.FloatTensor(states).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)

    # Apply normalization if enabled
    if normalize_data:
        state_mean = checkpoint['state_mean'].to(device) if checkpoint['state_mean'] is not None else None
        state_std = checkpoint['state_std'].to(device) if checkpoint['state_std'] is not None else None
        action_mean = checkpoint['action_mean'].to(device) if checkpoint['action_mean'] is not None else None
        action_std = checkpoint['action_std'].to(device) if checkpoint['action_std'] is not None else None

        if state_mean is not None and state_std is not None:
            states_tensor = (states_tensor - state_mean) / state_std
            next_states_tensor = (next_states_tensor - state_mean) / state_std
    else:
        action_mean = None
        action_std = None

    # Predict actions
    with torch.no_grad():
        actions_tensor = model.predict(states_tensor, next_states_tensor)

    # Convert to numpy
    if isinstance(actions_tensor, torch.Tensor):
        actions = actions_tensor.cpu().numpy()
    else:
        actions = actions_tensor

    # Denormalize actions if needed
    if normalize_data and action_mean is not None and action_std is not None:
        action_mean_np = action_mean.cpu().numpy()
        action_std_np = action_std.cpu().numpy()
        actions = actions * action_std_np + action_mean_np

    return actions


def create_action_translation_dataset(states, next_states, actions_src, actions_target, output_path):
    """
    Create and save action translation dataset in HDF5 format.

    Args:
        states: State observations (N, obs_dim)
        next_states: Next state observations (N, obs_dim)
        actions_src: Source domain actions (N, action_dim)
        actions_target: Target domain actions (N, action_dim)
        output_path: Path to save the dataset
    """
    print("\n" + "=" * 80)
    print("CREATING ACTION TRANSLATION DATASET")
    print("=" * 80)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create data group
        data_group = f.create_group('data')

        # Save data
        data_group.create_dataset('states', data=states.astype(np.float32), compression='gzip')
        data_group.create_dataset('next_states', data=next_states.astype(np.float32), compression='gzip')
        data_group.create_dataset('actions_src', data=actions_src.astype(np.float32), compression='gzip')
        data_group.create_dataset('actions_target', data=actions_target.astype(np.float32), compression='gzip')

        # Create meta group with dataset info
        meta_group = f.create_group('meta')
        meta_group.attrs['num_samples'] = len(states)
        meta_group.attrs['obs_dim'] = states.shape[1]
        meta_group.attrs['action_dim'] = actions_src.shape[1]
        meta_group.attrs['created_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Saved action translation dataset to {output_path}")
    print(f"  Number of samples: {len(states):,}")
    print(f"  State shape: {states.shape}")
    print(f"  Next state shape: {next_states.shape}")
    print(f"  Actions (source) shape: {actions_src.shape}")
    print(f"  Actions (target) shape: {actions_target.shape}")
    print("=" * 80 + "\n")

    return output_path


def relabel_transitions_with_inverse_dynamics(
    states, next_states,
    source_model, source_checkpoint,
    target_model, target_checkpoint,
    device='cuda',
    batch_size=1024,
    max_samples=None
):
    """
    Relabel transitions using source and target inverse dynamics models.

    Args:
        states: State observations from dataset
        next_states: Next state observations from dataset
        source_model: Source domain inverse dynamics model
        source_checkpoint: Source model checkpoint with normalization stats
        target_model: Target domain inverse dynamics model
        target_checkpoint: Target model checkpoint with normalization stats
        device: Device to run predictions on
        batch_size: Batch size for predictions
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        states, next_states, actions_src, actions_target as numpy arrays
    """
    print("\n" + "=" * 80)
    print("RELABELING TRANSITIONS WITH INVERSE DYNAMICS")
    print("=" * 80)

    # Limit samples if requested
    if max_samples is not None:
        num_samples = min(max_samples, len(states))
        states = states[:num_samples]
        next_states = next_states[:num_samples]
    else:
        num_samples = len(states)

    print(f"Processing {num_samples:,} transitions")
    print(f"Batch size: {batch_size}")

    actions_src = []
    actions_target = []

    # Process in batches with progress bar
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Relabeling batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_states = states[start_idx:end_idx]
        batch_next_states = next_states[start_idx:end_idx]

        # Predict actions with source model
        batch_actions_src = predict_with_normalization(
            source_model, source_checkpoint, batch_states, batch_next_states, device
        )

        # Predict actions with target model
        batch_actions_target = predict_with_normalization(
            target_model, target_checkpoint, batch_states, batch_next_states, device
        )

        actions_src.append(batch_actions_src)
        actions_target.append(batch_actions_target)

    # Concatenate all batches
    actions_src = np.concatenate(actions_src, axis=0)
    actions_target = np.concatenate(actions_target, axis=0)

    print(f"\nRelabeling complete!")
    print(f"  Processed {num_samples:,} transitions")
    print(f"  Source actions shape: {actions_src.shape}")
    print(f"  Target actions shape: {actions_target.shape}")

    # Print statistics
    print(f"\nAction Statistics:")
    print(f"  Source actions - mean: {np.mean(actions_src):.4f}, std: {np.std(actions_src):.4f}")
    print(f"  Target actions - mean: {np.mean(actions_target):.4f}, std: {np.std(actions_target):.4f}")
    print(f"  Action difference - mean: {np.mean(actions_target - actions_src):.4f}, std: {np.std(actions_target - actions_src):.4f}")
    print("=" * 80 + "\n")

    return states, next_states, actions_src, actions_target


def main():
    parser = argparse.ArgumentParser(
        description='Create action translation dataset by relabeling transitions with inverse dynamics models'
    )
    parser.add_argument('--source_dataset', type=str, required=True,
                       help='Path to source domain trajectory dataset HDF5 file')
    parser.add_argument('--source_model', type=str, required=True,
                       help='Path to source domain inverse dynamics model checkpoint (.pth)')
    parser.add_argument('--target_model', type=str, required=True,
                       help='Path to target domain inverse dynamics model checkpoint (.pth)')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML file (used for both models)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save the action translation dataset (default: same as source dataset)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for inverse dynamics predictions (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--filter_success_only', action='store_true',
                       help='Only use successful episodes from the dataset')
    parser.add_argument('--min_reward_threshold', type=float, default=None,
                       help='Minimum episode reward threshold (default: None)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.source_dataset):
        raise FileNotFoundError(f"Source dataset not found: {args.source_dataset}")
    if not os.path.exists(args.source_model):
        raise FileNotFoundError(f"Source model not found: {args.source_model}")
    if not os.path.exists(args.target_model):
        raise FileNotFoundError(f"Target model not found: {args.target_model}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")

    # Create output path
    if args.output_dir is None:
        output_dir = os.path.dirname(args.source_dataset)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'action_translation_dataset_{timestamp}.hdf5')

    print("\n" + "=" * 80)
    print("ACTION TRANSLATION DATASET CREATION")
    print("=" * 80)
    print(f"Source dataset: {args.source_dataset}")
    print(f"Source model: {args.source_model}")
    print(f"Target model: {args.target_model}")
    print(f"Model config: {args.model_config}")
    print(f"Output path: {output_path}")
    print(f"Device: {args.device}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,}")
    print("=" * 80 + "\n")

    # Load source dataset
    print("Loading source dataset...")
    states, _, next_states = load_trajectory_dataset(
        args.source_dataset,
        filter_success_only=args.filter_success_only,
        min_reward_threshold=args.min_reward_threshold
    )

    # Load inverse dynamics models
    print("\nLoading source domain inverse dynamics model...")
    source_model, source_checkpoint = load_inverse_dynamics_model(
        args.source_model, args.model_config, args.device
    )

    print("\nLoading target domain inverse dynamics model...")
    target_model, target_checkpoint = load_inverse_dynamics_model(
        args.target_model, args.model_config, args.device
    )

    # Relabel transitions
    states, next_states, actions_src, actions_target = relabel_transitions_with_inverse_dynamics(
        states, next_states,
        source_model, source_checkpoint,
        target_model, target_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    # Create and save action translation dataset
    create_action_translation_dataset(
        states, next_states, actions_src, actions_target, output_path
    )

    print("Action translation dataset creation completed successfully!")
    print(f"Dataset saved to: {output_path}\n")


if __name__ == "__main__":
    main()
