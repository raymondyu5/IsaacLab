"""Script to train a baseline BC model on action translation dataset.

This script trains a behavioral cloning (BC) model to predict target domain actions
directly from states, without using source domain actions. This serves as a baseline
to compare against the action translation approach.

The BC model learns: BC(s) → a_target

If the BC model performs well, it indicates that the states contain sufficient information
to predict target domain actions directly. This helps validate the inverse dynamics model
and provides a performance baseline.

Example usage:
    ./isaaclab.sh -p scripts/action_translation/train_bc_baseline.py \
        --dataset trajectory_data/action_translation/normal_to_slippery/action_translation_dataset_20251020_205908.hdf5 \
        --model_config configs/bc_baseline/flow_bc.yaml \
        --output_dir trained_models/bc_baseline
"""

import argparse
import os
import sys
import yaml
import torch
import datetime
import wandb

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('/home/raymond/projects/generative-policies')

from scripts.lib.training import (
    TrainConfig,
    load_processed_dataset,
    normalize_tensors,
    create_dataloaders,
    build_model_from_config,
    generic_training_loop
)


def main():
    parser = argparse.ArgumentParser(description='Train baseline BC model on action translation dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to action translation dataset HDF5 file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML file (e.g., configs/bc_baseline/flow_bc.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save trained model (default: same directory as dataset)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Wandb logging mode (default: online)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")

    # Load config
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config_dict = config.get('training', {})

    # Create TrainConfig from config file
    train_config = TrainConfig()
    train_config.update_from_dict(train_config_dict)

    # Setup output path
    model_name = model_config.get('name', 'bc_baseline')
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("BC BASELINE TRAINING")
    print("=" * 80)
    print(f"Model config: {args.model_config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output path: {model_output_path}")
    print(f"Device: {train_config.device}")
    print("=" * 80 + "\n")

    # Load dataset
    print("Loading processed dataset...")
    states, next_states, actions_src, actions_target, obs_dim, action_dim = load_processed_dataset(
        args.dataset
    )

    # Build model from config
    print(f"Building BC baseline model from config...\n")
    model = build_model_from_config(model_config, obs_dim, action_dim)

    # Prepare data tensors
    states_tensor = torch.FloatTensor(states)
    next_states_tensor = torch.FloatTensor(next_states)
    actions_target_tensor = torch.FloatTensor(actions_target)

    # Add dimension if needed
    if len(actions_target_tensor.shape) == 1:
        actions_target_tensor = actions_target_tensor.unsqueeze(1)

    # Normalize data
    # IMPORTANT: Only normalize observations, NOT actions
    # Actions should remain in their raw range to avoid range limitation bugs
    normalization_stats = {}
    if train_config.normalize_data:
        # Only normalize observations (states)
        tensors_to_normalize = {
            'states': states_tensor,
            'next_states': next_states_tensor,
        }

        normalized_tensors, stats = normalize_tensors(
            tensors_to_normalize,
            compute_stats_from=['states']  # Only compute stats from observations
        )

        states_tensor = normalized_tensors['states']
        next_states_tensor = normalized_tensors['next_states']

        # Save normalization stats for checkpoint
        normalization_stats = {
            'state_mean': stats['states'][0],
            'state_std': stats['states'][1],
            'action_target_mean': None,
            'action_target_std': None,
            'normalize_data': True
        }

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_tensors=[states_tensor, next_states_tensor, actions_target_tensor],
        val_tensors=None,  # Will split from train
        config=train_config
    )

    # Initialize wandb
    wandb_config = {
        "dataset_path": args.dataset,
        "config_path": args.model_config,
    }
    wandb_config.update(config)

    # Extract dataset info for tags
    wandb_tags = [model_name, 'bc_baseline', 'behavioral_cloning']
    if 'normal_to_slippery' in args.dataset:
        wandb_tags.append('normal_to_slippery')

    wandb.init(
        project="bc_baseline",
        name=f"{model_name}_{timestamp}",
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags
    )

    # Define forward functions for training
    def model_forward_fn(model, batch):
        """Forward pass for BC model during training.
        Note: FlowBC's forward method expects (obs, action_prior, action) but ignores action_prior."""
        states, next_states, actions_target = batch
        # FlowBC expects action_prior but doesn't use it - pass dummy tensor
        dummy_action_prior = torch.zeros_like(actions_target)
        return model(states, dummy_action_prior, actions_target)

    def model_predict_fn(model, batch):
        """Prediction function for BC model during validation."""
        states, next_states, actions_target = batch
        # FlowBC's predict method also expects action_prior but ignores it
        dummy_action_prior = torch.zeros_like(actions_target).cpu().numpy()
        predictions = model.predict(states.cpu().numpy(), dummy_action_prior)
        predictions = torch.tensor(predictions, device=states.device, dtype=states.dtype)
        return predictions, actions_target

    # Add dimension info to normalization stats for checkpoint
    dimension_info = {
        'obs_dim': obs_dim,
        'state_dim': obs_dim,  # Alias for compatibility
        'action_dim': action_dim
    }
    checkpoint_info = {**normalization_stats, **dimension_info}

    # Train model using generic training loop
    model, train_losses, val_losses = generic_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_forward_fn=model_forward_fn,
        model_predict_fn=model_predict_fn,
        output_path=model_output_path,
        normalization_stats=checkpoint_info,
        wandb_enabled=True
    )

    wandb.finish()

    print(f"Best model saved to: {model_output_path}")
    print("Training completed successfully!\n")


if __name__ == "__main__":
    main()