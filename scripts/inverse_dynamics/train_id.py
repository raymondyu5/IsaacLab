"""Script to train inverse dynamics model from trajectory data.

Example usage:
    ./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
        --config configs/inverse_dynamics/kuka_allegro_train_mlp.yaml \
        --train_dataset trajectory_data/train.hdf5 \
        --val_dataset trajectory_data/val.hdf5
"""

import argparse
import os
import sys
import yaml
import torch
import datetime
import wandb

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset import load_trajectory_dataset
from scripts.lib.training import (
    TrainConfig,
    build_model_from_config,
    normalize_tensors,
    create_dataloaders,
    log_model_architecture,
    generic_training_loop
)


def main():
    parser = argparse.ArgumentParser(description='Train inverse dynamics model from trajectory dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file')
    parser.add_argument('--train_dataset', type=str, required=True,
                       help='Path to training trajectory dataset HDF5 file')
    parser.add_argument('--val_dataset', type=str, required=True,
                       help='Path to validation trajectory dataset HDF5 file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save trained model (default: same as train dataset)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                       help='Wandb logging mode (default: online)')

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config_dict = config.get('training', {})
    data_config = config.get('data', {})

    # Create training config
    train_config = TrainConfig().update_from_dict(train_config_dict)

    # Setup output path
    model_name = model_config.get('name', 'inverse_dynamics_model')
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.train_dataset)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("INVERSE DYNAMICS TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset: {args.val_dataset}")
    print(f"Output path: {model_output_path}")
    print(f"Device: {train_config.device}")
    print("=" * 80 + "\n")

    # Load datasets
    print("Loading training dataset...")
    train_states, train_actions, train_next_states = load_trajectory_dataset(
        args.train_dataset,
        filter_success_only=data_config.get('filter_success_only', False),
        min_reward_threshold=data_config.get('min_reward_threshold', None)
    )

    print("Loading validation dataset...")
    val_states, val_actions, val_next_states = load_trajectory_dataset(
        args.val_dataset,
        filter_success_only=data_config.get('filter_success_only', False),
        min_reward_threshold=data_config.get('min_reward_threshold', None)
    )

    # Build model
    print(f"Building inverse dynamics model from config...\n")
    obs_dim = train_states.shape[1]
    action_dim = train_actions.shape[1] if len(train_actions.shape) > 1 else 1
    model = build_model_from_config(model_config, obs_dim, action_dim)

    # Prepare tensors as dict for normalization
    train_tensors_dict = {
        'states': torch.FloatTensor(train_states),
        'actions': torch.FloatTensor(train_actions),
        'next_states': torch.FloatTensor(train_next_states)
    }
    val_tensors_dict = {
        'states': torch.FloatTensor(val_states),
        'actions': torch.FloatTensor(val_actions),
        'next_states': torch.FloatTensor(val_next_states)
    }

    # Normalize data
    if train_config.normalize_data:
        train_tensors_dict, normalization_stats = normalize_tensors(train_tensors_dict, compute_stats_from=['states', 'actions'])
        val_tensors_dict, _ = normalize_tensors(val_tensors_dict, stats=normalization_stats)
    else:
        normalization_stats = None

    # Convert to list format for dataloaders
    train_tensors_list = [train_tensors_dict['states'], train_tensors_dict['actions'], train_tensors_dict['next_states']]
    val_tensors_list = [val_tensors_dict['states'], val_tensors_dict['actions'], val_tensors_dict['next_states']]

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_tensors_list, val_tensors_list, train_config)

    # Log model architecture
    log_model_architecture(model)

    # Setup wandb
    wandb_config = {
        "train_dataset_path": args.train_dataset,
        "val_dataset_path": args.val_dataset,
        "config_path": args.config,
    }
    wandb_config.update(config)

    dataset_basename = os.path.basename(args.train_dataset)
    wandb_tags = [model_name, 'inverse_dynamics']
    if 'Dexsuite' in dataset_basename:
        wandb_tags.append('dexsuite')
    if 'ShadowHand' in dataset_basename:
        wandb_tags.append('shadow_hand')

    wandb.init(
        project="inverse_dynamics_training",
        name=f"{model_name}_{timestamp}",
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags
    )

    # Define model forward function for training
    def model_forward(model, batch):
        states, actions, next_states = batch
        return model(obs=states, next_obs=next_states, action=actions)

    # Define model predict function for validation
    def model_predict(model, batch):
        states, actions, next_states = batch
        preds = model.predict(states, next_states)
        return torch.as_tensor(preds).to(actions.device), actions

    # Train model
    model = generic_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_forward_fn=model_forward,
        model_predict_fn=model_predict,
        output_path=model_output_path,
        normalization_stats=normalization_stats
    )

    wandb.finish()

    print(f"\nBest model saved to: {model_output_path}")
    print("Training completed successfully!\n")


if __name__ == "__main__":
    main()
