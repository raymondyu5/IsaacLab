"""Script to train action translator from action translation dataset.

This script trains a model to translate actions from a source domain (e.g., normal)
to a target domain (e.g., slippery) given the current state.

Example usage:
    ./isaaclab.sh -p scripts/action_translation/train_action_translator.py \
        --dataset trajectory_data/action_translation/normal_to_slippery.hdf5 \
        --model_config configs/action_translator/kuka_allegro_mlp.yaml \
        --output_dir trained_models/action_translator/normal_to_slippery
"""

import argparse
import os
import sys
import yaml
import torch
import datetime
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lib.training import (
    TrainConfig,
    load_action_translation_dataset,
    build_model_from_config,
    normalize_tensors,
    create_dataloaders,
    log_model_architecture,
    generic_training_loop
)


def main():
    parser = argparse.ArgumentParser(description='Train action translator model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to action translation dataset HDF5 file')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save trained model')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                       help='Wandb logging mode (default: online)')

    args = parser.parse_args()

    # Load model config
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config file not found: {args.model_config}")

    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config_dict = config.get('training', {})

    # Create training config and override with CLI args
    train_config = TrainConfig().update_from_dict(train_config_dict)
    train_config.val_split = args.val_split
    if args.num_epochs is not None:
        train_config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate

    # Setup output path
    model_name = model_config.get('name', 'action_translator')
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(args.output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("ACTION TRANSLATOR TRAINING")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model config: {args.model_config}")
    print(f"Output path: {model_output_path}")
    print(f"Device: {train_config.device}")
    print(f"Validation split: {train_config.val_split}")
    print("=" * 80 + "\n")

    # Load dataset
    states, next_states, actions_src, actions_target, obs_dim, action_dim = load_action_translation_dataset(args.dataset)

    # Build model
    print(f"Building action translator model from config...\n")
    model = build_model_from_config(model_config, obs_dim, action_dim)

    # Prepare tensors (we use states and target actions for translator: state -> action_target)
    all_tensors = (
        torch.FloatTensor(states),
        torch.FloatTensor(actions_target)
    )

    # Normalize data
    if train_config.normalize_data:
        all_tensors, normalization_stats = normalize_tensors(all_tensors, compute_stats_from=all_tensors)
    else:
        normalization_stats = None

    # Split into train/val
    total_samples = all_tensors[0].shape[0]
    val_size = int(total_samples * train_config.val_split)
    train_size = total_samples - val_size

    # Create random split
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_tensors = tuple(t[train_indices] for t in all_tensors)
    val_tensors = tuple(t[val_indices] for t in all_tensors)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_tensors, val_tensors, train_config)

    # Log model architecture
    log_model_architecture(model)

    # Setup wandb
    wandb_config = {
        "dataset_path": args.dataset,
        "model_config_path": args.model_config,
        "val_split": train_config.val_split,
    }
    wandb_config.update(config)

    dataset_basename = os.path.basename(args.dataset)
    wandb_tags = [model_name, 'action_translation']
    if 'normal_to_slippery' in dataset_basename:
        wandb_tags.extend(['normal', 'slippery'])

    wandb.init(
        project="action_translation_training",
        name=f"{model_name}_{timestamp}",
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags
    )

    # Define model forward function for training
    def model_forward(model, batch):
        states, actions_target = batch
        return model(obs=states, action=actions_target)

    # Define model predict function for validation
    def model_predict(model, batch):
        states, actions_target = batch
        preds = model.predict(states)
        return torch.as_tensor(preds).to(actions_target.device), actions_target

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
