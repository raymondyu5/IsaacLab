"""Train inverse dynamics model from Zarr trajectory data.

Supports both low-dimensional observations and pointcloud observations.

Example usage (low-dim only):
    ./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
        --config scripts/inverse_dynamics/configs/tomato_soup_can.yaml \
        --output_dir logs/inverse_dynamics/tomato

Example usage (with pointclouds):
    ./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
        --config scripts/inverse_dynamics/configs/tomato_soup_can_pcd.yaml \
        --output_dir logs/inverse_dynamics/tomato_pcd
"""

import argparse
import os
import sys
import yaml
import torch
import datetime
import wandb
from torch.utils.data import DataLoader

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ISAACLAB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)

from scripts.inverse_dynamics.lib.zarr_dataset import ZarrInverseDynamicsDataset
from scripts.inverse_dynamics.lib.training import (
    TrainConfig,
    build_model_from_config,
    log_model_architecture,
    generic_training_loop
)


def main():
    parser = argparse.ArgumentParser(description='Train inverse dynamics model from Zarr data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_project', type=str, default='inverse_dynamics')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    train_config_dict = config['training']
    data_config = config['data']

    train_config = TrainConfig().update_from_dict(train_config_dict)

    # Setup output
    model_name = model_config.get('name', 'id_model')
    output_dir = args.output_dir or os.path.join(ISAACLAB_ROOT, 'logs', 'inverse_dynamics', model_name)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("INVERSE DYNAMICS TRAINING - ZARR (SIMPLE)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data: {data_config['data_path']}")
    print(f"Output: {model_output_path}")
    print("=" * 80 + "\n")

    # Create datasets (auto-detects pointcloud vs low-dim based on image_keys)
    print("Creating datasets...")
    train_dataset = ZarrInverseDynamicsDataset(
        data_path=data_config['data_path'],
        obs_keys=data_config['obs_keys'],
        image_keys=data_config.get('image_keys'),  # None for low-dim only
        action_key=data_config.get('action_key', 'actions'),
        num_episodes=data_config.get('num_episodes', 1000),
        val_ratio=data_config.get('val_ratio', 0.1),
        train=True,
        seed=data_config.get('seed', 42),
        downsample_points=data_config.get('downsample_points', 2048),
        pcd_noise=data_config.get('pcd_noise', 0.02),
        noise_extrinsic=data_config.get('noise_extrinsic', True),
        noise_extrinsic_parameter=data_config.get('noise_extrinsic_parameter', [0.05, 0.2]),
        noise_keys=data_config.get('noise_keys'),
        noise_scale=data_config.get('noise_scale'),
    )

    val_dataset = ZarrInverseDynamicsDataset(
        data_path=data_config['data_path'],
        obs_keys=data_config['obs_keys'],
        image_keys=data_config.get('image_keys'),
        action_key=data_config.get('action_key', 'actions'),
        num_episodes=data_config.get('num_episodes', 1000),
        val_ratio=data_config.get('val_ratio', 0.1),
        train=False,
        seed=data_config.get('seed', 42),
        downsample_points=data_config.get('downsample_points', 2048),
        pcd_noise=data_config.get('pcd_noise', 0.02),
        noise_extrinsic=data_config.get('noise_extrinsic', True),
        noise_extrinsic_parameter=data_config.get('noise_extrinsic_parameter', [0.05, 0.2]),
        noise_keys=data_config.get('noise_keys'),
        noise_scale=data_config.get('noise_scale'),
    )

    # Check if using pointclouds
    use_pointcloud = train_dataset.use_pointcloud

    # Build model
    print("Building model...")
    if use_pointcloud:
        obs_dim = train_dataset.low_dim_obs_dim
    else:
        obs_dim = train_dataset.obs_dim
    action_dim = train_dataset.action_dim

    model = build_model_from_config(model_config, obs_dim, action_dim)
    log_model_architecture(model)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )

    # Normalization
    normalization_stats = None
    if train_config.normalize_data:
        print("Computing normalization stats...")
        stats = train_dataset.get_stats()
        normalization_stats = {
            'state_mean': torch.FloatTensor(stats['state'][0]),
            'state_std': torch.FloatTensor(stats['state'][1]),
            'action_mean': torch.FloatTensor(stats['action'][0]),
            'action_std': torch.FloatTensor(stats['action'][1]),
        }

    # Setup wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{model_name}_{timestamp}",
        mode=args.wandb_mode,
        config=config,
    )

    # Training functions (handle both tensor and dict formats)
    def model_forward(model, batch):
        states = batch['state']
        next_states = batch['next_state']
        actions = batch['action']

        if use_pointcloud:
            # Pointcloud format: state is dict with 'low_dim' and 'seg_pc'
            if train_config.normalize_data:
                state_mean = normalization_stats['state_mean'].to(states['low_dim'].device)
                state_std = normalization_stats['state_std'].to(states['low_dim'].device)
                action_mean = normalization_stats['action_mean'].to(actions.device)
                action_std = normalization_stats['action_std'].to(actions.device)

                # Normalize only low-dim observations (pointclouds are already in metric space)
                states = {
                    'low_dim': (states['low_dim'] - state_mean) / state_std,
                    'seg_pc': states['seg_pc']  # No normalization for PCD
                }
                next_states = {
                    'low_dim': (next_states['low_dim'] - state_mean) / state_std,
                    'seg_pc': next_states['seg_pc']
                }
                actions = (actions - action_mean) / action_std

            return model(obs=states, next_obs=next_states, action=actions)
        else:
            # Low-dim format: state is tensor
            if train_config.normalize_data:
                state_mean = normalization_stats['state_mean'].to(states.device)
                state_std = normalization_stats['state_std'].to(states.device)
                action_mean = normalization_stats['action_mean'].to(states.device)
                action_std = normalization_stats['action_std'].to(states.device)

                states = (states - state_mean) / state_std
                next_states = (next_states - state_mean) / state_std
                actions = (actions - action_mean) / action_std

            return model(obs=states, next_obs=next_states, action=actions)

    def model_predict(model, batch):
        states = batch['state']
        next_states = batch['next_state']
        actions = batch['action']

        if use_pointcloud:
            # Pointcloud format
            if train_config.normalize_data:
                state_mean = normalization_stats['state_mean'].to(states['low_dim'].device)
                state_std = normalization_stats['state_std'].to(states['low_dim'].device)
                action_mean = normalization_stats['action_mean'].to(actions.device)
                action_std = normalization_stats['action_std'].to(actions.device)

                states = {
                    'low_dim': (states['low_dim'] - state_mean) / state_std,
                    'seg_pc': states['seg_pc']
                }
                next_states = {
                    'low_dim': (next_states['low_dim'] - state_mean) / state_std,
                    'seg_pc': next_states['seg_pc']
                }
                actions_normalized = (actions - action_mean) / action_std
            else:
                actions_normalized = actions

            preds = model.predict(states, next_states)
            return torch.as_tensor(preds).to(actions_normalized.device), actions_normalized
        else:
            # Low-dim format
            if train_config.normalize_data:
                state_mean = normalization_stats['state_mean'].to(states.device)
                state_std = normalization_stats['state_std'].to(states.device)
                action_mean = normalization_stats['action_mean'].to(states.device)
                action_std = normalization_stats['action_std'].to(states.device)

                states = (states - state_mean) / state_std
                next_states = (next_states - state_mean) / state_std
                actions_normalized = (actions - action_mean) / action_std
            else:
                actions_normalized = actions

            preds = model.predict(states, next_states)
            return torch.as_tensor(preds).to(actions_normalized.device), actions_normalized

    # Train
    model = generic_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_forward_fn=model_forward,
        model_predict_fn=model_predict,
        output_path=model_output_path,
        normalization_stats=normalization_stats,
        wandb_enabled=True
    )

    wandb.finish()
    print(f"\nModel saved to: {model_output_path}\n")


if __name__ == "__main__":
    main()
