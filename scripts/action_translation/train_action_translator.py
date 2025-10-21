"""Script to train action translator from action translation dataset.

This script trains a model to translate actions from a source domain (e.g., normal)
to a target domain (e.g., slippery) given the current state.

Example usage:
    ./isaaclab.sh -p scripts/action_translation/train_action_translator.py \
        --dataset trajectory_data/action_translation/normal_to_slippery/action_translation_dataset_20251020_205908.hdf5 \
        --model_config configs/action_translator/kuka_allegro_mlp.yaml \
        --output_dir trained_models/action_translator/normal_to_slippery \
        --num_epochs 500 \
        --batch_size 256 \
        --learning_rate 1e-3
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import h5py
import numpy as np
from tqdm import tqdm
import wandb
import datetime
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('/home/raymond/projects/generative-policies')


def load_action_translation_dataset(dataset_path):
    """Load action translation dataset from HDF5 file."""
    print("=" * 80)
    print("LOADING ACTION TRANSLATION DATASET")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}\n")

    with h5py.File(dataset_path, 'r') as f:
        states = f['data']['states'][:]
        next_states = f['data']['next_states'][:]
        actions_src = f['data']['actions_src'][:]
        actions_target = f['data']['actions_target'][:]

        # Load metadata
        num_samples = f['meta'].attrs['num_samples']
        obs_dim = f['meta'].attrs['obs_dim']
        action_dim = f['meta'].attrs['action_dim']

    print(f"Dataset loaded successfully!")
    print(f"  Number of samples: {num_samples:,}")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  States shape: {states.shape}")
    print(f"  Actions (source) shape: {actions_src.shape}")
    print(f"  Actions (target) shape: {actions_target.shape}")
    print("=" * 80 + "\n")

    return states, next_states, actions_src, actions_target, obs_dim, action_dim


def build_action_translator_from_config(config, obs_dim, action_dim, load_checkpoint=False):
    """Build action translator model from config."""
    target_path = config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Get init parameters
    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')

    # Build model parameters
    model_params = {k: v for k, v in config.items() if k in init_params}
    model_params['obs_dim'] = obs_dim
    model_params['action_dim'] = action_dim

    # Create model
    model = model_class(**model_params)

    # Load checkpoint if requested
    if load_checkpoint and 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint_path = config['checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))

    return model


def train_action_translator(
    states, actions_src, actions_target,
    model, train_config, model_output_path=None
):
    """
    Train the Action Translator model.

    Args:
        states: State observations
        actions_src: Source domain actions (action priors)
        actions_target: Target domain actions (ground truth)
        model: The action translator model to train
        train_config: Dictionary containing all training hyperparameters
        model_output_path: Path to save the best model checkpoint
    """
    # Extract config values with defaults
    num_epochs = train_config.get('num_epochs', 100)
    learning_rate = train_config.get('learning_rate', 1e-3)
    batch_size = train_config.get('batch_size', 64)
    device = train_config.get('device', 'cuda')
    val_split = train_config.get('val_split', 0.2)
    save_every_n_epochs = train_config.get('save_every_n_epochs', 10)
    weight_decay = train_config.get('weight_decay', 1e-4)
    grad_clip = train_config.get('grad_clip', 1.0)
    lr_scheduler = train_config.get('lr_scheduler', 'cosine')
    warmup_epochs = train_config.get('warmup_epochs', 10)
    early_stopping_patience = train_config.get('early_stopping_patience', 50)
    normalize_data = train_config.get('normalize_data', True)
    use_mixed_precision = train_config.get('use_mixed_precision', True)

    print("=" * 80)
    print("TRAINING ACTION TRANSLATOR")
    print("=" * 80)

    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_src_tensor = torch.FloatTensor(actions_src)
    actions_target_tensor = torch.FloatTensor(actions_target)

    # Add action dimension if not present
    if len(actions_src_tensor.shape) == 1:
        actions_src_tensor = actions_src_tensor.unsqueeze(1)
    if len(actions_target_tensor.shape) == 1:
        actions_target_tensor = actions_target_tensor.unsqueeze(1)

    # Normalize data for stable training
    state_mean, state_std = None, None
    action_src_mean, action_src_std = None, None
    action_target_mean, action_target_std = None, None

    if normalize_data:
        print("\n" + "-" * 80)
        print("DATA NORMALIZATION")
        print("-" * 80)

        # Normalize states
        state_mean = states_tensor.mean(dim=0, keepdim=True)
        state_std = states_tensor.std(dim=0, keepdim=True) + 1e-8
        states_tensor = (states_tensor - state_mean) / state_std

        # Normalize source actions
        action_src_mean = actions_src_tensor.mean(dim=0, keepdim=True)
        action_src_std = actions_src_tensor.std(dim=0, keepdim=True) + 1e-8
        actions_src_tensor = (actions_src_tensor - action_src_mean) / action_src_std

        # Normalize target actions
        action_target_mean = actions_target_tensor.mean(dim=0, keepdim=True)
        action_target_std = actions_target_tensor.std(dim=0, keepdim=True) + 1e-8
        actions_target_tensor = (actions_target_tensor - action_target_mean) / action_target_std

        print(f"State mean: min={state_mean.min().item():.4f}, max={state_mean.max().item():.4f}")
        print(f"State std: min={state_std.min().item():.4f}, max={state_std.max().item():.4f}")
        print(f"Action (src) mean: min={action_src_mean.min().item():.4f}, max={action_src_mean.max().item():.4f}")
        print(f"Action (src) std: min={action_src_std.min().item():.4f}, max={action_src_std.max().item():.4f}")
        print(f"Action (target) mean: min={action_target_mean.min().item():.4f}, max={action_target_mean.max().item():.4f}")
        print(f"Action (target) std: min={action_target_std.min().item():.4f}, max={action_target_std.max().item():.4f}")
        print("-" * 80 + "\n")

    # Create dataset and split into train/val
    dataset = TensorDataset(states_tensor, actions_src_tensor, actions_target_tensor)

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders with num_workers for faster loading
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )

    print(f"\nDataset Split:")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    model.to(device)

    print(f"\n" + "-" * 80)
    print("MODEL ARCHITECTURE")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"\nLayer Breakdown:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                print(f"  {name}: {param_count:,} parameters")
    print("-" * 80 + "\n")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("-" * 80)
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"Optimizer: AdamW")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Weight decay (L2): {weight_decay}")
    print(f"Gradient clipping: {grad_clip}")
    print(f"LR Scheduler: {lr_scheduler}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Data normalization: {normalize_data}")
    print(f"Mixed precision: {use_mixed_precision and device == 'cuda'}")
    print("-" * 80 + "\n")

    # Learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=learning_rate * 0.01
        )
    elif lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    elif lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Warmup scheduler
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device == 'cuda' else None

    # Training loop
    train_losses = []
    val_losses = []

    # Checkpoint tracking
    best_val_loss = float('inf')
    epochs_since_save = 0
    epochs_without_improvement = 0

    # Create outer progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}",
                         position=1, leave=False, ncols=100)

        for batch_states, batch_actions_src, batch_actions_target in train_pbar:
            batch_states = batch_states.to(device)
            batch_actions_src = batch_actions_src.to(device)
            batch_actions_target = batch_actions_target.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # Forward pass: predict target action given state and source action
                    loss = model(
                        obs=batch_states,
                        action_prior=batch_actions_src,
                        action=batch_actions_target
                    )

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(
                    obs=batch_states,
                    action_prior=batch_actions_src,
                    action=batch_actions_target
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_train_loss/num_train_batches:.6f}'
            })

        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}",
                       position=1, leave=False, ncols=100)

        with torch.no_grad():
            for batch_states, batch_actions_src, batch_actions_target in val_pbar:
                batch_states = batch_states.to(device)
                batch_actions_src = batch_actions_src.to(device)
                batch_actions_target = batch_actions_target.to(device)

                # Forward pass
                loss = model(batch_states, batch_actions_src, batch_actions_target)

                epoch_val_loss += loss.item()
                num_val_batches += 1

                val_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_val_loss/num_val_batches:.6f}'
                })

        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # Track best validation loss and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"\n\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

        epochs_since_save += 1
        if model_output_path is not None and avg_val_loss <= best_val_loss and epochs_since_save >= save_every_n_epochs:
            # Save model state dict along with normalization statistics
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'state_mean': state_mean,
                'state_std': state_std,
                'action_src_mean': action_src_mean,
                'action_src_std': action_src_std,
                'action_target_mean': action_target_mean,
                'action_target_std': action_target_std,
                'normalize_data': normalize_data
            }
            torch.save(checkpoint, model_output_path)
            print(f"\nNew best validation loss: {avg_val_loss:.6f}. Checkpoint saved to {model_output_path}")
            epochs_since_save = 0

        # Warmup phase
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}',
            'best_val': f'{best_val_loss:.6f}',
            'lr': f'{new_lr:.2e}'
        })

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": new_lr,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement
        })

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {new_lr:.2e}")

    epoch_pbar.close()

    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80 + "\n")

    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train action translator from action translation dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to action translation dataset HDF5 file')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML file (e.g., configs/action_translator/kuka_allegro_mlp.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save trained model (default: same directory as dataset)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (overrides config)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                       help='Wandb logging mode (default: online)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")

    # Load model config
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config = config.get('training', {})

    # Override config with command line arguments
    if args.num_epochs is not None:
        train_config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        train_config['learning_rate'] = args.learning_rate
    if args.device is not None:
        train_config['device'] = args.device

    # Set defaults
    train_config.setdefault('device', 'cuda')
    train_config.setdefault('num_epochs', 500)
    train_config.setdefault('batch_size', 256)
    train_config.setdefault('learning_rate', 1e-3)

    # Set output path
    model_name = model_config.get('name', 'action_translator')
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("ACTION TRANSLATOR TRAINING")
    print("=" * 80)
    print(f"Model config: {args.model_config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output path: {model_output_path}")
    print(f"Device: {train_config['device']}")
    print("=" * 80 + "\n")

    # Load dataset
    print("Loading action translation dataset...")
    states, next_states, actions_src, actions_target, obs_dim, action_dim = load_action_translation_dataset(
        args.dataset
    )

    # Build model from config
    print(f"Building action translator from config...\n")
    model = build_action_translator_from_config(model_config, obs_dim, action_dim, load_checkpoint=False)

    # Initialize wandb
    wandb_config = {
        "dataset_path": args.dataset,
        "config_path": args.model_config,
    }
    wandb_config.update(config)

    # Extract dataset info for tags
    dataset_basename = os.path.basename(args.dataset)
    wandb_tags = [model_name, 'action_translator']
    if 'normal_to_slippery' in args.dataset:
        wandb_tags.append('normal_to_slippery')

    wandb.init(
        project="action_translator_training",
        name=f"{model_name}_{timestamp}",
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags
    )

    # Train model
    model, train_losses, val_losses = train_action_translator(
        states, actions_src, actions_target,
        model,
        train_config=train_config,
        model_output_path=model_output_path
    )

    wandb.finish()

    print(f"Best model saved to: {model_output_path}")
    print("Training completed successfully!\n")


if __name__ == "__main__":
    main()
