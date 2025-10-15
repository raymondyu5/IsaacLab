"""Script to train inverse dynamics model from trajectory data.

Example usage:
    # Train with config file
    ./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
        --config configs/inverse_dynamics/training_config.yaml \
        --dataset_path trajectory_data/trajectories_Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0_20251012_221227.hdf5

    # Train with custom output directory
    ./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
        --config configs/inverse_dynamics/training_config.yaml \
        --dataset_path trajectory_data/trajectories_Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0_20251012_221227.hdf5 \
        --output_dir trained_models/
"""

import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import datetime
import inspect

# Import utility functions
from utils.data_utils import load_trajectory_dataset


def train_inverse_dynamics(states, actions, next_states, model,
                          train_config, model_output_path=None):
    """Train the Inverse Dynamics model.

    Args:
        states: Training state observations
        actions: Training actions
        next_states: Training next state observations
        model: The inverse dynamics model to train
        train_config: Dictionary containing all training hyperparameters
        model_output_path: Path to save the best model checkpoint
    """
    # Extract config values with defaults
    num_epochs = train_config.get('num_epochs', 100)
    learning_rate = train_config.get('learning_rate', 1e-3)
    batch_size = train_config.get('batch_size', 64)
    device = train_config.get('device', 'cpu')
    val_split = train_config.get('val_split', 0.2)
    save_every_n_epochs = train_config.get('save_every_n_epochs', 1)
    weight_decay = train_config.get('weight_decay', 1e-4)
    grad_clip = train_config.get('grad_clip', 1.0)
    lr_scheduler = train_config.get('lr_scheduler', 'cosine')
    warmup_epochs = train_config.get('warmup_epochs', 10)
    early_stopping_patience = train_config.get('early_stopping_patience', 50)
    normalize_data = train_config.get('normalize_data', True)
    use_mixed_precision = train_config.get('use_mixed_precision', True)
    print("=" * 80)
    print("TRAINING INVERSE DYNAMICS MODEL")
    print("=" * 80)

    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    next_states_tensor = torch.FloatTensor(next_states)

    # Add action dimension if not present
    if len(actions_tensor.shape) == 1:
        actions_tensor = actions_tensor.unsqueeze(1)

    # Normalize data for stable training
    state_mean, state_std = None, None
    action_mean, action_std = None, None

    if normalize_data:
        print("\n" + "-" * 80)
        print("DATA NORMALIZATION")
        print("-" * 80)

        # Normalize states
        state_mean = states_tensor.mean(dim=0, keepdim=True)
        state_std = states_tensor.std(dim=0, keepdim=True) + 1e-8
        states_tensor = (states_tensor - state_mean) / state_std
        next_states_tensor = (next_states_tensor - state_mean) / state_std

        # Normalize actions
        action_mean = actions_tensor.mean(dim=0, keepdim=True)
        action_std = actions_tensor.std(dim=0, keepdim=True) + 1e-8
        actions_tensor = (actions_tensor - action_mean) / action_std

        print(f"State mean: min={state_mean.min().item():.4f}, max={state_mean.max().item():.4f}")
        print(f"State std: min={state_std.min().item():.4f}, max={state_std.max().item():.4f}")
        print(f"Action mean: min={action_mean.min().item():.4f}, max={action_mean.max().item():.4f}")
        print(f"Action std: min={action_std.min().item():.4f}, max={action_std.max().item():.4f}")
        print("-" * 80 + "\n")

    # Create dataset and split into train/val
    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders with num_workers for faster loading
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True if device == 'cuda' else False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True if device == 'cuda' else False)

    print(f"\nDataset Split:")
    print(f"  Train samples: {train_size:,}")
    print(f"  Validation samples: {val_size:,}")

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

    scheduler = None
    if lr_scheduler == 'cosine':
        # Cosine annealing with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=learning_rate * 0.01
        )
    elif lr_scheduler == 'plateau':
        # Reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    elif lr_scheduler == 'step':
        # Step decay
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

        for batch_states, batch_actions, batch_next_states in train_pbar:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_next_states = batch_next_states.to(device)

            optimizer.zero_grad()

            # M=xed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # loss for inverse dynamics p(a | s, s')
                    loss = model(obs=batch_states,
                                next_obs=batch_next_states,
                                action=batch_actions)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(obs=batch_states,
                            next_obs=batch_next_states,
                            action=batch_actions)

                # Backward pass with gradient scaling
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

        # Val
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}",
                       position=1, leave=False, ncols=100)

        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_pbar:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_states = batch_next_states.to(device)

                # Forward pass
                preds = model.predict(batch_states, batch_next_states)
                loss = torch.nn.functional.mse_loss(torch.as_tensor(preds).to(device), batch_actions)

                epoch_val_loss += loss.item()
                num_val_batches += 1

                val_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_val_loss/num_val_batches:.6f}'
                })

        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # best validation loss and early stopping
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
                'action_mean': action_mean,
                'action_std': action_std,
                'normalize_data': normalize_data
            }
            torch.save(checkpoint, model_output_path)
            print(f"\nNew best validation loss: {avg_val_loss:.6f}. Checkpoint saved to {model_output_path}")
            epochs_since_save = 0

        # warmup phase
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

        # send logs to wandb
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
    parser = argparse.ArgumentParser(description='Train inverse dynamics model from trajectory dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file containing model and training hyperparameters')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to trajectory dataset HDF5 file (from collect_trajectory_data.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save trained model (default: same directory as dataset)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                       help='Wandb logging mode (default: online)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    data_config = config.get('data', {})

    model_name = model_config.get('name', 'inverse_dynamics_model')
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = os.path.join(output_dir, f'{model_name}_{timestamp}.pth')

    print("\n" + "=" * 80)
    print("INVERSE DYNAMICS TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output path: {model_output_path}")
    print(f"Device: {train_config.get('device', 'cuda')}")
    print("=" * 80 + "\n")

    # Load dataset
    states, actions, next_states = load_trajectory_dataset(
        args.dataset_path,
        filter_success_only=data_config.get('filter_success_only', False),
        min_reward_threshold=data_config.get('min_reward_threshold', None)
    )

    # Build model from config
    print(f"Building inverse dynamics model from config...\n")
    target_path = model_config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    model_params = {k: v for k, v in model_config.items() if k in init_params}
    model = model_class(**model_params)

    wandb_config = {
        "dataset_path": args.dataset_path,
        "config_path": args.config,
    }
    wandb_config.update(config)

    # Extract task name from dataset path for tags (remove later)
    dataset_basename = os.path.basename(args.dataset_path)
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

    model, train_losses, val_losses = train_inverse_dynamics(
        states, actions, next_states, model,
        train_config=train_config,
        model_output_path=model_output_path
    )

    wandb.finish()

    print(f"Best model saved to: {model_output_path}")
    print("Training completed successfully!\n")


if __name__ == "__main__":
    main()
