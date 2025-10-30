"""Shared training utilities for all training scripts.

This module provides common functionality used across train_bc_sanity_check.py,
train_action_translator.py, and train_id.py to eliminate code duplication.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Tuple, Optional, Any, Callable
import h5py
import numpy as np
from tqdm import tqdm
import wandb
import datetime
import inspect
from dataclasses import dataclass, field, asdict


@dataclass
class TrainConfig:
    """Unified configuration for all training hyperparameters."""
    # Core training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = 'cuda'

    # Data parameters
    val_split: float = 0.2
    normalize_data: bool = True

    # Optimization parameters
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_scheduler: str = 'cosine'  # Options: 'cosine', 'plateau', 'step', None
    warmup_epochs: int = 10

    # Training control
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 50
    use_mixed_precision: bool = True

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update config values from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def load_processed_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load processed flat dataset from HDF5 file.

    Loads datasets created by process_trajectories.py for BC or Action Translation training.

    Supports two formats:
    1. Action translation format: data/states, data/next_states, data/actions_src, data/actions_target
    2. BC Markov format: states, actions, next_states (actions_src = actions)

    Returns:
        states: State observations
        next_states: Next state observations
        actions_src: Source domain actions
        actions_target: Target domain actions
        obs_dim: Observation dimension
        action_dim: Action dimension
    """
    print("=" * 80)
    print("LOADING PROCESSED DATASET")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}\n")

    with h5py.File(dataset_path, 'r') as f:
        # Check if this is BC Markov format (no 'data' group)
        if 'data' not in f and 'states' in f and 'actions' in f:
            print("[INFO] Detected BC Markov dataset format (Markovian states)")
            states = f['states'][:]
            next_states = f['next_states'][:]
            actions_target = f['actions'][:]
            actions_src = actions_target  # For BC, we don't use source actions

            # Load metadata
            num_samples = f.attrs['num_samples']
            obs_dim = f.attrs['state_dim']
            action_dim = f.attrs['action_dim']
            markovian = True
        else:
            # Original action translation format
            print("[INFO] Detected action translation dataset format (165D observations)")
            states = f['data']['states'][:]
            next_states = f['data']['next_states'][:]
            actions_src = f['data']['actions_src'][:]
            actions_target = f['data']['actions_target'][:]

            # Load metadata
            num_samples = f['meta'].attrs['num_samples']
            obs_dim = f['meta'].attrs['obs_dim']
            action_dim = f['meta'].attrs['action_dim']
            markovian = False

    print(f"Dataset loaded successfully!")
    print(f"  Number of samples: {num_samples:,}")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Markovian states (no action history): {markovian}")
    print(f"  States shape: {states.shape}")
    print(f"  Next states shape: {next_states.shape}")
    if not markovian:
        print(f"  Actions (source) shape: {actions_src.shape}")
    print(f"  Actions (target) shape: {actions_target.shape}")
    print("=" * 80 + "\n")

    return states, next_states, actions_src, actions_target, obs_dim, action_dim


def normalize_tensors(
    tensors: Dict[str, torch.Tensor],
    stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    compute_stats_from: Optional[List[str]] = None,
    epsilon: float = 1e-8
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Normalize tensors using provided or computed statistics.

    Args:
        tensors: Dictionary of tensors to normalize
        stats: Pre-computed (mean, std) tuples for each tensor key
        compute_stats_from: Keys to compute statistics from (training set)
        epsilon: Small value for numerical stability

    Returns:
        normalized_tensors: Dictionary of normalized tensors
        stats: Dictionary of (mean, std) tuples used for normalization
    """
    if stats is None:
        stats = {}
        compute_stats_from = compute_stats_from or list(tensors.keys())

        print("\n" + "-" * 80)
        print("DATA NORMALIZATION")
        print("-" * 80)

        for key in compute_stats_from:
            if key in tensors:
                tensor = tensors[key]
                mean = tensor.mean(dim=0, keepdim=True)
                std = tensor.std(dim=0, keepdim=True) + epsilon
                stats[key] = (mean, std)
                print(f"{key} mean: min={mean.min().item():.4f}, max={mean.max().item():.4f}")
                print(f"{key} std: min={std.min().item():.4f}, max={std.max().item():.4f}")

        print("-" * 80 + "\n")

    normalized = {}
    for key, tensor in tensors.items():
        if key in stats:
            mean, std = stats[key]
        elif key.startswith('next_') and key[5:] in stats:
            # Use same stats for next_states as states
            mean, std = stats[key[5:]]
        else:
            # No normalization for this tensor
            normalized[key] = tensor
            continue

        normalized[key] = (tensor - mean) / std

    return normalized, stats


def create_dataloaders(
    train_tensors: List[torch.Tensor],
    val_tensors: Optional[List[torch.Tensor]],
    config: TrainConfig,
    val_split: Optional[float] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        train_tensors: List of tensors for training data
        val_tensors: Optional list of tensors for validation data
        config: Training configuration
        val_split: If val_tensors is None, split this fraction from train

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    if val_tensors is None:
        # Split training data into train/val
        dataset = TensorDataset(*train_tensors)
        val_split = val_split or config.val_split

        total_size = len(dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)

    print(f"\nDataset Split:")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and config.device == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and config.device == 'cuda'
    )

    return train_loader, val_loader


def log_model_architecture(model: nn.Module):
    """Log model architecture and parameter counts."""
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


def setup_optimizers(
    model: nn.Module,
    config: TrainConfig
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler], Optional[optim.lr_scheduler._LRScheduler]]:
    """Setup optimizer and learning rate schedulers.

    Returns:
        optimizer: AdamW optimizer
        scheduler: Main learning rate scheduler (cosine/plateau/step)
        warmup_scheduler: Optional warmup scheduler
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print("-" * 80)
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"Optimizer: AdamW")
    print(f"Initial learning rate: {config.learning_rate}")
    print(f"Weight decay (L2): {config.weight_decay}")
    print(f"Gradient clipping: {config.grad_clip}")
    print(f"LR Scheduler: {config.lr_scheduler}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print(f"Data normalization: {config.normalize_data}")
    print(f"Mixed precision: {config.use_mixed_precision and config.device == 'cuda'}")
    print("-" * 80 + "\n")

    # Main scheduler
    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=config.learning_rate * 0.01
        )
    elif config.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    elif config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5
        )

    # Warmup scheduler
    warmup_scheduler = None
    if config.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs
        )

    return optimizer, scheduler, warmup_scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    num_epochs: int,
    model_forward_fn: Callable,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        config: Training configuration
        epoch: Current epoch number
        num_epochs: Total number of epochs
        model_forward_fn: Function that calls model.forward with correct arguments
        scaler: Optional gradient scaler for mixed precision

    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}",
                position=1, leave=False, ncols=100)

    for batch in pbar:
        batch = [b.to(config.device) for b in batch]
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = model_forward_fn(model, batch)

            scaler.scale(loss).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                # Check for NaN gradients
                if torch.isnan(grad_norm):
                    print(f"WARNING: NaN gradients detected, skipping update")
                    optimizer.zero_grad()
                    continue

            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model_forward_fn(model, batch)
            loss.backward()

            if config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                # Check for NaN gradients
                if torch.isnan(grad_norm):
                    print(f"WARNING: NaN gradients detected, skipping update")
                    optimizer.zero_grad()
                    continue

            optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'batch_loss': f'{loss.item():.6f}',
            'avg_loss': f'{epoch_loss/num_batches:.6f}'
        })

    return epoch_loss / num_batches


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
    epoch: int,
    num_epochs: int,
    model_predict_fn: Callable
) -> float:
    """Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        config: Training configuration
        epoch: Current epoch number
        num_epochs: Total number of epochs
        model_predict_fn: Function that returns (predictions, targets) for loss computation

    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}",
                position=1, leave=False, ncols=100)

    with torch.no_grad():
        for batch in pbar:
            batch = [b.to(config.device) for b in batch]

            predictions, targets = model_predict_fn(model, batch)
            loss = torch.nn.functional.mse_loss(predictions, targets)

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'batch_loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_loss/num_batches:.6f}'
            })

    return epoch_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    output_path: str,
    val_loss: float,
    normalization_stats: Optional[Dict] = None,
    additional_info: Optional[Dict] = None
):
    """Save model checkpoint with normalization statistics."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss
    }

    if normalization_stats:
        checkpoint.update(normalization_stats)

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint saved to {output_path} (val_loss: {val_loss:.6f})")


def build_model_from_config(config: Dict, obs_dim: int, action_dim: int, load_checkpoint: bool = False) -> nn.Module:
    """Build model dynamically from config with _target_ specification.

    Args:
        config: Model configuration with '_target_' key specifying class path
        obs_dim: Observation dimension
        action_dim: Action dimension
        load_checkpoint: Whether to load checkpoint if specified in config

    Returns:
        Instantiated model
    """
    if '_target_' not in config:
        raise ValueError("Model config must contain '_target_' key specifying the model class path")

    target_path = config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Get valid init parameters for the model class
    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')

    # Build model parameters, filtering to only valid ones
    model_params = {k: v for k, v in config.items() if k in init_params and k != '_target_'}

    # Add dimensions if they're valid parameters
    if 'obs_dim' in init_params:
        model_params['obs_dim'] = obs_dim
    if 'action_dim' in init_params:
        model_params['action_dim'] = action_dim

    # Create model
    model = model_class(**model_params)

    # Load checkpoint if requested
    if load_checkpoint and 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint_path = config['checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

    return model


def generic_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    model_forward_fn: Callable,
    model_predict_fn: Callable,
    output_path: Optional[str] = None,
    normalization_stats: Optional[Dict] = None,
    wandb_enabled: bool = True
) -> Tuple[nn.Module, List[float], List[float]]:
    """Generic training loop that can be used by all training scripts.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        model_forward_fn: Function(model, batch) -> loss for training
        model_predict_fn: Function(model, batch) -> (predictions, targets) for validation
        output_path: Path to save best model checkpoint
        normalization_stats: Optional normalization statistics to save with checkpoint
        wandb_enabled: Whether to log to wandb

    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    model = model.to(config.device)
    log_model_architecture(model)

    # Setup training
    optimizer, scheduler, warmup_scheduler = setup_optimizers(model, config)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision and config.device == 'cuda' else None

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epochs_since_save = 0

    # Epoch loop
    epoch_pbar = tqdm(range(config.num_epochs), desc="Epochs", position=0, leave=True)

    for epoch in epoch_pbar:
        # Train
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, config,
            epoch, config.num_epochs, model_forward_fn, scaler
        )
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss = validate_epoch(
            model, val_loader, config,
            epoch, config.num_epochs, model_predict_fn
        )
        val_losses.append(avg_val_loss)

        # Track best and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model immediately when validation improves
            if output_path:
                save_checkpoint(model, output_path, avg_val_loss, normalization_stats)
                epochs_since_save = 0
        else:
            epochs_without_improvement += 1
            epochs_since_save += 1

        # Early stopping
        if config.early_stopping_patience is not None and config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            print(f"\n\nEarly stopping triggered! No improvement for {config.early_stopping_patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

        # Periodic checkpoint saving (even if not improving)
        if output_path and epochs_since_save >= config.save_every_n_epochs:
            print(f"\nSaving periodic checkpoint (not best, but saving progress)...")
            save_checkpoint(model, output_path.replace('.pth', f'_epoch{epoch}.pth'), avg_val_loss, normalization_stats)
            epochs_since_save = 0

        # Learning rate scheduling
        if warmup_scheduler and epoch < config.warmup_epochs:
            warmup_scheduler.step()
        elif scheduler:
            if config.lr_scheduler == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}',
            'best_val': f'{best_val_loss:.6f}',
            'lr': f'{current_lr:.2e}'
        })

        # Log to wandb
        if wandb_enabled:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement
            })

        # Periodic console output
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")

    epoch_pbar.close()

    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80 + "\n")

    return model, train_losses, val_losses