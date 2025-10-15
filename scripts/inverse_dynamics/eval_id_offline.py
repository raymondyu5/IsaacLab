"""Script to evaluate inverse dynamics model by sequential evaluation on trajectories.

This script takes expert trajectories and uses the inverse dynamics model to predict
actions that should reproduce the expert states. It then compares the predictions with
the expert trajectory and saves a new dataset with ID-predicted actions.

The output dataset has the format (s, a_id_pred, s') where:
- s: current state (same as original)
- a_id_pred: action predicted by the inverse dynamics model
- s': next state (same as original)

Example usage:
    # Evaluate with a single trajectory
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_rollout.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --episode_idx 0

    # Evaluate and save dataset with ID actions
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_rollout.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --episode_idx 0 \
        --output_dataset trajectory_data/data_with_id_actions.hdf5

    # Evaluate and plot specific action dimensions
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_rollout.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --episode_idx 0 \
        --plot_dims 0 1 2 5 10 \
        --output_dir eval_results/

    # Evaluate and plot ALL action dimensions
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_rollout.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --episode_idx 0 \
        --plot_all_dims

    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_offline.py         --checkpoint trajectory_data/kuka_allegro_mlp_id_20251013_120530.pth         --config configs/inverse_dynamics/kuka_allegro_train_mlp.yaml         --dataset_path trajectory_data/test_openloop3.hdf5         --episode_idx 2 
"""

import argparse
import os
import yaml
import torch
import numpy as np
import h5py
import inspect
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def load_single_trajectory(dataset_path: str, episode_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single trajectory from the HDF5 dataset.

    Args:
        dataset_path: Path to HDF5 trajectory dataset file
        episode_idx: Index of the episode to load

    Returns:
        observations: (T, obs_dim) array of states
        actions: (T, action_dim) array of actions
    """
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        episode_names = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                              key=lambda x: int(x.split('_')[1]))

        if episode_idx >= len(episode_names):
            raise ValueError(f"Episode index {episode_idx} out of range. Dataset has {len(episode_names)} episodes.")

        ep_name = episode_names[episode_idx]
        ep_group = data_group[ep_name]

        obs = ep_group['obs'][:]  # (T, obs_dim)
        actions = ep_group['actions'][:]  # (T, action_dim)

        print(f"\nLoaded episode: {ep_name}")
        print(f"  Trajectory length: {len(obs)}")
        print(f"  Observation dim: {obs.shape[1]}")
        print(f"  Action dim: {actions.shape[1]}")

        # Check if success metadata exists
        if 'success' in ep_group.attrs:
            success = ep_group.attrs['success']
            print(f"  Success: {success}")

        return obs, actions


def save_trajectory_with_id_actions(
    dataset_path: str,
    output_path: str,
    episode_idx: int,
    id_predicted_actions: np.ndarray,
    expert_traj: np.ndarray
):
    """
    Save a new HDF5 dataset with the same format as the original, but with ID-predicted actions.

    This creates a dataset where:
    - States (s, s') remain the same as the original
    - Actions are replaced with the ID model's predicted actions (a_id_pred)
    - All other fields and metadata are preserved from the original

    Args:
        dataset_path: Path to original HDF5 dataset
        output_path: Path to save new HDF5 dataset with ID actions
        episode_idx: Index of episode that was evaluated
        id_predicted_actions: (T-1, action_dim) ID-predicted actions
        expert_traj: (T, obs_dim) expert trajectory states
    """
    print(f"\nSaving trajectory with ID-predicted actions to: {output_path}")

    # Open original dataset to read all data and structure
    with h5py.File(dataset_path, 'r') as f_in:
        data_group = f_in['data']
        episode_names = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                              key=lambda x: int(x.split('_')[1]))

        if episode_idx >= len(episode_names):
            raise ValueError(f"Episode index {episode_idx} out of range.")

        ep_name = episode_names[episode_idx]
        ep_group = data_group[ep_name]

        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Create data group
            out_data = f_out.create_group('data')

            # Copy data group attributes (like env_args, total, etc.)
            for attr_key, attr_val in data_group.attrs.items():
                # Update total to reflect we only have one episode
                if attr_key == 'total':
                    out_data.attrs[attr_key] = len(id_predicted_actions)
                else:
                    out_data.attrs[attr_key] = attr_val

            # Create demo group
            out_demo = out_data.create_group(ep_name)

            # Copy all attributes from original episode
            for attr_key, attr_val in ep_group.attrs.items():
                # Update num_samples to match the new trajectory length
                if attr_key == 'num_samples':
                    out_demo.attrs[attr_key] = len(id_predicted_actions)
                else:
                    out_demo.attrs[attr_key] = attr_val

            # Since ID model predicts T-1 actions, we need to trim states accordingly
            # We'll use states [0:T-1] as current states and [1:T] as next states
            # This gives us T-1 transitions with T-1 actions
            trimmed_traj = expert_traj[:-1]  # Remove last state to match action length

            # Save observations (trimmed to match action length)
            out_demo.create_dataset('obs', data=trimmed_traj)

            # Save ID-predicted actions
            out_demo.create_dataset('actions', data=id_predicted_actions)

            # Helper function to recursively copy groups
            def copy_group(src_group, dst_group, trim_length=None):
                """Recursively copy HDF5 groups and their contents."""
                for key in src_group.keys():
                    if isinstance(src_group[key], h5py.Group):
                        # Create subgroup and recursively copy
                        sub_group = dst_group.create_group(key)
                        copy_group(src_group[key], sub_group, trim_length)
                        # Copy group attributes
                        for attr_key, attr_val in src_group[key].attrs.items():
                            sub_group.attrs[attr_key] = attr_val
                    elif isinstance(src_group[key], h5py.Dataset):
                        # Copy dataset
                        data = src_group[key][:]
                        # Trim if needed and dataset length matches original trajectory
                        if trim_length is not None and len(data) == len(expert_traj):
                            data = data[:trim_length]
                        dst_group.create_dataset(key, data=data)

            # Copy other fields if they exist in the original
            for key in ep_group.keys():
                if key not in ['obs', 'actions']:
                    if isinstance(ep_group[key], h5py.Group):
                        # Create the group and recursively copy its contents
                        new_group = out_demo.create_group(key)
                        copy_group(ep_group[key], new_group, trim_length=len(id_predicted_actions))
                        # Copy group attributes
                        for attr_key, attr_val in ep_group[key].attrs.items():
                            new_group.attrs[attr_key] = attr_val
                    elif isinstance(ep_group[key], h5py.Dataset):
                        # For other fields, copy them directly or trim if needed
                        original_data = ep_group[key][:]

                        # If the field has the same length as original trajectory, trim it
                        if len(original_data) == len(expert_traj):
                            out_demo.create_dataset(key, data=original_data[:-1])
                        else:
                            # Keep as is
                            out_demo.create_dataset(key, data=original_data)

            # Copy root-level attributes if any
            for attr_key, attr_val in f_in.attrs.items():
                f_out.attrs[attr_key] = attr_val

    print(f"  Saved {len(id_predicted_actions)} transitions")
    print(f"  Observation shape: {trimmed_traj.shape}")
    print(f"  Action shape: {id_predicted_actions.shape}")
    print(f"Dataset saved successfully!")


def evaluate_sequential_with_inverse_dynamics(
    model,
    expert_traj: np.ndarray,
    expert_actions: np.ndarray,
    device: str = 'cpu',
    normalize_data: bool = True,
    state_mean: Optional[torch.Tensor] = None,
    state_std: Optional[torch.Tensor] = None,
    action_mean: Optional[torch.Tensor] = None,
    action_std: Optional[torch.Tensor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequentially evaluate inverse dynamics model on a trajectory.

    The inverse dynamics model predicts the action needed to get from
    state s_t to the next expert state s_{t+1}. We use the expert next states
    as targets to evaluate how well the model can predict the actions.

    Args:
        model: Trained inverse dynamics model
        expert_traj: Expert trajectory states (T, obs_dim)
        expert_actions: Expert actions (T, action_dim)
        device: Device to run inference on
        normalize_data: Whether to normalize data
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        action_mean: Mean for action denormalization
        action_std: Std for action denormalization

    Returns:
        predicted_actions: (T-1, action_dim) predicted actions
        expert_actions: (T-1, action_dim) ground truth actions (trimmed)
    """
    model.to(device)
    model.eval()

    T = len(expert_traj)
    predicted_actions = []

    print("\nSequentially evaluating trajectory with inverse dynamics model...")
    print(f"Trajectory length: {T}")

    with torch.no_grad():
        for t in range(T - 1):
            # Get current state and target next state
            current_state = expert_traj[t]
            target_next_state = expert_traj[t + 1]

            # Convert to tensors
            current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)  # (1, obs_dim)
            target_next_state_tensor = torch.FloatTensor(target_next_state).unsqueeze(0)  # (1, obs_dim)

            # Normalize if needed
            if normalize_data and state_mean is not None and state_std is not None:
                current_state_tensor = (current_state_tensor - state_mean) / state_std
                target_next_state_tensor = (target_next_state_tensor - state_mean) / state_std

            # Move to device
            current_state_tensor = current_state_tensor.to(device)
            target_next_state_tensor = target_next_state_tensor.to(device)

            # Predict action using inverse dynamics
            predicted_action = model.predict(current_state_tensor, target_next_state_tensor)

            # Denormalize action if needed
            if normalize_data and action_mean is not None and action_std is not None:
                predicted_action = predicted_action * action_std.cpu().numpy() + action_mean.cpu().numpy()

            predicted_actions.append(predicted_action[0])

    predicted_actions = np.array(predicted_actions)
    expert_actions_trimmed = expert_actions[:-1]  # Remove last action to match length

    print(f"Generated {len(predicted_actions)} action predictions")

    return predicted_actions, expert_actions_trimmed


def compute_sequential_metrics(predicted_actions: np.ndarray, expert_actions: np.ndarray) -> dict:
    """
    Compute metrics comparing predicted actions to expert actions.

    Args:
        predicted_actions: (T-1, action_dim) predicted actions
        expert_actions: (T-1, action_dim) expert actions

    Returns:
        Dictionary containing evaluation metrics
    """
    mse = np.mean((predicted_actions - expert_actions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_actions - expert_actions))

    # Per-dimension metrics
    mse_per_dim = np.mean((predicted_actions - expert_actions) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(predicted_actions - expert_actions), axis=0)

    # Correlation per dimension
    correlations = []
    for i in range(predicted_actions.shape[1]):
        corr = np.corrcoef(predicted_actions[:, i], expert_actions[:, i])[0, 1]
        correlations.append(corr)

    mean_correlation = np.mean(correlations)

    # R-squared
    ss_res = np.sum((expert_actions - predicted_actions) ** 2)
    ss_tot = np.sum((expert_actions - expert_actions.mean(axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_correlation': mean_correlation,
        'mse_per_dim': mse_per_dim,
        'mae_per_dim': mae_per_dim,
        'correlations': correlations
    }


def plot_action_comparison(
    predicted_actions: np.ndarray,
    expert_actions: np.ndarray,
    metrics: dict,
    output_path: str,
    plot_dims: Optional[List[int]] = None
):
    """
    Plot comparison between predicted and expert actions.

    Args:
        predicted_actions: (T-1, action_dim) predicted actions
        expert_actions: (T-1, action_dim) expert actions
        metrics: Dictionary of evaluation metrics
        output_path: Path to save plot
        plot_dims: List of action dimensions to plot (if None, plot all)
    """
    action_dim = predicted_actions.shape[1]

    if plot_dims is None:
        # If too many dimensions, just plot the first 6
        plot_dims = list(range(min(6, action_dim)))

    # Filter to valid dimensions
    plot_dims = [d for d in plot_dims if 0 <= d < action_dim]

    if len(plot_dims) == 0:
        print("No valid dimensions to plot.")
        return

    num_plots = len(plot_dims)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))

    if num_plots == 1:
        axes = [axes]

    timesteps = np.arange(len(predicted_actions))

    for idx, dim in enumerate(plot_dims):
        ax = axes[idx]

        ax.plot(timesteps, expert_actions[:, dim],
                label='Expert', color='blue', alpha=0.7, linewidth=2)
        ax.plot(timesteps, predicted_actions[:, dim],
                label='ID Model', color='red', alpha=0.7, linewidth=2, linestyle='--')

        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Action Dim {dim}')
        ax.set_title(f'Action Dimension {dim} - MSE: {metrics["mse_per_dim"][dim]:.6f}, '
                    f'Corr: {metrics["correlations"][dim]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Inverse Dynamics Action Prediction vs Expert\n'
                f'Overall MSE: {metrics["mse"]:.6f}, MAE: {metrics["mae"]:.6f}, '
                f'R²: {metrics["r2"]:.4f}', fontsize=14, y=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate inverse dynamics model by sequential evaluation on trajectories'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to trajectory dataset HDF5 file')
    parser.add_argument('--episode_idx', type=int, default=0,
                       help='Index of episode to evaluate (default: 0)')
    parser.add_argument('--plot_dims', type=int, nargs='+', default=None,
                       help='Action dimensions to plot (default: first 6)')
    parser.add_argument('--plot_all_dims', action='store_true',
                       help='Plot all action dimensions (overrides --plot_dims)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: same as checkpoint)')
    parser.add_argument('--output_dataset', type=str, default=None,
                       help='Path to save HDF5 dataset with ID-predicted actions (default: output_dir/id_dataset.hdf5)')

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config = config.get('training', {})

    # Setup output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_dir = os.path.join(output_dir, f'eval_sequential_{checkpoint_name}_ep{args.episode_idx}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("INVERSE DYNAMICS SEQUENTIAL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Episode index: {args.episode_idx}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {train_config.get('device', 'cuda')}")
    print("=" * 80)

    # Load trajectory
    expert_traj, expert_actions = load_single_trajectory(args.dataset_path, args.episode_idx)

    # Build model
    print(f"\nBuilding inverse dynamics model from config...")
    target_path = model_config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    model_params = {k: v for k, v in model_config.items() if k in init_params}
    model = model_class(**model_params)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            "Checkpoint is missing normalization statistics. "
            "Please retrain the model with the updated train_id.py script."
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']
    action_mean = checkpoint['action_mean']
    action_std = checkpoint['action_std']
    normalize_data = checkpoint.get('normalize_data', train_config.get('normalize_data', True))

    print("Checkpoint loaded successfully!")
    if normalize_data and state_mean is not None:
        print(f"\nUsing normalization statistics from training:")
        print(f"  State mean range: [{state_mean.min().item():.4f}, {state_mean.max().item():.4f}]")
        print(f"  State std range: [{state_std.min().item():.4f}, {state_std.max().item():.4f}]")
        print(f"  Action mean range: [{action_mean.min().item():.4f}, {action_mean.max().item():.4f}]")
        print(f"  Action std range: [{action_std.min().item():.4f}, {action_std.max().item():.4f}]")

    device = train_config.get('device', 'cuda')

    # Sequential evaluation with inverse dynamics
    predicted_actions, expert_actions_trimmed = evaluate_sequential_with_inverse_dynamics(
        model=model,
        expert_traj=expert_traj,
        expert_actions=expert_actions,
        device=device,
        normalize_data=normalize_data,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std
    )

    # Compute metrics
    print("\n" + "-" * 80)
    print("COMPUTING EVALUATION METRICS")
    print("-" * 80)

    metrics = compute_sequential_metrics(predicted_actions, expert_actions_trimmed)

    print("\nOverall Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  Mean Correlation: {metrics['mean_correlation']:.6f}")

    print("\nPer-Dimension MSE:")
    for i, mse in enumerate(metrics['mse_per_dim']):
        print(f"  Dim {i:2d}: {mse:.6f}")

    print("\nPer-Dimension Correlation:")
    for i, corr in enumerate(metrics['correlations']):
        print(f"  Dim {i:2d}: {corr:.4f}")

    print("-" * 80)

    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'sequential_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INVERSE DYNAMICS SEQUENTIAL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Episode index: {args.episode_idx}\n")
        f.write(f"Trajectory length: {len(expert_traj)}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  MSE:  {metrics['mse']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"  MAE:  {metrics['mae']:.6f}\n")
        f.write(f"  R²:   {metrics['r2']:.6f}\n")
        f.write(f"  Mean Correlation: {metrics['mean_correlation']:.6f}\n\n")
        f.write("Per-Dimension MSE:\n")
        for i, mse in enumerate(metrics['mse_per_dim']):
            f.write(f"  Dim {i:2d}: {mse:.6f}\n")
        f.write("\nPer-Dimension MAE:\n")
        for i, mae in enumerate(metrics['mae_per_dim']):
            f.write(f"  Dim {i:2d}: {mae:.6f}\n")
        f.write("\nPer-Dimension Correlation:\n")
        for i, corr in enumerate(metrics['correlations']):
            f.write(f"  Dim {i:2d}: {corr:.4f}\n")

    print(f"\nMetrics saved to: {metrics_path}")

    # Plot results
    plot_path = os.path.join(output_dir, 'action_comparison.png')

    # Determine which dimensions to plot
    plot_dims = args.plot_dims
    if args.plot_all_dims:
        action_dim = predicted_actions.shape[1]
        plot_dims = list(range(action_dim))
        print(f"\nPlotting all {action_dim} action dimensions...")

    plot_action_comparison(
        predicted_actions=predicted_actions,
        expert_actions=expert_actions_trimmed,
        metrics=metrics,
        output_path=plot_path,
        plot_dims=plot_dims
    )

    # Save dataset with ID-predicted actions
    output_dataset_path = args.output_dataset if args.output_dataset else os.path.join(output_dir, 'id_dataset.hdf5')
    save_trajectory_with_id_actions(
        dataset_path=args.dataset_path,
        output_path=output_dataset_path,
        episode_idx=args.episode_idx,
        id_predicted_actions=predicted_actions,
        expert_traj=expert_traj
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"ID dataset saved to: {output_dataset_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
