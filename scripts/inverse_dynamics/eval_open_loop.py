"""Script to evaluate inverse dynamics model with open-loop simulation.

This script loads expert trajectories and executes actions in the simulator to see
how well they track the expert trajectory. It supports two modes:

1. **expert mode**: Execute expert actions from the dataset (useful for debugging)
2. **id_model mode**: Execute actions predicted by the ID model (default)

This helps diagnose whether trajectory divergence is due to:
- Poor ID model predictions (if expert actions work well but ID actions don't)
- Inherent open-loop instability (if even expert actions diverge)

Example usage:
    # Evaluate ID model actions (default)
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_sim.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0 \
        --episode_idx 0 \
        --mode id_model

    # Evaluate expert actions (for comparison/debugging)
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_sim.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0 \
        --episode_idx 0 \
        --mode expert

    # With video and custom state dims
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id_sim.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0 \
        --episode_idx 0 \
        --mode expert \
        --plot_dims 0 1 2 5 10 \
        --video \
        --video_dir eval_videos/
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate inverse dynamics model with closed-loop simulation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (should be 1 for evaluation)")
parser.add_argument("--task", type=str, required=True, help="Name of the task environment")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained ID model checkpoint (.pth file)")
parser.add_argument("--config", type=str, required=True, help="Path to training config YAML file")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to trajectory dataset HDF5 file")
parser.add_argument("--episode_idx", type=int, default=0, help="Index of episode to evaluate (default: 0)")
parser.add_argument(
    "--plot_dims",
    type=int,
    nargs="+",
    default=None,
    help="State dimensions to plot (default: first 6)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save results (default: same as checkpoint)"
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video of the rollout"
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="./eval_id_videos",
    help="Directory to save recorded videos"
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["expert", "id_model"],
    default="id_model",
    help="Rollout mode: 'expert' for expert actions, 'id_model' for inverse dynamics predictions (default: id_model)"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Enable cameras if video recording is requested
if args_cli.video:
    args_cli.enable_cameras = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import yaml
import inspect
import h5py
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


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

        return obs, actions


def compute_state_metrics(id_states: np.ndarray, expert_states: np.ndarray) -> dict:
    """
    Compute metrics comparing ID model rollout states to expert states.

    Args:
        id_states: (T, obs_dim) states from ID model rollout
        expert_states: (T, obs_dim) expert states

    Returns:
        Dictionary containing evaluation metrics
    """
    mse = np.mean((id_states - expert_states) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(id_states - expert_states))

    # Per-dimension metrics
    mse_per_dim = np.mean((id_states - expert_states) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(id_states - expert_states), axis=0)

    # Correlation per dimension
    correlations = []
    for i in range(id_states.shape[1]):
        if np.std(id_states[:, i]) > 1e-8 and np.std(expert_states[:, i]) > 1e-8:
            corr = np.corrcoef(id_states[:, i], expert_states[:, i])[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

    mean_correlation = np.mean(correlations)

    # R-squared
    ss_res = np.sum((expert_states - id_states) ** 2)
    ss_tot = np.sum((expert_states - expert_states.mean(axis=0)) ** 2)
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


def plot_state_comparison(
    id_states: np.ndarray,
    expert_states: np.ndarray,
    metrics: dict,
    output_path: str,
    plot_dims: Optional[List[int]] = None
):
    """
    Plot comparison between ID model rollout and expert states.

    Args:
        id_states: (T, obs_dim) states from ID model rollout
        expert_states: (T, obs_dim) expert states
        metrics: Dictionary of evaluation metrics
        output_path: Path to save plot
        plot_dims: List of state dimensions to plot (if None, plot first 6)
    """
    obs_dim = id_states.shape[1]

    if plot_dims is None:
        # If too many dimensions, just plot the first 6
        plot_dims = list(range(min(6, obs_dim)))

    # Filter to valid dimensions
    plot_dims = [d for d in plot_dims if 0 <= d < obs_dim]

    if len(plot_dims) == 0:
        print("No valid dimensions to plot.")
        return

    num_plots = len(plot_dims)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))

    if num_plots == 1:
        axes = [axes]

    timesteps = np.arange(len(id_states))

    for idx, dim in enumerate(plot_dims):
        ax = axes[idx]

        ax.plot(timesteps, expert_states[:, dim],
                label='Expert', color='blue', alpha=0.7, linewidth=2)
        ax.plot(timesteps, id_states[:, dim],
                label='ID Model', color='red', alpha=0.7, linewidth=2, linestyle='--')

        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'State Dim {dim}')
        ax.set_title(f'State Dimension {dim} - MSE: {metrics["mse_per_dim"][dim]:.6f}, '
                    f'Corr: {metrics["correlations"][dim]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Inverse Dynamics Rollout vs Expert States\n'
                f'Overall MSE: {metrics["mse"]:.6f}, MAE: {metrics["mae"]:.6f}, '
                f'R²: {metrics["r2"]:.4f}', fontsize=14, y=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    """Evaluate inverse dynamics model with closed-loop simulation."""

    # Validate files exist
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.config):
        raise FileNotFoundError(f"Config file not found: {args_cli.config}")
    if not os.path.exists(args_cli.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args_cli.dataset_path}")

    # Load config
    with open(args_cli.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config = config.get('training', {})

    # Setup output directory
    output_dir = args_cli.output_dir if args_cli.output_dir else os.path.dirname(args_cli.checkpoint)
    checkpoint_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    mode_suffix = f"_{args_cli.mode}" if args_cli.mode == "expert" else ""
    output_dir = os.path.join(output_dir, f'eval_sim_{checkpoint_name}_ep{args_cli.episode_idx}{mode_suffix}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"OPEN-LOOP SIMULATION EVALUATION - MODE: {args_cli.mode.upper()}")
    print("=" * 80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Config: {args_cli.config}")
    print(f"Dataset: {args_cli.dataset_path}")
    print(f"Task: {args_cli.task}")
    print(f"Episode index: {args_cli.episode_idx}")
    print(f"Mode: {args_cli.mode}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {train_config.get('device', 'cuda')}")
    print("=" * 80)

    # Load trajectory and get env_id from dataset
    expert_traj, expert_actions = load_single_trajectory(args_cli.dataset_path, args_cli.episode_idx)
    traj_length = len(expert_traj)

    # Load env_id from dataset to spawn correct object variant
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_path)
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_data_temp = dataset_file_handler.load_episode(episode_names[args_cli.episode_idx], 'cpu')
    episode_env_id = episode_data_temp.env_id if episode_data_temp.env_id is not None else 0
    dataset_file_handler.close()

    print(f"\n[INFO] Episode recorded from env_id: {episode_env_id}")
    if episode_env_id > 0:
        print(f"[INFO] Creating {episode_env_id + 1} environments (only env {episode_env_id} will be active)")

    # Build inverse dynamics model
    print(f"\nBuilding inverse dynamics model from config...")
    target_path = model_config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    model_params = {k: v for k, v in model_config.items() if k in init_params}

    # Use CPU for model to avoid device conflicts
    if 'device' in model_params:
        model_params['device'] = 'cpu'

    model = model_class(**model_params)

    # Load checkpoint
    print(f"Loading checkpoint from: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded successfully!")

    # Prepare normalization statistics
    # normalize_data = train_config.get('normalize_data', True)
    # state_mean = None
    # state_std = None
    # action_mean = None
    # action_std = None

    # if normalize_data:
    #     print("\nComputing normalization statistics from trajectory...")
    #     expert_traj_tensor = torch.FloatTensor(expert_traj)
    #     expert_actions_tensor = torch.FloatTensor(expert_actions)

    #     state_mean = expert_traj_tensor.mean(dim=0, keepdim=True)
    #     state_std = expert_traj_tensor.std(dim=0, keepdim=True) + 1e-8
    #     action_mean = expert_actions_tensor.mean(dim=0, keepdim=True)
    #     action_std = expert_actions_tensor.std(dim=0, keepdim=True) + 1e-8

    #     print(f"  State mean range: [{state_mean.min().item():.4f}, {state_mean.max().item():.4f}]")
    #     print(f"  State std range: [{state_std.min().item():.4f}, {state_std.max().item():.4f}]")

    # ============================================================================
    # PREPARE ACTIONS BASED ON MODE
    # ============================================================================
    print("\n" + "=" * 80)
    if args_cli.mode == "expert":
        print("MODE: EXPERT ACTIONS (OPEN-LOOP)")
        print("=" * 80)
        print(f"Expert trajectory length: {traj_length}")
        print("[INFO] Using expert actions directly from dataset")

        # Use expert actions directly
        actions_to_execute = expert_actions[:traj_length - 1]
        print(f"✓ Loaded {len(actions_to_execute)} expert actions")
        print("=" * 80)

    else:  # id_model mode
        print("MODE: ID MODEL PREDICTIONS (OPEN-LOOP)")
        print("=" * 80)
        print(f"Expert trajectory length: {traj_length}")
        print("[INFO] Computing actions using ID model on dataset state transitions")

        predicted_actions = []

        with torch.inference_mode():
            for t in range(traj_length - 1):
                # Use expert state transitions (s_t, s_{t+1}) to predict action
                current_state = expert_traj[t]
                target_next_state = expert_traj[t + 1]

                # Convert to tensors
                current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)  # (1, obs_dim)
                target_next_state_tensor = torch.FloatTensor(target_next_state).unsqueeze(0)  # (1, obs_dim)

                # Normalize if needed
                # if normalize_data and state_mean is not None and state_std is not None:
                #     current_state_tensor = (current_state_tensor - state_mean) / state_std
                #     target_next_state_tensor = (target_next_state_tensor - state_mean) / state_std

                # Predict action using inverse dynamics
                predicted_action = model.predict(current_state_tensor, target_next_state_tensor)

                # Denormalize action if needed
                # if normalize_data and action_mean is not None and action_std is not None:
                #     predicted_action = predicted_action * action_std.cpu().numpy() + action_mean.cpu().numpy()

                predicted_actions.append(predicted_action[0])

                if (t + 1) % 50 == 0:
                    print(f"  Predicted action {t + 1}/{traj_length - 1}")

        actions_to_execute = np.array(predicted_actions)
        print(f"✓ Generated {len(actions_to_execute)} action predictions")

        # Compute action prediction errors for reporting
        expert_actions_trimmed = expert_actions[:len(actions_to_execute)]
        action_errors = np.linalg.norm(actions_to_execute - expert_actions_trimmed, axis=1)
        print(f"\nAction prediction quality:")
        print(f"  Mean L2 error: {action_errors.mean():.4f}")
        print(f"  Median L2 error: {np.median(action_errors):.4f}")
        print("=" * 80)

    # ============================================================================
    # Setup environment
    # ============================================================================
    # Create enough environments so that episode_env_id exists (MultiAssetSpawner uses env_id % num_assets)
    # We'll use episode_env_id + 1 environments and only interact with the last one
    num_envs_needed = episode_env_id + 1
    print(f"\nSetting up environment: {args_cli.task}")
    print(f"[INFO] Creating {num_envs_needed} environments (will only use env {episode_env_id})")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs_needed)

    # Set seed to match replay (default is usually 42)
    # This ensures the MultiAssetSpawner selects the same object variants
    if not hasattr(env_cfg, 'seed') or env_cfg.seed is None:
        env_cfg.seed = 42
    print(f"[INFO] Environment seed: {env_cfg.seed}")

    # Disable recorders for clean evaluation
    env_cfg.recorders = {}

    # Disable command resampling
    if hasattr(env_cfg, 'commands'):
        for attr_name in dir(env_cfg.commands):
            attr = getattr(env_cfg.commands, attr_name)
            if hasattr(attr, 'resampling_time_range'):
                attr.resampling_time_range = (1000000.0, 1000000.0)
                print(f"[INFO] Disabled command resampling for '{attr_name}'")
            if hasattr(attr, 'debug_vis'):
                attr.debug_vis = False

    # Remove termination-dependent rewards
    if hasattr(env_cfg, 'rewards'):
        reward_terms_to_remove = []
        for attr_name in dir(env_cfg.rewards):
            attr = getattr(env_cfg.rewards, attr_name)
            if hasattr(attr, 'func') and hasattr(attr, 'params'):
                if isinstance(attr.params, dict) and 'term_keys' in attr.params:
                    reward_terms_to_remove.append(attr_name)
        for term_name in reward_terms_to_remove:
            print(f"[INFO] Removing reward term '{term_name}' (depends on terminations)")
            delattr(env_cfg.rewards, term_name)

    env_cfg.terminations = {}

    print("\n[INFO] Configuring deterministic environment...")

    # Disable all randomization for deterministic replay
    env_cfg.events = {}

    if hasattr(env_cfg, 'curriculum'):
        env_cfg.curriculum = {}

    # Disable observation noise
    if hasattr(env_cfg, 'observations'):
        for group_name in ['policy', 'critic']:
            if hasattr(env_cfg.observations, group_name):
                group = getattr(env_cfg.observations, group_name)
                for term_name in dir(group):
                    if not term_name.startswith('_'):
                        term = getattr(group, term_name)
                        if hasattr(term, 'noise'):
                            term.noise = None

    # Disable action noise only for directrlenvs
    if hasattr(env_cfg, 'actions'):
        for action_name in dir(env_cfg.actions):
            if not action_name.startswith('_'):
                action = getattr(env_cfg.actions, action_name)
                if hasattr(action, 'noise'):
                    action.noise = None

    # Configure viewer to follow the correct environment
    if args_cli.video and hasattr(env_cfg, 'viewer') and env_cfg.viewer is not None:
        env_cfg.viewer.env_index = episode_env_id

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped

    # Setup video recording
    video_frames = [] if args_cli.video else None
    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)

    # Load initial state from dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_path)
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_data = dataset_file_handler.load_episode(episode_names[args_cli.episode_idx], env.device)

    # Reset environment to initial state
    initial_state = episode_data.get_initial_state()
    env.reset()
    env.reset_to(initial_state, torch.tensor([episode_env_id], device=env.device), is_relative=True)

    # Get initial observation from environment after reset
    obs_after_reset = env.observation_manager.compute()
    if isinstance(obs_after_reset, dict):
        obs_tensor_after_reset = obs_after_reset['policy'] if 'policy' in obs_after_reset else obs_after_reset[list(obs_after_reset.keys())[0]]
        initial_obs = obs_tensor_after_reset[episode_env_id].cpu().numpy()
    else:
        obs_tensor_after_reset = obs_after_reset
        initial_obs = obs_tensor_after_reset[episode_env_id].cpu().numpy()

    # Capture initial frame
    if args_cli.video:
        frame = env.render()
        if frame is not None:
            if isinstance(frame, (list, tuple)) and len(frame) > episode_env_id:
                video_frames.append(frame[episode_env_id])
            else:
                video_frames.append(frame)


    # Execute actions in simulation (open-loop execution)
    print("\n" + "=" * 80)
    print("EXECUTING ACTIONS IN SIMULATION")
    print("=" * 80)
    print(f"Executing {len(actions_to_execute)} actions open-loop...")

    id_states = []
    id_actions = []

    # Use the actual observation from the environment after reset
    id_states.append(initial_obs)

    for t in range(len(actions_to_execute)):
        # Use the action (either expert or ID-predicted)
        action = actions_to_execute[t]
        id_actions.append(action)

        # Execute action in simulator
        actions_all_envs = torch.zeros(num_envs_needed, action.shape[0], device=env.device)
        actions_all_envs[episode_env_id] = torch.FloatTensor(action).to(env.device)


        obs, rewards, terminated, truncated, info = env.step(actions_all_envs)

        # Extract observation for the active environment
        if isinstance(obs, dict):
            current_obs = obs['policy'][episode_env_id].cpu().numpy() if 'policy' in obs else obs[list(obs.keys())[0]][episode_env_id].cpu().numpy()
        else:
            current_obs = obs[episode_env_id].cpu().numpy()

        id_states.append(current_obs)

        # Capture frame for video
        if args_cli.video:
            frame = env.render()
            if frame is not None:
                # Extract frame for the specific env we're using
                if isinstance(frame, (list, tuple)) and len(frame) > episode_env_id:
                    video_frames.append(frame[episode_env_id])
                else:
                    video_frames.append(frame)

        if (t + 1) % 50 == 0:
            print(f"  Executed step {t + 1}/{len(actions_to_execute)}")

    print(f"\nCompleted rollout: {len(id_states)} states, {len(id_actions)} actions")

    # Convert to numpy arrays
    id_states = np.array(id_states)
    id_actions = np.array(id_actions)

    # Compute metrics
    print("\n" + "-" * 80)
    print("COMPUTING EVALUATION METRICS")
    print("-" * 80)

    # Trim expert trajectory to match ID model length (in case rollout was shorter)
    min_length = min(len(id_states), len(expert_traj))
    id_states = id_states[:min_length]
    expert_traj_trimmed = expert_traj[:min_length]

    metrics = compute_state_metrics(id_states, expert_traj_trimmed)

    print("\nState Tracking Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  Mean Correlation: {metrics['mean_correlation']:.6f}")

    print("\nPer-Dimension MSE (first 10):")
    for i, mse in enumerate(metrics['mse_per_dim'][:10]):
        print(f"  Dim {i:2d}: {mse:.6f}")

    print("\nPer-Dimension Correlation (first 10):")
    for i, corr in enumerate(metrics['correlations'][:10]):
        print(f"  Dim {i:2d}: {corr:.4f}")

    print("-" * 80)

    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'sim_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INVERSE DYNAMICS CLOSED-LOOP SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {args_cli.dataset_path}\n")
        f.write(f"Episode index: {args_cli.episode_idx}\n")
        f.write(f"Target trajectory length: {traj_length}\n")
        f.write(f"Actual rollout length: {len(id_states)}\n\n")
        f.write("State Tracking Metrics:\n")
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
    plot_path = os.path.join(output_dir, 'state_comparison.png')
    plot_state_comparison(
        id_states=id_states,
        expert_states=expert_traj_trimmed,
        metrics=metrics,
        output_path=plot_path,
        plot_dims=args_cli.plot_dims
    )

    # Save video
    if args_cli.video and len(video_frames) > 1:
        video_path = os.path.join(args_cli.video_dir, f'eval_sim_ep{args_cli.episode_idx}.mp4')
        print(f"\nSaving video ({len(video_frames)} frames)...")
        try:
            import imageio
            imageio.mimsave(video_path, video_frames, fps=30, macro_block_size=1)
            print(f"  ✓ Video saved to: {video_path}")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")

    # Close environment
    env.close()
    dataset_file_handler.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
