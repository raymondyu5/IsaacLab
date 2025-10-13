"""Script to evaluate a trained inverse dynamics model on an offline dataset.

Example usage:
    # Evaluate with config file (recommended)
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5

    # Evaluate on a subset of the dataset
    ./isaaclab.sh -p scripts/inverse_dynamics/eval_id.py \
        --checkpoint trajectory_data/kuka_allegro_flow_id_20251012_235017.pth \
        --config configs/inverse_dynamics/kuka_allegro_train_flow.yaml \
        --dataset_path trajectory_data/data.hdf5 \
        --num_episodes 100
"""

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import inspect

from utils.data_utils import load_trajectory_dataset
from utils.eval_utils import compute_evaluation_metrics, plot_evaluation_results


def evaluate_inverse_dynamics(states, actions, next_states, model,
                               eval_config, output_dir=None):
    """
    Evaluate the inverse dynamics model on the provided dataset.

    Args:
        states: Evaluation state observations
        actions: Ground truth actions
        next_states: Evaluation next state observations
        model: The trained inverse dynamics model
        eval_config: Dictionary containing evaluation settings
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary containing evaluation metrics
    """
    print("=" * 80)
    print("EVALUATING INVERSE DYNAMICS MODEL")
    print("=" * 80)

    batch_size = eval_config.get('batch_size', 256)
    device = eval_config.get('device', 'cpu')
    normalize_data = eval_config.get('normalize_data', True)
    state_mean = eval_config.get('state_mean', None)
    state_std = eval_config.get('state_std', None)
    action_mean = eval_config.get('action_mean', None)
    action_std = eval_config.get('action_std', None)

    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    next_states_tensor = torch.FloatTensor(next_states)

    if len(actions_tensor.shape) == 1:
        actions_tensor = actions_tensor.unsqueeze(1)

    if normalize_data:
        print("\n" + "-" * 80)
        print("DATA NORMALIZATION")
        print("-" * 80)

        if state_mean is not None and state_std is not None:
            print("Using provided normalization statistics (from training)")
            states_tensor = (states_tensor - state_mean) / state_std
            next_states_tensor = (next_states_tensor - state_mean) / state_std
        else:
            print("Computing normalization statistics from evaluation data")
            state_mean = states_tensor.mean(dim=0, keepdim=True)
            state_std = states_tensor.std(dim=0, keepdim=True) + 1e-8
            states_tensor = (states_tensor - state_mean) / state_std
            next_states_tensor = (next_states_tensor - state_mean) / state_std

        print("-" * 80 + "\n")

    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True if device == 'cuda' else False)

    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truth = []

    print("Running inference on evaluation dataset...")
    with torch.no_grad():
        for batch_states, batch_actions, batch_next_states in tqdm(dataloader, desc="Evaluating"):
            batch_states = batch_states.to(device)
            batch_next_states = batch_next_states.to(device)

            preds = model.predict(batch_states, batch_next_states)

            all_predictions.append(preds)
            all_ground_truth.append(batch_actions.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)

    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    print("\n" + "-" * 80)
    print("COMPUTING EVALUATION METRICS")
    print("-" * 80)

    metrics = compute_evaluation_metrics(predictions, ground_truth)

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

    print("-" * 80 + "\n")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'evaluation_results.png')
        plot_evaluation_results(predictions, ground_truth, metrics, plot_path)

        # Save metrics to file
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INVERSE DYNAMICS MODEL EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  MSE:  {metrics['mse']:.6f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MAE:  {metrics['mae']:.6f}\n")
            f.write(f"  R²:   {metrics['r2']:.6f}\n")
            f.write(f"  Mean Correlation: {metrics['mean_correlation']:.6f}\n\n")
            f.write("Per-Dimension MSE:\n")
            for i, mse in enumerate(metrics['mse_per_dim']):
                f.write(f"  Dim {i:2d}: {mse:.6f}\n")
            f.write("\nPer-Dimension Correlation:\n")
            for i, corr in enumerate(metrics['correlations']):
                f.write(f"  Dim {i:2d}: {corr:.4f}\n")

        print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained inverse dynamics model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file (used for model architecture)')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to evaluation dataset HDF5 file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results (default: same directory as checkpoint)')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes to evaluate (default: all episodes)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation (default: 256)')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    data_config = config.get('data', {})

    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_dir = os.path.join(output_dir, f'eval_{checkpoint_name}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("INVERSE DYNAMICS MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {train_config.get('device', 'cuda')}")
    print("=" * 80 + "\n")

    states, actions, next_states = load_trajectory_dataset(
        args.dataset_path,
        filter_success_only=data_config.get('filter_success_only', False),
        min_reward_threshold=data_config.get('min_reward_threshold', None),
        num_episodes=args.num_episodes
    )

    print(f"Building inverse dynamics model from config...\n")
    target_path = model_config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    model_params = {k: v for k, v in model_config.items() if k in init_params}
    model = model_class(**model_params)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!\n")

    eval_config = {
        'batch_size': args.batch_size,
        'device': train_config.get('device', 'cuda'),
        'normalize_data': train_config.get('normalize_data', True),
        'state_mean': None,
        'state_std': None,
        'action_mean': None,
        'action_std': None,
    }

    metrics = evaluate_inverse_dynamics(
        states, actions, next_states, model,
        eval_config=eval_config,
        output_dir=output_dir
    )

    print(f"Evaluation results saved to: {output_dir}")
    print("Evaluation completed successfully!\n")


if __name__ == "__main__":
    main()
