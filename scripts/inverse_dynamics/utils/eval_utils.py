"""Evaluation utilities for inverse dynamics models."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def compute_evaluation_metrics(predictions, ground_truth):
    """
    Compute various evaluation metrics comparing predictions to ground truth.

    Args:
        predictions: Predicted actions (N, action_dim)
        ground_truth: Ground truth actions (N, action_dim)

    Returns:
        Dictionary containing various metrics
    """
    # Overall metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truth))

    # Per-dimension metrics
    mse_per_dim = np.mean((predictions - ground_truth) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(predictions - ground_truth), axis=0)

    # Correlation per dimension
    correlations = []
    for i in range(predictions.shape[1]):
        if np.std(predictions[:, i]) > 1e-8 and np.std(ground_truth[:, i]) > 1e-8:
            corr, _ = pearsonr(predictions[:, i], ground_truth[:, i])
            correlations.append(corr)
        else:
            correlations.append(0.0)
    correlations = np.array(correlations)

    # R-squared (coefficient of determination)
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_per_dim': mse_per_dim,
        'mae_per_dim': mae_per_dim,
        'correlations': correlations,
        'mean_correlation': np.mean(correlations)
    }

    return metrics


def plot_evaluation_results(predictions, ground_truth, metrics, output_path=None, additional_dims=None):
    """
    Create visualization plots for evaluation results.

    Args:
        predictions: Predicted actions (N, action_dim)
        ground_truth: Ground truth actions (N, action_dim)
        metrics: Dictionary of computed metrics
        output_path: Path to save the plot (if None, displays instead)
        additional_dims: List of additional dimension indices to plot separately (e.g., [14, 16, 22])
    """
    action_dim = predictions.shape[1]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Per-dimension MSE
    plt.subplot(3, 3, 1)
    plt.bar(range(action_dim), metrics['mse_per_dim'])
    plt.xlabel('Action Dimension')
    plt.ylabel('MSE')
    plt.title('Per-Dimension MSE')
    plt.xticks(range(action_dim))
    plt.grid(True, alpha=0.3)

    # 2. Per-dimension MAE
    plt.subplot(3, 3, 2)
    plt.bar(range(action_dim), metrics['mae_per_dim'])
    plt.xlabel('Action Dimension')
    plt.ylabel('MAE')
    plt.title('Per-Dimension MAE')
    plt.xticks(range(action_dim))
    plt.grid(True, alpha=0.3)

    # 3. Per-dimension correlation
    plt.subplot(3, 3, 3)
    plt.bar(range(action_dim), metrics['correlations'])
    plt.xlabel('Action Dimension')
    plt.ylabel('Pearson Correlation')
    plt.title('Per-Dimension Correlation')
    plt.xticks(range(action_dim))
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.3)

    # 4-9. Scatter plots for first 6 dimensions
    num_scatter = min(6, action_dim)
    for i in range(num_scatter):
        plt.subplot(3, 3, 4 + i)
        plt.scatter(ground_truth[:, i], predictions[:, i], alpha=0.3, s=1)

        # Add y=x line
        min_val = min(ground_truth[:, i].min(), predictions[:, i].min())
        max_val = max(ground_truth[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        plt.xlabel(f'Ground Truth Dim {i}')
        plt.ylabel(f'Predicted Dim {i}')
        plt.title(f'Dimension {i} (corr={metrics["correlations"][i]:.3f})')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()

    # Create additional plot for user-specified dimensions
    if additional_dims is not None and len(additional_dims) > 0:
        # Filter to only valid dimensions
        available_dims = [d for d in additional_dims if d < action_dim]

        if len(available_dims) > 0:
            # Calculate grid layout
            n_dims = len(available_dims)
            n_cols = min(3, n_dims)
            n_rows = (n_dims + n_cols - 1) // n_cols

            fig2 = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

            for idx, dim in enumerate(available_dims):
                plt.subplot(n_rows, n_cols, idx + 1)
                plt.scatter(ground_truth[:, dim], predictions[:, dim], alpha=0.3, s=1)

                # Add y=x line
                min_val = min(ground_truth[:, dim].min(), predictions[:, dim].min())
                max_val = max(ground_truth[:, dim].max(), predictions[:, dim].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

                plt.xlabel(f'Ground Truth Dim {dim}')
                plt.ylabel(f'Predicted Dim {dim}')
                plt.title(f'Dimension {dim}\ncorr={metrics["correlations"][dim]:.3f}, MSE={metrics["mse_per_dim"][dim]:.6f}')
                plt.grid(True, alpha=0.3)
                plt.legend()

            plt.tight_layout()

            if output_path is not None:
                additional_path = output_path.replace('.png', '_additional_dims.png')
                plt.savefig(additional_path, dpi=150, bbox_inches='tight')
                print(f"Additional dimensions plot saved to: {additional_path}")
            else:
                plt.show()

            plt.close()
