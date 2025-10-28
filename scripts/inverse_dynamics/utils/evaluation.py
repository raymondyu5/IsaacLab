"""Evaluation utilities for inverse dynamics models."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def compute_evaluation_metrics(predictions, ground_truth):
    """Compute evaluation metrics comparing predictions to ground truth."""
    errors = predictions - ground_truth

    # Overall metrics
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # Per-dimension
    mse_per_dim = np.mean(errors ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    mae_per_dim = np.mean(np.abs(errors), axis=0)

    # Correlations
    correlations = []
    for i in range(predictions.shape[1]):
        if np.std(predictions[:, i]) > 1e-8 and np.std(ground_truth[:, i]) > 1e-8:
            corr, _ = pearsonr(predictions[:, i], ground_truth[:, i])
            correlations.append(corr)
        else:
            correlations.append(0.0)
    correlations = np.array(correlations)

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    # Normalized metrics (for interpretability)
    action_range = ground_truth.max(axis=0) - ground_truth.min(axis=0)
    normalized_rmse = rmse_per_dim / (action_range + 1e-8)
    normalized_mae = mae_per_dim / (action_range + 1e-8)

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'mse_per_dim': mse_per_dim, 'rmse_per_dim': rmse_per_dim, 'mae_per_dim': mae_per_dim,
        'normalized_rmse': normalized_rmse, 'normalized_mae': normalized_mae,
        'correlations': correlations, 'mean_correlation': np.mean(correlations),
    }


def plot_evaluation_results(predictions, ground_truth, metrics, output_path=None):
    """Create visualization plots for evaluation results."""
    action_dim = predictions.shape[1]
    fig = plt.figure(figsize=(20, 12))

    # 1. Normalized RMSE
    plt.subplot(3, 3, 1)
    plt.bar(range(action_dim), metrics['normalized_rmse'] * 100)
    plt.xlabel('Action Dimension')
    plt.ylabel('RMSE (% of range)')
    plt.title('Normalized RMSE')
    plt.xticks(range(action_dim))
    plt.grid(True, alpha=0.3)

    # 2. Normalized MAE
    plt.subplot(3, 3, 2)
    plt.bar(range(action_dim), metrics['normalized_mae'] * 100)
    plt.xlabel('Action Dimension')
    plt.ylabel('MAE (% of range)')
    plt.title('Normalized MAE')
    plt.xticks(range(action_dim))
    plt.grid(True, alpha=0.3)

    # 3. Correlations
    plt.subplot(3, 3, 3)
    plt.bar(range(action_dim), metrics['correlations'])
    plt.xlabel('Action Dimension')
    plt.ylabel('Pearson Correlation')
    plt.title('Per-Dimension Correlation')
    plt.xticks(range(action_dim))
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.3)

    # 4-9. Scatter plots for first 6 dimensions
    for i in range(min(6, action_dim)):
        plt.subplot(3, 3, 4 + i)
        plt.scatter(ground_truth[:, i], predictions[:, i], alpha=0.3, s=1)
        min_val = min(ground_truth[:, i].min(), predictions[:, i].min())
        max_val = max(ground_truth[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel(f'Ground Truth Dim {i}')
        plt.ylabel(f'Predicted Dim {i}')
        plt.title(f'Dim {i} (corr={metrics["correlations"][i]:.3f})')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()
