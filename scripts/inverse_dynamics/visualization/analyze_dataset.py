"""Analyze and visualize dataset statistics.

Usage:
    python analyze_dataset.py --data_path <path> --max_episodes 10
"""

import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def analyze_episode_stats(data_path: str, max_episodes: int = None, key: str = 'seg_pc'):
    """Analyze statistics across episodes.

    Args:
        data_path: Path to directory containing episode_N.zarr folders
        max_episodes: Maximum number of episodes to analyze
        key: Pointcloud key to analyze

    Returns:
        stats: Dictionary containing various statistics
    """
    all_files = os.listdir(data_path)
    episode_dirs = sorted(
        [f for f in all_files if f.startswith('episode_') and f.endswith('.zarr')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    if max_episodes:
        episode_dirs = episode_dirs[:max_episodes]

    print(f"Analyzing {len(episode_dirs)} episodes...")

    episode_lengths = []
    num_points_per_episode = []
    point_counts_distribution = []

    for ep_dir in tqdm(episode_dirs, desc="Loading episodes"):
        zarr_path = os.path.join(data_path, ep_dir)

        try:
            data = zarr.open(zarr_path, mode='r')

            # Check actions to get episode length
            actions = np.array(data['data/actions'])
            episode_lengths.append(len(actions))

            # Load pointcloud
            pcd_data = np.array(data[f'data/{key}'])  # (T, N, 3)

            # Count points per timestep
            for t in range(len(pcd_data)):
                num_points = pcd_data[t].shape[0]
                point_counts_distribution.append(num_points)

            # Average points for this episode
            avg_points = np.mean([pcd_data[t].shape[0] for t in range(len(pcd_data))])
            num_points_per_episode.append(avg_points)

        except Exception as e:
            print(f"Error loading {zarr_path}: {e}")
            continue

    stats = {
        'episode_lengths': np.array(episode_lengths),
        'num_points_per_episode': np.array(num_points_per_episode),
        'point_counts_distribution': np.array(point_counts_distribution),
        'num_episodes': len(episode_dirs)
    }

    return stats


def plot_stats(stats: dict, save_path: str = None):
    """Plot dataset statistics.

    Args:
        stats: Statistics dictionary from analyze_episode_stats
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode lengths
    ax = axes[0, 0]
    ax.hist(stats['episode_lengths'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Episode Length (timesteps)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.axvline(np.mean(stats['episode_lengths']), color='r', linestyle='--',
              label=f'Mean: {np.mean(stats["episode_lengths"]):.1f}')
    ax.legend()

    # Average points per episode
    ax = axes[0, 1]
    ax.plot(stats['num_points_per_episode'], marker='o', markersize=3, linestyle='-', alpha=0.6)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Average Points per Timestep')
    ax.set_title('Average Points per Episode')
    ax.axhline(np.mean(stats['num_points_per_episode']), color='r', linestyle='--',
              label=f'Mean: {np.mean(stats["num_points_per_episode"]):.0f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Point count distribution (histogram)
    ax = axes[1, 0]
    ax.hist(stats['point_counts_distribution'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Count (timesteps)')
    ax.set_title('Point Count Distribution Across All Timesteps')
    ax.axvline(np.mean(stats['point_counts_distribution']), color='r', linestyle='--',
              label=f'Mean: {np.mean(stats["point_counts_distribution"]):.0f}')
    ax.axvline(np.median(stats['point_counts_distribution']), color='g', linestyle='--',
              label=f'Median: {np.median(stats["point_counts_distribution"]):.0f}')
    ax.legend()

    # Box plot of point counts
    ax = axes[1, 1]
    ax.boxplot(stats['point_counts_distribution'], vert=True)
    ax.set_ylabel('Number of Points')
    ax.set_title('Point Count Statistics')
    ax.grid(True, alpha=0.3, axis='y')

    # Add text with summary stats
    summary_text = f"""
    Episodes: {stats['num_episodes']}
    Total timesteps: {len(stats['point_counts_distribution']):,}

    Points per timestep:
      Min: {np.min(stats['point_counts_distribution']):,}
      Max: {np.max(stats['point_counts_distribution']):,}
      Mean: {np.mean(stats['point_counts_distribution']):.0f}
      Std: {np.std(stats['point_counts_distribution']):.0f}
    """
    ax.text(1.5, np.median(stats['point_counts_distribution']), summary_text,
           fontsize=9, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics plot to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Episodes analyzed: {stats['num_episodes']}")
    print(f"Total timesteps: {len(stats['point_counts_distribution']):,}")
    print(f"\nEpisode lengths:")
    print(f"  Min: {np.min(stats['episode_lengths'])}")
    print(f"  Max: {np.max(stats['episode_lengths'])}")
    print(f"  Mean: {np.mean(stats['episode_lengths']):.1f} ± {np.std(stats['episode_lengths']):.1f}")
    print(f"\nPoints per timestep:")
    print(f"  Min: {np.min(stats['point_counts_distribution']):,}")
    print(f"  Max: {np.max(stats['point_counts_distribution']):,}")
    print(f"  Mean: {np.mean(stats['point_counts_distribution']):.0f} ± {np.std(stats['point_counts_distribution']):.0f}")
    print(f"  Median: {np.median(stats['point_counts_distribution']):.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to directory containing episode_N.zarr folders')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum number of episodes to analyze (default: all)')
    parser.add_argument('--key', type=str, default='seg_pc',
                       help='Pointcloud key to analyze (default: seg_pc)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save path for statistics plot (optional)')

    args = parser.parse_args()

    stats = analyze_episode_stats(args.data_path, args.max_episodes, args.key)
    plot_stats(stats, args.save)


if __name__ == '__main__':
    main()
