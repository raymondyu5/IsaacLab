"""
Script to split trajectory data into separate datasets by action dimensions.

This script reads an HDF5 trajectory file and splits the actions into multiple
separate HDF5 files based on specified action dimension ranges. This is useful
for training separate policies for different parts of a robot (e.g., arm vs hand).

Usage:
    # Split Kuka-Allegro data into arm (0-6) and hand (7-22) actions
    python split_trajectory_data.py \\
        --input trajectory_data.h5 \\
        --arm_indices 0 6 \\
        --hand_indices 7 22

    # Custom split with different names
    python split_trajectory_data.py \\
        --input trajectory_data.h5 \\
        --splits arm:0:6 hand:7:22 gripper:23:24

    # Verify split without creating files
    python split_trajectory_data.py \\
        --input trajectory_data.h5 \\
        --arm_indices 0 6 \\
        --hand_indices 7 22 \\
        --dry_run
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def parse_split_arg(split_str):
    """Parse a split argument in the format 'name:start:end'.

    Args:
        split_str: String in format "name:start:end" or "name:start:end:inclusive"

    Returns:
        tuple: (name, start_idx, end_idx) where end_idx is exclusive
    """
    parts = split_str.split(':')
    if len(parts) < 3:
        raise ValueError(f"Split argument must be in format 'name:start:end', got: {split_str}")

    name = parts[0]
    start = int(parts[1])
    end = int(parts[2])

    # Handle inclusive flag (optional 4th parameter)
    if len(parts) == 4 and parts[3].lower() == 'inclusive':
        end += 1  # Make end exclusive

    return name, start, end


def get_split_configs(args):
    """Build split configurations from command-line arguments.

    Returns:
        dict: {split_name: (start_idx, end_idx)} where end_idx is exclusive
    """
    splits = {}

    # Parse --splits arguments
    if args.splits:
        for split_str in args.splits:
            name, start, end = parse_split_arg(split_str)
            splits[name] = (start, end)

    # Parse legacy --arm_indices and --hand_indices arguments
    if args.arm_indices:
        if len(args.arm_indices) != 2:
            raise ValueError("--arm_indices requires exactly 2 values: start and end")
        splits['arm'] = (args.arm_indices[0], args.arm_indices[1] + 1)  # Make exclusive

    if args.hand_indices:
        if len(args.hand_indices) != 2:
            raise ValueError("--hand_indices requires exactly 2 values: start and end")
        splits['hand'] = (args.hand_indices[0], args.hand_indices[1] + 1)  # Make exclusive

    if not splits:
        raise ValueError("No splits specified. Use --splits, --arm_indices, or --hand_indices")

    return splits


def validate_splits(splits, total_action_dim):
    """Validate that split indices are within bounds and don't overlap.

    Args:
        splits: dict of {name: (start, end)}
        total_action_dim: Total number of action dimensions
    """
    print("\nValidating splits...")

    for name, (start, end) in splits.items():
        if start < 0 or end > total_action_dim:
            raise ValueError(
                f"Split '{name}' has indices [{start}:{end}) outside valid range [0:{total_action_dim})"
            )
        if start >= end:
            raise ValueError(f"Split '{name}' has invalid range: start={start} >= end={end}")

        print(f"  ✓ {name}: indices [{start}:{end}) = {end - start} dimensions")

    # Check for overlaps
    all_indices = set()
    for name, (start, end) in splits.items():
        indices = set(range(start, end))
        overlap = all_indices & indices
        if overlap:
            raise ValueError(f"Split '{name}' overlaps with previous splits at indices: {sorted(overlap)}")
        all_indices.update(indices)

    print(f"  ✓ No overlaps detected")
    print(f"  ✓ Covering {len(all_indices)}/{total_action_dim} total action dimensions")


def analyze_dataset(input_path):
    """Analyze the input dataset structure and dimensions.

    Returns:
        dict: Dataset statistics and format information
    """
    with h5py.File(input_path, 'r') as f:
        # Try to detect dataset format
        # Format 1: episode_0, episode_1, ... (direct episodes)
        # Format 2: data/demo_0, data/demo_1, ... (robomimic-style)

        episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

        # Check for robomimic-style format: data/demo_X
        if not episode_keys and 'data' in f:
            data_group = f['data']
            episode_keys = sorted([f'data/{k}' for k in data_group.keys() if k.startswith('demo_')])
            is_robomimic_format = True
        else:
            is_robomimic_format = False

        if not episode_keys:
            raise ValueError(
                f"No episodes found in {input_path}. "
                f"Expected format: 'episode_X' or 'data/demo_X' groups. "
                f"Found keys: {list(f.keys())}"
            )

        # Get dimensions from first episode
        first_ep = f[episode_keys[0]]

        # Handle different action field names
        action_key = 'actions' if 'actions' in first_ep else 'processed_actions'
        obs_key = 'observations' if 'observations' in first_ep else 'obs'

        actions_shape = first_ep[action_key].shape
        obs_shape = first_ep[obs_key].shape if obs_key in first_ep else (0, 0)

        stats = {
            'num_episodes': len(episode_keys),
            'action_dim': actions_shape[1],
            'obs_dim': obs_shape[1] if len(obs_shape) > 1 else obs_shape[0],
            'avg_episode_length': np.mean([len(f[k][action_key]) for k in episode_keys]),
            'episode_keys': episode_keys,
            'is_robomimic_format': is_robomimic_format,
            'action_key': action_key,
            'obs_key': obs_key,
        }

        return stats


def split_trajectory_data(input_path, output_dir, splits):
    """Split trajectory data into separate files based on action dimensions.

    Args:
        input_path: Path to input HDF5 file
        output_dir: Directory to save split files
        splits: dict of {split_name: (start_idx, end_idx)}
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"Splitting Trajectory Data")
    print(f"{'='*60}")
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")

    # Analyze input dataset
    print(f"\nAnalyzing input dataset...")
    stats = analyze_dataset(input_path)
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Action dimensions: {stats['action_dim']}")
    print(f"  Observation dimensions: {stats['obs_dim']}")
    print(f"  Avg episode length: {stats['avg_episode_length']:.1f} steps")

    # Validate splits
    validate_splits(splits, stats['action_dim'])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open input file
    with h5py.File(input_path, 'r') as input_file:

        # Create output files
        output_files = {}
        for name in splits.keys():
            output_path = output_dir / f"trajectory_{name}_data.h5"
            output_files[name] = h5py.File(output_path, 'w')
            print(f"\nCreating: {output_path}")

        try:
            # Process each episode
            print(f"\nProcessing {stats['num_episodes']} episodes...")
            action_key = stats['action_key']
            obs_key = stats['obs_key']

            for i, episode_key in enumerate(stats['episode_keys'], 1):
                input_episode = input_file[episode_key]

                # Get full data (use detected keys)
                actions = input_episode[action_key][:]
                observations = input_episode[obs_key][:] if obs_key in input_episode else None
                rewards = input_episode['rewards'][:] if 'rewards' in input_episode else None

                # Handle dones/terminals
                if 'dones' in input_episode:
                    dones = input_episode['dones'][:]
                elif 'terminals' in input_episode:
                    dones = input_episode['terminals'][:]
                else:
                    # Create synthetic dones (mark last step as done)
                    dones = np.zeros(len(actions), dtype=bool)
                    dones[-1] = True

                # Split and save to each output file
                for split_name, (start, end) in splits.items():
                    # Determine output episode key (strip 'data/' prefix if robomimic format)
                    if stats['is_robomimic_format']:
                        # For robomimic format, preserve the data/demo_X structure
                        if not episode_key.startswith('data/'):
                            output_key = episode_key
                        else:
                            # Create nested structure: data/demo_X
                            demo_name = episode_key.split('/')[-1]
                            if 'data' not in output_files[split_name]:
                                output_files[split_name].create_group('data')
                            output_episode = output_files[split_name]['data'].create_group(demo_name)
                    else:
                        # Direct episode format
                        output_episode = output_files[split_name].create_group(episode_key)

                    # Save split actions
                    split_actions = actions[:, start:end]
                    output_episode.create_dataset('actions', data=split_actions, compression='gzip')

                    # Copy observations, rewards, dones if they exist
                    if observations is not None:
                        output_episode.create_dataset('observations', data=observations, compression='gzip')
                    if rewards is not None:
                        output_episode.create_dataset('rewards', data=rewards, compression='gzip')
                    output_episode.create_dataset('dones', data=dones, compression='gzip')

                    # Copy additional data structures (like states, initial_state)
                    for key in input_episode.keys():
                        if key not in [action_key, obs_key, 'rewards', 'dones', 'terminals']:
                            # Recursively copy groups
                            if isinstance(input_episode[key], h5py.Group):
                                def copy_group(src, dst, name):
                                    if name not in dst:
                                        dst.create_group(name)
                                    for k in src[name].keys():
                                        if isinstance(src[name][k], h5py.Group):
                                            copy_group(src[name], dst[name], k)
                                        else:
                                            dst[name].create_dataset(k, data=src[name][k][:])
                                copy_group(input_episode, output_episode, key)
                            else:
                                # Copy dataset
                                output_episode.create_dataset(key, data=input_episode[key][:])

                # Progress indicator
                if i % 10 == 0 or i == stats['num_episodes']:
                    print(f"  Processed {i}/{stats['num_episodes']} episodes", end='\r')

            print(f"\n  ✓ All episodes processed successfully")

        finally:
            # Close all output files
            for name, f in output_files.items():
                f.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Splitting Complete!")
    print(f"{'='*60}")
    for split_name, (start, end) in splits.items():
        output_path = output_dir / f"trajectory_{name}_data.h5"
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  {split_name}: {output_path}")
        print(f"    - Action dims: [{start}:{end}) = {end - start} DOFs")
        print(f"    - File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Split trajectory data by action dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split Kuka-Allegro data (arm: 0-6, hand: 7-22)
  python split_trajectory_data.py \\
      --input trajectory_data.h5 \\
      --arm_indices 0 6 \\
      --hand_indices 7 22

  # Custom splits using --splits argument
  python split_trajectory_data.py \\
      --input trajectory_data.h5 \\
      --splits arm:0:7 hand:7:23

  # Multiple custom splits
  python split_trajectory_data.py \\
      --input trajectory_data.h5 \\
      --splits left_arm:0:7 right_arm:7:14 torso:14:17

  # Dry run to preview without creating files
  python split_trajectory_data.py \\
      --input trajectory_data.h5 \\
      --arm_indices 0 6 \\
      --hand_indices 7 22 \\
      --dry_run
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input HDF5 trajectory file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save split files (default: same as input file directory)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        help='Split specifications in format "name:start:end" (end is exclusive). '
             'Example: --splits arm:0:7 hand:7:23'
    )

    parser.add_argument(
        '--arm_indices',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Arm action indices (inclusive). Example: --arm_indices 0 6'
    )

    parser.add_argument(
        '--hand_indices',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Hand action indices (inclusive). Example: --hand_indices 7 22'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be done without actually creating files'
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(args.input).parent

    # Get split configurations
    splits = get_split_configs(args)

    # Run the split
    split_trajectory_data(
        input_path=args.input,
        output_dir=args.output_dir,
        splits=splits,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
