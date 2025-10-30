"""
Process raw trajectory datasets into flat format for BC training.

Takes raw HDF5 datasets (with demo_X structure) and converts them to flat format:
- Extracts 77D Markovian states using StateExtractor
- Concatenates all demos into single arrays
- Saves as states, next_states, actions for BC training

Example usage:
    python scripts/data_collection/process_trajectories_for_bc.py \
        --input trajectory_data/trajectories_train.hdf5 \
        --output trajectory_data/bc_dataset_train.hdf5
"""

import argparse
import h5py
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.state_extraction import StateExtractor


def process_dataset(input_path: str, output_path: str):
    """Process raw trajectory dataset into flat BC training format.

    Args:
        input_path: Path to raw trajectory HDF5 file
        output_path: Path to save processed flat HDF5 file
    """
    print("=" * 80)
    print("PROCESSING TRAJECTORIES FOR BC TRAINING")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    extractor = StateExtractor()

    all_states = []
    all_next_states = []
    all_actions = []

    with h5py.File(input_path, 'r') as f:
        # Get all demo keys
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
        print(f"Found {len(demo_keys)} demos")

        for demo_key in demo_keys:
            demo = f['data'][demo_key]

            try:
                # Extract states using StateExtractor
                states, actions = extractor.extract_from_dataset(demo)

                # Create next_states (shift by 1)
                next_states = np.roll(states, -1, axis=0)
                # Last next_state is invalid, remove it
                states = states[:-1]
                next_states = next_states[:-1]
                actions = actions[:-1]

                all_states.append(states)
                all_next_states.append(next_states)
                all_actions.append(actions)

            except Exception as e:
                print(f"Warning: Failed to process {demo_key}: {e}")
                continue

        print(f"Successfully processed {len(all_states)} demos")

    # Concatenate all demos
    all_states = np.concatenate(all_states, axis=0)
    all_next_states = np.concatenate(all_next_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    print()
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total transitions: {len(all_states)}")
    print(f"State dimension:   {all_states.shape[1]}D")
    print(f"Action dimension:  {all_actions.shape[1]}D")
    print()
    print(f"States shape:      {all_states.shape}")
    print(f"Next states shape: {all_next_states.shape}")
    print(f"Actions shape:     {all_actions.shape}")
    print("=" * 80)
    print()

    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('next_states', data=all_next_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')

        # Add metadata attributes
        f.attrs['num_samples'] = len(all_states)
        f.attrs['state_dim'] = all_states.shape[1]
        f.attrs['action_dim'] = all_actions.shape[1]

    print("Done!")
    print()


def main():
    parser = argparse.ArgumentParser(description='Process raw trajectories for BC training')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw trajectory HDF5 file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save processed HDF5 file (default: input_bc.hdf5)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_bc{input_path.suffix}")

    # Process dataset
    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()