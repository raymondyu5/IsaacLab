"""Data loading utilities for trajectory-based datasets (ID/BC/AT training)."""

import os
import random

import h5py
import numpy as np
import torch

from .state_extraction import extract_from_dataset


def load_trajectory_dataset(dataset_path, filter_success_only=False, min_reward_threshold=None, num_episodes=None, use_observations=False):
    """
    Load trajectory dataset from HDF5 file and convert to transitions.

    Loads episode-based trajectory data collected by collect_trajectory_data.py
    and converts it into state-action-next_state transitions for inverse dynamics training.
    Automatically filters terminal transitions (last step of each episode).

    Expected format: policy_data structure with instantaneous states
    - demo['policy_data']['state']: (T, state_dim) instantaneous positional states
    - demo['policy_data']['obs']: (T, obs_dim) full observations (for action translator)
    - demo['actions']: (T, action_dim) actions

    Args:
        dataset_path: Path to HDF5 trajectory dataset file
        filter_success_only: If True, only load episodes marked as successful
        min_reward_threshold: If provided, only load episodes with total reward >= threshold
        num_episodes: If provided, only load the first N episodes (after filtering)
        use_observations: If True, return 165D observations instead of 90D states (for action translation)

    Returns:
        states, actions, next_states as numpy arrays (or observations if use_observations=True)
    """
    print("=" * 80)
    print("LOADING TRAJECTORY DATASET")
    if use_observations:
        print("(USING FULL 165D OBSERVATIONS FOR ACTION TRANSLATION)")
    print("=" * 80)

    all_states = []
    all_actions = []
    all_next_states = []
    total_episodes = 0
    loaded_episodes = 0
    filtered_episodes = 0
    total_transitions = 0
    filtered_terminals = 0

    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']

        # Find all episodes (demo_0, demo_1, ...)
        episode_names = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                               key=lambda x: int(x.split('_')[1]))
        total_episodes = len(episode_names)

        print(f"\nFound {total_episodes} episodes in dataset")
        print(f"Dataset path: {dataset_path}")
        print(f"Expected format: policy_data with instantaneous states")

        if num_episodes is not None:
            print(f"Limiting to first {num_episodes} episodes (after filtering)")

        for ep_name in episode_names:
            # Check if we've loaded enough episodes
            if num_episodes is not None and loaded_episodes >= num_episodes:
                break

            ep_group = data_group[ep_name]

            # Check filtering criteria
            should_load = True

            if filter_success_only:
                success = ep_group.attrs.get('success', False)
                if not success:
                    should_load = False
                    filtered_episodes += 1

            if min_reward_threshold is not None and 'rewards' in ep_group:
                total_reward = np.sum(ep_group['rewards'][:])
                if total_reward < min_reward_threshold:
                    should_load = False
                    filtered_episodes += 1

            if not should_load:
                continue

            # Extract states or observations from policy_data format
            if use_observations:
                # Extract full 165D observations
                if 'policy_data' not in ep_group or 'obs' not in ep_group['policy_data']:
                    raise KeyError(f"Expected 'policy_data/obs' in {ep_name}")
                states_full = np.array(ep_group['policy_data']['obs'])
                actions_full = np.array(ep_group['actions'])
            else:
                # Extract 90D Markovian states
                states_full, actions_full = extract_from_dataset(ep_group)

            # Create transitions: (s_t, a_t, s_{t+1})
            if len(states_full) > 1:
                states = states_full[:-1]       # s_0 to s_{T-1}
                next_states = states_full[1:]   # s_1 to s_T
                ep_actions = actions_full[:-1]  # a_0 to a_{T-1}

                all_states.append(states)
                all_actions.append(ep_actions)
                all_next_states.append(next_states)

                total_transitions += len(states)
                filtered_terminals += 1
                loaded_episodes += 1

    if len(all_states) == 0:
        raise ValueError("No valid episodes found in dataset after filtering!")

    # Concatenate all episodes
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    next_states = np.concatenate(all_next_states, axis=0)

    obs_dim = states.shape[1]
    action_dim = actions.shape[1]

    print(f"\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total episodes in file: {total_episodes}")
    print(f"Loaded episodes: {loaded_episodes}")
    print(f"Filtered episodes: {filtered_episodes}")
    print(f"Valid transitions: {total_transitions:,}")
    print(f"Filtered terminal transitions: {filtered_terminals}")
    print(f"\nDimensions:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"\nArray shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Next states: {next_states.shape}")
    print("=" * 80 + "\n")

    return states, actions, next_states


def truncate_episodes(input_file: str, max_steps: int) -> str:
    """
    Truncate episodes in HDF5 file to maximum number of steps.

    Args:
        input_file: Path to input HDF5 file
        max_steps: Maximum steps to keep per episode

    Returns:
        Path to truncated file
    """
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_truncated{max_steps}.hdf5"

    print(f"Truncating episodes to {max_steps} steps...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    with h5py.File(input_file, 'r') as f_in:
        with h5py.File(output_file, 'w') as f_out:
            # Copy attributes
            for attr_name in f_in.attrs:
                f_out.attrs[attr_name] = f_in.attrs[attr_name]

            # Create data group
            if 'data' in f_in:
                data_in = f_in['data']
                data_out = f_out.create_group('data')

                total_episodes = 0
                total_steps_original = 0
                total_steps_truncated = 0

                # Process each episode
                for ep_key in data_in:
                    if not ep_key.startswith('demo_'):
                        continue

                    episode = data_in[ep_key]

                    # Get episode length
                    if 'actions' in episode:
                        ep_length = len(episode['actions'])
                    else:
                        continue

                    total_steps_original += ep_length

                    # Create new episode in output
                    new_episode = data_out.create_group(ep_key)

                    # Copy data, truncating to max_steps
                    for dataset_name in episode.keys():
                        item = episode[dataset_name]

                        if isinstance(item, h5py.Group):
                            # Handle groups (like initial_state, states, policy_data)
                            new_group = new_episode.create_group(dataset_name)

                            # Recursively copy group contents
                            def copy_group_contents(src_group, dst_group, truncate=True):
                                # Copy attributes
                                for attr_name in src_group.attrs:
                                    dst_group.attrs[attr_name] = src_group.attrs[attr_name]

                                for key in src_group.keys():
                                    if isinstance(src_group[key], h5py.Dataset):
                                        data = src_group[key][:]
                                        # Only truncate if it's time-series data and truncate flag is True
                                        if truncate and dataset_name != 'initial_state' and len(data.shape) > 0 and len(data) > max_steps:
                                            dst_group.create_dataset(key, data=data[:max_steps])
                                        else:
                                            dst_group.create_dataset(key, data=data)
                                    elif isinstance(src_group[key], h5py.Group):
                                        # Nested group
                                        sub_group = dst_group.create_group(key)
                                        copy_group_contents(src_group[key], sub_group, truncate)

                            # Copy the group recursively
                            copy_group_contents(item, new_group, truncate=(dataset_name != 'initial_state'))

                        elif isinstance(item, h5py.Dataset):
                            # Handle regular datasets (like actions, rewards)
                            data = item[:]

                            if len(data.shape) == 0:
                                # Scalar data - keep as is
                                new_episode.create_dataset(dataset_name, data=data)
                            elif len(data) > max_steps:
                                # Truncate time-series data
                                new_episode.create_dataset(dataset_name, data=data[:max_steps])
                                if dataset_name == 'actions':
                                    total_steps_truncated += max_steps
                            else:
                                # Keep all if shorter than max_steps
                                new_episode.create_dataset(dataset_name, data=data)
                                if dataset_name == 'actions':
                                    total_steps_truncated += len(data)

                        else:
                            raise TypeError(f"Unknown type for '{dataset_name}' in episode '{ep_key}': {type(item)}")

                    total_episodes += 1

                print(f"  Processed {total_episodes} episodes")
                print(f"  Original total steps: {total_steps_original}")
                print(f"  Truncated total steps: {total_steps_truncated}")
                print(f"  Reduction: {(1 - total_steps_truncated/total_steps_original)*100:.1f}%")

    return output_file


def split_dataset_into_train_val_test(
    source_hdf5_path: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_mode: str = "sequential",
) -> dict:
    """
    Splits a single HDF5 dataset into train/val/test files.

    Each output file will contain:
    - Subset of episodes based on split ratios
    - Full environment metadata (identical across all splits)
    - All episode-level metadata (seed, env_id, success)

    Args:
        source_hdf5_path: Path to the source HDF5 dataset file
        output_dir: Directory to save the split files
        train_ratio: Fraction of episodes for training (e.g., 0.7)
        val_ratio: Fraction of episodes for validation (e.g., 0.15)
        test_ratio: Fraction of episodes for test (e.g., 0.15)
        split_mode: How to assign episodes - "sequential" or "random"

    Returns:
        Dictionary with statistics about the split
    """
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    import json

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not torch.isclose(torch.tensor(total_ratio), torch.tensor(1.0), atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Open source file
    source_handler = HDF5DatasetFileHandler()
    source_handler.open(source_hdf5_path, mode="r")

    episode_names = list(source_handler.get_episode_names())
    num_episodes = len(episode_names)

    if num_episodes == 0:
        print("[WARNING] Source dataset is empty. No splits created.")
        source_handler.close()
        return {}

    print(f"\n[INFO] Splitting {num_episodes} episodes into train/val/test...")
    print(f"  Ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"  Mode: {split_mode}")

    # Calculate split sizes
    num_train = int(num_episodes * train_ratio)
    num_val = int(num_episodes * val_ratio)
    num_test = num_episodes - num_train - num_val  # Remainder goes to test

    # Assign episodes to splits
    if split_mode == "sequential":
        train_episodes = episode_names[:num_train]
        val_episodes = episode_names[num_train : num_train + num_val]
        test_episodes = episode_names[num_train + num_val :]
    elif split_mode == "random":
        shuffled_episodes = episode_names.copy()
        random.shuffle(shuffled_episodes)
        train_episodes = shuffled_episodes[:num_train]
        val_episodes = shuffled_episodes[num_train : num_train + num_val]
        test_episodes = shuffled_episodes[num_train + num_val :]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # Create output filenames
    base_name = os.path.splitext(os.path.basename(source_hdf5_path))[0]
    train_path = os.path.join(output_dir, f"{base_name}_train.hdf5")
    val_path = os.path.join(output_dir, f"{base_name}_val.hdf5")
    test_path = os.path.join(output_dir, f"{base_name}_test.hdf5")

    # Get environment metadata from source
    with h5py.File(source_hdf5_path, "r") as src_file:
        env_args = src_file["data"].attrs.get("env_args", "{}")

    # Helper function to create split file
    def create_split_file(split_path, episode_list, split_name):
        if len(episode_list) == 0:
            print(f"[WARNING] No episodes for {split_name} split. Skipping file creation.")
            return {"total": 0, "successful": 0, "failed": 0}

        handler = HDF5DatasetFileHandler()
        handler.create(split_path.replace(".hdf5", ""))  # create() adds .hdf5 extension

        # Copy environment metadata
        handler._env_args = json.loads(env_args)
        handler._hdf5_data_group.attrs["env_args"] = env_args

        successful_count = 0
        failed_count = 0

        for ep_name in episode_list:
            episode = source_handler.load_episode(ep_name, device="cpu")
            if episode is not None:
                handler.write_episode(episode)
                if episode.success:
                    successful_count += 1
                else:
                    failed_count += 1

        handler.flush()
        handler.close()

        print(f"  {split_name}: {len(episode_list)} episodes ({successful_count} successful, {failed_count} failed)")
        print(f"    Saved to: {split_path}")

        return {"total": len(episode_list), "successful": successful_count, "failed": failed_count}

    # Create split files
    train_stats = create_split_file(train_path, train_episodes, "Train")
    val_stats = create_split_file(val_path, val_episodes, "Validation")
    test_stats = create_split_file(test_path, test_episodes, "Test")

    source_handler.close()

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
    }
