"""Data loading utilities for inverse dynamics training and evaluation."""

import h5py
import numpy as np

from .states import extract_states_from_demo


def load_trajectory_dataset(dataset_path, filter_success_only=False, min_reward_threshold=None, num_episodes=None):
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

    Returns:
        states, actions, next_states as numpy arrays
    """
    print("=" * 80)
    print("LOADING TRAJECTORY DATASET")
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

            # Extract states from policy_data format
            states_full, actions_full = extract_states_from_demo(ep_group)

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
