# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to replay trajectory data collected by collect_trajectory_data.py.

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/replay_trajectory_data.py     --dataset_file trajectory_data/trajectories_Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0_20251012_221227.hdf5     --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0     --num_envs 1     --video     --video_dir ./replay_videos     --headless --validate_states

"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay trajectory data from Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help="Validate if the states match between loaded from datasets and replayed. Only valid if --num_envs is 1.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record videos of replayed trajectories.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="./replay_videos",
    help="Directory to save recorded videos.",
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

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset (with batch dim from get_state).
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        if asset_type not in runtime_state:
            continue
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                # Dataset state comes with batch dimension [1, N] from get_state, so squeeze it
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name].squeeze(0)
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match: "
                        f"dataset={dataset_asset_state.shape}, runtime={runtime_asset_state.shape}"
                    )
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i].item() - runtime_asset_state[i].item()) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i].item():.6f}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i].item():.6f}\r\n"
    return states_matched, output_log


def main():
    """Replay episodes loaded from a file."""

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # Determine required number of environments based on stored env_ids
    # For MultiAssetSpawner to select correct objects, we need to create all environments up to max env_id
    episode_names = list(dataset_file_handler.get_episode_names())
    max_env_id = 0
    env_id_map = {}  # Maps episode index to env_id

    for idx in episode_indices_to_replay:
        if idx < len(episode_names):
            ep_data = dataset_file_handler.load_episode(episode_names[idx], args_cli.device)
            if ep_data.env_id is not None:
                env_id_map[idx] = ep_data.env_id
                max_env_id = max(max_env_id, ep_data.env_id)
            else:
                print("env_id not stored")
                env_id_map[idx] = 0  # Default to 0 if not stored

    # Create enough environments to include the highest env_id
    # This is required for MultiAssetSpawner determinism (uses env_id % num_assets)
    num_envs = max(args_cli.num_envs, max_env_id + 1)

    if num_envs != args_cli.num_envs:
        print(f"[INFO] Adjusted num_envs from {args_cli.num_envs} to {num_envs} for deterministic object spawning")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Set seed for deterministic object spawning (must match eval_id_sim)
    if not hasattr(env_cfg, 'seed') or env_cfg.seed is None:
        env_cfg.seed = 42
    print(f"[INFO] Environment seed: {env_cfg.seed}")

    # Disable all recorders for clean replay
    env_cfg.recorders = {}

    # Disable command resampling to prevent goal markers from flying around
    if hasattr(env_cfg, 'commands'):
        for attr_name in dir(env_cfg.commands):
            attr = getattr(env_cfg.commands, attr_name)
            # Check if it's a command config with resampling_time_range
            if hasattr(attr, 'resampling_time_range'):
                # Set resampling time to a very large value to effectively disable it
                attr.resampling_time_range = (1000000.0, 1000000.0)
                print(f"[INFO] Disabled command resampling for '{attr_name}'")
            # Disable debug visualization to reduce visual clutter
            if hasattr(attr, 'debug_vis'):
                attr.debug_vis = False

    # Remove rewards that depend on terminations before disabling terminations
    if hasattr(env_cfg, 'rewards'):
        # Scan through all reward terms and remove those that depend on terminations
        reward_terms_to_remove = []
        for attr_name in dir(env_cfg.rewards):
            attr = getattr(env_cfg.rewards, attr_name)
            # Check if it's a reward term config
            if hasattr(attr, 'func') and hasattr(attr, 'params'):
                # Check if params has 'term_keys' which indicates dependency on termination
                if isinstance(attr.params, dict) and 'term_keys' in attr.params:
                    reward_terms_to_remove.append(attr_name)

        # Remove identified reward terms
        for term_name in reward_terms_to_remove:
            print(f"[INFO] Removing reward term '{term_name}' (depends on terminations)")
            delattr(env_cfg.rewards, term_name)

    env_cfg.terminations = {}

    # Disable all randomization for deterministic replay
    print("\n[INFO] Configuring deterministic environment...")
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

    # Disable action noise for directrlenvs
    if hasattr(env_cfg, 'actions'):
        for action_name in dir(env_cfg.actions):
            if not action_name.startswith('_'):
                action = getattr(env_cfg.actions, action_name)
                if hasattr(action, 'noise'):
                    action.noise = None

    # Configure viewer to follow the correct environment
    if args_cli.video and len(episode_indices_to_replay) == 1:
        first_ep_idx = list(episode_indices_to_replay)[0]
        target_env_id = env_id_map.get(first_ep_idx, 0)
        if hasattr(env_cfg, 'viewer') and env_cfg.viewer is not None:
            env_cfg.viewer.env_index = target_env_id

    # Create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    # delete? not sure what this is for
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # Setup video recording directory
    if args_cli.video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        print(f"[INFO] Videos will be saved to: {args_cli.video_dir}")

    # Reset before starting
    env.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0

    with torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            env_episode_index_map = {index: None for index in range(num_envs)}  # Track which episode is in which env
            video_frames_map = {index: [] for index in range(num_envs)} if args_cli.video else None
            first_loop = True
            has_next_action = True

            while has_next_action:
                # Initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False

                # Process episodes by their original env_id to ensure correct object spawning
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()

                    if env_next_action is None:
                        # Find an episode that should be played in this env_id
                        next_episode_index = None

                        # First, try to find an episode that matches this env_id
                        episodes_to_check = episode_indices_to_replay.copy()
                        for idx in episodes_to_check:
                            if idx in env_id_map and env_id_map[idx] == env_id:
                                next_episode_index = idx
                                episode_indices_to_replay.remove(idx)
                                break

                        if next_episode_index is not None and next_episode_index < episode_count:
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count :4}: Loading episode #{next_episode_index} (env_id={env_id}) to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            env_episode_index_map[env_id] = next_episode_index  # Track which episode is loaded

                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)

                            # Verify initial state (only for first episode)
                            if replayed_episode_count == 1:
                                try:
                                    obj_pos_w = env.scene['object'].data.root_pos_w[env_id]
                                    env_origin = env.scene.env_origins[env_id]
                                    obj_pos_local = obj_pos_w - env_origin
                                    print(f"\nInitial state - Object local position: [{obj_pos_local[0]:.4f}, {obj_pos_local[1]:.4f}, {obj_pos_local[2]:.4f}]")
                                except (KeyError, AttributeError):
                                    pass

                            # Capture first frame for video
                            if args_cli.video:
                                frame = env.render()
                                if frame is not None and env_id < len(frame):
                                    video_frames_map[env_id] = [frame]

                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True

                    actions[env_id] = env_next_action

                if first_loop:
                    first_loop = False
                else:
                    # Step the environment
                    env.step(actions)

                    # For deterministic replay, reset to the recorded state after each step
                    # This ensures we follow the exact trajectory from the dataset
                    for env_id in range(num_envs):
                        if not env_episode_data_map[env_id].is_empty():
                            # Get the next state from dataset (this is the state AFTER the action was applied)
                            next_state = env_episode_data_map[env_id].get_next_state()
                            if next_state is not None:
                                # Reset this environment to the exact recorded state
                                env.reset_to(next_state, torch.tensor([env_id], device=env.device), is_relative=True)

                    # Capture video frames (after state reset)
                    if args_cli.video:
                        frame = env.render()
                        if frame is not None:
                            # Check if frame is an array of frames (multiple envs) or single frame
                            if isinstance(frame, (list, tuple)) and len(frame) > 1:
                                # Multiple frames, one per env
                                for env_id in range(num_envs):
                                    if len(video_frames_map[env_id]) > 0 and env_id < len(frame):
                                        video_frames_map[env_id].append(frame[env_id])
                            else:
                                # Single frame for all envs (or just one env)
                                for env_id in range(num_envs):
                                    if len(video_frames_map[env_id]) > 0:
                                        video_frames_map[env_id].append(frame)

                    # Validate states if enabled
                    if state_validation_enabled:
                        # After reset_to, the current state should match the dataset
                        # Get current runtime state
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        # Get previous state from dataset (we already advanced the index in reset loop)
                        # So we need to check the state we just set
                        prev_state_idx = env_episode_data_map[0].next_state_index - 1
                        if prev_state_idx >= 0:
                            dataset_state = env_episode_data_map[0].get_state(prev_state_idx)
                            if dataset_state is not None:
                                print(
                                    f"Validating states at step: {prev_state_idx :4}",
                                    end="",
                                )
                                states_matched, comparison_log = compare_states(dataset_state, current_runtime_state, 0)
                                if states_matched:
                                    print("\t- matched.")
                                else:
                                    print("\t- mismatched.")
                                    print(comparison_log)

            # Save videos for completed episodes
            if args_cli.video:
                for env_id in range(num_envs):
                    num_frames = len(video_frames_map[env_id])
                    episode_idx = env_episode_index_map[env_id]

                    if num_frames > 1 and episode_idx is not None:  # More than just initial frame
                        video_path = os.path.join(args_cli.video_dir, f"episode_{episode_idx:06d}.mp4")
                        print(f"Saving video for episode {episode_idx} ({num_frames} frames)...")
                        try:
                            import imageio
                            # Stack frames if needed
                            frames = video_frames_map[env_id]
                            imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
                            print(f"Video saved to: {video_path}")
                        except Exception as e:
                            print(f"Failed to save video: {e}")

            break

    # Close environment after replay is complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
