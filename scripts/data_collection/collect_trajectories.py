"""
Script to collect trajectory data from a trained RL policy.

Collects complete trajectories with automatic reward filtering and optional train/val/test splitting.
Supports episode truncation for balanced datasets.

Example usage:

./isaaclab.sh -p scripts/data_collection/collect_trajectories.py \
    --task Isaac-Dexsuite-Kuka-Allegro-Lift-IK-Play-v0 \
    --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/model_4750_source.pt \
    --num_trajectories 5000 \
    --num_envs 128 \
    --output_dir ./trajectory_data \
    --min_reward 10 \
    --max_steps_per_episode 75 \
    --split_dataset \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --split_mode random \
    --headless
"""

import argparse
import sys
import os
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning', 'rsl_rl'))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect trajectory data from a trained RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_trajectories", type=int, default=100, help="Total number of trajectories to collect.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--output_dir", type=str, default="./trajectory_data", help="Output directory for trajectory data.")
parser.add_argument("--max_episode_length", type=int, default=None, help="Maximum episode length (if overriding).")
parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    default=None,
    help="Maximum steps to save per episode (for balanced datasets). E.g., 75 to focus on grasping phase.",
)
parser.add_argument(
    "--min_reward",
    type=float,
    default=None,
    help="Minimum total episode reward to save (filters out low-reward episodes).",
)
parser.add_argument(
    "--max_reward",
    type=float,
    default=None,
    help="Maximum total episode reward to save (filters out high-reward episodes).",
)
parser.add_argument(
    "--split_dataset",
    action="store_true",
    default=False,
    help="Split collected dataset into train/val/test splits.",
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.7,
    help="Fraction of episodes for training split (default: 0.7).",
)
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.15,
    help="Fraction of episodes for validation split (default: 0.15).",
)
parser.add_argument(
    "--test_ratio",
    type=float,
    default=0.15,
    help="Fraction of episodes for test split (default: 0.15).",
)
parser.add_argument(
    "--split_mode",
    type=str,
    default="sequential",
    choices=["sequential", "random"],
    help="How to assign episodes to splits: 'sequential' or 'random' (default: sequential).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from tqdm import tqdm

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import utility functions for dataset manipulation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.trajectory_dataset import truncate_episodes, split_dataset_into_train_val_test


class PostStepRewardsRecorder(RecorderTerm):
    """Custom recorder term that records rewards at the end of each step."""

    def record_post_step(self):
        """Record rewards after each environment step."""
        return "rewards", self._env.reward_buf


class PreStepProprioObservationsRecorder(RecorderTerm):
    """Custom recorder term that records proprio group observations in each step."""

    def record_pre_step(self):
        """Record proprio observations before each environment step."""
        return "proprio_obs", self._env.obs_buf["proprio"]


class PreStepPerceptionObservationsRecorder(RecorderTerm):
    """Custom recorder term that records perception group observations in each step."""

    def record_pre_step(self):
        """Record perception observations before each environment step."""
        return "perception_obs", self._env.obs_buf["perception"]


# Global variable to track reward-based success (accessed by termination function)
_reward_success_tracker = None


class RewardBasedSuccess:
    """helper to track cumulative rewards and determine success based on threshold.

    mark episodes as successful if their total reward is within [min_reward, max_reward],
    enabling filtering with EXPORT_SUCCEEDED_ONLY mode.
    """

    def __init__(self, num_envs: int, min_reward: float = None, max_reward: float = None, device: str = "cuda"):
        self.min_reward = min_reward if min_reward is not None else float('-inf')
        self.max_reward = max_reward if max_reward is not None else float('inf')
        self.device = device
        self.cumulative_rewards = torch.zeros(num_envs, device=device)

    def add_rewards(self, rewards: torch.Tensor):
        """accumulate rewards for the current step."""
        self.cumulative_rewards += rewards

    def check_success(self) -> torch.Tensor:
        """check if current cumulative rewards meet threshold.

        Returns:
            boolean tensor indicating which episodes are within [min_reward, max_reward]
        """
        above_min = self.cumulative_rewards >= self.min_reward
        below_max = self.cumulative_rewards <= self.max_reward
        return above_min & below_max

    def reset(self, env_ids: torch.Tensor):
        """reset cumulative rewards for completed episodes.

        Args:
            env_ids: IDs of environments that are resetting
        """
        self.cumulative_rewards[env_ids] = 0.0


def reward_threshold_success(env, min_reward: float = None, max_reward: float = None) -> torch.Tensor:
    """termination function that checks if cumulative reward is within threshold range.

    used by the TerminationManager to mark episodes as successful
    based on total reward, enabling filtering with EXPORT_SUCCEEDED_ONLY mode.

    IMPORTANT: This should only return True when episodes are actually terminating,
    not during the episode. Otherwise episodes get marked successful prematurely.

    Args:
        env: The environment instance
        min_reward: Minimum reward threshold (inclusive)
        max_reward: Maximum reward threshold (inclusive)

    Returns:
        Boolean tensor indicating which episodes are successful (only for terminating episodes)
    """
    global _reward_success_tracker
    if _reward_success_tracker is not None:
        # Only check success for episodes that are actually done
        # This prevents marking episodes as successful before they complete
        done_mask = env.termination_manager.terminated | env.termination_manager.time_outs
        success_mask = _reward_success_tracker.check_success()
        # Return True only for episodes that are both done AND meet the reward threshold
        return done_mask & success_mask
    else:
        # never mark as successful
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    # get task name & prepare checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configs
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set seed
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    # override max episode length if specified
    if args_cli.max_episode_length is not None:
        env_cfg.episode_length_s = args_cli.max_episode_length * env_cfg.sim.dt * env_cfg.decimation

    # get checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create output directory
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"trajectories_{task_name}_{timestamp}.hdf5")
    output_file_name = os.path.splitext(os.path.basename(output_file))[0]
    print(f"[INFO] Trajectories will be saved to: {output_file}")

    # RecorderManager to automatically record actions, states, and rewards
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # Set export mode based on reward filtering
    if args_cli.min_reward is not None or args_cli.max_reward is not None:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        # Build filter description
        filter_desc = []
        if args_cli.min_reward is not None:
            filter_desc.append(f"reward >= {args_cli.min_reward}")
        if args_cli.max_reward is not None:
            filter_desc.append(f"reward <= {args_cli.max_reward}")
        print(f"[INFO] Reward filtering enabled: Only saving episodes with {' and '.join(filter_desc)}")

        # add a "success" termination term that checks reward threshold
        # used by RecorderManager to determine which episodes to export
        from isaaclab.managers import TerminationTermCfg
        if not hasattr(env_cfg, 'terminations') or env_cfg.terminations is None:
            from isaaclab.utils import configclass
            @configclass
            class TerminationsCfg:
                pass
            env_cfg.terminations = TerminationsCfg()

        # add custom success termination based on reward threshold
        env_cfg.terminations.success = TerminationTermCfg(
            func=reward_threshold_success,
            params={"min_reward": args_cli.min_reward, "max_reward": args_cli.max_reward}
        )
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        print(f"[INFO] Saving all episodes (no reward filtering)")

    # Add custom rewards recorder
    env_cfg.recorders.record_rewards = RecorderTermCfg(class_type=PostStepRewardsRecorder)

    # Add custom proprio observations recorder (fingertips, contact forces, etc.)
    # Note: The attribute name becomes the directory name in the dataset
    env_cfg.recorders.record_proprio_data = RecorderTermCfg(class_type=PreStepProprioObservationsRecorder)

    # Add custom perception observations recorder (point cloud, visual features, etc.)
    env_cfg.recorders.record_perception_data = RecorderTermCfg(class_type=PreStepPerceptionObservationsRecorder)

    # Create isaac environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap around environment for rsl-rl
    env_wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env_wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get environment parameters
    physics_dt = env.unwrapped.physics_dt
    step_dt = env.unwrapped.step_dt
    max_episode_length = env.unwrapped.max_episode_length

    print(f"\n[INFO] Environment Configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Num Envs: {env_cfg.scene.num_envs}")
    print(f"  Seed: {env_cfg.seed}")
    print(f"  Physics dt: {physics_dt}")
    print(f"  Step dt: {step_dt}")
    print(f"  Max episode length: {max_episode_length}")
    print(f"  Target trajectories: {args_cli.num_trajectories}")
    print(f"  Output file: {output_file}")

    # Reset environment to start collection
    obs = env_wrapper.get_observations()

    print(f"[INFO] Starting trajectory collection...")
    print(f"[INFO] RecorderManager will automatically save actions, observations, states, and rewards")

    # Track episode rewards and lengths for each environment
    episode_rewards = torch.zeros(env_cfg.scene.num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(env_cfg.scene.num_envs, dtype=torch.int32, device=env.unwrapped.device)

    # Initialize reward-based success tracker if filtering is enabled
    global _reward_success_tracker
    if args_cli.min_reward is not None or args_cli.max_reward is not None:
        _reward_success_tracker = RewardBasedSuccess(
            num_envs=env_cfg.scene.num_envs,
            min_reward=args_cli.min_reward,
            max_reward=args_cli.max_reward,
            device=env.unwrapped.device
        )

    pbar = tqdm(total=args_cli.num_trajectories, desc="Collecting trajectories", unit="traj")

    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)

            obs, rewards, dones, extras = env_wrapper.step(actions)

            # Accumulate rewards and step counts for each environment
            episode_rewards += rewards
            episode_lengths += 1

            # Track rewards for success filtering if enabled
            if _reward_success_tracker is not None:
                _reward_success_tracker.add_rewards(rewards)

            # Handle episode completions
            if dones.any():
                done_env_ids = torch.where(dones)[0]

                # If reward filtering is enabled, reset tracker for completed episodes
                if _reward_success_tracker is not None:
                    _reward_success_tracker.reset(done_env_ids)

                for env_id in done_env_ids:
                    env_id_int = env_id.item()
                    total_reward = episode_rewards[env_id_int].item()
                    length = episode_lengths[env_id_int].item()

                    # Determine if this episode will be saved
                    if args_cli.min_reward is not None or args_cli.max_reward is not None:
                        min_ok = total_reward >= args_cli.min_reward if args_cli.min_reward is not None else True
                        max_ok = total_reward <= args_cli.max_reward if args_cli.max_reward is not None else True
                        will_save = min_ok and max_ok
                        status = "SAVED" if will_save else "FILTERED"
                    else:
                        status = "SAVED"
                    # print sum meta data
                    pbar.write(f"[INFO] Episode completed (env {env_id_int}): "
                               f"length = {length} steps, "
                               f"reward = {total_reward:.2f} [{status}]")

                    # reset counters for this environment
                    episode_rewards[env_id_int] = 0.0
                    episode_lengths[env_id_int] = 0

                # Update progress bar based on total completed episodes
                total_episodes = env.unwrapped.recorder_manager.exported_successful_episode_count + env.unwrapped.recorder_manager.exported_failed_episode_count
                pbar.n = total_episodes
                pbar.refresh()

            # Check total episodes collected across all environments
            total_episodes = env.unwrapped.recorder_manager.exported_successful_episode_count + env.unwrapped.recorder_manager.exported_failed_episode_count

            if total_episodes >= args_cli.num_trajectories:
                pbar.close()
                print(f"[INFO] Target of {args_cli.num_trajectories} trajectories reached. Stopping collection.")
                break

    # Close progress bar if still open
    if pbar is not None and not pbar.disable:
        pbar.close()

    # final stats
    total_successful = env.unwrapped.recorder_manager.exported_successful_episode_count
    total_failed = env.unwrapped.recorder_manager.exported_failed_episode_count

    env.close()

    print(f"\n{'='*80}")
    print(f"TRAJECTORY COLLECTION COMPLETED")
    print(f"{'='*80}")
    print(f"Total trajectories collected: {total_successful + total_failed}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}\n")

    if args_cli.max_steps_per_episode is not None:
        print(f"\n{'='*80}")
        print(f"TRUNCATING EPISODES TO {args_cli.max_steps_per_episode} STEPS")
        print(f"{'='*80}")
        truncated_file = truncate_episodes(output_file, args_cli.max_steps_per_episode)
        if truncated_file:
            output_file = truncated_file
            print(f"Truncated dataset saved to: {output_file}")
        print(f"{'='*80}\n")

    if args_cli.split_dataset:
        if not os.path.exists(output_file):
            print(f"[WARNING] Cannot split dataset: source file not found: {output_file}")
        elif total_successful + total_failed == 0:
            print(f"[WARNING] Cannot split dataset: no episodes were collected")
        else:
            print(f"\n{'='*80}")
            print(f"SPLITTING DATASET INTO TRAIN/VAL/TEST")
            print(f"{'='*80}")

            split_stats = split_dataset_into_train_val_test(
                source_hdf5_path=output_file,
                output_dir=output_dir,
                train_ratio=args_cli.train_ratio,
                val_ratio=args_cli.val_ratio,
                test_ratio=args_cli.test_ratio,
                split_mode=args_cli.split_mode,
            )

            if split_stats:
                print(f"\n{'='*80}")
                print(f"DATASET SPLIT COMPLETED")
                print(f"{'='*80}")
                print(f"Source: {output_file}")
                print(f"Total episodes: {total_successful + total_failed}")
                print(f"\nTrain split: {os.path.basename(split_stats['train_path'])}")
                print(f"  Episodes: {split_stats['train']['total']} "
                      f"({100 * split_stats['train']['total'] / (total_successful + total_failed):.1f}%)")
                print(f"  Successful: {split_stats['train']['successful']}")
                print(f"  Failed: {split_stats['train']['failed']}")
                print(f"\nValidation split: {os.path.basename(split_stats['val_path'])}")
                print(f"  Episodes: {split_stats['val']['total']} "
                      f"({100 * split_stats['val']['total'] / (total_successful + total_failed):.1f}%)")
                print(f"  Successful: {split_stats['val']['successful']}")
                print(f"  Failed: {split_stats['val']['failed']}")
                print(f"\nTest split: {os.path.basename(split_stats['test_path'])}")
                print(f"  Episodes: {split_stats['test']['total']} "
                      f"({100 * split_stats['test']['total'] / (total_successful + total_failed):.1f}%)")
                print(f"  Successful: {split_stats['test']['successful']}")
                print(f"  Failed: {split_stats['test']['failed']}")
                print(f"{'='*80}\n")
            else:
                print(f"[WARNING] Dataset splitting failed or produced no output")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
