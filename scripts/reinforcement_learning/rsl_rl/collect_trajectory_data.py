"""
Script to collect trajectory data from a trained RL policy.

operates simialrly to play.py but collects complete trajectories without recording any env transitions/resets

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/collect_trajectory_data.py     --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0     --checkpoint /home/raymond/projects/policy_translation/IsaacLab/logs/rsl_rl/dexsuite_kuka_allegro/2025-10-08_13-57-00/model_14999.pt     --num_trajectories 30   --num_envs 16     --output_dir ./trajectory_data --min_reward 10 --headless 
"""

import argparse
import sys
from isaaclab.app import AppLauncher

# local imports
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
    "--constrain_object_spawn",
    action="store_true",
    default=True,
    help="Constrain object spawning to table surface.",
)
parser.add_argument(
    "--min_reward",
    type=float,
    default=None,
    help="Minimum total episode reward to save (filters out low-reward episodes).",
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


class PostStepRewardsRecorder(RecorderTerm):
    """Custom recorder term that records rewards at the end of each step."""

    def record_post_step(self):
        """Record rewards after each environment step."""
        return "rewards", self._env.reward_buf


# Global variable to track reward-based success (accessed by termination function)
_reward_success_tracker = None


class RewardBasedSuccess:
    """helper to track cumulative rewards and determine success based on threshold.

    mark episodes as successful if their total reward exceeds a threshold,
    enabling filtering with EXPORT_SUCCEEDED_ONLY mode.
    """

    def __init__(self, num_envs: int, min_reward: float, device: str):
        self.min_reward = min_reward
        self.device = device
        self.cumulative_rewards = torch.zeros(num_envs, device=device)

    def add_rewards(self, rewards: torch.Tensor):
        """accumulate rewards for the current step."""
        self.cumulative_rewards += rewards

    def check_success(self) -> torch.Tensor:
        """check if current cumulative rewards meet threshold.

        Returns:
            boolean tensor indicating which episodes currently meet the threshold
        """
        return self.cumulative_rewards >= self.min_reward

    def reset(self, env_ids: torch.Tensor):
        """reset cumulative rewards for completed episodes.

        Args:
            env_ids: IDs of environments that are resetting
        """
        self.cumulative_rewards[env_ids] = 0.0


def reward_threshold_success(env, min_reward: float) -> torch.Tensor:
    """termination function that checks if cumulative reward meets threshold.

    used by the TerminationManager to mark episodes as successful
    based on total reward, enabling filtering with EXPORT_SUCCEEDED_ONLY mode.

    Args:
        env: The environment instance
        min_reward: Minimum reward threshold

    Returns:
        Boolean tensor indicating which episodes are successful
    """
    global _reward_success_tracker
    if _reward_success_tracker is not None:
        return _reward_success_tracker.check_success()
    else:
        # never mark as successful
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    # get task name & prepare checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # vverride configs
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set seed
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    # override max episode length if specified
    if args_cli.max_episode_length is not None:
        env_cfg.episode_length_s = args_cli.max_episode_length * env_cfg.sim.dt * env_cfg.decimation

    # Constrain object spawning since
    # table is at z=0.235, height=0.04, so top surface is at z=0.255
    # obj initial position is at (-0.55, 0.1, 0.35)
    # modify the reset_object event to spawn objects closer to the table surface
    if args_cli.constrain_object_spawn:
        if hasattr(env_cfg, 'events') and env_cfg.events is not None:
            if hasattr(env_cfg.events, 'reset_object'):
                print(f"[INFO] Constraining object spawn to table surface")
                env_cfg.events.reset_object.params["pose_range"] = {
                    "x": [-0.15, 0.15],      # Narrower x range centered on table
                    "y": [-0.15, 0.15],      # Narrower y range
                    "z": [0.0, 0.05],        # Very small z range to keep on table (relative to init pos)
                    "roll": [-0.3, 0.3],     # Smaller rotation range
                    "pitch": [-0.3, 0.3],    # Smaller rotation range
                    "yaw": [-3.14, 3.14],    # Keep full yaw rotation
                }

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
    if args_cli.min_reward is not None:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        print(f"[INFO] Reward filtering enabled: Only saving episodes with total reward >= {args_cli.min_reward}")

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
            params={"min_reward": args_cli.min_reward}
        )
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        print(f"[INFO] Saving all episodes (no reward filtering)")

    # Add custom rewards recorder
    env_cfg.recorders.record_rewards = RecorderTermCfg(class_type=PostStepRewardsRecorder)

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
    if args_cli.min_reward is not None:
        _reward_success_tracker = RewardBasedSuccess(
            num_envs=env_cfg.scene.num_envs,
            min_reward=args_cli.min_reward,
            device=env.unwrapped.device
        )

    with torch.inference_mode():
        while simulation_app.is_running():
            # Get actions from policy
            actions = policy(obs)

            # Step environment (RecorderManager will automatically records data during this step)
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

                # Print reward and length for each completed episode
                for env_id in done_env_ids:
                    env_id_int = env_id.item()
                    total_reward = episode_rewards[env_id_int].item()
                    length = episode_lengths[env_id_int].item()

                    # Determine if this episode will be saved
                    if args_cli.min_reward is not None:
                        will_save = total_reward >= args_cli.min_reward
                        status = "SAVED" if will_save else "FILTERED"
                    else:
                        status = "SAVED"
                    # print sum meta data
                    print(f"[INFO] Episode completed (env {env_id_int}): "
                          f"length = {length} steps, "
                          f"reward = {total_reward:.2f} [{status}]")

                    # reset counters for this environment
                    episode_rewards[env_id_int] = 0.0
                    episode_lengths[env_id_int] = 0

            # Check total episodes collected across all environments (check every step!)
            total_episodes = env.unwrapped.recorder_manager.exported_successful_episode_count + \
                           env.unwrapped.recorder_manager.exported_failed_episode_count

            # Stop when we've collected enough trajectories
            if total_episodes >= args_cli.num_trajectories:
                print(f"[INFO] Target of {args_cli.num_trajectories} trajectories reached. Stopping collection.")
                break

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


if __name__ == "__main__":
    main()
    simulation_app.close()
