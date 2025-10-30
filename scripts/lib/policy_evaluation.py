""" 
utils for online eval

This module provides shared utilities for evaluating policies in Isaac Lab environments,
including metrics tracking, policy wrappers, and evaluation loops.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any


class EvaluationMetrics:
    """Class to track evaluation metrics across episodes.

    Tracks success rate, position/orientation errors, rewards, and episode lengths
    during policy evaluation in Isaac Lab environments.
    """

    def __init__(self, num_envs: int, device: torch.device):
        """Initialize metrics tracker.

        Args:
            num_envs: Number of parallel environments
            device: Torch device (cuda or cpu)
        """
        self.num_envs = num_envs
        self.device = device

        # Episode-level tracking (completed episodes)
        self.episode_successes = []
        self.episode_position_errors = []
        self.episode_orientation_errors = []
        self.episode_rewards = []
        self.episode_lengths = []

        # Current episode tracking (per environment)
        self.current_episode_rewards = torch.zeros(num_envs, device=device)
        self.current_episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.current_episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episodes_completed = 0

    def update(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict, env):
        """Update metrics with current step information.

        Args:
            rewards: Reward tensor for current step, shape (num_envs,)
            dones: Done flags for current step, shape (num_envs,)
            infos: Info dict from environment
            env: Isaac Lab environment instance
        """
        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        # Check for success at every step
        if hasattr(env.unwrapped, "command_manager"):
            cmd_manager = env.unwrapped.command_manager
            if hasattr(cmd_manager, "_terms") and "object_pose" in cmd_manager._terms:
                pose_cmd = cmd_manager._terms["object_pose"]
                pos_errors = pose_cmd.metrics["position_error"]
                # Mark as success if position error < 0.05m
                success_this_step = pos_errors < 0.05
                self.current_episode_success |= success_this_step

        # Check for completed episodes
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_indices:
                idx_item = idx.item()

                # Get final position/orientation errors
                if hasattr(env.unwrapped, "command_manager"):
                    cmd_manager = env.unwrapped.command_manager
                    if hasattr(cmd_manager, "_terms") and "object_pose" in cmd_manager._terms:
                        pose_cmd = cmd_manager._terms["object_pose"]
                        pos_error = pose_cmd.metrics["position_error"][idx_item].item()
                        ori_error = pose_cmd.metrics["orientation_error"][idx_item].item()
                        success = self.current_episode_success[idx_item].item()

                        self.episode_successes.append(success)
                        self.episode_position_errors.append(pos_error)
                        self.episode_orientation_errors.append(ori_error)
                    else:
                        self.episode_successes.append(False)
                        self.episode_position_errors.append(float("nan"))
                        self.episode_orientation_errors.append(float("nan"))
                else:
                    self.episode_successes.append(False)
                    self.episode_position_errors.append(float("nan"))
                    self.episode_orientation_errors.append(float("nan"))

                self.episode_rewards.append(self.current_episode_rewards[idx_item].item())
                self.episode_lengths.append(self.current_episode_lengths[idx_item].item())

                # Reset tracking for this environment
                self.current_episode_rewards[idx_item] = 0
                self.current_episode_lengths[idx_item] = 0
                self.current_episode_success[idx_item] = False
                self.episodes_completed += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of evaluation.

        Returns:
            Dictionary containing:
                - episodes_completed: Total episodes finished
                - success_rate: Mean success rate (%)
                - success_rate_std: Std of success rate (%)
                - success_rate_ci_95: 95% confidence interval (%)
                - position_error: Dict with mean, std, min, max
                - orientation_error: Dict with mean, std, min, max
                - episode_reward: Dict with mean, std, min, max
                - episode_length: Dict with mean, std, min, max
        """
        if not self.episode_successes:
            return {"error": "No episodes completed", "episodes_completed": 0}

        successes = np.array(self.episode_successes)
        pos_errors = np.array(self.episode_position_errors)
        ori_errors = np.array(self.episode_orientation_errors)
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        valid_pos = ~np.isnan(pos_errors)
        valid_ori = ~np.isnan(ori_errors)

        summary = {
            "episodes_completed": len(successes),
            "success_rate": float(np.mean(successes) * 100),
            "success_rate_std": float(np.std(successes) * 100),
            "num_successes": int(np.sum(successes)),
            "num_failures": int(len(successes) - np.sum(successes)),
            "position_error": {
                "mean": float(np.mean(pos_errors[valid_pos])) if valid_pos.any() else None,
                "std": float(np.std(pos_errors[valid_pos])) if valid_pos.any() else None,
                "min": float(np.min(pos_errors[valid_pos])) if valid_pos.any() else None,
                "max": float(np.max(pos_errors[valid_pos])) if valid_pos.any() else None,
            },
            "orientation_error": {
                "mean": float(np.mean(ori_errors[valid_ori])) if valid_ori.any() else None,
                "std": float(np.std(ori_errors[valid_ori])) if valid_ori.any() else None,
                "min": float(np.min(ori_errors[valid_ori])) if valid_ori.any() else None,
                "max": float(np.max(ori_errors[valid_ori])) if valid_ori.any() else None,
            },
            "episode_reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
            "episode_length": {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
            },
        }

        # Calculate 95% CI for success rate
        n = len(successes)
        if n > 1:
            se = np.sqrt(np.mean(successes) * (1 - np.mean(successes)) / n)
            ci_95 = 1.96 * se * 100
            summary["success_rate_ci_95"] = float(ci_95)

        return summary

    def print_summary(self):
        """Print evaluation summary to console in a formatted way."""
        summary = self.get_summary()

        if "error" in summary:
            print(f"\n[ERROR] {summary['error']}")
            return

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Episodes Completed: {summary['episodes_completed']}")
        print(f"\nSuccess Rate: {summary['success_rate']:.2f}% ± {summary['success_rate_std']:.2f}%")
        if "success_rate_ci_95" in summary:
            print(f"  95% CI: ±{summary['success_rate_ci_95']:.2f}%")
        print(f"  Successes: {summary['num_successes']}")
        print(f"  Failures: {summary['num_failures']}")

        if summary["position_error"]["mean"] is not None:
            print(f"\nPosition Error (meters):")
            print(f"  Mean: {summary['position_error']['mean']:.4f} ± {summary['position_error']['std']:.4f}")
            print(f"  Min:  {summary['position_error']['min']:.4f}")
            print(f"  Max:  {summary['position_error']['max']:.4f}")

        if summary["orientation_error"]["mean"] is not None:
            print(f"\nOrientation Error (radians):")
            print(f"  Mean: {summary['orientation_error']['mean']:.4f} ± {summary['orientation_error']['std']:.4f}")
            print(f"  Min:  {summary['orientation_error']['min']:.4f}")
            print(f"  Max:  {summary['orientation_error']['max']:.4f}")

        print(f"\nEpisode Reward:")
        print(f"  Mean: {summary['episode_reward']['mean']:.2f} ± {summary['episode_reward']['std']:.2f}")
        print(f"  Min:  {summary['episode_reward']['min']:.2f}")
        print(f"  Max:  {summary['episode_reward']['max']:.2f}")

        print(f"\nEpisode Length:")
        print(f"  Mean: {summary['episode_length']['mean']:.1f} ± {summary['episode_length']['std']:.1f}")
        print(f"  Min:  {summary['episode_length']['min']}")
        print(f"  Max:  {summary['episode_length']['max']}")
        print("=" * 80 + "\n")


def run_policy_evaluation(
    env,
    policy,
    num_episodes: int,
    simulation_app=None,
    real_time: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run policy evaluation in Isaac Lab environment.

    This is the unified evaluation loop that works with any policy (AT, BC, or baseline).
    Handles the complete evaluation workflow including metrics tracking, progress display,
    and results saving.

    Args:
        env: Isaac Lab environment (RslRlVecEnvWrapper)
        policy: Policy object with __call__(obs) -> actions
            - For BC: BCPolicy instance
            - For AT: TranslatorPolicyWrapper or raw policy
            - For baseline: OnPolicyRunner.get_inference_policy()
        num_episodes: Total number of episodes to evaluate
        simulation_app: Isaac simulation app (for checking is_running())
        real_time: Whether to run in real-time (with sleep)
        metadata: Optional metadata dict to include in results
        save_path: Optional path to save results JSON (None = no save)

    Returns:
        Dictionary containing evaluation summary statistics with metadata
    """
    import time
    import json

    num_envs = env.num_envs
    device = env.device

    # Initialize metrics tracker
    metrics = EvaluationMetrics(num_envs, device)

    print(f"\nStarting evaluation: {num_episodes} episodes across {num_envs} parallel environments")

    dt = env.unwrapped.step_dt
    timestep = 0
    start_time = time.time()

    # Get initial observations
    obs = env.get_observations()

    # Evaluation loop
    while metrics.episodes_completed < num_episodes:
        # Check if simulation is still running (if app provided)
        if simulation_app is not None and not simulation_app.is_running():
            print("\nSimulation app stopped, ending evaluation early")
            break

        step_start = time.time()

        # Run everything in inference mode
        with torch.inference_mode():
            # Agent stepping
            actions = policy(obs)

            # Env stepping
            obs, rewards, dones, infos = env.step(actions)

        # Update metrics
        metrics.update(rewards, dones, infos, env)

        timestep += 1

        # Print progress every 100 steps
        if timestep % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {timestep} | Episodes completed: {metrics.episodes_completed}/{num_episodes} | "
                f"Elapsed: {elapsed:.1f}s",
                end='\r'
            )

        # Time delay for real-time evaluation
        if real_time:
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Print final summary
    total_time = time.time() - start_time
    print(f"\n\nEvaluation completed in {total_time:.1f}s | Total steps: {timestep} | Steps/sec: {timestep / total_time:.1f}")

    # Get summary
    summary = metrics.get_summary()
    metrics.print_summary()

    # Add metadata if provided
    if metadata is not None:
        summary["metadata"] = metadata

    # Add timing info
    if "metadata" not in summary:
        summary["metadata"] = {}
    summary["metadata"]["total_time"] = total_time
    summary["metadata"]["total_steps"] = timestep

    # Save results if requested
    if save_path is not None:
        import os

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        summary_serializable = convert_to_json_serializable(summary)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary_serializable, f, indent=2)
        print(f"Results saved to: {save_path}")

    return summary
