# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RL agent from RSL-RL with detailed metrics and success rate tracking."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use deterministic evaluation (fixed seed sequence for reproducibility).",
)
parser.add_argument(
    "--save_results", type=str, default=None, help="Path to save evaluation results (JSON format)."
)
parser.add_argument(
    "--visualize_obs",
    action="store_true",
    default=False,
    help="Visualize the first 3 observation dimensions in real-time.",
)
parser.add_argument(
    "--visualize_episode",
    type=int,
    default=None,
    help="Record observations only for a specific episode number (1-indexed, e.g., 1 = first episode, 5 = fifth episode). If None, records all episodes.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from collections import defaultdict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class ObservationVisualizer:
    """Class to collect and save observation data as plots."""

    def __init__(self, num_envs: int, save_dir: str, target_episode: int = None, env=None):
        """
        Args:
            num_envs: Number of parallel environments.
            save_dir: Directory to save the plots.
            target_episode: If specified, only record this episode number (1-indexed). None = record all.
            env: Environment instance to extract true joint positions from (optional).
        """
        self.num_envs = num_envs
        self.save_dir = save_dir
        self.target_episode = target_episode
        self.env = env

        # Store all observations from the first environment
        self.obs_history = []

        # Track current episode number for env 0 (starts at 1, increments when episodes complete)
        self.current_episode = 1
        # Start recording immediately if no target specified, or if target is episode 1
        self.is_recording = (target_episode is None) or (target_episode == 1)
        self.recording_started = False

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        if target_episode is not None:
            if target_episode < 1:
                raise ValueError(f"Episode number must be >= 1, got {target_episode}")
            print(f"[INFO] Will record observations only for episode {target_episode}")

    def update(self, obs, dones):
        """Collect observations (first 3 proprio joint values).

        Args:
            obs: Observation (can be TensorDict or Tensor).
            dones: Done flags to track episode boundaries.
        """
        # Only record if we're on the target episode or recording all episodes
        if self.is_recording:
            # Handle TensorDict - extract the 'proprio' term
            if hasattr(obs, 'keys'):
                # Debug: print available keys on first call
                if not hasattr(self, '_printed_keys'):
                    print(f"[DEBUG] Available observation keys: {list(obs.keys())}")
                    self._printed_keys = True

                # TensorDict - extract the 'proprio' key which contains proprioceptive observations
                if 'proprio' in obs:
                    obs_tensor = obs['proprio']
                    if not hasattr(self, '_printed_proprio_shape'):
                        print(f"[DEBUG] Proprio shape: {obs_tensor.shape}")
                        print(f"[DEBUG] First 30 proprio values (env 0): {obs_tensor[0, :30].cpu().numpy()}")
                        print(f"[DEBUG] Values at indices 100-130: {obs_tensor[0, 100:130].cpu().numpy()}")
                        print(f"[DEBUG] Min/Max/Mean of all proprio: {obs_tensor[0].min():.4f} / {obs_tensor[0].max():.4f} / {obs_tensor[0].mean():.4f}")
                        self._printed_proprio_shape = True
                elif 'policy' in obs:
                    # Fallback to 'policy' if 'proprio' doesn't exist
                    obs_tensor = obs['policy']
                    if not hasattr(self, '_warned_no_proprio'):
                        print("[WARNING] No 'proprio' key found, using 'policy' observations instead")
                        print(f"[DEBUG] Policy obs shape: {obs_tensor.shape}")
                        print(f"[DEBUG] First 10 policy values (env 0): {obs_tensor[0, :10].cpu().numpy()}")
                        self._warned_no_proprio = True
                else:
                    # If no 'policy' or 'proprio' key, try to get the first key
                    first_key = list(obs.keys())[0]
                    obs_tensor = obs[first_key]
                    if not hasattr(self, '_warned_fallback'):
                        print(f"[WARNING] No 'proprio' or 'policy' key found, using '{first_key}' instead")
                        print(f"[DEBUG] {first_key} obs shape: {obs_tensor.shape}")
                        self._warned_fallback = True
            else:
                # Regular tensor
                obs_tensor = obs

            # Option 1: Extract from observation tensor (might include history/corruption)
            obs_from_tensor = obs_tensor[0, :3].cpu().numpy()

            # Option 2: Extract directly from environment robot state (ground truth)
            if self.env is not None and hasattr(self.env, 'unwrapped'):
                try:
                    # Get actual joint positions from the robot articulation
                    robot = self.env.unwrapped.scene["robot"]
                    true_joint_pos = robot.data.joint_pos[0, :3].cpu().numpy()
                    obs_np = true_joint_pos

                    # Debug: compare observation vs ground truth
                    if not hasattr(self, '_compared_obs_vs_truth'):
                        print(f"[DEBUG] Obs from tensor [0:3]: {obs_from_tensor}")
                        print(f"[DEBUG] True joint pos [0:3]: {true_joint_pos}")
                        print(f"[DEBUG] Difference: {obs_from_tensor - true_joint_pos}")
                        self._compared_obs_vs_truth = True
                except Exception as e:
                    print(f"[WARNING] Could not extract ground truth joint positions: {e}")
                    obs_np = obs_from_tensor
            else:
                obs_np = obs_from_tensor

            self.obs_history.append(obs_np)

        # Check if episode ended for env 0
        if dones[0]:
            self.current_episode += 1

            # If we just finished recording the target episode, stop recording
            if self.target_episode is not None and self.is_recording:
                self.is_recording = False
                print(f"[INFO] Finished recording episode {self.target_episode} ({len(self.obs_history)} steps)")

            # If we're about to start the target episode, start recording
            if self.target_episode is not None and self.current_episode == self.target_episode:
                self.is_recording = True
                print(f"[INFO] Starting to record episode {self.target_episode}")

    def save_plots(self):
        """Save collected observations as plots."""
        if len(self.obs_history) == 0:
            print("[WARNING] No observations collected, skipping visualization")
            return

        obs_array = np.array(self.obs_history)  # Shape: (timesteps, 3)

        # Save raw observation data for later comparison
        data_path = os.path.join(self.save_dir, "observation_data.npy")
        np.save(data_path, obs_array)
        print(f"[INFO] Raw observation data saved to: {data_path}")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle("Proprio Observations (First 3 Joints - Environment 0)", fontsize=14)

        timesteps = np.arange(len(self.obs_history))

        for i, ax in enumerate(axes):
            ax.plot(timesteps, obs_array[:, i], "b-", linewidth=1.0, alpha=0.8)
            ax.set_ylabel(f"Joint {i}", fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestep", fontsize=10)
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(self.save_dir, "observation_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Observation visualization saved to: {save_path}")

        # Also save individual plots for each joint
        for i in range(3):
            fig_single, ax = plt.subplots(figsize=(12, 4))
            ax.plot(timesteps, obs_array[:, i], "b-", linewidth=1.0, alpha=0.8)
            ax.set_xlabel("Timestep", fontsize=10)
            ax.set_ylabel(f"Joint {i} Position", fontsize=10)
            ax.set_title(f"Joint {i} Position Over Time", fontsize=12)
            ax.grid(True, alpha=0.3)

            save_path_single = os.path.join(self.save_dir, f"joint_{i}.png")
            plt.savefig(save_path_single, dpi=150, bbox_inches="tight")
            plt.close(fig_single)

        plt.close(fig)
        print(f"[INFO] Individual joint plots saved to: {self.save_dir}/joint_*.png")


class EvaluationMetrics:
    """Class to track evaluation metrics across episodes."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device

        # Episode-level tracking
        self.episode_successes = []
        self.episode_position_errors = []
        self.episode_orientation_errors = []
        self.episode_rewards = []
        self.episode_lengths = []

        # Current episode tracking
        self.current_episode_rewards = torch.zeros(num_envs, device=device)
        self.current_episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.current_episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)  # Track if success achieved at any point
        self.episodes_completed = 0

    def update(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict, env):
        """Update metrics with current step information."""
        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        # Check for success at every step (not just when done)
        if hasattr(env.unwrapped, "command_manager"):
            cmd_manager = env.unwrapped.command_manager
            if hasattr(cmd_manager, "_terms") and "object_pose" in cmd_manager._terms:
                pose_cmd = cmd_manager._terms["object_pose"]
                pos_errors = pose_cmd.metrics["position_error"]
                # Mark as success if position error < 0.05m at any point during episode
                success_this_step = pos_errors < 0.05
                self.current_episode_success |= success_this_step

        # Check for completed episodes
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_indices:
                idx_item = idx.item()

                # Get final position/orientation errors for reporting
                if hasattr(env.unwrapped, "command_manager"):
                    cmd_manager = env.unwrapped.command_manager
                    if hasattr(cmd_manager, "_terms") and "object_pose" in cmd_manager._terms:
                        pose_cmd = cmd_manager._terms["object_pose"]
                        pos_error = pose_cmd.metrics["position_error"][idx_item].item()
                        ori_error = pose_cmd.metrics["orientation_error"][idx_item].item()

                        # Use the success flag we tracked throughout the episode
                        success = self.current_episode_success[idx_item].item()

                        self.episode_successes.append(success)
                        self.episode_position_errors.append(pos_error)
                        self.episode_orientation_errors.append(ori_error)
                    else:
                        # Fallback if command manager structure is different
                        self.episode_successes.append(False)
                        self.episode_position_errors.append(float("nan"))
                        self.episode_orientation_errors.append(float("nan"))
                else:
                    # No command manager available
                    self.episode_successes.append(False)
                    self.episode_position_errors.append(float("nan"))
                    self.episode_orientation_errors.append(float("nan"))

                self.episode_rewards.append(self.current_episode_rewards[idx_item].item())
                self.episode_lengths.append(self.current_episode_lengths[idx_item].item())

                # Reset current episode tracking for this environment
                self.current_episode_rewards[idx_item] = 0
                self.current_episode_lengths[idx_item] = 0
                self.current_episode_success[idx_item] = False

                self.episodes_completed += 1

    def get_summary(self) -> dict:
        """Get summary statistics of evaluation."""
        if not self.episode_successes:
            return {
                "error": "No episodes completed",
                "episodes_completed": 0,
            }

        successes = np.array(self.episode_successes)
        pos_errors = np.array(self.episode_position_errors)
        ori_errors = np.array(self.episode_orientation_errors)
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        # Filter out NaN values for error metrics
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

        # Calculate 95% confidence interval for success rate
        n = len(successes)
        if n > 1:
            se = np.sqrt(np.mean(successes) * (1 - np.mean(successes)) / n)
            ci_95 = 1.96 * se * 100  # Convert to percentage
            summary["success_rate_ci_95"] = float(ci_95)

        return summary

    def print_summary(self):
        """Print evaluation summary to console."""
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

        print(f"\nEpisode Length (steps):")
        print(f"  Mean: {summary['episode_length']['mean']:.1f} ± {summary['episode_length']['std']:.1f}")
        print(f"  Min:  {summary['episode_length']['min']}")
        print(f"  Max:  {summary['episode_length']['max']}")
        print("=" * 80 + "\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate RSL-RL agent with detailed metrics."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    if args_cli.deterministic:
        # Use fixed seed for deterministic evaluation
        base_seed = agent_cfg.seed if agent_cfg.seed is not None else 42
        env_cfg.seed = base_seed
        print(f"[INFO] Running deterministic evaluation with base seed: {base_seed}")
    else:
        env_cfg.seed = agent_cfg.seed
        print(f"[INFO] Running randomized evaluation with seed: {env_cfg.seed}")

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "evaluation"),
            "step_trigger": lambda step: step % args_cli.video_length == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Initialize metrics tracker
    metrics = EvaluationMetrics(num_envs=env_cfg.scene.num_envs, device=env.unwrapped.device)

    # Initialize observation visualizer if requested
    visualizer = None
    if args_cli.visualize_obs:
        print("[INFO] Enabling observation visualization for first 3 proprio joints")
        print("[INFO] Will extract ground truth joint positions from environment for accurate comparison")
        vis_dir = os.path.join(log_dir, "observation_plots")
        visualizer = ObservationVisualizer(
            num_envs=env_cfg.scene.num_envs,
            save_dir=vis_dir,
            target_episode=args_cli.visualize_episode,
            env=env
        )

    # Calculate target number of total episodes
    target_episodes = args_cli.num_episodes
    print(f"\n[INFO] Starting evaluation for {target_episodes} episodes across {env_cfg.scene.num_envs} parallel environments")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    start_time = time.time()

    # simulate environment until we reach target number of episodes
    while simulation_app.is_running() and metrics.episodes_completed < target_episodes:
        step_start = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

        # update metrics
        metrics.update(rewards, dones, infos, env)

        # update visualization if enabled
        if visualizer is not None:
            visualizer.update(obs, dones)

        timestep += 1

        # Print progress every 100 steps
        if timestep % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"[INFO] Step {timestep} | Episodes completed: {metrics.episodes_completed}/{target_episodes} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # time delay for real-time evaluation
        if args_cli.real_time:
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Print final summary
    total_time = time.time() - start_time
    print(f"\n[INFO] Evaluation completed in {total_time:.1f} seconds")
    print(f"[INFO] Total steps: {timestep}")
    print(f"[INFO] Steps per second: {timestep / total_time:.1f}")

    metrics.print_summary()

    # Save results to file if requested
    if args_cli.save_results:
        summary = metrics.get_summary()
        summary["metadata"] = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_envs": env_cfg.scene.num_envs,
            "target_episodes": target_episodes,
            "deterministic": args_cli.deterministic,
            "seed": env_cfg.seed,
            "total_time": total_time,
            "total_steps": timestep,
        }

        save_path = args_cli.save_results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Results saved to: {save_path}")

    # save visualization plots if enabled
    if visualizer is not None:
        visualizer.save_plots()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
