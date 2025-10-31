
import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Add the base evaluation script's directory to path for cli_args
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning', 'rsl_rl'))

# local imports
import cli_args  # isort: skip

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate policy with action translation.")

# Action translation specific arguments
parser.add_argument(
    "--source_task",
    type=str,
    required=True,
    help="Name of the source task (where the policy was trained, e.g., Isaac-Dexsuite-Kuka-Allegro-Reorient-Play)"
)
parser.add_argument(
    "--source_checkpoint",
    type=str,
    default=None,
    help="Path to source policy checkpoint (e.g., logs/rsl_rl/dexsuite_kuka_allegro/model_4750_source.pt)"
)
parser.add_argument(
    "--translator_config",
    type=str,
    default=None,
    help="Path to action translator config YAML file"
)
parser.add_argument(
    "--translator_checkpoint",
    type=str,
    default=None,
    help="Path to trained action translator checkpoint (.pth)"
)
parser.add_argument(
    "--no_translation",
    action="store_true",
    help="Disable action translation (baseline evaluation)"
)

# Standard evaluation arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the target task (environment to evaluate on).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Validate required arguments
if args_cli.task is None:
    raise ValueError("--task argument is required")
if args_cli.source_task is None:
    raise ValueError("--source_task argument is required")

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Validate arguments
if not args_cli.no_translation:
    if args_cli.translator_config is None or args_cli.translator_checkpoint is None:
        raise ValueError(
            "Must provide both --translator_config and --translator_checkpoint. "
            "Use --no_translation for baseline evaluation."
        )

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Add path for lib imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import unified evaluation from lib
from scripts.lib.policy_evaluation import EvaluationMetrics, run_policy_evaluation

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate policy with action translation."""

    # Get task names
    target_task_name = args_cli.task.split(":")[-1]
    source_task_name = args_cli.source_task.split(":")[-1]

    print("\n" + "=" * 80)
    print("POLICY EVALUATION WITH ACTION TRANSLATION")
    print("=" * 80)
    print(f"Source task (policy trained on): {source_task_name}")
    print(f"Target task (evaluating on): {target_task_name}")
    print(f"Action translation: {'DISABLED (baseline)' if args_cli.no_translation else 'ENABLED'}")
    print("=" * 80 + "\n")

    # Override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # Set the environment seed
    if args_cli.deterministic:
        base_seed = agent_cfg.seed if agent_cfg.seed is not None else 42
        env_cfg.seed = base_seed
    else:
        env_cfg.seed = agent_cfg.seed

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Get checkpoint path for SOURCE task
    if args_cli.source_checkpoint:
        resume_path = retrieve_file_path(args_cli.source_checkpoint)
        print(f"Loading source policy from specified checkpoint: {resume_path}")
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"Loading source policy from auto-detected checkpoint: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # Create evaluation-specific log directory
    eval_suffix = "no_translation" if args_cli.no_translation else "with_translation"
    eval_log_dir = os.path.join(log_dir, f"eval_{target_task_name}_{eval_suffix}")
    os.makedirs(eval_log_dir, exist_ok=True)
    env_cfg.log_dir = eval_log_dir

    if not args_cli.no_translation:
        print(f"Translator config: {args_cli.translator_config}")
        print(f"Translator checkpoint: {args_cli.translator_checkpoint}")

    # Create Isaac environment (TARGET task)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_length == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("Recording videos during evaluation")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load source policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # Obtain the trained policy for inference
    source_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Wrap with action translator if enabled
    if args_cli.no_translation:
        print("Using source policy directly (no translation)")
        policy = source_policy
    else:
        print("Wrapping source policy with action translator")

        # Import translator wrapper here to avoid initialization issues
        translator_path = os.path.dirname(__file__)
        if translator_path not in sys.path:
            sys.path.insert(0, translator_path)
        from translator_policy_wrapper import TranslatorPolicyWrapper

        # Wrap policy with action translator
        policy = TranslatorPolicyWrapper(
            source_policy=source_policy,
            translator_config=args_cli.translator_config,
            translator_checkpoint=args_cli.translator_checkpoint,
            env=env.unwrapped,
            device=env.unwrapped.device,
            verbose=True
        )

    # Prepare metadata for results
    metadata = {
        "source_task": source_task_name,
        "target_task": target_task_name,
        "action_translation_enabled": not args_cli.no_translation,
        "translator_config": args_cli.translator_config if not args_cli.no_translation else None,
        "translator_checkpoint": args_cli.translator_checkpoint if not args_cli.no_translation else None,
        "checkpoint": resume_path,
        "num_envs": env_cfg.scene.num_envs,
        "target_episodes": args_cli.num_episodes,
        "deterministic": args_cli.deterministic,
        "seed": env_cfg.seed,
    }

    # Auto-save path
    results_filename = f"eval_results_{eval_suffix}.json"
    save_path = args_cli.save_results if args_cli.save_results else os.path.join(eval_log_dir, results_filename)

    # Run unified evaluation (replaces entire loop above!)
    summary = run_policy_evaluation(
        env=env,
        policy=policy,
        num_episodes=args_cli.num_episodes,
        simulation_app=simulation_app,
        real_time=args_cli.real_time,
        metadata=metadata,
        save_path=save_path
    )

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
