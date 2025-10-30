"""Script to evaluate BC model as a policy in the target environment.

This script evaluates a behavioral cloning model trained on (state, action_target) pairs
by deploying it as a policy in the target domain environment.

This serves as an upper bound for action translation performance - if BC performs well,
it indicates the inverse dynamics model produces good target actions.

Example usage:
    ./isaaclab.sh -p scripts/behavior_cloning/evaluate_bc_policy.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Slippery-Play-v0 \
        --bc_checkpoint trained_models/bc_baseline_20241029.pth \
        --model_config configs/bc_baseline/flow_bc.yaml \
        --num_envs 32 \
        --num_episodes 100 \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate BC model as a policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (target environment).")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="Path to BC model checkpoint (.pth)")
parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML file")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use deterministic evaluation (fixed seed).",
)
parser.add_argument(
    "--save_results", type=str, default=None, help="Path to save evaluation results (JSON)."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('/home/raymond/projects/generative-policies')

from scripts.lib.policy_evaluation import run_policy_evaluation
from scripts.lib.state_extraction import extract_from_env
from scripts.lib.training import build_model_from_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Evaluate BC model as a policy."""

    task_name = args_cli.task.split(":")[-1]

    print("\n" + "=" * 80)
    print("BC POLICY EVALUATION")
    print("=" * 80)
    print(f"Task: {task_name}")
    print(f"BC checkpoint: {args_cli.bc_checkpoint}")
    print(f"Model config: {args_cli.model_config}")
    print("=" * 80 + "\n")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.deterministic:
        base_seed = args_cli.seed if args_cli.seed is not None else 42
        env_cfg.seed = base_seed
        print(f"[INFO] Deterministic evaluation with seed: {base_seed}")
    else:
        env_cfg.seed = args_cli.seed if args_cli.seed is not None else 42

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    eval_log_dir = os.path.join("logs", "bc_evaluation", task_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    print(f"[INFO] Logs: {eval_log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_length == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    print(f"[INFO] Created environment with {env.num_envs} parallel envs")

    # Load BC checkpoint
    device = env.unwrapped.device
    checkpoint = torch.load(args_cli.bc_checkpoint, map_location=device, weights_only=False)

    # Get dimensions from checkpoint
    obs_dim = checkpoint.get('obs_dim', checkpoint.get('state_dim', 165))
    action_dim = checkpoint.get('action_dim', env.num_actions)

    print(f"[INFO] BC model: obs_dim={obs_dim}, action_dim={action_dim}")

    # Load model config
    with open(args_cli.model_config, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config.get('model', {})

    # Build model
    model = build_model_from_config(model_config, obs_dim, action_dim)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[INFO] Loaded BC model from checkpoint")

    class SimplePolicy:
        def __call__(self, obs):
            if hasattr(obs, 'get'):
                obs_tensor = obs.get("policy", next(iter(obs.values())))
            else:
                obs_tensor = obs

            # Extract Markovian state if model expects it (e.g., 134D from 165D obs)
            if obs_dim != 165 and obs_tensor.shape[-1] == 165:
                obs_tensor = extract_from_env(env.unwrapped, obs_tensor)

            dummy_prior = torch.zeros((obs_tensor.shape[0], action_dim), device=device)
            actions = model.predict(obs_tensor, dummy_prior)

            if isinstance(actions, torch.Tensor):
                return actions
            else:
                return torch.tensor(actions, device=device, dtype=torch.float32)

    bc_policy = SimplePolicy()

    metadata = {
        "task": task_name,
        "bc_checkpoint": args_cli.bc_checkpoint,
        "model_config": args_cli.model_config,
        "num_envs": env_cfg.scene.num_envs,
        "target_episodes": args_cli.num_episodes,
        "deterministic": args_cli.deterministic,
        "seed": env_cfg.seed,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }

    save_path = args_cli.save_results or os.path.join(eval_log_dir, "bc_eval_results.json")

    summary = run_policy_evaluation(
        env=env,
        policy=bc_policy,
        num_episodes=args_cli.num_episodes,
        simulation_app=simulation_app,
        real_time=args_cli.real_time,
        metadata=metadata,
        save_path=save_path
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
