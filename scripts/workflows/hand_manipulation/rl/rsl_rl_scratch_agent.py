# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
from hydra.utils import instantiate
# add argparse arguments
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

parser.add_argument(
    "--action_framework",
    default=None,
)

parser.add_argument(
    "--use_dict_obs",
    action="store_true",
)

parser.add_argument(
    "--vae_path",
    default=None,
)

parser.add_argument(
    "--diffusion_path",
    default=None,
)

parser.add_argument(
    "--latent_dim",
    default=32,
    type=int,
)
parser.add_argument(
    "--use_relative_finger_pose",
    action="store_true",
)

parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)

parser.add_argument(
    "--bc_dir",
    default=None,
    type=str,
)

parser.add_argument(
    "--use_residual_action",
    action="store_true",
)

parser.add_argument(
    "--use_chunk_action",
    action="store_true",
)
parser.add_argument(
    "--use_interpolate_chunk",
    action="store_true",
)
parser.add_argument(
    "--residual_step",
    default=1,
    type=int,
)

parser.add_argument(
    "--resume",
    action="store_true",
)

parser.add_argument(
    "--checkpoint",
    default=None,
)

parser.add_argument("--video",
                    action="store_true",
                    default=False,
                    help="Record videos during training.")

parser.add_argument("--video_length",
                    type=int,
                    default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval",
                    type=int,
                    default=10000,
                    help="Interval between video recordings (in steps).")

parser.add_argument("--distributed",
                    action="store_true",
                    default=False,
                    help="Run training with multiple GPUs or nodes.")

parser.add_argument(
    "--rl_type",
    default="ppo",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(
        RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat", "-p", "-m", "pip", "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}"
        ]
    else:
        cmd = [
            "./isaaclab.sh", "-p", "-m", "pip", "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}"
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n")
    exit(1)
"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from scripts.workflows.hand_manipulation.env.rl_env.rsl_utilis.runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from scripts.workflows.hand_manipulation.env.rl_env.rsl_rl_wrapper import RslRlVecEnvWrapper
from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def setup_env(args_cli, save_config, agent_cfg):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.resume = args_cli.resume

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    return gym.make(args_cli.task,
                    cfg=env_cfg,
                    render_mode="rgb_array"
                    if args_cli.video else None).unwrapped, agent_cfg


def main():
    """Train with RSL-RL agent."""

    cfg = OmegaConf.load(args_cli.rl_config)

    # Instantiate the class from _target_
    agent_cfg = instantiate(cfg)

    # parse configuration
    env_cfg, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    env_cfg["params"]["add_right_hand"] = args_cli.add_right_hand
    env_cfg["params"]["add_left_hand"] = args_cli.add_left_hand
    env_cfg["params"]["num_envs"] = args_cli.num_envs
    env_cfg["params"]["rl_train"] = True
    env_cfg["params"]["sample_points"] = True

    env, agent_cfg = setup_env(args_cli, env_cfg, agent_cfg)
    env.reset()

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(args_cli.log_dir,
                                          agent_cfg.checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    wandb_run = setup_wandb(args_cli,
                            "rsl_rl_ppo",
                            tags=None,
                            project="isaaclab")

    rl_env = RLDatawrapperEnv(
        env,
        env_cfg,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for rsl-rl
    rl_agent_env = RslRlVecEnvWrapper(rl_env,
                                      clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(rl_agent_env,
                            agent_cfg.to_dict(),
                            log_dir=args_cli.log_dir,
                            device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(args_cli.log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(args_cli.log_dir, "params", "agent.yaml"),
              agent_cfg)
    dump_pickle(os.path.join(args_cli.log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(args_cli.log_dir, "params", "agent.pkl"),
                agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
