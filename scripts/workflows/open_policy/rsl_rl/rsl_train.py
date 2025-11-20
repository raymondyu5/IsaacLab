# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
from scripts.reinforcement_learning.rsl_rl import cli_args
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
# add argparse arguments

parser.add_argument("--eval_freq",
                    type=int,
                    default=50,
                    help="Seed used for the environment")
parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from scripts.rsl_rl.on_policy_runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped, env_cfg


def main():
    """Train with RSL-RL agent."""

    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    env, _ = setup_env(args_cli, save_config)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    agent_cfg.max_iterations = (args_cli.max_iterations
                                if args_cli.max_iterations is not None else
                                agent_cfg.max_iterations)

    # specify directory for logging experiments
    log_root_path = os.path.join(args_cli.log_dir, "rsl_rl",
                                 agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env.reset()

    open_policy_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run,
                                          agent_cfg.load_checkpoint)

    # wrap around environment for rsl-rl
    open_policy_rl_env = RslRlVecEnvWrapper(open_policy_env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(open_policy_rl_env,
                            agent_cfg.to_dict(),
                            log_dir=log_dir,
                            device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # policy = runner.get_inference_policy(device=env.unwrapped.device)
    # open_policy_env.agent = policy

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
