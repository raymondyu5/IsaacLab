# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import numpy as np

parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations",
                    type=int,
                    default=None,
                    help="RL Policy training iterations.")
parser.add_argument("--checkpoint", default=None, help="checkpoint path")
parser.add_argument("--algo_name", default="ppo", help="type of algorithm")
parser.add_argument(
    "--base_normalized",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--scale_factor",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--base_scale_factor",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--normalize_action_path",
    type=str,
    default=None,
)
# launch omniverse app
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym

from scripts.workflows.open_policy.task.rl_replay_datawrapper import RLReplayDatawrapper

import os

from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_tasks.utils.hydra import hydra_task_config
from scripts.workflows.open_policy.utils.sb3_residual_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import imageio
import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback

import random

from scripts.workflows.open_policy.task.rl_openvla_wrapper import RLDatawrapperEnv
from scripts.sb3.ppo import PPO
from scripts.sb3.sac import SAC
import sys

sys.path.append("submodule/robomimic_openrt")
import robomimic.utils.file_utils as FileUtils


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Load YAML file
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    # Convert to Box
    agent_cfg = Box(yaml_data)

    agent_cfg[
        "seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg[
            "seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg[
            "n_steps"] * args_cli.num_envs

    #set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    agent_cfg.seed = agent_cfg["seed"]
    # agent_cfg.sim.device = args_cli.device if args_cli.device is not None else "cpu"

    # directory for logging into
    log_dir = args_cli.log_dir
    # dump the configuration into log-directory

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    # reinstantiate the environment
    save_config["params"]["Task"]["action_range"] = np.array(
        save_config["params"]["Task"]["action_range"])

    save_config["params"]["Task"]["action_range"] *= args_cli.scale_factor
    openvla_env = RLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    base_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=args_cli.device, verbose=True)
    base_policy.start_episode()
    base_policy.policy.set_eval()

    # wrap around environment for stable baselines
    openvla_rl_env = Sb3VecEnvWrapper(openvla_env, base_policy, args_cli)

    # create agent from stable baselines

    setup_wandb(args_cli,
                "bridge_eggplant_ppo_bc",
                tags=None,
                project="isaaclab")
    if args_cli.algo_name == "sac":
        agent = SAC(policy_arch, openvla_rl_env, verbose=1, **agent_cfg)
    else:
        agent = PPO(policy_arch, openvla_rl_env, verbose=1, **agent_cfg)

    agent.learn(
        total_timesteps=n_timesteps,
        callback=WandbCallback(
            model_save_freq=5,
            model_save_path=str(args_cli.log_dir +
                                f"/residual_{args_cli.algo_name}"),
            eval_env_fn=None,
            eval_freq=10,
            eval_cam_names=None,
        ),
    )

    # save the final model

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
