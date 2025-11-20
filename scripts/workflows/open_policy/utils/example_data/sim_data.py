"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import os
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_tasks.utils.hydra import hydra_task_config
from scripts.workflows.open_policy.utils.sb3_wrapper import Sb3VecEnvWrapper, process_sb3_cfg
import os
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device="cuda",
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():

    # parse configuration

    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    data_file = os.listdir("logs/trash_action/")

    for file in data_file:
        data = np.load(f"logs/trash_action/{file}")
        print("new episode")
        env.reset()
        all_actions = torch.as_tensor(data,
                                      dtype=torch.float32).to(env.env.device)
        for per_action in all_actions:
            env.step(per_action.unsqueeze(0))


if __name__ == "__main__":
    main()
