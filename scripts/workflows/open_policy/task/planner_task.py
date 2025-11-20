# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--grasp_mode",
                    type=str,
                    default=None,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--pick_only",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--approach_only",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
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

from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    save_config["params"]["RL_Train"]["initial"] = False

    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()

    planner_env = PlannerGrasp(env, args_cli, save_config)

    for i in range(args_cli.num_demos):

        last_obs = planner_env.reset()

        planner_env.step_planner(last_obs, skip_frame=1)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
