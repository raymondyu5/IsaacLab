# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

import argparse

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

from isaaclab_tasks.utils import parse_env_cfg
from scripts.workflows.vlm_failure.stack_block.utils.llm_data_wrapper import LLMDatawrapper


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
    # create environment
    replay_normalized_actions = args_cli.replay
    if replay_normalized_actions:
        save_config["params"]["Camera"]["initial"] = True

    env = setup_env(args_cli, save_config)
    collision_checker = False
    collect_data = True

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    multi_env = LLMDatawrapper(
        env,
        collision_checker=collision_checker,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        collect_data=collect_data,
        args_cli=args_cli,
        env_config=save_config,
        filter_keys=["segmentation", "seg_pc", "seg_rgb"],
        use_joint_pos="IK" not in args_cli.task,
        load_path=args_cli.load_path,
        save_path=args_cli.save_path,
        failure_attempt=2,
    )

    env.reset()
    stop = False

    # simulate environment
    while not stop:
        observation = multi_env.reset_demo_env()
        if observation is not None:
            multi_env.correct_env_espiode(observation)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
