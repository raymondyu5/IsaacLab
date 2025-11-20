# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import pinocchio as pin
import numpy as np
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
import cv2
from scripts.sb3.rl_algo_wrapper import rl_parser

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(rl_parser)
# parse the arguments
args_cli = rl_parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import torch

# from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp

# from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
# import isaaclab.utils.math as math_utils
import matplotlib.pyplot as plt
from tools.visualization_utils import *

from tools.trash.amazon.test_space_mouse import SpaceMouseExpert


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.workflows.hand_manipulation.env.datagen.render_noise_wrapper import RenderNoiseWrapper


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["use_teleop"] = True

    save_config["params"]["real_eval_mode"] = True

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()
    for i in range(10):
        env.step(torch.as_tensor(env.action_space.sample()).to(env.device))
    env.reset()

    render_env = RenderNoiseWrapper(env, save_config, args_cli)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            render_env.step_env()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
