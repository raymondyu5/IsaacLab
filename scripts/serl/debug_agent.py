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
from scripts.serl.utils.serl_algo_wrapper import serl_parser, FLAGS
import cv2
import yaml

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(serl_parser)
# parse the arguments
args_cli = serl_parser.parse_args()

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


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.serl.utils.demo_utils import make_zarr_replay_buffer

from scripts.serl.env.residual_rl_wrapper import ResidualRLDatawrapperEnv


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    # save_config["params"]["real_eval_mode"] = True
    save_config["params"]["Camera"]["random_pose"] = True
    # obs_keys = [
    #     'right_contact_obs', 'right_ee_pose', 'right_hand_joint_pos',
    #     'right_manipulated_object_pose', 'right_object_in_tip',
    #     'right_target_object_pose'
    # ]
    # demo_path = "logs/data_0705/retarget_visionpro_data/rl_data/state/tomato_soup_can/"
    # with open(
    #         "source/config/task/hand_env/leap_franka/residual_serl_env_ycb.yaml",
    #         'r') as file:
    #     env_config = yaml.safe_load(file)

    # action_range = env_config["params"]["Task"]["action_range"]

    # replay_buffer, normazlied_action, action_bounds = make_zarr_replay_buffer(
    #     demo_path, obs_keys, action_range=action_range)

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()
    for i in range(10):
        env.step(torch.as_tensor(env.action_space.sample()).to(env.device))
    env.reset()

    # action = action_bounds[:, 0] + (normazlied_action + 1.0) / 2.0 * (
    #     action_bounds[:, 1] - action_bounds[:, 0])

    rl_env = ResidualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    while simulation_app.is_running():
        rl_env.debug_step()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
