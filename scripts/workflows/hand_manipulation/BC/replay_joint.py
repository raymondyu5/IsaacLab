# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""
import numpy as np
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument("--add_right_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--add_left_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--random_camera_pose",
    action="store_true",
)
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

# from scripts.workflows.open_policy.task.planner_grasp import PlannerGrasp

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import isaaclab.utils.math as math_utils
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


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["eval_mode"] = True
    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose

    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    joint_limits = env.scene["right_hand"]._data.joint_limits

    # print joint limits
    index = 0

    joint_list = torch.as_tensor([
        [
            1.84093851, -0.88725557, -1.98390265, -2.06911664, -1.21120858,
            2.38840473, 1.41848584
        ],
        [
            0.82490296, 0.96857195, -1.06084403, -1.90676334, 1.07986801,
            2.30624568, 0.27548056
        ],
        [
            0.90948371, 0.96904613, -1.02947448, -1.81323898, 1.08382605,
            2.31114998, 0.84822807
        ],
        [
            0.92059689, 1.08801488, -1.2761323, -2.13270649, 1.32119531,
            2.25459435, -0.31275837
        ],
    ]).to(env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = (torch.rand(env.action_space.shape,
                                  device=env.unwrapped.device) * 2 - 1) * 0
            index += 1
            actions[:, :7] = joint_list[index % joint_list.shape[0]]

            obs, rewards, terminated, time_outs, extras = env.step(actions)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
