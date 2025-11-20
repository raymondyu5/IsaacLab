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

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    # create environment

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    if save_config["params"]["arm_type"] is not None:
        if "IK" in args_cli.task:
            init_pose = torch.as_tensor(
                save_config["params"]["init_ee_pose"] +
                [0] * save_config["params"]["num_hand_joints"]).to(
                    env.device).unsqueeze(0)
        else:
            init_pose = torch.as_tensor(
                save_config["params"]["reset_joint_pose"] +
                [0] * save_config["params"]["num_hand_joints"]).to(
                    env.device).unsqueeze(0)

        for i in range(10):
            env.step(init_pose)

    from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
    arm_motion_env = ArmMotionPlannerEnv(
        env,
        args_cli,
        save_config,
    )

    arm_motion_env.ik_planner.ik_solver.fk(
        env.scene["right_hand"].root_physx_view.get_dof_positions()[:, :7])
    state = env.scene["right_palm_lower"]._data.root_state_w[:, :7].clone()
    state = torch.as_tensor(save_config["params"]["init_ee_pose"]).to(
        env.device).unsqueeze(0)
    arm_qpos = arm_motion_env.ik_plan_motion(state)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.rand(env.action_space.shape,
                                 device=env.unwrapped.device) * 0.0

            actions[:, 0:7] = arm_qpos
            # print(actions[:, 0:7])

            # actions = default_actions.clone()
            #apply actions
            obs, rewards, terminated, time_outs, extras = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
