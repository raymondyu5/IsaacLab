# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

parser.add_argument(
    "--robot_name",
    default="leap",
)

parser.add_argument("--add_right_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--add_left_hand",
                    action="store_true",
                    help="Disable fabric and use USD I/O operations.")

parser.add_argument(
    "--save_data",
    action="store_true",
)
parser.add_argument(
    "--collision_checker",
    action="store_true",
)

parser.add_argument(
    "--data_type",
    default="dexycb",
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


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.workflows.hand_manipulation.env.teleop_env.dexdata_gen import DexDataGen


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    env.reset()
    if save_config["params"]["arm_type"] is not None:
        if "IK" in args_cli.task:
            if "Rel" not in args_cli.task:
                init_pose = torch.as_tensor(
                    save_config["params"]["init_ee_pose"] +
                    [0] * save_config["params"]["num_hand_joints"]).to(
                        env.device).unsqueeze(0)
            else:

                init_pose = torch.zeros(env.action_space.shape).to(env.device)
                init_joint_pose = torch.as_tensor(
                    save_config["params"]["reset_joint_pose"] +
                    [0] * save_config["params"]["num_hand_joints"]).to(
                        env.device).unsqueeze(0)
                env.scene["right_hand"]._root_physx_view.set_dof_positions(
                    init_joint_pose,
                    indices=torch.arange(env.num_envs).to(env.device))
        else:
            init_pose = torch.as_tensor(
                save_config["params"]["reset_joint_pose"] +
                [0] * save_config["params"]["num_hand_joints"]).to(
                    env.device).unsqueeze(0)

        for i in range(10):
            env.step(init_pose)

    DexDataGenerative_env = DexDataGen(
        args_cli,
        save_config,
        env,
    )

    while simulation_app.is_running():
        DexDataGenerative_env.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
