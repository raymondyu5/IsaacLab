# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser
import os
import numpy as np

rl_parser.add_argument(
    "--use_failure",
    action="store_true",
)

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

import imageio


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


from scripts.workflows.hand_manipulation.env.bc_env.bc_env_wrapper import HandBCEnvWrapper


def evaluate_bc_policy(args_cli, policy_env, count, result_path):
    success_or_not = []
    success_count = 0
    init = False
    while simulation_app.is_running():

        success = policy_env.step()

        # if not init:
        #     init = True
        #     continue

        if count >= 2000:
            return

        # if success is not None:

        #     if len(torch.where(~success)[0]) > 0:

        #         failure_index = torch.where(~success)[0]
        #         for fail_id in failure_index:

        #             video_writer = imageio.get_writer(
        #                 os.path.join(result_path,
        #                              f"video_{count+fail_id}_fail.mp4"),
        #                 fps=30,
        #             )

        #             for image in policy_env.diffusion_env.image_buffer:
        #                 video_writer.append_data(image[fail_id])
        #             video_writer.close()

        #             success_or_not.append(
        #                 success.to(torch.uint8).cpu().numpy())
        # if count % 5 == 0:
        #     success_id = torch.where(success)[0]
        #     if len(success_id) > 0:
        #         import random
        #         success_id = random.choice(success_id)
        #         video_writer = imageio.get_writer(
        #             os.path.join(result_path,
        #                          f"video_{count+success_id}_success.mp4"),
        #             fps=30,
        #         )

        #         for image in policy_env.diffusion_env.image_buffer:
        #             video_writer.append_data(image[success_id])
        #         video_writer.close()

        count += len(success)

        success_count += success.sum().item()
        print(args_cli.action_framework, args_cli.target_object_name,
              "Succuss Rate: ", success_count / count, count)


def collect_bc_policy(args_cli, policy_env):

    count = 0
    success = policy_env.step()

    while count < (args_cli.num_demos + policy_env.env.num_envs):

        success = policy_env.step()
        count += policy_env.env.num_envs


def analyze_bc_policy(args_cli, policy_env, result_path):
    analysis_dict = []
    for coord in policy_env.diffusion_env.object_coords:

        success = policy_env.step()
        analysis_dict.append([coord.tolist(), success])
        print("Coordinate: ", coord.tolist(), " Success Rate: ", success)
        np.save(result_path + "/analysis.npy",
                np.array(analysis_dict, dtype=object))


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs

    if args_cli.target_object_name is not None:
        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    save_config["params"]["Camera"][
        "random_pose"] = args_cli.random_camera_pose
    save_config["params"]["action_framework"] = args_cli.action_framework
    save_config["params"]["eval_mode"] = True
    if save_config["params"].get("adr") is not None:
        save_config["params"]["adr"]["init"] = args_cli.adr

    # if args_cli.action_framework == "state_diffusion":
    #     # save_config["params"]["eval_mode"] = True
    #     save_config["params"]["Camera"][
    #         "initial"] = False  # disable initial camera pose

    save_config["params"]["real_eval_mode"] = args_cli.real_eval_mode
    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    obs, _ = env.reset()

    # for i in range(2):

    #     for i in range(30):
    #         env.step(torch.as_tensor(env.action_space.sample()).to(env.device))
    #     last_obs = env.reset()

    policy_env = HandBCEnvWrapper(env, save_config, args_cli)
    count = 0

    result_path = os.path.join(
        args_cli.log_dir, "eval_results", args_cli.target_object_name
        if args_cli.target_object_name is not None else "all")
    os.makedirs(result_path, exist_ok=True)

    if args_cli.analysis:

        analyze_bc_policy(
            args_cli,
            policy_env,
            result_path,
        )
    elif args_cli.save_path is not None:
        collect_bc_policy(
            args_cli,
            policy_env,
        )
    else:
        evaluate_bc_policy(
            args_cli,
            policy_env,
            count,
            result_path,
        )

    # set the action to zero

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
