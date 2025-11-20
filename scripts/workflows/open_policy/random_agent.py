# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
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
from isaacsim.core.utils.extensions import enable_extension


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

    env = setup_env(args_cli, save_config)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    last_obs, _ = env.reset()
    import os
    os.makedirs(f"{args_cli.log_dir}/mesh", exist_ok=True)

    # env.scene["robot"]
    # from pxr import Usd, UsdGeom
    # from curobo.util.usd_helper import UsdHelper
    # import trimesh
    # import numpy as np
    # usd_help = UsdHelper()
    # usd_help.load_stage(env.scene.stage)
    # all_items = usd_help.stage.Traverse()
    # prim = [
    #     prim for prim in all_items
    #     if prim.IsA(UsdGeom.Mesh) and "panda" in prim.GetPath().pathString
    # ]

    # for p in prim:
    #     if (p.GetAttribute("points").Get() is not None):

    #         faces = list(p.GetAttribute("faceVertexIndices").Get())

    #         faces = np.array(faces).reshape(-1, 3)
    #         vertices = np.array(list(p.GetAttribute("points").Get())).reshape(
    #             -1, 3)
    #         mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #         name = {p.GetPath().pathString.replace("/", "_")}
    #         mesh.export(f"{args_cli.log_dir}/mesh/{name}.obj")

    # simulate environment
    dof = torch.tensor([[
        -3.9149e-09, -6.2832e-01, -4.7145e-08, -2.5133e+00, 1.5885e-06,
        1.8850e+00, -0.0, 2.2977e-02, 2.4186e-02, -1.5064e-03, -1.5263e-03,
        -3.2472e-02, 3.3726e-02, -3.2033e-02, -3.3312e-02
    ]],
                       device=env.device)

    while simulation_app.is_running():

        for i in range(100):

            # env.scene["robot"].root_physx_view.set_dof_positions(
            #     dof,
            #     torch.arange(1).to(env.device))

            actions = 0 * torch.rand(env.action_space.shape,
                                     device=env.unwrapped.device)
            actions[:, -1] = 1.0
            obs, rewards, terminated, time_outs, extras = env.step(actions)

            # import cv2
            # cv2.imwrite(
            #     "obs.png",
            #     obs["policy"]["gs_image"][0][0].cpu().numpy()[:, :, ::-1])

        # apply actions
        env.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
