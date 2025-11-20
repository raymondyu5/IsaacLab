import imageio.v3 as iio
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

enable_extension("omni.isaac.debug_draw")
from omni.isaac.debug_draw import _debug_draw

from scripts.workflows.open_policy.task.openvla_env import OpenVlaEvalEnv

from scripts.workflows.utils.client.openvla_client import OpenVlaClient
import imageio


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


# Path to the video file
video_path = "/home/ensu/Downloads/3_True.mp4"

# Create a directory to save the extracted frames (optional)
import os

output_dir = "logs/"
os.makedirs(output_dir, exist_ok=True)

# Read the video and extract frames
reader = iio.imiter(
    video_path, plugin="pyav")  # Use the PyAV plugin for better performance

frame_count = 0
images_buffer = []
for frame in reader:
    # Save each frame as an image
    # frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    # iio.imwrite(frame_path, frame)
    # frame_count += 1
    # import pdb
    # pdb.set_trace()
    # print(f"Saved {frame_path}")
    images_buffer.append(frame)


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

    env.reset()

    openval_env = OpenVlaEvalEnv(
        env,
        save_config,
        prompt=
        "In: What action should the robot take to put eggplant into yellow basket?\nOut:",
        use_relative_pose=True if "Rel" in args_cli.task else False,
        inference_client=OpenVlaClient(args_cli.client_url),
        render_camera=save_config["params"]["Task"]["render_camera"])
    count = 0

    # simulate environment
    while simulation_app.is_running():
        # gs_env.step()
        last_obs = openval_env.reset()
        action_buffer = []

        for index, image in enumerate(images_buffer):
            print(index)
            response = openval_env.inference_client.step(
                "In: What action should the robot take to put eggplant into yellow basket?\nOut:",
                image[:, :, ::-1])
            actions = torch.as_tensor(response["action"],
                                      device=env.device).unsqueeze(0)
            action_buffer.append(actions)

            obs, rewards, terminated, time_outs, extras = env.step(actions)

            print(obs["policy"]["ee_pose"][:, :3] -
                  last_obs["policy"]["ee_pose"][:, :3])
            last_obs = obs
        action_buffer = torch.cat(action_buffer, dim=0)
        import numpy as np
        np.save("action_buffer.npy", action_buffer.cpu().numpy())


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
