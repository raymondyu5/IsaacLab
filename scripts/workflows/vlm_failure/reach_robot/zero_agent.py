from isaacsim import SimulationApp
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
from scripts.workflows.vlm_failure.reach_robot.task.plan_env import MotionPlanReach


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def main():
    # parse configuration
    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment

    env = setup_env(args_cli, save_config)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    last_obs, _ = env.reset()

    motionplaner_env = MotionPlanReach(
        env,
        save_config,
        args_cli,
        use_ik_pose=True if "Rel" in args_cli.task else False)
    motionplaner_env.reset_env()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            last_obs = motionplaner_env.reset_env()
            motionplaner_env.step_motion(last_obs)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
