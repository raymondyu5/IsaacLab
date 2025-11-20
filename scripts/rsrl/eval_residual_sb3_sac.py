import os

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.sb3.rl_algo_wrapper import rl_parser, initalize_rl_env
import sys
# launch omniverse app
import yaml
import gymnasium as gym
import torch

AppLauncher.add_app_launcher_args(rl_parser)
args_cli, _ = rl_parser.parse_known_args(sys.argv[1:])

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from scripts.serl.env.residual_rl_wrapper import ResidualRLDatawrapperEnv


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    return gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None).unwrapped


from scripts.rsrl.env.sb3_wrapper import Sb3EnvWrapper, obs_keys
from scripts.rsrl.agent.residual_sac import ResidualSAC
from scripts.sb3.td3 import TD3


def main():
    """Zero actions agent with Isaac Lab environment."""

    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["rl_train"] = True

    save_config["params"]["real_eval_mode"] = True
    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    if "Joint-Rel" in args_cli.task:
        args_cli.log_dir = os.path.join(args_cli.log_dir, "joint_pose")

    env = setup_env(args_cli, save_config)
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample() * 0.0

    obs, _ = env.reset()
    video_kwargs = {
        "video_folder": os.path.join(args_cli.log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    rl_env = ResidualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
        obs_keys=obs_keys,
        eval_mode=True)

    # wrap around environment for stable baselines
    rl_agent_env = Sb3EnvWrapper(rl_env)
    checkpoint_path = args_cli.checkpoint

    if args_cli.rl_type == "td3":
        agent = TD3
    else:
        agent = agent

    if "zip" in checkpoint_path:

        agent = agent.load(checkpoint_path,
                           rl_agent_env,
                           print_system_info=False)

    else:

        agent = agent

    rl_env.reset()

    for i in range(10):
        rl_env.env.step(torch.zeros((env.num_envs, 22), device=rl_env.device))
    rl_env.reset()
    rl_env.init_eval_result_folder()

    while simulation_app.is_running():
        # run everything in inference mode

        with torch.inference_mode():

            if "zip" in checkpoint_path:
                # print("eval base policy success rate")
                # success_or_not, _ = rl_env.eval_checkpoint(agent,
                #                                            rl_agent_env,
                #                                            bc_eval=True)
                print("eval rl success rate")

                success_or_not, stop = rl_env.eval_checkpoint(
                    agent, rl_agent_env)
                # success_or_not = rl_env.eval_symmetry(agent, last_obs)
            else:
                all_ckpt = os.listdir(checkpoint_path)
                zip_ckpt = [f for f in all_ckpt if f.endswith(".zip")]
                zip_ckpt.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
                success_or_not = rl_env.eval_all_checkpoint(
                    agent, rl_agent_env, zip_ckpt[0])
                rl_env.eval_env.eval_iter = 0

                for model_path in zip_ckpt:

                    print(f"Evaluating {model_path}")

                    success_or_not = rl_env.eval_all_checkpoint(
                        agent, rl_agent_env, model_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
