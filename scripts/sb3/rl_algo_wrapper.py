import sys

sys.path.append("submodule/stable-baselines3")

from scripts.sb3.ppo import PPO
from scripts.sb3.rfs.RFSTD3 import RFSTD3
from scripts.sb3.rfs.DSRLTD3 import DSRLTD3
from scripts.sb3.ppo_bc import PPOBC
from scripts.sb3.sac import SAC
from scripts.rsrl.agent.residual_sac import ResidualSAC
from scripts.sb3.td3 import TD3
from scripts.sb3.rfs.DSRL import DSRL
from scripts.sb3.sb3_rlpd import SACRLPD
import numpy as np
import os
import random
from datetime import datetime

import os
import yaml
from box import Box
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
from scripts.workflows.utils.parse_setting import parser
import copy
import torch

rl_parser = copy.deepcopy(parser)
rl_parser.add_argument("--seed",
                       type=int,
                       default=None,
                       help="Seed used for the environment")
rl_parser.add_argument("--max_iterations",
                       type=int,
                       default=None,
                       help="RL Policy training iterations.")

rl_parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
rl_parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

rl_parser.add_argument(
    "--action_framework",
    default=None,
)

rl_parser.add_argument(
    "--diffusion_path",
    default=None,
)
rl_parser.add_argument(
    "--vae_path",
    default=None,
)

rl_parser.add_argument(
    "--latent_dim",
    default=32,
    type=int,
)
rl_parser.add_argument(
    "--use_relative_finger_pose",
    action="store_true",
)
rl_parser.add_argument(
    "--rl_type",
    default="ppo",
)

rl_parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)

rl_parser.add_argument(
    "--bc_dir",
    default=None,
    type=str,
)

rl_parser.add_argument(
    "--use_residual_action",
    action="store_true",
)

rl_parser.add_argument(
    "--use_chunk_action",
    action="store_true",
)
rl_parser.add_argument(
    "--use_interpolate_chunk",
    action="store_true",
)
rl_parser.add_argument(
    "--residual_step",
    default=1,
    type=int,
)

rl_parser.add_argument(
    "--resume",
    action="store_true",
)

rl_parser.add_argument(
    "--checkpoint",
    default=None,
)

rl_parser.add_argument("--video",
                       action="store_true",
                       default=False,
                       help="Record videos during training.")

rl_parser.add_argument("--video_length",
                       type=int,
                       default=200,
                       help="Length of the recorded video (in steps).")
rl_parser.add_argument("--video_interval",
                       type=int,
                       default=10000,
                       help="Interval between video recordings (in steps).")

rl_parser.add_argument("--distributed",
                       action="store_true",
                       default=False,
                       help="Run training with multiple GPUs or nodes.")
rl_parser.add_argument(
    "--use_visual_obs",
    action="store_true",
)

rl_parser.add_argument(
    "--target_object_name",
    type=str,
    default=None,  # Options: tomato_soup_can, banana, cereal_box, etc.
)

rl_parser.add_argument(
    "--use_base_action",
    action="store_true",
)

rl_parser.add_argument(
    "--random_camera_pose",
    action="store_true",
)

rl_parser.add_argument(
    "--diffusion_checkpoint",
    type=str,
    default="latest",  # Options: open_loop, close_loop, replay
)

rl_parser.add_argument(
    "--eval_mode",
    type=str,
    default="close_loop",  # Options: close_loop, close_loop, replay
)

rl_parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help=
    "When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

rl_parser.add_argument(
    "--collect_relative_finger_pose",
    action="store_true",
)

rl_parser.add_argument(
    "--real_eval_mode",
    action="store_true",
)

rl_parser.add_argument(
    "--analysis",
    action="store_true",
)
rl_parser.add_argument("--failure_ratio", default=0.0)

rl_parser.add_argument(
    "--action_range",
    nargs="+",  # means accept 1 or more args
    type=float,  # convert each to float
    default=None)

rl_parser.add_argument(
    "--demo_path",
    type=str,
    default=None,  # Options: tomato_soup_can, banana, cereal_box, etc.
)
rl_parser.add_argument(
    "--load_critic",
    action="store_true",
)

rl_parser.add_argument(
    "--load_actor",
    action="store_true",
)

rl_parser.add_argument(
    "--compute_sdf",
    action="store_true",
)

rl_parser.add_argument(
    "--eval_disturbance",
    action="store_true",
)

rl_parser.add_argument(
    "--use_residual_only",
    action="store_true",
)

rl_parser.add_argument(
    "--revert_action",
    action="store_true",
)
rl_parser.add_argument(
    "--adr",
    action="store_true",
)


# add dsrl
def initalize_rl_env(args_cli,
                     rl_agent_env,
                     save_dir,
                     rl_env=None,
                     wamrup=False,
                     model_save_freq=50):
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    agent_cfg = Box(yaml_data)
    agent_cfg[
        "seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg[
            "seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg[
            "n_steps"] * args_cli.num_envs
    agent_cfg.seed = agent_cfg["seed"]
    from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import process_sb3_cfg
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create agent from stable baselines
    exp_name = [f"{args_cli.rl_type}"]
    if "Joint" in args_cli.task:
        exp_name.append("Joint")
    if "Abs" in args_cli.task:
        exp_name.append("abs")
    else:
        exp_name.append("rel")
    if args_cli.action_framework is not None:
        exp_name.append(args_cli.action_framework)

    if args_cli.use_residual_action:
        exp_name.append("residual")
    if args_cli.use_chunk_action:
        exp_name.append("chunk")

    if args_cli.rl_type == "ppo":

        agent = PPO(policy_arch,
                    rl_agent_env,
                    verbose=1,
                    gpu_buffer=False,
                    **agent_cfg)

    elif args_cli.rl_type == "sac":
        agent = SAC(policy_arch, rl_agent_env, verbose=1, **agent_cfg)
    elif args_cli.rl_type == "residual_sac":
        agent = ResidualSAC(policy_arch, rl_agent_env, verbose=1, **agent_cfg)
    elif args_cli.rl_type == "td3":

        from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
        if agent_cfg.get("action_noise", None) is not None:
            agent_cfg["action_noise"] = VectorizedActionNoise(
                NormalActionNoise(np.zeros(rl_agent_env.action_space.shape) +
                                  agent_cfg["action_noise"]["mean"],
                                  sigma=agent_cfg["action_noise"]["std"] *
                                  np.ones(rl_agent_env.action_space.shape)),
                rl_agent_env.num_envs)
        agent = TD3(
            policy_arch,
            rl_agent_env,
            verbose=1,
            freeze_critic=args_cli.load_critic,
            freeze_actor=args_cli.load_actor,
            **agent_cfg,
        )
    elif args_cli.rl_type == "rfstd3":

        from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
        if agent_cfg.get("action_noise", None) is not None:
            agent_cfg["action_noise"] = VectorizedActionNoise(
                NormalActionNoise(np.zeros(rl_agent_env.action_space.shape) +
                                  agent_cfg["action_noise"]["mean"],
                                  sigma=agent_cfg["action_noise"]["std"] *
                                  np.ones(rl_agent_env.action_space.shape)),
                rl_agent_env.num_envs)

        agent = RFSTD3(
            policy_arch,
            rl_agent_env,
            verbose=1,
            freeze_critic=args_cli.load_critic,
            freeze_actor=args_cli.load_actor,
            **agent_cfg,
        )

    rollout_id = 0
    if args_cli.resume:

        agent = agent.load(args_cli.checkpoint,
                           rl_agent_env,
                           print_system_info=True,
                           load_actor_only=args_cli.load_actor,
                           load_critic_only=args_cli.load_critic)
        setattr(agent, "freeze_critic", args_cli.load_critic)
        setattr(agent, "freeze_actor", args_cli.load_actor)
        rollout_id = int(
            args_cli.checkpoint.split("/")[-1].split(".")[0].split("_")[-1])
    # if wamrup:
    #     rl_env.warmup(agent, bc_eval=True)

    setup_wandb(args_cli, "_".join(exp_name), tags=None, project="isaaclab_rl")
    if rl_env is not None:
        wandb_name = "_".join(exp_name)
        save_dir += f"/{wandb_name}"

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        iteration=rollout_id,
        callback=WandbCallback(
            model_save_freq=model_save_freq,
            video_folder=os.path.join(save_dir, "videos", "train"),
            model_save_path=str(save_dir + f"/{args_cli.rl_type}"),
            eval_env_fn=rl_agent_env,
            eval_freq=1000,
            eval_cam_names=None,
            viz_point_cloud=False,
            viz_pc_env=None,
            rollout_id=rollout_id),
    )


def initalize_rfs_env(args_cli,
                      rl_agent_env,
                      save_dir,
                      rl_env=None,
                      diffusion_model=False,
                      diffusion_obs_space=None,
                      diffusion_action_space=None,
                      model_save_freq=50):
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    with open(args_cli.rl_config, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    agent_cfg = Box(yaml_data)
    agent_cfg[
        "seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg[
            "seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg[
            "n_steps"] * args_cli.num_envs
    agent_cfg.seed = agent_cfg["seed"]
    from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import process_sb3_cfg
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create agent from stable baselines
    exp_name = [f"{args_cli.rl_type}"]

    if args_cli.use_residual_action:
        exp_name.append("residual")
    if args_cli.use_chunk_action:
        exp_name.append("chunk")

    if args_cli.rl_type == "rfstd3":

        from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
        if agent_cfg.get("action_noise", None) is not None:
            agent_cfg["action_noise"] = VectorizedActionNoise(
                NormalActionNoise(np.zeros(rl_agent_env.action_space.shape) +
                                  agent_cfg["action_noise"]["mean"],
                                  sigma=agent_cfg["action_noise"]["std"] *
                                  np.ones(rl_agent_env.action_space.shape)),
                rl_agent_env.num_envs)

        agent = RFSTD3(
            policy_arch,
            rl_agent_env,
            verbose=1,
            freeze_critic=args_cli.load_critic,
            freeze_actor=args_cli.load_actor,
            **agent_cfg,
        )
        exp_name.append("rsf")
    elif args_cli.rl_type == "dsrl":
        agent = DSRL(
            policy_arch,
            rl_agent_env,
            verbose=1,
            diffusion_model=diffusion_model,
            diffusion_obs_space=diffusion_obs_space,
            diffusion_action_space=diffusion_action_space,
            **agent_cfg,
        )
        exp_name.append("dsrl")
    elif args_cli.rl_type == "dsrltd3":
        agent = DSRLTD3(
            policy_arch,
            rl_agent_env,
            verbose=1,
            diffusion_model=diffusion_model,
            diffusion_obs_space=diffusion_obs_space,
            diffusion_action_space=diffusion_action_space,
            **agent_cfg,
        )
        exp_name.append("dsrltd3")
    elif args_cli.rl_type == "td3":
        from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
        if agent_cfg.get("action_noise", None) is not None:
            agent_cfg["action_noise"] = VectorizedActionNoise(
                NormalActionNoise(np.zeros(rl_agent_env.action_space.shape) +
                                  agent_cfg["action_noise"]["mean"],
                                  sigma=agent_cfg["action_noise"]["std"] *
                                  np.ones(rl_agent_env.action_space.shape)),
                rl_agent_env.num_envs)
        agent = TD3(
            policy_arch,
            rl_agent_env,
            verbose=1,
            freeze_critic=args_cli.load_critic,
            freeze_actor=args_cli.load_actor,
            **agent_cfg,
        )
        exp_name.append("td3")

    rollout_id = 0
    if args_cli.resume:

        agent = agent.load(args_cli.checkpoint,
                           rl_agent_env,
                           print_system_info=True,
                           load_actor_only=args_cli.load_actor,
                           load_critic_only=args_cli.load_critic)

        rollout_id = int(
            args_cli.checkpoint.split("/")[-1].split(".")[0].split("_")[-1])
    # if wamrup:
    #     rl_env.warmup(agent, bc_eval=True)

    setup_wandb(args_cli, "_".join(exp_name), tags=None, project="isaaclab_rl")
    if rl_env is not None:
        wandb_name = "_".join(exp_name)
        save_dir += f"/{wandb_name}"

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        callback=WandbCallback(
            model_save_freq=model_save_freq,
            video_folder=os.path.join(save_dir, "videos", "train"),
            model_save_path=str(save_dir + f"/{args_cli.rl_type}"),
            eval_env_fn=rl_agent_env,
            eval_freq=1000,
            eval_cam_names=None,
            viz_point_cloud=False,
            viz_pc_env=None,
            rollout_id=rollout_id),
    )
