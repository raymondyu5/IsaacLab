import torch

torch.set_default_dtype(torch.float32)  # tensors default to float32
torch.set_default_tensor_type(torch.FloatTensor)  # also enforces CPU tensors

import sys

sys.path.append("submodule/stable-baselines3")

from scripts.sb3.td3_bc import TD3BC
from scripts.rsrl.utils.td3bc_utils import *

from scripts.rsrl.utils.td3bc_loader import DataLoader
from scripts.utils.rl_utils.env.diffusion_wrapper import DiffusionWrapper
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
import os

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="TD3-BC training with DummyEnv")

    parser.add_argument("--use_pcd_obs",
                        action="store_true",
                        help="Use point cloud observations")
    parser.add_argument("--use_visual_obs",
                        action="store_true",
                        help="Use visual observations (rgb/depth)")
    parser.add_argument("--data_path",
                        type=str,
                        default=None,
                        help="Path to offline dataset")
    parser.add_argument("--target_cam_id",
                        type=str,
                        nargs="+",
                        default=["CL8384200N1"],
                        help="Target camera IDs (space separated list)")

    parser.add_argument(
        "--lowdim_keys",
        type=str,
        nargs="+",
        default=["joint_positions", "gripper_position", "base_action"],
        help="Low-dimensional keys")
    parser.add_argument("--rollout_id",
                        type=int,
                        default=0,
                        help="Rollout ID for saving logs/checkpoints")
    parser.add_argument("--save_dir",
                        type=str,
                        default="logs",
                        help="Directory to save logs and models")
    parser.add_argument("--q_type",
                        type=str,
                        choices=["sum", "res", "concat"],
                        default="res",
                        help="Q-function type")

    parser.add_argument("--load_list",
                        nargs='+',
                        default=None,
                        help="List of load paths")

    parser.add_argument("--use_latent_noise", action="store_true")
    parser.add_argument("--use_latent_noise_only", action="store_true")
    parser.add_argument("--num_demo", type=int, default=10)

    return parser.parse_args()


class DummyEnv:

    def __init__(
        self,
        use_pcd_obs=False,
        use_visual_obs=False,
        data_path=None,
        target_cam_id=None,
        lowdim_keys=["joint_positions", "gripper_position", "base_action"],
        rollout_id=0,
        save_dir="logs",
        q_type="sum",
        use_latent_noise=False,
        diffusion_path=None,
        diffusion_ckpt=None,
    ):

        if use_visual_obs:

            policy_cfg = td3_cfg | visual_policy_kwargs
            observation_space = visual_observation_space
            visual_keys = ["rgb"]
        if use_pcd_obs:
            policy_cfg = td3_cfg | pcd_policy_kwargs
            observation_space = pcd_observation_space
            visual_keys = ["seg_pc"]

        policy_arch = policy_cfg.pop("policy")
        policy_cfg.pop("seed")
        n_timesteps = policy_cfg.pop("n_timesteps")

        policy_cfg["policy_kwargs"].update({"q_type": q_type})
        latent_model = None

        if use_latent_noise:

            latent_model = DiffusionWrapper(diffusion_path=diffusion_path,
                                            target_cam_id=target_cam_id,
                                            diffusion_ckpt=diffusion_ckpt)

        if not args.use_latent_noise:
            action_space = action_space
        else:
            if args.use_latent_noise_only:
                action_space = only_latent_action_space
            else:
                action_space = latent_action_space

        self.algo = TD3BC(
            policy_arch,
            observation_space=observation_space,
            action_space=action_space,
            verbose=1,
            latent_model=latent_model,
            lambda_bc=0.2 if not args.use_latent_noise_only else 0.0,
            use_latent_noise_only=args.use_latent_noise_only,
            **policy_cfg)
        self.use_latent_noise = use_latent_noise

        self.data_loader = DataLoader(
            self.algo.replay_buffer,
            visual_keys=visual_keys,
            lowdim_keys=lowdim_keys,
            data_path=data_path,
            target_cam_id=target_cam_id,
            load_list=args.load_list,
            use_latent_noise=use_latent_noise,
            use_latent_noise_only=args.use_latent_noise_only,
            num_demo=args.num_demo)

        setup_wandb(policy_cfg, "real_rl", tags=None, project="real_rl")
        self.callback = WandbCallback(model_save_freq=100,
                                      video_folder=None,
                                      model_save_path=str(save_dir),
                                      eval_freq=1000,
                                      eval_cam_names=None,
                                      viz_point_cloud=False,
                                      viz_pc_env=None,
                                      rollout_id=rollout_id)
        total_timesteps = 300

        total_timesteps, callback = self.algo._setup_learn(
            total_timesteps,
            self.callback,
            reset_num_timesteps=True,
            tb_log_name="SAC",
            progress_bar=False,
        )

        callback.on_training_start(locals(), globals())
        self.learn_policy()

    def learn_policy(self):
        total_timesteps = 200
        while self.algo.num_timesteps < total_timesteps:

            self.algo.train(self.algo,
                            batch_size=self.algo.batch_size,
                            gradient_steps=self.algo.gradient_steps,
                            callback=self.callback)

            self.algo.num_timesteps += 1


if __name__ == "__main__":
    args = get_args()

    env = DummyEnv(
        use_pcd_obs=args.use_pcd_obs,
        use_visual_obs=args.use_visual_obs,
        data_path=args.data_path,
        target_cam_id=args.target_cam_id,
        lowdim_keys=args.lowdim_keys,
        rollout_id=args.rollout_id,
        save_dir=args.save_dir,
        q_type=args.q_type,
        use_latent_noise=args.use_latent_noise,
        diffusion_path=
        "logs/trash/base_policy/zeroshot_plush/cfm/pcd_cfm/horizon_1_nobs_1",
        diffusion_ckpt="checkpoint_300.ckpt",
    )

    print(env.algo)
