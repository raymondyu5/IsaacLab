import torch

torch.set_default_dtype(torch.float32)  # tensors default to float32
torch.set_default_tensor_type(torch.FloatTensor)  # also enforces CPU tensors

import sys

sys.path.append("submodule/stable-baselines3")

from scripts.sb3.td3_bc import TD3BC
from scripts.rsrl.utils.td3bc_utils import *

from scripts.rsrl.utils.td3bc_loader import DataLoader

from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
import os

import argparse
from scripts.offline_rl.utils.dataloader_utils import populate_data_store_from_zarr_to_sb3, ReplayBuffer


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

    parser.add_argument("--lowdim_keys",
                        type=str,
                        nargs="+",
                        default=[
                            'right_hand_joint_pos',
                            "right_manipulated_object_pose",
                            "right_target_object_pose"
                        ],
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
    parser.add_argument("--num_demo", type=int, default=100)

    return parser.parse_args()


class DummyEnv:

    def __init__(self,
                 use_pcd_obs=False,
                 use_visual_obs=False,
                 data_path=None,
                 target_cam_id=None,
                 lowdim_keys=[
                     'right_hand_joint_pos', "right_manipulated_object_pose",
                     "right_target_object_pose"
                 ],
                 rollout_id=0,
                 save_dir="logs",
                 q_type="sum",
                 use_latent_noise=False,
                 num_demo=10,
                 demo_path=None):

        self.action_space = action_space
        if use_visual_obs:

            policy_cfg = td3_cfg | visual_policy_kwargs
            observation_space = visual_observation_space
            visual_keys = ["rgb"]
        elif use_pcd_obs:
            policy_cfg = td3_cfg | pcd_policy_kwargs
            observation_space = pcd_observation_space
            visual_keys = ["seg_pc"]
        else:

            td3_cfg["policy"] = "MlpPolicy"
            policy_cfg = td3_cfg | state_policy_kwargs
            observation_space = lowdim_observation_space

        policy_arch = policy_cfg.pop("policy")
        policy_cfg.pop("seed")
        n_timesteps = policy_cfg.pop("n_timesteps")

        policy_cfg["policy_kwargs"].update({"q_type": q_type})
        latent_model = None

        self.algo = TD3BC(policy_arch,
                          observation_space=observation_space,
                          action_space=action_space
                          if not use_latent_noise else latent_action_space,
                          verbose=1,
                          latent_model=latent_model,
                          lambda_bc=0.0,
                          **policy_cfg)
        populate_data_store_from_zarr_to_sb3(self.algo.replay_buffer,
                                             demo_path,
                                             lowdim_keys,
                                             num_demos=num_demo)

        setup_wandb(policy_cfg, "real_rl", tags=None, project="real_rl")
        self.callback = WandbCallback(model_save_freq=100,
                                      video_folder=None,
                                      model_save_path=str(save_dir),
                                      eval_freq=1000,
                                      eval_cam_names=None,
                                      viz_point_cloud=False,
                                      viz_pc_env=None,
                                      rollout_id=rollout_id)
        total_timesteps = 1000

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
        num_demo=args.num_demo,
        demo_path=
        "logs/data_0705/retarget_visionpro_data/rl_data/data/ours/image/bunny")

    print(env.algo)
