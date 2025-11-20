import numpy as np

import torch

import sys
import os

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation)
import time


class DiffusionVerifyWrapper:

    def __init__(self, env, args_cli, num_hand_joints=16):
        self.args_cli = args_cli
        self.device = args_cli.device
        self.env = env
        self.num_hand_joints = num_hand_joints
        self.hand_side = "left" if args_cli.add_left_hand else "right"
        self.load_diffusion_model()
        self.actions_buffer = None
        self.raw_data = None
        self.demo_count = 0
        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)
        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=2,
            horizon_K=self.policy.horizon,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.num_hand_joints,
        )

    def load_diffusion_model(self):

        checkpoint = os.path.join(self.args_cli.vae_path, "checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']
        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_unet_hand_policy.TrainCFMUnetLowdimWorkspace"

        cfg.policy.num_inference_steps = 3
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, )
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device(self.device)
        self.policy.to(device)
        self.policy.eval()
        self.chunk_size = self.policy.n_obs_steps

    def verify_diffusion(self, ):

        self.env.reset()
        print("new demo")
        demo_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device).to(torch.float32)

        raw_hand_pose = demo_actions[..., -self.num_hand_joints:].clone()
        num_chunks = raw_hand_pose.shape[0] - self.policy.horizon + 1

        self.temporal_obs_buffer.reset(num_chunks * self.policy.horizon,
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(num_chunks * self.policy.horizon,
                                          self.env.num_envs)

        step_count = 0
        self.add_obs_to_buffer(step_count)

        with torch.no_grad():
            for index in range(1, num_chunks - 1):

                obs_list = torch.as_tensor(
                    raw_hand_pose[index - 1:index - 1 + self.chunk_size]).to(
                        self.device).to(torch.float32).reshape(
                            1, -1, self.num_hand_joints)

                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                obs_dict = {
                    "obs":
                    torch.cat(
                        [
                            obs_list.clone(), normalized_joint_des[-1].reshape(
                                1, -1, self.num_hand_joints)
                            # (torch.rand(
                            #     size=(1, self.chunk_size, self.num_hand_joints),
                            #     dtype=obs_list.dtype) * 2 - 1).to(self.device) *
                            # 0.2
                        ],
                        dim=0)
                }

                # obs_dict = {"obs": normalized_joint_des}

                noise = (torch.randn(size=(2, self.policy.horizon,
                                           self.num_hand_joints),
                                     dtype=normalized_joint_des.dtype,
                                     device=normalized_joint_des.device,
                                     generator=None))

                recontructed_hand_pose = extract_finger_joints(
                    self.policy.predict_action(obs_dict, noise)["action_pred"],
                    self.joint_limits)

                action = torch.zeros(self.env.action_space.shape).to(
                    self.device)

                for i in range(int(recontructed_hand_pose.shape[1])):

                    action[:, 6:] = recontructed_hand_pose[:, i]

                    demo_action = action.clone()

                    self.env.step(demo_action)
                    step_count += 1
                    self.add_obs_to_buffer(step_count)
                    torch.cuda.empty_cache()
        self.demo_count = torch.randint(0, len(self.raw_data), (1, )).item()

    def add_obs_to_buffer(self, index):
        state = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        joint_min = self.joint_limits[:, 0]  # shape [16]
        joint_max = self.joint_limits[:, 1]  # shape [16]
        normalized_joints = ((state - joint_min[None, :]) /
                             (joint_max - joint_min)[None, :]) * 2 - 1

        self.temporal_obs_buffer.add_obs(index, normalized_joints)

    def random_play_diffusion(self):
        self.env.reset()

        self.temporal_obs_buffer.reset(200 * self.policy.horizon,
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(200 * self.policy.horizon,
                                          self.env.num_envs)
        for i in range(20):
            actions = torch.zeros(self.env.action_space.shape).to(self.device)
            self.env.step(actions)

        step_count = 0
        self.add_obs_to_buffer(step_count)

        for index in range(180):

            with torch.no_grad():

                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                obs_dict = {"obs": normalized_joint_des}

                noise = (torch.randn(size=(self.env.num_envs,
                                           self.policy.horizon,
                                           self.num_hand_joints),
                                     dtype=normalized_joint_des.dtype,
                                     device=normalized_joint_des.device,
                                     generator=None))

                reconstructed_hand_actions = extract_finger_joints(
                    self.policy.predict_action(obs_dict, noise)["action_pred"],
                    self.joint_limits)

                # self.temporal_action_buffer.add_prediction(
                #     index - 1, reconstructed_hand_actions)
                # hand_action = self.temporal_action_buffer.compute_action()

            action = torch.zeros(self.env.action_space.shape).to(self.device)

            # for i in range(int(reconstructed_hand_actions.shape[1])):
            action[:, 6:] = reconstructed_hand_actions[:, -1]
            # action[:, 6:] = hand_action

            self.env.step(action)
            step_count += 1

            self.add_obs_to_buffer(step_count)
