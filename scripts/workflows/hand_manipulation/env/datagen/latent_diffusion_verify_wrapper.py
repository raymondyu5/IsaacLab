import numpy as np

import torch

import sys
import os
import einops

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation, load_latent_action, load_config)

from pythae.models import AutoModel


class LatentDiffusionVerifyWrapper:

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
            horizon_K=self.policy.input_dim_h,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.num_hand_joints,
        )
        self.chunk_size = self.policy.horizon

    def load_diffusion_model(self):

        checkpoint = os.path.join(f"{self.args_cli.vae_path}/checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, args_cli=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload,
                               exclude_keys=["normalizer"],
                               include_keys=None)
        self.policy = workspace.model

        self.max_latent_value = self.policy.max_latent_value
        self.min_latent_value = self.policy.min_latent_value

    def verify_diffusion(self, ):

        self.env.reset()
        print("new demo")

        demo_latent_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device).to(torch.float32)[..., -self.num_hand_joints:]

        demo_gt_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"][
                ...,
                -self.num_hand_joints:])).to(self.device).to(torch.float32)

        self.temporal_action_buffer.reset(demo_gt_actions.shape[0])

        num_chunks = (demo_latent_actions.shape[0] - self.policy.input_dim_h +
                      1)
        self.temporal_obs_buffer.reset(num_chunks * self.policy.input_dim_h,
                                       self.env.num_envs)

        with torch.no_grad():
            step_count = 0
            self.add_obs_to_buffer(step_count)

            for index in range(1, num_chunks - 1):

                act_list = torch.as_tensor(
                    demo_latent_actions[index:index + self.chunk_size]).to(
                        self.device).to(torch.float32).reshape(
                            1, -1, self.num_hand_joints)
                obs_list = torch.as_tensor(
                    demo_latent_actions[index - 1:index - 1 +
                                        self.chunk_size]).to(self.device).to(
                                            torch.float32).reshape(
                                                1, -1, self.num_hand_joints)
                result = self.policy.eval_step({
                    "action": act_list,
                    "obs": obs_list
                })

                reconstruct_action = einops.rearrange(result["dec_out"],
                                                      "N (T A) -> N T A",
                                                      A=16)
                reconstruct_action = self.policy.normalizer[
                    "action"].unnormalize(reconstruct_action)

                gt_actions_chunks = extract_finger_joints(
                    torch.as_tensor(
                        demo_gt_actions[index:index + self.chunk_size]).to(
                            self.device).to(torch.float32).reshape(
                                -1, self.num_hand_joints), self.joint_limits)

                recontructed_hand_pose = extract_finger_joints(
                    reconstruct_action.to(self.device), self.joint_limits)

                for i in range(recontructed_hand_pose.shape[1]):
                    action = torch.zeros(self.env.action_space.shape).to(
                        self.device)

                    action[0, 6:] = recontructed_hand_pose[:, i].unsqueeze(0)
                    action[1, 6:] = gt_actions_chunks[i]

                    demo_action = action.clone()

                    self.env.step(demo_action)

                    step_count += 1
                    self.add_obs_to_buffer(step_count)

                torch.cuda.empty_cache()
        self.demo_count += 1

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

        self.temporal_obs_buffer.reset(200 * self.policy.input_dim_h,
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(200 * self.policy.input_dim_h,
                                          self.env.num_envs)

        for i in range(20):
            actions = torch.zeros(self.env.action_space.shape).to(self.device)
            # actions[:, -self.num_hand_joints:] = extract_finger_joints(
            #     torch.as_tensor(
            #         np.array(self.raw_data[f"demo_{self.demo_count}"]
            #                  ["actions"])[..., -self.num_hand_joints:]).to(
            #                      self.device).to(torch.float32)[0],
            #     self.joint_limits)
            self.env.step(actions)

        step_count = 0
        print("need to reset")

        for index in range(180):

            with torch.no_grad():
                self.add_obs_to_buffer(step_count)

                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                latent_dim = self.policy.n_latent_dims
                T_down = self.policy.downsampled_input_h

                sample = torch.rand(
                    (self.env.num_envs, latent_dim * T_down)).to(
                        self.policy.device) * (
                            self.max_latent_value -
                            self.min_latent_value) + self.min_latent_value

                action = self.policy.get_action_from_latent(
                    sample.to(torch.float32), normalized_joint_des[:, 0])
                reconstruct_action = einops.rearrange(action,
                                                      "N (T A) -> N T A",
                                                      A=self.num_hand_joints)
                reconstruct_action = self.policy.normalizer[
                    "action"].unnormalize(reconstruct_action).clip(-1, 1)

                recontructed_hand_pose = extract_finger_joints(
                    reconstruct_action.to(self.device), self.joint_limits)
            action = torch.zeros(self.env.action_space.shape).to(self.device)

            # for i in range(0, int(recontructed_hand_pose.shape[1])):
            # action[:, 6:] = recontructed_hand_pose[:, i]
            action[:, 6:] = recontructed_hand_pose[:, 0]

            self.env.step(action)
            step_count += 1

            # self.add_obs_to_buffer(step_count)
