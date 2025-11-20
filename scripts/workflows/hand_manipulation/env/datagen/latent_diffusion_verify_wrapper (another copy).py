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
    TemporalEnsembleBufferObservation, load_latent_action, load_config)
import time

sys.path.append("submodule/benchmark_VAE/src")
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
            horizon_K=self.policy.horizon,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.num_hand_joints,
        )

    def load_diffusion_model(self):

        checkpoint = os.path.join(self.args_cli.diffusion_path, "checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg.policy.num_inference_steps = 6
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
        self.chunk_size = cfg.horizon

        vae_dir = self.args_cli.vae_path

        all_dirs = [
            d for d in os.listdir(vae_dir)
            if os.path.isdir(os.path.join(vae_dir, d))
        ]

        last_training = sorted(all_dirs)[-1]

        model_dir = os.path.join(vae_dir, last_training, "final_model")

        self.latent_model = AutoModel.load_from_folder(
            model_dir, device=self.device).eval().to(self.device)
        self.latent_action = load_latent_action(vae_dir)
        self.latent_action_config = load_config(vae_dir, self.device)

    def verify_diffusion(self, ):

        self.env.reset()
        print("new demo")
        demo_latent_actions = torch.as_tensor(
            np.array(
                self.latent_action[f"demo_{self.demo_count}"]["actions"])).to(
                    self.device).to(torch.float32)
        demo_state = torch.as_tensor(
            np.array(self.latent_action[f"demo_{self.demo_count}"]["obs"]
                     ["state"])).to(self.device).to(torch.float32)

        demo_gt_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"][
                ...,
                -self.num_hand_joints:])).to(self.device).to(torch.float32)

        latent_hand_pose = demo_latent_actions[...,
                                               -self.num_hand_joints:].clone()
        self.temporal_action_buffer.reset(demo_gt_actions.shape[0])

        num_chunks = (demo_latent_actions.shape[0] - self.policy.horizon + 1)
        self.temporal_obs_buffer.reset(num_chunks * self.policy.horizon,
                                       self.env.num_envs)

        with torch.no_grad():
            step_count = 0
            self.add_obs_to_buffer(step_count)

            for index in range(0, num_chunks - 1):

                obs_list = torch.as_tensor(
                    demo_state[index:index + self.chunk_size]).to(
                        self.device).to(torch.float32).reshape(
                            1, -1, self.num_hand_joints)

                gt_actions_chunks = extract_finger_joints(
                    torch.as_tensor(
                        demo_gt_actions[index:index + self.chunk_size]).to(
                            self.device).to(torch.float32).reshape(
                                -1, self.num_hand_joints), self.joint_limits)

                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                obs_dict = {
                    "obs":
                    torch.cat(
                        [
                            obs_list.clone(),
                            (
                                normalized_joint_des[1]
                                #  + 0.03 *
                                #  (torch.rand_like(normalized_joint_des[-1],
                                #                   device=normalized_joint_des.device) *
                                #   2 - 1)
                            ).unsqueeze(0).repeat_interleave(
                                int(obs_list.shape[1] /
                                    normalized_joint_des.shape[1]),
                                dim=1)
                        ],
                        dim=0)
                }

                noise = (torch.randn(size=(2, self.policy.horizon,
                                           self.latent_action_config[-1]),
                                     dtype=obs_list.dtype,
                                     device=obs_list.device,
                                     generator=None))
                noise[-1] = noise[0].clone()

                latent_action = self.policy.predict_action(
                    obs_dict, noise)["action_pred"]
                flat_latent = latent_action.view(-1, latent_action.shape[-1])
                reconstruct_action = self.latent_model.decode_rl_action(
                    latent_action.reshape(-1,
                                          latent_action.shape[-1])).reshape(
                                              latent_action.shape[0], -1,
                                              self.num_hand_joints)

                recontructed_hand_pose = extract_finger_joints(
                    reconstruct_action, self.joint_limits)

                for i in range(1):
                    action = torch.zeros(self.env.action_space.shape).to(
                        self.device)

                    action[:2, 6:] = recontructed_hand_pose[:, i]
                    # action[0, 6:] = gt_actions_chunks[:, i]

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

        self.temporal_obs_buffer.reset(200 * self.policy.horizon,
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(200 * self.policy.horizon,
                                          self.env.num_envs)

        for i in range(20):
            actions = torch.zeros(self.env.action_space.shape).to(self.device)
            actions[:, -self.num_hand_joints:] = extract_finger_joints(
                torch.as_tensor(
                    np.array(self.raw_data[f"demo_{self.demo_count}"]
                             ["actions"])[..., -self.num_hand_joints:]).to(
                                 self.device).to(torch.float32)[0],
                self.joint_limits)
            self.env.step(actions)
        # self.env.scene[
        #     f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
        #         rest_joint_pose,
        #         torch.arange(self.env.num_envs, device=self.env.device))

        step_count = 0
        print("need to reset")
        import pdb
        pdb.set_trace()

        for index in range(200):

            with torch.no_grad():
                self.add_obs_to_buffer(step_count)

                normalized_joint_des = self.temporal_obs_buffer.compute_obs(
                ).clone()

                obs_dict = {"obs": normalized_joint_des}

                noise = (torch.randn(size=(self.env.num_envs,
                                           self.policy.horizon,
                                           self.latent_action_config[-1]),
                                     dtype=normalized_joint_des.dtype,
                                     device=normalized_joint_des.device,
                                     generator=None))

                latent_action = self.policy.predict_action(
                    obs_dict, noise)["action_pred"]
                flat_latent = latent_action.view(-1, latent_action.shape[-1])
                reconstruct_action = self.latent_model.decode_rl_action(
                    latent_action.reshape(-1,
                                          latent_action.shape[-1])).reshape(
                                              latent_action.shape[0], -1,
                                              self.num_hand_joints)

                recontructed_hand_pose = extract_finger_joints(
                    reconstruct_action, self.joint_limits)

                # self.temporal_action_buffer.add_prediction(
                #     index - 1, reconstructed_hand_actions)
                # hand_action = self.temporal_action_buffer.compute_action()

            action = torch.zeros(self.env.action_space.shape).to(self.device)

            # for i in range(0, int(recontructed_hand_pose.shape[1])):
            action[:, 6:] = recontructed_hand_pose[:, 1]
            # action[:, 6:] = reconstructed_hand_actions[:, 0]

            self.env.step(action)
            step_count += 1

            self.add_obs_to_buffer(step_count)
