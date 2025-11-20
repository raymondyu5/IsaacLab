import torch
import numpy as np

import os
import sys

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra

import copy
import gymnasium as gym

import time

from scripts.workflows.hand_manipulation.env.bc_env.image_bc_diffusion_wrapper import ImageBCDiffusionWrapper
from scripts.workflows.hand_manipulation.env.bc_env.pcd_bc_diffusion_wrapper import PCDBCDiffusionWrapper

from scripts.workflows.hand_manipulation.env.bc_env.state_bc_diffusion_wrapper import StateBCDiffusionWrapper
from isaaclab.envs import mdp

from scripts.workflows.hand_manipulation.env.bc_env.bc_env_wrapper import HandBCEnvWrapper

import yaml


class ResidualEnv(HandBCEnvWrapper):
    """Residual Environment for RL training with residual actions."""

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
    ):
        self.env = env
        self.device = self.env.unwrapped.device
        self.args_cli = args_cli
        self.use_relative_pose = use_relative_pose
        self.env_config = env_config

        self.init_setting()

        super().__init__(
            env.unwrapped,
            env_config,
            args_cli,
            use_relative_pose=use_relative_pose,
        )

        low = np.zeros(self.env.unwrapped.action_space.shape) - 1.0
        high = np.zeros(self.env.unwrapped.action_space.shape) + 1.0

        # Create a new Box
        self.action_space = gym.spaces.Box(
            low=low, high=high, dtype=self.env.unwrapped.action_space.dtype)
        self.observation_space = self.env.unwrapped.observation_space
        self.action_framework = copy.deepcopy(args_cli.action_framework)
        self.use_multi_finger = True

        if self.args_cli.use_base_action:
            self.use_last_action = False
            self.last_diffusion_action_dim = 16
            self.num_diffusion_horizon = self.diffusion_env.policy.horizon

            action_space = self.num_diffusion_horizon * self.last_diffusion_action_dim + 4  # 22 for arm + hand
            self.env.unwrapped.action_space = gym.spaces.Box(
                low=np.ones(action_space) * -1.0,
                high=np.ones(action_space) * 1.0,
                dtype=self.env.unwrapped.action_space.dtype)
            self.step = self.step_residual_base_actions

        else:
            self.last_diffusion_action_dim = 22

            self.use_last_action = True

            if not self.use_multi_finger:

                self.env.unwrapped.action_space = gym.spaces.Box(
                    low=np.ones(4) * -1.0,
                    high=np.ones(4) * 1.0,
                    dtype=self.env.unwrapped.action_space.dtype)
            else:
                self.env.unwrapped.action_space = gym.spaces.Box(
                    low=np.ones(7) * -1.0,
                    high=np.ones(7) * 1.0,
                    dtype=self.env.unwrapped.action_space.dtype)

            self.step = self.step_residual_actions
        self.finger_id = torch.as_tensor([
            0,
            4,
            8,
            12,
            1,
            3,
            7,
            9,
            2,
            6,
            10,
            14,
            5,
            11,
            13,
            15,
        ]).to(self.device)

        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

        self.success_rate = 0.0
        self.difficulty_level = 0.0
        self.finish_warmup = False

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]

        arm_action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

        first_six = arm_action_bound[:6]
        nonzero_mask = first_six != 0
        nonzero_vals = first_six[nonzero_mask]  # values
        self.rl_action_id = torch.nonzero(nonzero_mask).squeeze(-1)  # indices
        rl_action_id_hand = torch.arange(6, 22, device=arm_action_bound.device)
        self.rl_action_id = torch.cat([self.rl_action_id, rl_action_id_hand],
                                      dim=0)

        last_val = arm_action_bound[-1].expand(16)

        # --- Combine ---
        action_bound = torch.cat([nonzero_vals, last_val], dim=0)
        arm_action_limit = torch.stack([-action_bound, action_bound], dim=1)

        self.lower_bound = arm_action_limit[:, 0]
        self.upper_bound = arm_action_limit[:, 1]
        self.whole_uper_bound = torch.zeros(22).to(self.device)
        self.whole_uper_bound[-len(self.upper_bound):] = self.upper_bound

        self.hand_side = "right" if self.args_cli.add_right_hand else "left"

        init_joint_pose = self.env_config["params"][
            f"{self.hand_side}_reset_joint_pose"] + [0] * self.num_hand_joints
        self.init_joint_pose = torch.as_tensor(init_joint_pose).unsqueeze(
            0).to(self.device).repeat_interleave(self.env.unwrapped.num_envs,
                                                 dim=0)
        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

    def reset(self):

        next_obs, info = self.env.unwrapped.reset()
        # self.reset_curriculum()

        for i in range(10):
            # self.reset_robot_joints()
            actions = torch.zeros((self.env.num_envs, 22),
                                  dtype=torch.float32,
                                  device=self.device)
            next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                actions)

        if self.args_cli.action_framework is not None:
            self.last_diffusion_obs = self.get_diffusion_obs(
                next_obs["policy"])

            with torch.no_grad():

                self.diffusion_action = self.diffusion_env.policy.predict_action(
                    self.last_diffusion_obs)["action_pred"][:, 0].clone()
                if self.use_last_action:
                    next_obs["policy"]["last_action"] = self.diffusion_action[
                        ..., -self.last_diffusion_action_dim:]
        self.total_dense_rewards = 0.0
        self.total_sparse_rewards = 0.0
        self.start_time = time.time()
        self.last_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[..., -16:].clone()
        self.success_rate = 0.0
        self.finish_warmup = False
        self.last_obs = copy.deepcopy(next_obs)

        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.scene.env_origins

        self.init_object_height = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, 2]

        return next_obs, info, self.diffusion_action.clone()

    def _process_obs(
        self,
        obs_dict: torch.Tensor | dict[str, torch.Tensor],
    ):

        obs_buffer = []
        if isinstance(obs_dict, dict):
            for key, value in obs_dict.items():
                if key in ["seg_pc", "rgb"]:
                    continue

            # for key in self.diffusion_env.obs_key:
            #     value = obs_dict[key]
                obs_buffer.append(value)
            obs = torch.cat(obs_buffer, dim=-1)
        else:
            obs = obs_dict
        return obs

    def construct_reward(self, ):

        ############# height reward #############

        threshold_low = 0.25
        threshold_high = 0.4

        object_height = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, 2]
        height_reward = torch.clip(
            (object_height - self.init_object_height - 0.02) / threshold_low,
            0, 1) * 2
        # + ((object_height - self.init_object_height - 0.02)
        #                  > 0.0).float() * 0.3

        # scale goes from 1 at 0.35 to 0 at 0.5
        # scale = (threshold_high - object_height) / (threshold_high -
        #                                             threshold_low)
        # scale = torch.clamp(scale, min=0.0, max=1.0)

        # # if object_height <= threshold_low → keep reward unchanged (scale=1)
        # scale = torch.where(object_height > threshold_low, scale,
        #                     torch.ones_like(scale))

        # # apply shrink
        # height_reward = height_reward * scale

        ############# height reward #############

        ##### palm close reward #####
        palm_pose = self.env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w

        nonlift = height_reward < 0.05
        palm_pos = palm_pose[:, :3]
        obj_pos = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3]

        # weights: [wx, wy, wz]
        weights = torch.tensor([2.0, 2.0, 1.0], device=palm_pos.device)

        # weighted difference
        diff = (palm_pos - obj_pos) * weights

        # weighted distance
        height_diff = torch.linalg.norm(diff, dim=-1)

        max_dist = 0.3  # e.g., 10 cm
        norm_diff = torch.clamp(height_diff / max_dist, 0.0, 1.0)

        # convert to reward: 1 when aligned, 0 when far apart
        close_reward = (1.0 - norm_diff) * nonlift.float()
        close_reward = torch.where(nonlift, close_reward,
                                   torch.ones_like(close_reward))
        ##### palm close reward #####

        ### finger pose reward ####

        finger_action = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_pos[..., -16:].clone()

        finger_reward = (finger_action.mean(dim=-1) > 0.8).float() * (
            close_reward > 0.8).float() * (height_reward > 0.5).float()
        ### finger pose reward ####

        #### total reward
        reward = height_reward * 1.5 + close_reward * 0.8  #+ finger_reward

        return reward

    def step_residual_base_actions(self, action, bc_eval=False):

        if not bc_eval:

            latent_noise = action[:, :self.last_diffusion_action_dim].reshape(
                self.env.num_envs, self.num_diffusion_horizon, -1)

            with torch.no_grad():
                self.last_diffusion_obs = self.get_diffusion_obs(
                    self.last_obs["policy"])
                raw_latent_noise = torch.randn(
                    (self.env.num_envs, self.num_diffusion_horizon,
                     22)).to(self.device)

                raw_latent_noise[..., -latent_noise.shape[-1]:] = latent_noise

                self.diffusion_action = self.diffusion_env.policy.predict_action(
                    self.last_diffusion_obs,
                    raw_latent_noise)["action_pred"][:, 0].clone()

            reconstructed_actions = self.process_residual_action(
                action[:, self.last_diffusion_action_dim:].clip(-1, 1))

            base_action = self.diffusion_action.clone()

        else:

            with torch.no_grad():
                self.last_diffusion_obs = self.get_diffusion_obs(
                    self.last_obs["policy"])
            self.diffusion_action = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs, )["action_pred"][:, 0].clone()

            reconstructed_actions = self.diffusion_action.clone()
            self.finish_warmup = True
            base_action = self.diffusion_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
            reconstructed_actions.to(self.device))
        sparse_reward, terminated, time_outs = self.after_step(
            obs, rewards, terminated, time_outs)
        self.last_obs = copy.deepcopy(obs)
        return obs, sparse_reward, terminated, time_outs, extras, [
            torch.cat([
                base_action.clone(),
                self.whole_uper_bound.unsqueeze(0).repeat_interleave(
                    self.env.num_envs, 0)
            ],
                      dim=0),
            torch.cat([
                self.diffusion_action.clone(),
                self.whole_uper_bound.unsqueeze(0).repeat_interleave(
                    self.env.num_envs, 0)
            ],
                      dim=0)
        ]

    def process_residual_action(self, action):

        delta_pose = action.clip(-1, 1)

        delta_hand_arm_actions = torch.zeros(
            (self.env.num_envs, 22)).to(self.device)

        if not self.use_multi_finger:

            delta_hand_arm_actions[:, -16:] = (delta_pose[..., -1] + 1) / 2 * (
                self.upper_bound[-16:] -
                self.lower_bound[-16:]) + self.lower_bound[-16:]
            num_finger = 1
        else:
            finger_action = delta_pose[:, -4:]
            joint_action = finger_action.repeat_interleave(4, dim=1)
            delta_finger_action = (joint_action + 1) / 2 * (
                self.upper_bound[-16:] -
                self.lower_bound[-16:]) + self.lower_bound[-16:]

            delta_hand_arm_actions[:,
                                   -16:] = delta_finger_action[:,
                                                               self.finger_id]
            num_finger = 4

        if len(self.rl_action_id) > 16:
            arm_action_id = self.rl_action_id[:-16]
            delta_hand_arm_actions[:, arm_action_id] = (
                delta_pose[..., :-num_finger] +
                1) / 2 * (self.upper_bound[arm_action_id] -
                          self.lower_bound[arm_action_id]
                          ) + self.lower_bound[arm_action_id]

        reconstructed_actions = self.diffusion_action.clone()

        reconstructed_actions += delta_hand_arm_actions.clone()
        return reconstructed_actions

    def step_residual_actions(self, action, bc_eval=False):

        if not bc_eval:

            reconstructed_actions = self.process_residual_action(action)

            base_action = self.diffusion_action.clone()

        else:
            reconstructed_actions = self.diffusion_action.clone()
            self.finish_warmup = True
            base_action = self.diffusion_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
            reconstructed_actions.to(self.device))
        # self.last_obs = copy.deepcopy(obs)
        self.last_diffusion_obs = self.get_diffusion_obs(obs["policy"])
        with torch.no_grad():

            self.diffusion_action = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs)["action_pred"][:, 0].clone()
            # self.diffusion_action[..., -16:] += (torch.rand(
            #     (self.env.num_envs, 16)).to(self.device) * 2 - 1) * 0.10
            obs["policy"]["last_action"] = self.diffusion_action[
                ..., -self.last_diffusion_action_dim:]
        sparse_reward, terminated, time_outs = self.after_step(
            obs, rewards, terminated, time_outs)

        return obs, sparse_reward, terminated, time_outs, extras, [
            torch.cat([
                base_action.clone(),
                self.whole_uper_bound.unsqueeze(0).repeat_interleave(
                    self.env.num_envs, 0)
            ],
                      dim=0),
            torch.cat([
                self.diffusion_action.clone(),
                self.whole_uper_bound.unsqueeze(0).repeat_interleave(
                    self.env.num_envs, 0)
            ],
                      dim=0)
        ]

    def after_step(self, obs, rewards, terminated, time_outs):

        sparse_reward = self.construct_reward()
        self.total_dense_rewards += rewards.sum().item()
        self.total_sparse_rewards += sparse_reward  #.sum().item()

        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:

            self.total_dense_rewards /= self.env.unwrapped.num_envs
            self.total_sparse_rewards /= self.env.unwrapped.num_envs

            self.success_rate = (
                obs["policy"][f"{self.hand_side}_manipulated_object_pose"][:,
                                                                           2]
                > 0.20).float().mean().item()

            self.total_sparse_rewards = self.total_sparse_rewards.mean().item()
            # self.success_rate = self.evaluate_success().sum().item(
            # ) / self.env.unwrapped.num_envs
            print(
                f"[INFO]: Total dense rewards: {self.total_dense_rewards:.2f} and Total sparse reward:{self.total_sparse_rewards:.2f} for {self.env.unwrapped.episode_length_buf[0]} steps, with success rate: {self.success_rate:.2f} with time {time.time() - self.start_time:.2f} seconds"
            )
            self.total_sparse_rewards = 0
            self.total_dense_rewards = 0
            terminated[:] = False
            time_outs[:] = True
        return sparse_reward, terminated, time_outs

    # def construct_reward(self, ):

    #     ## object height reward

    #     object_height = self.env.scene[
    #         f"{self.hand_side}_hand_object"]._data.root_state_w[:, 2]
    #     height_reward = torch.clip(
    #         (object_height - self.init_object_height - 0.02) / object_height,
    #         0, 1) * 2
    #     palm_pose = self.env.scene[
    #         f"{self.hand_side}_palm_lower"]._data.root_state_w

    #     ### palm height reward
    #     nonlift = height_reward < 0.05
    #     height_diff = torch.abs(palm_pose[:, 2] - object_height)

    #     max_dist = 0.2  # e.g., 10 cm
    #     norm_diff = torch.clamp(height_diff / max_dist, 0.0, 1.0)

    #     # convert to reward: 1 when aligned, 0 when far apart
    #     close_reward = (1.0 - norm_diff) * nonlift.float()
    #     close_reward = torch.where(nonlift, close_reward,
    #                                torch.ones_like(close_reward))

    #     ### finger pose reward

    #     finger_action = self.env.scene[
    #         f"{self.hand_side}_hand"]._data.joint_pos[..., -16:].clone()

    #     # normalize finger closeness: assume joint range [0=open, 1=closed]
    #     finger_norm = torch.clamp(finger_action, 0.0, 1.0)

    #     # --- Distance thresholds ---
    #     # if palm is farther than 5 cm → encourage open fingers (reward 0 if closing)
    #     far_mask = (height_diff > 0.05)

    #     # if palm is close (<= 3 cm) → encourage closing
    #     close_mask = (height_diff < 0.03)

    #     # between 3–5 cm → interpolate linearly
    #     mid_mask = (~far_mask) & (~close_mask)

    #     # --- Finger reward ---
    #     # encourage open when far
    #     open_reward = (1.0 - finger_norm.mean(dim=-1)) * far_mask.float()

    #     # encourage close when very near
    #     close_reward_fingers = finger_norm.mean(dim=-1) * close_mask.float()

    #     # interpolate in mid zone
    #     mid_alpha = (0.05 - height_diff) / (0.05 - 0.03
    #                                         )  # goes 0→1 as dist 5→3 cm
    #     mid_alpha = torch.clamp(mid_alpha, 0.0, 1.0)
    #     mid_reward = (finger_norm.mean(dim=-1) * mid_alpha) * mid_mask.float()

    #     finger_reward = open_reward + close_reward_fingers + mid_reward

    #     #### total reward
    #     reward = height_reward + close_reward  #+ finger_reward
    #     return reward
