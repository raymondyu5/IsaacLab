from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer)


class SingleArmRLStep:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device
        self.init_setting()
        if self.args_cli.add_right_hand:
            self.hand_side = "right"
        elif self.args_cli.add_left_hand:
            self.hand_side = "left"

    def init_setting(self, ):

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]

        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low
        if self.args_cli.action_framework is not None:
            if self.args_cli.action_framework == "pca":
                self.eigen_vectors = None
                self.min_pca_values = None
                self.max_pca_values = None
                self.pca_D_mean = None
                self.pca_D_std = None
                self.num_finger_actions = None
            if "vae" in self.args_cli.action_framework:
                self.vae_model = None
                self.num_finger_actions = None
                self.vae_model_setting = None

    def reset(self):

        obs, info = self.env.reset()

        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs, info

    def step_env(self, actions, base_action=None):

        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        clip_actions = (clip_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound

        if self.args_cli.use_relative_finger_pose:

            clip_actions *= self.env.step_dt * 1

            clip_actions[:, -self.
                         num_hand_joints:] += self.pre_finger_action.clone()
        else:
            clip_actions[:, :-self.num_hand_joints] *= self.env.step_dt * 1

        obs, rewards, terminated, time_outs, extras = self.env.step(
            clip_actions)

        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.num_hand_joints:]

        rewards_tensor = rewards
        # if torch.isnan(rewards_tensor).any():
        #     print("Warning: rewards contain NaN values!")

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_vaegrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1) * 0.0

        with torch.no_grad():
            hand_actions = clip_actions[:, -self.num_finger_actions:].clone()

            vae_values = (hand_actions + 1) / 2 * (
                self.vae_model_setting[1] -
                self.vae_model_setting[0]) + self.vae_model_setting[0]

            reconstruct_finger_pose = self.vae_model.decode_action(vae_values)
            if self.vae_model_setting[2] is not None:
                reconstruct_finger_pose = dataset_denrormalizer(
                    reconstruct_finger_pose, self.vae_model_setting[3],
                    self.vae_model_setting[4])

        arm_actions = (clip_actions[:, :-self.num_finger_actions] +
                       1) / 2 * (self.upper_bound[:-self.num_hand_joints] -
                                 self.lower_bound[:-self.num_hand_joints]
                                 ) + self.lower_bound[:-self.num_hand_joints]

        vae_reconstructed_actions = torch.cat(
            [arm_actions, reconstruct_finger_pose], dim=1)
        vae_reconstructed_actions[:, :-self.
                                  num_hand_joints] *= self.env.step_dt * 1

        obs, rewards, terminated, time_outs, extras = self.env.step(
            vae_reconstructed_actions)
        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_eigengrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        arm_actions = (clip_actions[:, :-self.num_finger_actions] +
                       1) / 2 * (self.upper_bound[:-self.num_hand_joints] -
                                 self.lower_bound[:-self.num_hand_joints]
                                 ) + self.lower_bound[:-self.num_hand_joints]
        arm_actions *= self.env.step_dt * 1

        hand_actions = reconstruct_hand_pose_from_normalized_action(
            clip_actions[:, 6:] * 0.5, self.eigen_vectors, self.min_pca_values,
            self.max_pca_values, self.pca_D_mean, self.pca_D_std)
        pca_reconstructed_actions = torch.cat([arm_actions, hand_actions],
                                              dim=1)
        obs, rewards, terminated, time_outs, extras = self.env.step(
            pca_reconstructed_actions)
        # rewards_tensor = rewards
        # if torch.isnan(rewards_tensor).any():
        #     print("Warning: rewards contain NaN values!")

        return obs, rewards, terminated, time_outs, extras, clip_actions
