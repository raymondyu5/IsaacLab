from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv


class BimanulRLStep:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device
        self.init_setting()

    def init_setting(self, ):

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]
        self.num_arm_actions = int(self.env.action_space.shape[-1] / 2 -
                                   self.num_hand_joints)
        self.lower_bound = self.env.action_space.low
        self.upper_bound = self.env.action_space.high

        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low
        if self.args_cli.action_framework is not None:
            if self.args_cli.action_framework == "pca":
                self.eigen_vectors_right, self.min_pca_values_right, self.max_pca_values_right, self.pca_D_mean_right, self.pca_D_std_right = None, None, None, None, None
                self.eigen_vectors_left, self.min_pca_values_left, self.max_pca_values_left, self.pca_D_mean_left, self.pca_D_std_left = None, None, None, None, None
            if "vae" in self.args_cli.action_framework:
                self.vae_model_left, self.num_finger_actions_left = None, 0
                self.vae_model_right, self.num_finger_actions_right = None, 0
                self.max_latent_value_right, self.min_latent_value_right = None, None
                self.max_latent_value_left, self.min_latent_value_left = None, None

    def reset(self):

        obs, info = self.env.reset()

        self.pre_left_finger_action = self.env.scene[
            f"left_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

        self.pre_right_finger_action = self.env.scene[
            f"right_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

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
            clip_actions[:, self.num_arm_actions:self.num_arm_actions + self.
                         num_hand_joints] += self.pre_left_finger_action.clone(
                         )  #left robot finger

            clip_actions[:, -self.
                         num_hand_joints:] += self.pre_right_finger_action.clone(
                         )

        else:
            clip_actions[:, :-self.num_hand_joints] *= self.env.step_dt * 1

        obs, rewards, terminated, time_outs, extras = self.env.step(
            clip_actions)

        self.pre_right_finger_action = self.env.scene[
            f"right_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

        self.pre_left_finger_action = self.env.scene[
            f"left_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

        rewards_tensor = rewards
        # if torch.isnan(rewards_tensor).any():
        #     print("Warning: rewards contain NaN values!")

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def procees_vae_actions(self, clip_actions, vae_model, start_index,
                            end_index, hand_side):
        max_latent_value = getattr(vae_model, f"max_latent_value_{hand_side}")
        min_latent_value = getattr(vae_model, f"min_latent_value_{hand_side}")
        clip_hand_actions = (clip_actions[:, start_index:end_index] +
                             1) / 2 * (max_latent_value -
                                       min_latent_value) + min_latent_value
        pre_hand_action = getattr(self,
                                  f"pre_{hand_side}_finger_action").clone()

        reconstruct_finger_pose = vae_model.decoder(
            pre_hand_action.clone(),
            clip_hand_actions.clone() * self.args_cli.action_scale,
        )
        return reconstruct_finger_pose

    def step_vaegrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        with torch.no_grad():
            reconstruct_left_finger_pose = self.procees_vae_actions(
                clip_actions,
                self.vae_model_left,
                self.num_arm_actions,
                self.num_arm_actions + self.num_finger_actions_left,
                hand_side="left")
            reconstruct_right_finger_pose = self.procees_vae_actions(
                clip_actions,
                self.vae_model_right,
                -self.num_finger_actions_right,
                None,
                hand_side="right")

        left_arm_actions = (
            (clip_actions[:, :self.num_arm_actions] + 1) / 2 *
            (self.upper_bound[:self.num_arm_actions] -
             self.lower_bound[:self.num_arm_actions]) +
            self.lower_bound[:self.num_arm_actions]) * self.env.step_dt * 1

        right_arm_actions = (
            (clip_actions[:, self.num_arm_actions +
                          self.num_finger_actions_left:self.num_arm_actions +
                          self.num_finger_actions_left + self.num_arm_actions]
             + 1) / 2 * (self.upper_bound[:self.num_arm_actions] -
                         self.lower_bound[:self.num_arm_actions]) +
            self.lower_bound[:self.num_arm_actions]) * self.env.step_dt * 1
        # arm_actions *= self.env.step_dt * 1

        if self.args_cli.use_relative_finger_pose:
            reconstruct_left_finger_pose += self.pre_left_finger_action.clone()
            reconstruct_right_finger_pose += self.pre_right_finger_action.clone(
            )

        vae_reconstructed_actions = torch.cat([
            left_arm_actions, reconstruct_left_finger_pose, right_arm_actions,
            reconstruct_right_finger_pose
        ],
                                              dim=1)

        obs, rewards, terminated, time_outs, extras = self.env.step(
            vae_reconstructed_actions)

        self.pre_right_finger_action = self.env.scene[
            f"right_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

        self.pre_left_finger_action = self.env.scene[
            f"left_hand"].data.joint_pos[:, -self.num_hand_joints:].clone()

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_eigengrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        left_arm_actions = (
            (clip_actions[:, :self.num_arm_actions] + 1) / 2 *
            (self.upper_bound[:self.num_arm_actions] -
             self.lower_bound[:self.num_arm_actions]) +
            self.lower_bound[:self.num_arm_actions]) * self.env.step_dt * 1
        right_arm_actions = (
            (clip_actions[:, self.num_arm_actions +
                          self.num_finger_actions_left:self.num_arm_actions +
                          self.num_finger_actions_left + self.num_arm_actions]
             + 1) / 2 * (self.upper_bound[:self.num_arm_actions] -
                         self.lower_bound[:self.num_arm_actions]) +
            self.lower_bound[:self.num_arm_actions]) * self.env.step_dt * 1

        left_hand_actions = reconstruct_hand_pose_from_normalized_action(
            clip_actions[:, self.num_arm_actions:self.num_arm_actions +
                         self.num_finger_actions_left],
            self.eigen_vectors_left, self.min_pca_values_left,
            self.max_pca_values_left, self.pca_D_mean_left,
            self.pca_D_std_left)
        right_hand_actions = reconstruct_hand_pose_from_normalized_action(
            clip_actions[:, -self.num_finger_actions_right:],
            self.eigen_vectors_right, self.min_pca_values_right,
            self.max_pca_values_right, self.pca_D_mean_right,
            self.pca_D_std_right)
        pca_reconstructed_actions = torch.cat([
            left_arm_actions, left_hand_actions, right_arm_actions,
            right_hand_actions
        ],
                                              dim=1)
        obs, rewards, terminated, time_outs, extras = self.env.step(
            pca_reconstructed_actions)

        return obs, rewards, terminated, time_outs, extras, clip_actions
