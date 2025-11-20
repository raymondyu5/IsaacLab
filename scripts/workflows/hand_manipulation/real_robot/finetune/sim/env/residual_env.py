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


class ResidualEnv:
    """Residual Environment for RL training with residual actions."""

    def __init__(self,
                 env,
                 env_config,
                 args_cli,
                 use_relative_pose=False,
                 diffusion_device="cuda",
                 use_residual=True,
                 residual_delta=0.3,
                 diffuclties_range=10.0):
        self.env = env
        self.device = self.env.unwrapped.device
        self.args_cli = args_cli
        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.residual_delta = residual_delta
        self.use_base_action = args_cli.use_base_action

        self.diffuclties_range = diffuclties_range

        self.init_setting()
        self.diffusion_device = diffusion_device

        # self.load_diffusion_model(self.diffusion_device)

        original_box = self.env.unwrapped.action_space  # shape: (10, 22)

        # Get horizon
        self.diffusion_horizon = self.diffusion_env.policy.horizon

        self.signle_action_dim = original_box.low.shape[-1]

        if use_residual:
            self.delta_horzion = 1
        else:
            self.delta_horzion = 0

        horizon = self.diffusion_horizon

        if not self.use_base_action:
            horizon = 0
        low = np.zeros(((horizon + self.delta_horzion) *
                        22)) - 1  # shape: (10, 22 * horizon)
        high = np.zeros(
            ((horizon + self.delta_horzion) * 22)) + 1  # same shape

        # Create a new Box
        self.action_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=original_box.dtype)
        self.observation_space = self.env.unwrapped.observation_space
        self.action_framework = copy.deepcopy(args_cli.action_framework)
        self.args_cli.action_framework = None

        if self.use_base_action:
            self.step = self.step_base_actions
        else:
            self.step = self.step_residual_actions

        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)
        # self.target_object_region = env_config["params"]["RigidObject"][
        #     self.args_cli.target_object_name]["pose_range"]

        self.success_rate = 0.0
        self.difficulty_level = 0.0
        self.finish_warmup = False

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]

        arm_action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

        arm_action_limit = torch.stack([
            torch.tensor([-arm_action_bound[0] * self.residual_delta] * 3 +
                         [-arm_action_bound[1] * self.residual_delta] * 3 +
                         [-arm_action_bound[-1] / self.env.unwrapped.step_dt] *
                         self.num_hand_joints,
                         device=self.device),
            torch.tensor([arm_action_bound[0] * self.residual_delta] * 3 +
                         [arm_action_bound[1] * self.residual_delta] * 3 +
                         [arm_action_bound[-1] / self.env.unwrapped.step_dt] *
                         self.num_hand_joints,
                         device=self.device)
        ],
                                       dim=1)
        self.lower_bound = arm_action_limit[:, 0]
        self.upper_bound = arm_action_limit[:, 1]

        self.hand_side = "right" if self.args_cli.add_right_hand else "left"

        init_joint_pose = self.env_config["params"][
            f"{self.hand_side}_reset_joint_pose"] + [0] * self.num_hand_joints
        self.init_joint_pose = torch.as_tensor(init_joint_pose).unsqueeze(
            0).to(self.device).repeat_interleave(self.env.unwrapped.num_envs,
                                                 dim=0)
        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

        if self.args_cli.action_framework == "image_diffusion":

            self.diffusion_env = ImageBCDiffusionWrapper(
                self.env,
                self.env_config,
                self.args_cli,
            )
            delattr(self.diffusion_env, "temporal_image_buffer")
        elif self.args_cli.action_framework == "pcd_diffusion":

            self.diffusion_env = PCDBCDiffusionWrapper(
                self.env,
                self.env_config,
                self.args_cli,
            )
            self.diffusion_visual_encoder = self.diffusion_env.policy.obs_encoder
            self.diffusion_visual_dim = self.diffusion_env.policy.obs_encoder.key_model_map[
                "seg_pc"].mlp_global[-1].fc.out_features
            delattr(self.diffusion_env, "temporal_image_buffer")

        elif self.args_cli.action_framework == "state_diffusion":

            self.diffusion_env = StateBCDiffusionWrapper(
                self.env,
                self.env_config,
                self.args_cli,
            )

    def _process_shared_obs(self, obs_dict):

        image_obs = []
        agent_obs = []
        for key in obs_dict.keys():
            if key in ["seg_pc", "rgb"]:
                image_obs.append(obs_dict[key])
            else:
                agent_obs.append(obs_dict[key])

        image_obs = torch.cat(image_obs, dim=1)
        agent_obs = torch.cat(agent_obs, dim=-1)

        with torch.no_grad():

            share_encoder_obs = self.diffusion_visual_encoder({
                "seg_pc":
                image_obs,
                "agent_pos":
                agent_obs
            })
        return {"policy": share_encoder_obs}

    def reset_curriculum(self):

        if self.success_rate is None or not self.finish_warmup:
            return
        if self.success_rate > 0.7:
            self.difficulty_level += 1.0
        print("[INFO] Difficulty level: ", self.difficulty_level)
        scale = np.clip(self.difficulty_level / self.diffuclties_range, 0.0,
                        1.0)

        scaled_region = {
            key: [(torch.tensor(v) * scale).float() for v in vals]
            for key, vals in self.target_object_region.items()
        }
        mdp.reset_rigid_articulation(self.env, self.env_ids,
                                     f"{self.hand_side}_hand_object",
                                     scaled_region)

    def reset(self):

        next_obs, info = self.env.unwrapped.reset()
        # self.reset_curriculum()

        for i in range(10):
            # self.reset_robot_joints()
            actions = torch.zeros(self.env.unwrapped.action_space.shape,
                                  dtype=torch.float32,
                                  device=self.device)
            next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                actions)

        self.last_diffusion_obs = self.get_diffusion_obs(next_obs["policy"])

        self.total_rewards = 0.0
        self.start_time = time.time()

        with torch.no_grad():

            self.diffussion_action = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs)["action_pred"][:, 0].clone()
            next_obs["policy"]["last_action"] = self.diffussion_action
        return next_obs, info

    def get_diffusion_obs(self, obs):

        obs_demo = []

        for key in self.diffusion_env.obs_key:

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)

        if self.action_framework in ["pcd_diffusion", "image_diffusion"]:

            image_demo = []
            for key in self.diffusion_env.image_key:

                image_demo.append(obs[key])

            image_demo = torch.cat(image_demo, dim=0)

            obs_dict = {
                "agent_pos": obs_demo.unsqueeze(1),
                "seg_pc": image_demo.unsqueeze(1),
            }
        elif self.action_framework == "state_diffusion":
            obs_dict = {
                "obs": obs_demo.unsqueeze(1),
            }

        return obs_dict

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

    def evaluate_success(self, ):

        object_pose = self.env.unwrapped.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.unwrapped.scene.env_origins
        lift_or_not = (object_pose[:, 2] > 0.40)
        overhigh_or_not = (object_pose[:, 2] < 0.65)
        outofbox_or_not = ((object_pose[:, 0] < 0.65) &
                           (object_pose[:, 0] > 0.3) &
                           (object_pose[:, 1] < 0.3) &
                           (object_pose[:, 1] > -0.3))
        success = lift_or_not & overhigh_or_not & outofbox_or_not
        return success

    def step_base_actions(self, action, bc_eval=False):

        clip_actions = action.clone()

        with torch.no_grad():
            if self.use_base_action and not bc_eval:

                robot_noise = clip_actions.clone()[
                    ..., self.delta_horzion * self.signle_action_dim:].reshape(
                        self.env.unwrapped.num_envs, self.diffusion_horizon,
                        22)
            else:
                robot_noise = None
            self.finish_warmup = True
            reconstructed_actions = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs, robot_noise)["action_pred"][:, 0]

        if self.delta_horzion > 0 and not bc_eval:

            delta_pose = clip_actions[..., :self.delta_horzion *
                                      self.signle_action_dim]
            delta_pose = delta_pose.clip(-1, 1)

            delta_hand_arm_actions = (delta_pose + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            delta_hand_arm_actions *= self.env.unwrapped.step_dt * 1

            reconstructed_actions += delta_hand_arm_actions

        obs, rewards, terminated, time_outs, extras = self.env.step(
            reconstructed_actions.to(self.device))
        # self.last_obs = copy.deepcopy(obs)
        self.last_diffusion_obs = self.get_diffusion_obs(obs["policy"])

        self.total_rewards += rewards.sum().item()

        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            self.total_rewards /= self.env.unwrapped.num_envs
            self.success_rate = self.evaluate_success().sum().item(
            ) / self.env.unwrapped.num_envs
            print(
                f"[INFO]: Total rewards: {self.total_rewards:.4f} for {self.env.unwrapped.episode_length_buf[0]} steps, with success rate: {self.success_rate:.2f} with time {time.time() - self.start_time:.2f} seconds"
            )

        return obs, rewards, terminated, time_outs, extras, None

    def step_residual_actions(self, action, bc_eval=False):

        clip_actions = action.clone()

        if not bc_eval:

            delta_pose = clip_actions[..., :self.delta_horzion *
                                      self.signle_action_dim]
            delta_pose = delta_pose.clip(-1, 1)

            delta_hand_arm_actions = (delta_pose + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            delta_hand_arm_actions *= self.env.unwrapped.step_dt * 1

            reconstructed_actions = self.diffussion_action + delta_hand_arm_actions
        else:
            reconstructed_actions = self.diffussion_action.clone()
            self.finish_warmup = True

        obs, rewards, terminated, time_outs, extras = self.env.step(
            reconstructed_actions.to(self.device))
        # self.last_obs = copy.deepcopy(obs)
        self.last_diffusion_obs = self.get_diffusion_obs(obs["policy"])
        with torch.no_grad():

            self.diffussion_action = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs)["action_pred"][:, 0].clone()
            obs["policy"]["last_action"] = self.diffussion_action

        self.total_rewards += rewards.sum().item()

        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:

            self.total_rewards /= self.env.unwrapped.num_envs
            self.success_rate = self.evaluate_success().sum().item(
            ) / self.env.unwrapped.num_envs
            print(
                f"[INFO]: Total rewards: {self.total_rewards:.4f} for {self.env.unwrapped.episode_length_buf[0]} steps, with success rate: {self.success_rate:.2f} with time {time.time() - self.start_time:.2f} seconds"
            )

        return obs, rewards, terminated, time_outs, extras, None
