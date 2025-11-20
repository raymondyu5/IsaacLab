from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

# from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import sys
import einops

sys.path.append("submodule/benchmark_VAE/src")

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
import yaml
from pythae.models import AutoModel
import os
import isaaclab.utils.math as math_utils
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer, load_config,
    extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation)
import copy

import gymnasium as gym


class RLReactiveVAEStepWrapper:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device
        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)
        self.init_setting()
        self.save_data = False

    def init_setting(self, ):
        self.raw_action_space = self.env.unwrapped.action_space.shape[-1]

        self.object_name = self.env_config["params"][
            "target_manipulated_object"]

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]
        self.num_arm_actions = self.env.unwrapped.action_space.shape[
            -1] - self.num_hand_joints

        if self.args_cli.add_right_hand:
            self.hand_side = "right"
        else:
            self.hand_side = "left"

        if self.args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                self.args_cli,
                self.env_config,
                filter_keys=[],
                load_path=self.args_cli.load_path,
                save_path=self.args_cli.save_path,
                use_fps=False,
                use_joint_pos=False
                if "joint" not in self.args_cli.task else True,
            )
            reset_buffer(self)

        else:
            self.collector_interface = None
        if self.args_cli.action_framework == "reactive_vae":
            self.load_reactive_vae_model()
            self.step = self.step_reactive_vae_env
            if self.args_cli.use_chunk_action:
                if self.args_cli.use_interpolate_chunk:
                    self.step = self.step_reactive_vae_chunk_interpolate_env
                    self.cur_chunk_index = 0

                else:
                    self.step = self.step_reactive_vae_chunk_env

            self.num_diffusion_finger_actions = self.latent_dim * self.T_down
            if self.args_cli.use_residual_action:
                self.num_finger_actions = self.num_diffusion_finger_actions + self.num_hand_joints
            else:
                self.num_finger_actions = self.num_diffusion_finger_actions
            self.num_arm_actions = 6

            self.env.unwrapped.action_space = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(self.env.unwrapped.num_envs,
                       self.num_finger_actions + 6),
                dtype=np.float32)

        self.init_planner()

        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

    def load_reactive_vae_model(self):

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
        self.reactive_vae_model = workspace.model
        self.reactive_vae_model.decoder.to(self.device)

        self.max_latent_value = self.reactive_vae_model.max_latent_value
        self.min_latent_value = self.reactive_vae_model.min_latent_value
        self.chunk_size = self.reactive_vae_model.n_obs_steps
        self.latent_dim = self.reactive_vae_model.n_latent_dims
        self.T_down = self.reactive_vae_model.downsampled_input_h

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.env.unwrapped.num_envs,
            horizon_K=self.chunk_size,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.unwrapped.num_envs,
            horizon_K=1,
            obs_dim=self.num_hand_joints,
        )

    def init_bound(self):

        # if self.args_cli.action_framework is None:
        if self.args_cli.use_relative_finger_pose:

            arm_action_bound = torch.as_tensor(
                self.env_config["params"]["Task"]["action_range"]).to(
                    self.device)

            arm_action_limit = torch.stack([
                torch.tensor(
                    [-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3 +
                    [-arm_action_bound[2]] * self.num_hand_joints,
                    device=self.device),
                torch.tensor(
                    [arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3 +
                    [arm_action_bound[2]] * self.num_hand_joints,
                    device=self.device)
            ],
                                           dim=1)
        else:
            arm_action_bound = torch.as_tensor(
                self.env_config["params"]["Task"]["action_range"]).to(
                    self.device)
            hand_finger_limit = self.env.unwrapped.scene[
                f"{self.hand_side}_hand"]._data.joint_limits[
                    0, -self.num_hand_joints:]
            arm_action_limit = torch.stack([
                torch.tensor(
                    [-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3,
                    device=self.device),
                torch.tensor(
                    [arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3,
                    device=self.device)
            ],
                                           dim=1)

            arm_action_limit = torch.cat([arm_action_limit, hand_finger_limit],
                                         dim=0)
        if self.args_cli.use_residual_action:

            residual_hand_action = torch.ones(self.num_hand_joints).to(
                self.device) * arm_action_bound[-1]
            self.lower_bound = arm_action_limit[:, 0]
            self.upper_bound = arm_action_limit[:, 1]
            self.residual_lower_bound = -residual_hand_action.clone()
            self.residual_upper_bound = residual_hand_action.clone()

        else:
            self.lower_bound = arm_action_limit[:, 0]
            self.upper_bound = arm_action_limit[:, 1]

        self.action_bound = torch.stack([self.lower_bound, self.upper_bound],
                                        dim=1).to(self.device)

    def init_planner(self):
        self.init_bound()

        self.horizon = self.env_config["params"]["Task"]["horizon"]

    def save_data_to_buffer(self, last_obs, hand_arm_actions, rewards, does):

        body_state = self.env.scene[
            f"{self.hand_side}_hand"]._data.body_state_w
        last_obs["policy"]["body_state"] = body_state

        if "IK" in self.args_cli.task:

            ee_quat_des = self.env.unwrapped.action_manager._terms[
                f"{self.hand_side}_arm_action"]._ik_controller.ee_quat_des.clone(
                )
            ee_pos_des = self.env.unwrapped.action_manager._terms[
                f"{self.hand_side}_arm_action"]._ik_controller.ee_pos_des.clone(
                )
            joint_pos_des = self.env.unwrapped.action_manager._terms[
                f"{self.hand_side}_arm_action"].joint_pos_des.clone()
            finger_pos_des = self.env.unwrapped.action_manager._terms[
                f"{self.hand_side}_hand_action"].processed_actions.clone()
            last_obs["policy"]["ee_control_action"] = torch.cat(
                [ee_pos_des, ee_quat_des, finger_pos_des], dim=-1)
            last_obs["policy"]["joint_control_action"] = torch.cat(
                [joint_pos_des, finger_pos_des], dim=-1)

            last_obs["policy"]["delta_ee_control_action"] = torch.cat([
                hand_arm_actions[:, :self.num_arm_actions].clone(),
                finger_pos_des
            ],
                                                                      dim=-1)

        update_buffer(self,
                      None,
                      last_obs,
                      hand_arm_actions,
                      rewards,
                      does,
                      does,
                      convert_to_cpu=True)

    def reset(self):

        obs, info = self.env.reset()

        self.temporal_obs_buffer.reset(
            self.env.unwrapped.max_episode_length *
            self.reactive_vae_model.horizon, self.env.unwrapped.num_envs)
        self.temporal_action_buffer.reset(
            self.env.unwrapped.max_episode_length *
            self.reactive_vae_model.horizon, self.env.unwrapped.num_envs)
        self.add_obs_to_buffer(self.env.unwrapped.episode_length_buf[0],
                               reset=True)

        if self.args_cli.use_interpolate_chunk:
            self.cur_chunk_index = 0
        for i in range(10):
            obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.unwrapped.num_envs,
                            self.raw_action_space).to(device=self.device))
        self.pre_finger_action = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs, info

    def add_obs_to_buffer(self, index, reset=False):
        state = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        joint_min = self.joint_limits[:, 0]  # shape [16]
        joint_max = self.joint_limits[:, 1]  # shape [16]
        normalized_joints = ((state - joint_min[None, :]) /
                             (joint_max - joint_min)[None, :]) * 2 - 1
        if reset:
            self.temporal_obs_buffer.add_obs(index, normalized_joints * 0)
        else:

            self.temporal_obs_buffer.add_obs(index, normalized_joints)

    def get_action(self, hand_noise):

        hand_noise = (hand_noise + 1) / 2 * (
            (self.max_latent_value -
             self.min_latent_value)) + self.min_latent_value
        normalized_joint_des = self.temporal_obs_buffer.compute_obs().clone()

        with torch.no_grad():

            action = self.reactive_vae_model.get_action_from_latent(
                hand_noise.to(torch.float32), normalized_joint_des)
            reconstruct_action = einops.rearrange(action,
                                                  "N (T A) -> N T A",
                                                  A=self.num_hand_joints)
            reconstructed_hand_actions = self.reactive_vae_model.normalizer[
                "action"].unnormalize(reconstruct_action).to(self.device).clip(
                    -1, 1)

        return reconstructed_hand_actions

    def step_reactive_vae_env(self, actions, base_action=None):

        clip_actions = actions.clone()

        hand_noise = clip_actions[:,
                                  -self.num_diffusion_finger_actions:].clone(
                                  ).reshape(self.env.unwrapped.num_envs,
                                            self.latent_dim *
                                            self.T_down).clip(-1, 1)
        reconstructed_hand_actions = self.get_action(hand_noise)

        arm_actions = clip_actions[:, :-self.num_finger_actions].clone().clip(
            -1, 1)

        hand_arm_actions = torch.cat(
            [arm_actions, reconstructed_hand_actions[:, -1]], dim=1)

        hand_arm_actions = (hand_arm_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        hand_arm_actions[:, :-self.
                         num_hand_joints] *= self.env.unwrapped.step_dt * 1

        if self.args_cli.use_residual_action:
            residual_hand_action = clip_actions[:, -self.
                                                num_diffusion_finger_actions -
                                                self.num_hand_joints:-self.
                                                num_diffusion_finger_actions].clone(
                                                )  #residual action
            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound
            hand_arm_actions[:, -self.
                             num_hand_joints:] += residual_hand_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.step(
            hand_arm_actions)

        if self.save_data:
            self.save_data_to_buffer(self.last_obs, hand_arm_actions, rewards,
                                     terminated | time_outs)
            self.last_obs = copy.deepcopy(obs)

        torch.cuda.empty_cache()

        dones = terminated | time_outs
        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            dones[:] = True
            terminated[:] = False
            time_outs[:] = True
            sucess = self.eval_success(obs)
            print("success: ",
                  sucess.sum().item() / self.env.unwrapped.num_envs)

        else:
            dones[:] = False
            terminated[:] = False
            time_outs[:] = False

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_reactive_vae_chunk_env(self, actions, base_action=None):

        clip_actions = actions.clone()

        hand_noise = clip_actions[:,
                                  -self.num_diffusion_finger_actions:].clone(
                                  ).reshape(self.env.unwrapped.num_envs,
                                            self.latent_dim *
                                            self.T_down)  # last chunk
        reconstructed_hand_actions = self.get_action(hand_noise)
        rewards = torch.zeros(self.env.unwrapped.num_envs,
                              dtype=torch.float32,
                              device=self.device)

        if self.args_cli.use_residual_action:
            residual_hand_action = clip_actions[:, -self.
                                                num_diffusion_finger_actions -
                                                self.num_hand_joints:-self.
                                                num_diffusion_finger_actions].clone(
                                                )  #residual action

            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound

        for action_idx in range(1):  #reconstructed_hand_actions.shape[1]

            arm_actions = clip_actions[:, :-self.num_finger_actions].clone(
            ).clip(-1, 1)

            hand_arm_actions = torch.cat(
                [arm_actions, reconstructed_hand_actions[:, action_idx]],
                dim=1)

            hand_arm_actions = (hand_arm_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound

            hand_arm_actions[:, :-self.
                             num_hand_joints] *= self.env.unwrapped.step_dt * 1
            if self.args_cli.use_residual_action:
                hand_arm_actions[:, -self.
                                 num_hand_joints:] += residual_hand_action.clone(
                                 )

            obs, next_rewards, terminated, time_outs, extras = self.env.step(
                hand_arm_actions)
            if self.save_data:
                self.save_data_to_buffer(self.last_obs, hand_arm_actions,
                                         rewards, terminated | time_outs)
                self.last_obs = copy.deepcopy(obs)

            dones = terminated | time_outs
            if dones[0]:
                break
            rewards += next_rewards

            torch.cuda.empty_cache()

        dones = terminated | time_outs
        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            dones[:] = True
            terminated[:] = False
            time_outs[:] = True

        else:
            dones[:] = False
            terminated[:] = False
            time_outs[:] = False

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_reactive_vae_chunk_interpolate_env(self,
                                                actions,
                                                base_action=None):

        clip_actions = actions.clone()
        if self.cur_chunk_index % self.reactive_vae_model.horizon == 0:  # update the buffer when using all the chunk actions

            hand_noise = clip_actions[:, -self.
                                      num_diffusion_finger_actions:].clone(
                                      ).reshape(self.env.unwrapped.num_envs,
                                                self.latent_dim * self.T_down)
            normalized_joint_des = self.temporal_obs_buffer.compute_obs(
            ).clone()

            obs_dict = {"obs": normalized_joint_des}

            with torch.no_grad():

                self.reconstructed_hand_actions = self.reactive_vae_model.predict_action(
                    obs_dict, hand_noise)["action_pred"]

        if self.args_cli.use_residual_action:

            residual_hand_action = clip_actions[:, -self.
                                                num_diffusion_finger_actions -
                                                self.num_hand_joints:-self.
                                                num_diffusion_finger_actions].clone(
                                                )  #residual action

            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound

        arm_actions = clip_actions[:, :-self.num_finger_actions].clone().clip(
            -1, 1)

        hand_arm_actions = torch.cat([
            arm_actions,
            self.reconstructed_hand_actions[:, self.cur_chunk_index %
                                            self.reactive_vae_model.horizon]
        ],
                                     dim=1)

        hand_arm_actions = (hand_arm_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        hand_arm_actions[:, :-self.
                         num_hand_joints] *= self.env.unwrapped.step_dt * 1
        if self.args_cli.use_residual_action:
            hand_arm_actions[:, -self.
                             num_hand_joints:] += residual_hand_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.step(
            hand_arm_actions)
        if self.save_data:
            self.save_data_to_buffer(self.last_obs, hand_arm_actions, rewards,
                                     terminated | time_outs)
            self.last_obs = copy.deepcopy(obs)

        self.cur_chunk_index += 1
        torch.cuda.empty_cache()

        dones = terminated | time_outs
        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            dones[:] = True
            terminated[:] = False
            time_outs[:] = True

        else:
            dones[:] = False
            terminated[:] = False
            time_outs[:] = False

        return obs, rewards, terminated, time_outs, extras, clip_actions
