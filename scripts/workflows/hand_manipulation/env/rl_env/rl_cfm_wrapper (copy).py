from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import sys

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
from isaaclab.envs import ManagerBasedRLEnv
import time


class RLCFMStepWrapper:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config

        self.use_joint_pose = True if "Play" in args_cli.task else False

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

        self.object_name = self.env_config["params"][
            "target_manipulated_object"]

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]
        self.num_arm_actions = self.env.action_space.shape[
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
                # load_path=self.args_cli.load_path,
                save_path=self.args_cli.save_path,
                use_fps=False,
                use_joint_pos=False
                if "joint" not in self.args_cli.task else True,
                normalize_action=False,
            )

        else:
            self.collector_interface = None

        self.load_diffusion_model()
        self.step = self.step_diffusion_env
        if self.args_cli.use_chunk_action:
            if self.args_cli.use_interpolate_chunk:
                self.step = self.step_diffusion_chunk_interpolate_env
                self.cur_chunk_index = 0

            else:
                self.step = self.step_diffusion_chunk_env

        self.num_diffusion_finger_actions = self.num_hand_joints * self.diffusion_model.horizon
        if self.args_cli.use_residual_action:
            self.residual_step = self.args_cli.residual_step
            self.num_finger_actions = self.num_diffusion_finger_actions + self.num_hand_joints * self.residual_step
        else:
            self.num_finger_actions = self.num_diffusion_finger_actions
        self.num_arm_actions = 6
        self.num_arm_joints = 7

        self.init_planner()

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

    def load_diffusion_model(self):
        checkpoint = os.path.join(self.args_cli.diffusion_path, "checkpoints",
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
        self.diffusion_model = workspace.model
        if cfg.training.use_ema:
            self.diffusion_model = workspace.ema_model

        device = torch.device(self.device)
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
        self.chunk_size = self.diffusion_model.n_obs_steps

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.env.num_envs,
            horizon_K=self.chunk_size,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.diffusion_model.n_obs_steps,
            obs_dim=self.num_hand_joints,
        )

    def init_bound(self):

        # if self.args_cli.action_framework is None:
        if self.args_cli.use_relative_finger_pose and not self.use_joint_pose:

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
        elif self.use_joint_pose:

            arm_action_bound = torch.as_tensor(
                self.env_config["params"]["Task"]["joint_range"]).to(
                    self.device)

            arm_action_limit = torch.stack([
                torch.tensor([-arm_action_bound[0]] * 7, device=self.device),
                torch.tensor([arm_action_bound[0]] * 7, device=self.device)
            ],
                                           dim=1)
        else:
            arm_action_bound = torch.as_tensor(
                self.env_config["params"]["Task"]["action_range"]).to(
                    self.device)
            hand_finger_limit = self.env.scene[
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

    def reset(self):

        obs, info = self.env.reset()

        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()
        import pdb
        pdb.set_trace()

        for i in range(10):

            obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.action_space.shape, device=self.device))

        self.temporal_obs_buffer.reset(
            self.env.max_episode_length * self.diffusion_model.horizon,
            self.env.num_envs)
        self.temporal_action_buffer.reset(
            self.env.max_episode_length * self.diffusion_model.horizon,
            self.env.num_envs)
        self.add_obs_to_buffer(self.env.episode_length_buf[0])
        if self.args_cli.use_interpolate_chunk:
            self.cur_chunk_index = 0
        self.last_obs = None
        self.last_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[..., -16:].clone()

        return obs, info

    def add_obs_to_buffer(self, index):
        state = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        joint_min = self.joint_limits[:, 0]  # shape [16]
        joint_max = self.joint_limits[:, 1]  # shape [16]
        normalized_joints = ((state - joint_min[None, :]) /
                             (joint_max - joint_min)[None, :]) * 2 - 1

        self.temporal_obs_buffer.add_obs(index, normalized_joints)

    def step_diffusion_env(self, actions, base_action=None):

        clip_actions = actions.clone()

        hand_noise = clip_actions[:,
                                  -self.num_diffusion_finger_actions:].clone(
                                  ).reshape(self.env.num_envs,
                                            self.diffusion_model.horizon,
                                            self.num_hand_joints)
        normalized_joint_des = self.temporal_obs_buffer.compute_obs().clone()

        obs_dict = {"obs": normalized_joint_des}
        with torch.no_grad():

            reconstructed_hand_actions = self.diffusion_model.predict_action(
                obs_dict, hand_noise)["action_pred"]

        arm_actions = clip_actions[:, :-self.num_finger_actions].clone().clip(
            -1, 1)

        hand_arm_actions = torch.cat(
            [arm_actions, reconstructed_hand_actions[:, 1]], dim=1)

        hand_arm_actions = (hand_arm_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        hand_arm_actions[:, :-self.num_hand_joints] *= self.env.step_dt * 1
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

        self.add_obs_to_buffer(self.env.episode_length_buf[0])
        torch.cuda.empty_cache()

        return obs, rewards, terminated, time_outs, extras, hand_arm_actions

    def step_diffusion_chunk_env(self, actions, base_action=None):

        clip_actions = actions.clone()

        hand_noise = clip_actions[:,
                                  -self.num_diffusion_finger_actions:].clone(
                                  ).reshape(self.env.num_envs,
                                            self.diffusion_model.horizon,
                                            self.num_hand_joints)  # last chunk
        normalized_joint_des = self.temporal_obs_buffer.compute_obs().clone()

        obs_dict = {"obs": normalized_joint_des}

        with torch.no_grad():

            reconstructed_hand_actions = self.diffusion_model.predict_action(
                obs_dict, hand_noise)["action_pred"]
        rewards = torch.zeros(self.env.num_envs,
                              dtype=torch.float32,
                              device=self.device)

        if self.args_cli.use_residual_action:
            residual_hand_action = clip_actions[:, -self.
                                                num_diffusion_finger_actions -
                                                self.num_hand_joints *
                                                self.residual_step:-self.
                                                num_diffusion_finger_actions].clone(
                                                ).reshape(
                                                    self.env.num_envs,
                                                    self.residual_step,
                                                    self.num_hand_joints)

            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound

        raw_actions = []

        for action_idx in range(reconstructed_hand_actions.shape[1]):

            arm_actions = clip_actions[:, :-self.num_finger_actions].clone(
            ).clip(-1, 1)

            hand_arm_actions = torch.cat(
                [arm_actions, reconstructed_hand_actions[:, action_idx]],
                dim=1)

            hand_arm_actions = (hand_arm_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            hand_arm_actions[:, :-self.num_hand_joints] *= self.env.step_dt * 1
            if self.args_cli.use_residual_action:
                hand_arm_actions[:, -self.
                                 num_hand_joints:] += residual_hand_action.clone(
                                 )[:,
                                   min(action_idx, self.residual_step - 1)]
            if self.save_data:

                if self.eval_iter < 3:
                    hand_arm_actions[:, -self.num_hand_joints:] = 0.0
            obs, next_rewards, terminated, time_outs, extras = self.env.step(
                hand_arm_actions)
            if self.args_cli.collect_relative_finger_pose:
                hand_arm_actions[:, -self.
                                 num_hand_joints:] -= self.last_finger_pose.clone(
                                 )
            self.last_finger_pose = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[..., -16:].clone()

            raw_actions.append(hand_arm_actions.clone())
            # print(torch.min(hand_arm_actions.clone()[:, :6]),
            #       torch.max(hand_arm_actions.clone()[:, :6]))

            if self.save_data:
                self.save_data_to_buffer(self.last_obs, hand_arm_actions,
                                         next_rewards, terminated | time_outs)
                self.last_obs = copy.deepcopy(obs)

            dones = terminated | time_outs
            # print(self.env.env.episode_length_buf[0],
            #       self.env.env.max_episode_length - 1)

            if self.env.unwrapped.episode_length_buf[
                    0] == self.env.max_episode_length - 1:
                dones[:] = True
                terminated[:] = False
                time_outs[:] = True

                break
            else:
                dones[:] = False
                terminated[:] = False
                time_outs[:] = False
            rewards += next_rewards

            self.add_obs_to_buffer(self.env.episode_length_buf[0])
            torch.cuda.empty_cache()

        return obs, rewards, terminated, time_outs, extras, torch.stack(
            raw_actions, dim=1)

    def step_diffusion_chunk_interpolate_env(self, actions, base_action=None):

        clip_actions = actions.clone()
        if self.cur_chunk_index % self.diffusion_model.horizon == 0:  # update the buffer when using all the chunk actions

            hand_noise = clip_actions[:, -self.
                                      num_diffusion_finger_actions:].clone(
                                      ).reshape(self.env.num_envs,
                                                self.diffusion_model.horizon,
                                                self.num_hand_joints)
            normalized_joint_des = self.temporal_obs_buffer.compute_obs(
            ).clone()

            obs_dict = {"obs": normalized_joint_des}

            with torch.no_grad():

                self.reconstructed_hand_actions = self.diffusion_model.predict_action(
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
                                            self.diffusion_model.horizon]
        ],
                                     dim=1)

        hand_arm_actions = (hand_arm_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        hand_arm_actions[:, :-self.num_hand_joints] *= self.env.step_dt * 1
        if self.args_cli.use_residual_action:
            hand_arm_actions[:, -self.
                             num_hand_joints:] += residual_hand_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.step(
            hand_arm_actions)

        self.add_obs_to_buffer(self.env.episode_length_buf[0])
        self.cur_chunk_index += 1
        torch.cuda.empty_cache()

        return obs, rewards, terminated, time_outs, extras, hand_arm_actions

    def save_data_to_buffer(self, last_obs, hand_arm_actions, rewards, does):

        ee_quat_des = self.env.action_manager._terms[
            f"{self.hand_side}_arm_action"]._ik_controller.ee_quat_des.clone()
        ee_pos_des = self.env.action_manager._terms[
            f"{self.hand_side}_arm_action"]._ik_controller.ee_pos_des.clone()
        joint_pos_des = self.env.action_manager._terms[
            f"{self.hand_side}_arm_action"].joint_pos_des.clone()
        finger_pos_des = self.env.action_manager._terms[
            f"{self.hand_side}_hand_action"].processed_actions.clone()
        last_obs["policy"]["ee_control_action"] = torch.cat(
            [ee_pos_des, ee_quat_des, finger_pos_des], dim=-1)
        last_obs["policy"]["joint_control_action"] = torch.cat(
            [joint_pos_des, finger_pos_des], dim=-1)

        last_obs["policy"]["delta_ee_control_action"] = torch.cat([
            hand_arm_actions[:, :self.num_arm_actions].clone(), finger_pos_des
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
