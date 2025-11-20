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
# from pythae.models import AutoModel
import os
import isaaclab.utils.math as math_utils
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer, load_config,
    extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation)
import copy
from isaaclab.envs import ManagerBasedRLEnv
import time
import gymnasium as gym


class RLCFMStepWrapper:

    def __init__(self, args_cli, env_config, env, eval_split=1):

        self.args_cli = args_cli
        self.env_config = env_config

        self.use_joint_pose = True if "Joint-Rel" in args_cli.task else False
        self.eval_split = eval_split
        self.task = "place" if "Place" in args_cli.task else "grasp"

        self.env = env
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
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

        self.load_diffusion_model()

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

        self.step = self.step_diffusion_chunk_env
        self.residual_step = self.args_cli.residual_step

        self.num_diffusion_finger_actions = self.diffusion_model.action_dim * self.diffusion_model.horizon

        # num_arm_actions = 6 if self.use_joint_pose else 7
        self.args_cli.residual_step = self.args_cli.residual_step

        if self.args_cli.use_residual_action and not self.args_cli.use_residual_only:

            self.num_finger_actions = self.num_diffusion_finger_actions + self.num_hand_joints * 1
            self.residual_step = self.args_cli.residual_step
            # num_arm_actions = num_arm_actions * self.diffusion_horizon + 6

        elif self.args_cli.use_residual_only:

            self.num_finger_actions = self.num_hand_joints * 1
            self.residual_step = self.args_cli.residual_step

        else:
            self.num_finger_actions = self.num_diffusion_finger_actions

        self.num_arm_actions = 6
        self.num_arm_joints = 7

        self.init_planner()

        if "Abs" in self.args_cli.task and not self.args_cli.revert_action:
            if self.args_cli.use_residual_only:  # residual rl

                self.env.unwrapped.action_space = gym.spaces.Box(
                    low=-10,
                    high=10,
                    shape=(self.num_envs, self.num_hand_joints + 6),
                    dtype=np.float32)
            elif self.args_cli.use_residual_action:  # residual + diffusion

                self.env.unwrapped.action_space = gym.spaces.Box(
                    low=-10,
                    high=10,
                    shape=(self.num_envs,
                           self.raw_action_space * self.diffusion_horizon +
                           self.num_hand_joints + 6),
                    dtype=np.float32)
            else:  # diffusion only

                self.env.unwrapped.action_space = gym.spaces.Box(
                    low=-10,
                    high=10,
                    shape=(self.num_envs,
                           self.raw_action_space * self.diffusion_horizon),
                    dtype=np.float32)

        else:

            self.env.unwrapped.action_space = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(self.num_envs, self.raw_action_space -
                       self.num_hand_joints + self.num_finger_actions),
                dtype=np.float32)

        self.env_ids = torch.arange(self.num_envs).to(self.device)

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

        self.diffusion_obs_keys = cfg.dataset.obs_key

        # get policy from workspace
        self.diffusion_model = workspace.model
        if cfg.training.use_ema:
            self.diffusion_model = workspace.ema_model

        device = torch.device(self.device)
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
        self.chunk_size = self.diffusion_model.n_obs_steps

        self.diffusion_horizon = self.diffusion_model.horizon

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.num_envs,
            horizon_K=self.chunk_size,
            action_dim=self.num_hand_joints,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.num_envs,
            horizon_K=self.diffusion_model.n_obs_steps,
            obs_dim=self.num_hand_joints,
        )
        self.diffusion_cfg = cfg
        self.diffusion_obs_dim = self.diffusion_model.obs_dim

        self.diffusion_obs_space = gym.spaces.Box(low=-np.inf,
                                                  high=np.inf,
                                                  shape=(
                                                      self.diffusion_horizon,
                                                      self.diffusion_obs_dim,
                                                  ),
                                                  dtype=np.float32)
        self.diffusion_action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.diffusion_horizon,
                self.raw_action_space,
            ),
            dtype=np.float32,
        )

    def init_bound(self):

        hand_finger_limit = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"]._data.joint_limits[
                0, -self.num_hand_joints:]

        if self.use_joint_pose:

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

        for i in range(10):

            if "Rel" in self.args_cli.task:

                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.unwrapped.num_envs,
                                self.raw_action_space).to(device=self.device))
            if "Abs" in self.args_cli.task:

                # if self.args_cli.real_eval_mode:

                link7_pose = torch.as_tensor(
                    [[0.500, -0.000, 0.500, 0.0, 9.2460e-01, -3.8094e-01,
                      0.0]]).to(self.device).repeat_interleave(
                          self.env.unwrapped.num_envs, dim=0)
                # else:

                #     link7_pose = self.env.unwrapped.scene[
                #         f"{self.hand_side}_hand"]._data.randomize_ee_pose[:, :
                #                                                           7].clone(
                #                                                           )

                final_ee_pose = torch.cat([
                    link7_pose,
                    torch.zeros(
                        (self.env.unwrapped.num_envs, 16)).to(self.device)
                ],
                                          dim=-1)

                obs, rewards, terminated, time_outs, extras = self.env.step(
                    final_ee_pose)

        self.temporal_obs_buffer.reset(
            self.env.unwrapped.max_episode_length * self.residual_step,
            self.num_envs)
        self.temporal_action_buffer.reset(
            self.env.unwrapped.max_episode_length * self.residual_step,
            self.num_envs)
        self.add_obs_to_buffer(self.env.unwrapped.episode_length_buf[0])
        if self.args_cli.use_interpolate_chunk:
            self.cur_chunk_index = 0
        self.last_obs = obs
        self.last_finger_pose = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[..., -16:].clone()

        if self.save_data:

            self.random_hand_index = torch.randint(
                60,
                100,
                (self.env.num_envs, ),
            ).to(self.device)

        self.pre_finger_action = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs, info

    def add_obs_to_buffer(self, index):
        state = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        joint_min = self.joint_limits[:, 0]  # shape [16]
        joint_max = self.joint_limits[:, 1]  # shape [16]
        normalized_joints = ((state - joint_min[None, :]) /
                             (joint_max - joint_min)[None, :]) * 2 - 1

        self.temporal_obs_buffer.add_obs(index, normalized_joints)

    def save_data_to_buffer(self, last_obs, hand_arm_actions, rewards, does):

        body_state = self.env.scene[
            f"{self.hand_side}_hand"]._data.body_state_w
        last_obs["policy"]["body_state"] = body_state

        # if "IK" in self.args_cli.task:

        # ee_quat_des = self.env.unwrapped.action_manager._terms[
        #     f"{self.hand_side}_arm_action"]._ik_controller.ee_quat_des.clone(
        #     )
        # ee_pos_des = self.env.unwrapped.action_manager._terms[
        #     f"{self.hand_side}_arm_action"]._ik_controller.ee_pos_des.clone(
        #     )
        # joint_pos_des = self.env.unwrapped.action_manager._terms[
        #     f"{self.hand_side}_arm_action"].joint_pos_des.clone()
        # finger_pos_des = self.env.unwrapped.action_manager._terms[
        #     f"{self.hand_side}_hand_action"].processed_actions.clone()
        # last_obs["policy"]["ee_control_action"] = torch.cat(
        #     [ee_pos_des, ee_quat_des, finger_pos_des], dim=-1)
        # last_obs["policy"]["joint_control_action"] = torch.cat(
        #     [joint_pos_des, finger_pos_des], dim=-1)

        # last_obs["policy"]["delta_ee_control_action"] = torch.cat([
        #     hand_arm_actions[:, :self.num_arm_actions].clone(),
        #     finger_pos_des
        # ],
        #   dim=-1)

        update_buffer(self,
                      None,
                      last_obs,
                      hand_arm_actions,
                      rewards,
                      does,
                      does,
                      convert_to_cpu=True)

    def reconstruct_hand_action(self, diffusion_obs, clip_actions):

        if not self.args_cli.use_residual_only:

            if "Abs" in self.args_cli.task and not self.args_cli.revert_action:
                hand_noise = clip_actions[:, -(7 + self.num_hand_joints) *
                                          self.diffusion_horizon:].clone(
                                          ).reshape(
                                              clip_actions.shape[0],
                                              self.diffusion_model.horizon,
                                              7 + self.num_hand_joints
                                          )  # last chunk

            else:

                hand_noise = clip_actions[:, -self.
                                          num_diffusion_finger_actions:].clone(
                                          ).reshape(
                                              clip_actions.shape[0],
                                              self.diffusion_model.horizon,
                                              -1)  # last chunk
        else:

            hand_noise = None

        obs_dict = {"obs": diffusion_obs}

        with torch.no_grad():

            reconstructed_hand_actions = self.diffusion_model.predict_action(
                obs_dict, hand_noise)["action_pred"]

        residual_hand_action = None

        if self.args_cli.use_residual_action and not self.args_cli.use_residual_only:  # residual + diffusion

            if "Abs" in self.args_cli.task and not self.args_cli.revert_action:
                residual_hand_action = clip_actions[:, :self.num_hand_joints +
                                                    6].clone()

            else:

                residual_hand_action = clip_actions[:, -self.
                                                    num_diffusion_finger_actions
                                                    - self.num_hand_joints *
                                                    1:-self.
                                                    num_diffusion_finger_actions].clone(
                                                    ).reshape(
                                                        clip_actions.shape[0],
                                                        1,
                                                        self.num_hand_joints)

        elif self.args_cli.use_residual_only:

            residual_hand_action = clip_actions[:, -self.num_hand_joints *
                                                1:].clone()

        if ("Rel" in self.args_cli.task or self.args_cli.revert_action
            ) and self.args_cli.use_residual_action:

            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound

        return reconstructed_hand_actions, residual_hand_action

    def get_diffused_action(self, clip_actions, reconstructed_hand_actions,
                            residual_hand_action, action_idx):

        clip_actions = clip_actions.clone().clip(-1, 1)

        if "Rel" in self.args_cli.task or self.args_cli.revert_action:

            if reconstructed_hand_actions.shape[-1] > self.num_hand_joints:
                hand_arm_actions = reconstructed_hand_actions.clone()[
                    :,
                    action_idx,
                ]

            else:

                arm_actions = clip_actions[:, :-self.num_finger_actions].clone(
                ).clip(-1, 1)

                hand_arm_actions = torch.cat([
                    arm_actions,
                    reconstructed_hand_actions[:, action_idx,
                                               -self.num_hand_joints:]
                ],
                                             dim=1)

                hand_arm_actions = (hand_arm_actions + 1) / 2 * (
                    self.upper_bound - self.lower_bound) + self.lower_bound
                # if not self.use_joint_pose:
                hand_arm_actions[:, :-self.
                                 num_hand_joints] *= self.env.unwrapped.step_dt * 1

        if "Rel" in self.args_cli.task or self.args_cli.revert_action:

            if self.args_cli.use_residual_action and not self.args_cli.use_residual_only:

                hand_arm_actions[:, -self.
                                 num_hand_joints:] += residual_hand_action.clone(
                                 )[:, min(action_idx, 0)]

            elif self.args_cli.use_residual_only:

                hand_arm_actions[:, -self.
                                 num_hand_joints:] += residual_hand_action.clone(
                                 )

        else:

            if self.args_cli.use_residual_action:

                delta_action = (clip_actions[:, :self.num_hand_joints + 6] +
                                1) / 2 * (self.upper_bound -
                                          self.lower_bound) + self.lower_bound
                arm_actions = clip_actions[:, :6].clone().clip(-1, 1)
                reconstructed_arm_pose = reconstructed_hand_actions[:, 0, :
                                                                    7].clone()
                hand_actions = reconstructed_hand_actions[:, 0, -self.
                                                          num_hand_joints:].clone(
                                                          )
                target_pos, target_rot = math_utils.apply_delta_pose(
                    reconstructed_arm_pose[:, :3], reconstructed_arm_pose[:,
                                                                          3:7],
                    delta_action[:, :6] * self.env.unwrapped.step_dt * 1)

                hand_arm_actions = torch.cat([
                    target_pos, target_rot, hand_actions +
                    (clip_actions[:, 6:6 + self.num_hand_joints] + 1) / 2 *
                    self.env_config["params"]["Task"]["action_range"][2] -
                    self.env_config["params"]["Task"]["action_range"][2]
                ],
                                             dim=1)

            else:

                hand_arm_actions = reconstructed_hand_actions[:, 0].clone()
        return hand_arm_actions

    def process_action(self, clip_actions, reconstructed_hand_actions,
                       residual_hand_action, action_idx):
        hand_arm_actions = self.get_diffused_action(
            clip_actions, reconstructed_hand_actions, residual_hand_action,
            action_idx)
        # hand_arm_actions2 = self.get_diffused_action(
        #     clip_actions, reconstructed_hand_actions, residual_hand_action,
        #     action_idx)

        if self.save_data or self.eval_mode:
            if self.task == "place":
                if "Rel" in self.args_cli.task:

                    success_index = self.eval_success(self.last_obs)
                    if success_index.sum() > 0:

                        hand_arm_actions[success_index, :] = 0.0

        if "Abs" in self.args_cli.task and self.args_cli.revert_action:
            link7_pose = self.env.unwrapped.scene[
                f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :
                                                                    7].clone()
            link7_pose[:, :3] -= self.env.unwrapped.scene.env_origins

            if self.task == "place":

                pick_object_state = self.env.unwrapped.scene[
                    f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3]
                place_target_state = self.env.unwrapped.scene[
                    f"{self.hand_side}_hand_place_object"]._data.root_state_w[:, :
                                                                              3]
                dist = torch.linalg.norm(pick_object_state[:, :2] -
                                         place_target_state[:, :2],
                                         dim=1)
                close_mask = dist < 0.075
                hand_arm_actions[close_mask, :] = 0.0

            ee_pose = math_utils.apply_delta_pose(link7_pose[:, :3],
                                                  link7_pose[:, 3:7],
                                                  hand_arm_actions[:, :6])
            execute_robot_action = torch.cat([
                ee_pose[0], ee_pose[1],
                hand_arm_actions[:, -self.num_hand_joints:]
            ],
                                             dim=1)

            obs, next_rewards, terminated, time_outs, extras = self.env.step(
                execute_robot_action)
            _, terminated, time_outs = self.process_data(
                execute_robot_action, next_rewards, terminated, time_outs, obs)

        else:

            obs, next_rewards, terminated, time_outs, extras = self.env.step(
                hand_arm_actions)

            _, terminated, time_outs = self.process_data(
                hand_arm_actions, next_rewards, terminated, time_outs, obs)

        return hand_arm_actions, obs, next_rewards, terminated, time_outs, extras

    def process_data(self, hand_arm_actions, next_rewards, terminated,
                     time_outs, obs):

        if self.save_data:
            self.save_data_to_buffer(self.last_obs, hand_arm_actions,
                                     next_rewards, terminated | time_outs)
            self.last_obs = copy.deepcopy(obs)

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
        return dones, terminated, time_outs

    def get_diffusion_obs(self):

        if isinstance(self.diffusion_obs_keys, str):
            normalized_joint_des = self.temporal_obs_buffer.compute_obs(
            ).clone()

            diffusion_obs = normalized_joint_des
        else:

            lowdim_obs = []

            for key in self.diffusion_obs_keys:
                lowdim_obs.append(self.last_obs["policy"][key])
            torch.cat(lowdim_obs, dim=1)
            diffusion_obs = torch.cat(lowdim_obs, dim=1)[:, None]
        return diffusion_obs

    def step_diffusion_chunk_env(self, actions, base_action=None):

        clip_actions = actions.clone().clip(-1, 1)

        rewards = torch.zeros(self.num_envs,
                              dtype=torch.float32,
                              device=self.device)
        self.last_diffusion_obs = self.get_diffusion_obs()

        reconstructed_hand_actions, residual_hand_action = self.reconstruct_hand_action(
            self.last_diffusion_obs, clip_actions)

        raw_actions = []
        num_steps = (self.residual_step if self.args_cli.use_residual_action
                     else 2 if self.args_cli.use_chunk_action else 1)

        for action_idx in range(num_steps):

            hand_arm_actions, obs, next_rewards, terminated, time_outs, extras = self.process_action(
                clip_actions, reconstructed_hand_actions, residual_hand_action,
                action_idx)
            self.last_obs = copy.deepcopy(obs)

            self.last_finger_pose = self.env.unwrapped.scene[
                f"{self.hand_side}_hand"].data.joint_pos[..., -16:].clone()

            raw_actions.append(hand_arm_actions.clone())

            rewards += next_rewards

            self.add_obs_to_buffer(self.env.unwrapped.episode_length_buf[0])
            torch.cuda.empty_cache()

            if self.env.unwrapped.episode_length_buf[
                    0] == self.env.unwrapped.max_episode_length - 1:

                eval_sucess_flag = self.eval_success(obs)
                print("Eval Success Rate: ",
                      eval_sucess_flag.sum().item() / self.num_envs)

                break

        return obs, rewards, terminated, time_outs, extras, torch.stack(
            raw_actions, dim=1), self.get_diffusion_obs()
