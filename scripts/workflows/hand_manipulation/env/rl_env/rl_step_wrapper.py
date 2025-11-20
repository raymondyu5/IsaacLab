from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data, decode_latent_action
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

# from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import sys
import os

# Get the absolute path to the isaaclab root directory
# This handles both container (/workspace/isaaclab) and host paths
script_dir = os.path.dirname(os.path.abspath(__file__))
isaaclab_root = os.path.abspath(os.path.join(script_dir, "../../../../../"))

benchmark_vae_path = os.path.join(isaaclab_root, "submodule/benchmark_VAE/src")
diffusion_policy_path = os.path.join(isaaclab_root, "submodule/diffusion_policy")

if benchmark_vae_path not in sys.path:
    sys.path.append(benchmark_vae_path)
if diffusion_policy_path not in sys.path:
    sys.path.append(diffusion_policy_path)
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
import gymnasium as gym


class RLStepWrapper:

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

    def save_data_to_buffer(self, last_obs, hand_arm_actions, rewards, does):

        body_state = self.env.scene[
            f"{self.hand_side}_hand"]._data.body_state_w
        last_obs["policy"]["body_state"] = body_state

        update_buffer(self,
                      None,
                      last_obs,
                      hand_arm_actions,
                      rewards,
                      does,
                      does,
                      convert_to_cpu=True)

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

        if self.args_cli.action_framework == "pca":
            self.eigen_vectors, self.min_pca_values, self.max_pca_values, self.pca_D_mean, self.pca_D_std = load_pca_data(
                self.args_cli.vae_path, self.device)
            self.num_pca_finger_actions = self.eigen_vectors.shape[0]
            self.step = self.step_eigengrasp
            if self.args_cli.use_residual_action:
                self.num_finger_actions = self.num_pca_finger_actions + self.num_hand_joints
            else:
                self.num_finger_actions = self.num_pca_finger_actions

            self.env.unwrapped.action_space = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(self.env.unwrapped.num_envs,
                       self.num_pca_finger_actions + 6),
                dtype=np.float32)

        elif self.args_cli.action_framework == "vae":

            self.vae_model, self.vae_model_setting = self.load_vae(
                self.hand_side)

            self.num_vae_finger_actions = self.vae_model_setting[-1]
            if self.args_cli.use_residual_action:
                self.num_finger_actions = self.num_vae_finger_actions + self.num_hand_joints
            else:
                self.num_finger_actions = self.num_vae_finger_actions

            self.step = self.step_vaegrasp

        else:
            self.num_finger_actions = self.num_hand_joints
            self.step = self.step_env

        self.init_planner()

        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

    def load_vae(self, hand_side="left"):

        vae_checkpoint = self.args_cli.vae_path.replace("right", hand_side)
        vae_checkpoint = vae_checkpoint.replace("left", hand_side)

        all_dirs = [
            d for d in os.listdir(vae_checkpoint)
            if os.path.isdir(os.path.join(vae_checkpoint, d))
        ]
        last_training = sorted(all_dirs)[-1]

        vae_model_right = AutoModel.load_from_folder(
            os.path.join(vae_checkpoint, last_training, 'final_model'),
            device=self.device).to(self.device)
        vae_model_setting = load_config(self.args_cli.vae_path, to_torch=True)
        return vae_model_right, vae_model_setting

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

        # if self.env_config["params"]["arm_type"] is not None:

        # self.arm_motion_env = ArmMotionPlannerEnv(
        #     self.env,
        #     self.args_cli,
        #     self.env_config,
        # )

        # init_ee_pose = torch.as_tensor(
        #     self.env_config["params"]["init_ee_pose"]).to(
        #         self.device).unsqueeze(0)

        # init_arm_qpos = self.arm_motion_env.ik_plan_motion(
        #     init_ee_pose)[0].repeat_interleave(self.env.num_envs, dim=0)
        # init_hand_qpos = torch.zeros(
        #     (self.env.num_envs, self.num_hand_joints)).to(self.device)

        # self.init_robot_qpos = torch.cat([init_arm_qpos, init_hand_qpos],
        #                                  dim=1).to(self.device)

        # self.env.scene[
        #     f"{self.hand_side}_hand"].data.reset_joint_pos = self.init_robot_qpos

        self.horizon = self.env_config["params"]["Task"]["horizon"]

    def reset_robot_joints(self, ):

        init_joint_pose = self.env_config["params"][
            f"{self.hand_side}_reset_joint_pose"] + [0] * self.num_hand_joints

        self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                torch.as_tensor(init_joint_pose).unsqueeze(0).to(
                    self.device).repeat_interleave(self.env.unwrapped.num_envs,
                                                   dim=0),
                indices=self.env_ids)

    def reset(self):

        obs, info = self.env.reset()

        for i in range(10):
            # self.reset_robot_joints()
            obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.unwrapped.num_envs,
                            self.raw_action_space).to(device=self.device))
        self.pre_finger_action = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs, info

    def vis_obs(self, obs):
        # Create a figure and subplots
        plt.figure(figsize=(12, 8))
        obs = obs.cpu().numpy()

        for i in range(obs.shape[1]):
            plt.plot(obs[:, i], label=f'Plot {i+1}')

        # Add labels, legend, and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Visualization of N Plots')
        plt.legend()
        plt.show()

    def step_env(self, actions, base_action=None):

        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        if self.args_cli.use_relative_finger_pose:
            clip_actions = (clip_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound

            clip_actions *= self.env.unwrapped.step_dt * 1
            clip_actions[:, -self.
                         num_hand_joints:] += self.pre_finger_action.clone()
        else:
            raw_action = math_utils.denormalize_action(
                clip_actions.clone(), self.action_bound,
                self.env.unwrapped.step_dt)
            raw_action[...,
                       -self.num_hand_joints:] /= self.env.unwrapped.step_dt

            clip_actions = raw_action.clone()

        obs, rewards, terminated, time_outs, extras = self.env.step(
            clip_actions)

        if self.save_data:
            self.save_data_to_buffer(self.last_obs, clip_actions, rewards,
                                     terminated | time_outs)
            self.last_obs = obs

        self.pre_finger_action = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.num_hand_joints:]

        rewards_tensor = rewards
        if torch.isnan(rewards_tensor).any():
            print("Warning: rewards contain NaN values!")
        dones = terminated | time_outs
        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            dones[:] = True
            terminated[:] = False
            time_outs[:] = True
            success = self.eval_success(obs)
            print("success eval:",
                  success.sum().item() / self.env.unwrapped.num_envs)

        else:
            dones[:] = False
            terminated[:] = False
            time_outs[:] = False

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def step_vaegrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        with torch.no_grad():
            hand_actions = clip_actions[:,
                                        -self.num_vae_finger_actions:].clone()

            vae_values = (hand_actions + 1) / 2 * (
                self.vae_model_setting[1] -
                self.vae_model_setting[0]) + self.vae_model_setting[0]

            reconstruct_finger_pose = self.vae_model.decode_rl_action(
                vae_values)

        vae_reconstructed_actions = torch.cat([
            clip_actions[:, :-self.num_finger_actions], reconstruct_finger_pose
        ],
                                              dim=1)

        vae_reconstructed_actions = math_utils.denormalize_action(
            vae_reconstructed_actions.clone(), self.action_bound,
            self.env.unwrapped.step_dt)
        vae_reconstructed_actions[:, -self.
                                  num_hand_joints:] /= self.env.unwrapped.step_dt

        if self.args_cli.use_residual_action:

            residual_hand_action = clip_actions[:,
                                                -self.num_finger_actions:-self.
                                                num_vae_finger_actions].clone(
                                                )  #residual action
            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound
            vae_reconstructed_actions[:, -self.
                                      num_hand_joints:] += residual_hand_action.clone(
                                      )

        obs, rewards, terminated, time_outs, extras = self.env.step(
            vae_reconstructed_actions)
        self.pre_finger_action = self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        if self.save_data:
            self.save_data_to_buffer(self.last_obs, clip_actions, rewards,
                                     terminated | time_outs)
            self.last_obs = obs

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

    def step_eigengrasp(self, actions, base_action=None):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)

        hand_actions = reconstruct_hand_pose_from_normalized_action(
            clip_actions[:, -self.num_pca_finger_actions:], self.eigen_vectors,
            self.min_pca_values, self.max_pca_values, self.pca_D_mean,
            self.pca_D_std)  #.clip(-1, 1)

        pca_reconstructed_actions = torch.cat(
            [clip_actions[:, :-self.num_finger_actions], hand_actions], dim=1)

        pca_reconstructed_actions = (pca_reconstructed_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound

        pca_reconstructed_actions[:, :-self.
                                  num_hand_joints] *= self.env.unwrapped.step_dt * 1

        if self.args_cli.use_residual_action:

            residual_hand_action = clip_actions[:,
                                                -self.num_finger_actions:-self.
                                                num_pca_finger_actions].clone(
                                                )  #residual action
            residual_hand_action = (residual_hand_action + 1) / 2 * (
                self.residual_upper_bound -
                self.residual_lower_bound) + self.residual_lower_bound
            pca_reconstructed_actions[:, -self.
                                      num_hand_joints:] += residual_hand_action.clone(
                                      )

        obs, rewards, terminated, time_outs, extras = self.env.step(
            pca_reconstructed_actions)
        if self.save_data:
            self.save_data_to_buffer(self.last_obs, clip_actions, rewards,
                                     terminated | time_outs)
            self.last_obs = obs

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
