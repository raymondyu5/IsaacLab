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


class IBRLSimEnv:
    """Residual Environment for RL training with residual actions."""

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        diffusion_device="cuda",
        use_residual=True,
        residual_delta=1.0,
    ):
        self.env = env
        self.device = self.env.device
        self.args_cli = args_cli
        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.residual_delta = residual_delta
        self.use_base_action = args_cli.use_base_action
        self.use_residual = use_residual

        self.init_setting()
        self.diffusion_device = diffusion_device
        self.action_framework = args_cli.action_framework

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]

        arm_action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

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
        self.lower_bound = arm_action_limit[:, 0]
        self.upper_bound = arm_action_limit[:, 1]

        self.hand_side = "right" if self.args_cli.add_right_hand else "left"

        init_joint_pose = self.env_config["params"][
            f"{self.hand_side}_reset_joint_pose"] + [0] * self.num_hand_joints
        self.init_joint_pose = torch.as_tensor(init_joint_pose).unsqueeze(
            0).to(self.device).repeat_interleave(self.env.num_envs, dim=0)
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

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
        setattr(self.diffusion_env, "get_diffusion_obs",
                self.get_diffusion_obs)
        setattr(self.diffusion_env, "evaluate_success", self.evaluate_success)

    def reset(self, concatenate_obs=False):

        next_obs, info = self.env.reset()

        if "IK" in self.args_cli.task:
            # self.reset_robot_joints()
            actions = torch.zeros(self.env.unwrapped.action_space.shape,
                                  dtype=torch.float32,
                                  device=self.device)

        else:

            link7_pose = torch.as_tensor([[
                0.500, -0.000, 0.500, 0.0, 9.2460e-01, -3.8094e-01, 0.0
            ]]).to(self.device).repeat_interleave(self.env.unwrapped.num_envs,
                                                  dim=0)

            actions = torch.cat([
                link7_pose,
                torch.zeros((self.env.unwrapped.num_envs, 16)).to(self.device)
            ],
                                dim=-1)
        next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
            actions)

        self.last_hand_joint_pose = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_pos[:, -self.
                                                      num_hand_joints:].clone(
                                                      )

        return self._process_obs(next_obs["policy"], concatenate_obs), info

    def get_diffusion_obs(self, obs):

        obs_demo = []

        for key in self.diffusion_env.obs_key:

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)

        if self.action_framework in ["pcd_diffusion", "image_diffusion"]:

            image_demo = []

            for key in self.diffusion_env.image_key:
                if self.action_framework in ["pcd_diffusion"]:

                    image_demo.append(obs[key])

                    obs_dict = {
                        "agent_pos": obs_demo.unsqueeze(1),
                        "seg_pc": image_demo.unsqueeze(1),
                    }
                else:

                    transfomred_image = self.diffusion_env.image_transform(
                        obs["rgb"][:, 0].permute(0, 3, 1, 2))
                    image_demo.append(transfomred_image)
                    image_demo = torch.cat(image_demo, dim=0)
                    obs_dict = {
                        "agent_pos": obs_demo.unsqueeze(1),
                        self.diffusion_env.image_key[0]:
                        image_demo.unsqueeze(1),
                    }

        elif self.action_framework == "state_diffusion":
            obs_dict = {
                "obs": obs_demo.unsqueeze(1),
            }

        return obs_dict

    def _process_obs(self,
                     obs_dict: torch.Tensor | dict[str, torch.Tensor],
                     concatenate_obs: bool = True):
        if not concatenate_obs:
            return obs_dict
        obs = []

        if isinstance(obs_dict, dict):
            for key, value in obs_dict.items():
                if key in ["seg_pc", "rgb"]:
                    continue

                if torch.isnan(value).any().item():
                    import pdb
                    pdb.set_trace()

                obs.append(value)

            obs = torch.cat(obs, -1)

        else:
            obs = obs_dict
        return obs

    def evaluate_success(self, ):

        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.scene.env_origins
        lift_or_not = (object_pose[:, 2] > 0.40)
        overhigh_or_not = (object_pose[:, 2] < 0.65)
        outofbox_or_not = ((object_pose[:, 0] < 0.65) &
                           (object_pose[:, 0] > 0.3) &
                           (object_pose[:, 1] < 0.3) &
                           (object_pose[:, 1] > -0.3))
        success = lift_or_not & overhigh_or_not & outofbox_or_not
        return success

    def warm_up(self, replay_buffer, concatenate_obs=False):
        obs, info = self.reset()

        obs, rewards, terminated, time_outs, extras = self.env.step(
            reconstructed_actions.to(self.device))
        # self.last_obs = copy.deepcopy(obs)
        self.last_diffusion_obs = self.get_diffusion_obs(obs["policy"])
        with torch.no_grad():

            self.diffussion_action = self.diffusion_env.policy.predict_action(
                self.last_diffusion_obs)["action_pred"][:, 0].clone()
            # obs["policy"]["last_action"] = self.diffussion_action

        self.total_rewards += rewards.sum().item()
        success = self.evaluate_success().sum().item()

        if self.env.episode_length_buf[0] == self.env.max_episode_length - 1:

            terminated[:] = False
            time_outs[:] = True
            self.total_rewards /= self.env.num_envs
            self.success_rate = success / self.env.num_envs
            print(
                f"[INFO]: Total rewards: {self.total_rewards:.4f} for {self.env.episode_length_buf[0]} steps, with success rate: {self.success_rate:.2f}"
            )
        else:
            terminated[:] = False
            time_outs[:] = False
        done = terminated | time_outs

        return self._process_obs(obs["policy"],
                                 concatenate_obs), rewards, done, success

    def step(self, action, concatenate_obs=False):
        with torch.no_grad():

            clip_actions = action.clone()

            delta_pose = clip_actions.clip(-1, 1)

            delta_hand_arm_actions = (delta_pose + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            # delta_hand_arm_actions *= self.env.step_dt * 1

            delta_hand_arm_actions[:, -self.
                                   num_hand_joints:] += self.last_hand_joint_pose.clone(
                                   )

            obs, rewards, terminated, time_outs, extras = self.env.step(
                delta_hand_arm_actions.to(self.device).detach().to(
                    torch.float32))

            self.total_rewards += rewards.sum().item()
            success = self.evaluate_success().sum().item()

            if self.env.episode_length_buf[
                    0] == self.env.max_episode_length - 1:

                terminated[:] = False
                time_outs[:] = True
                self.total_rewards /= self.env.num_envs
                self.success_rate = success / self.env.num_envs
                print(
                    f"[INFO]: Total rewards: {self.total_rewards:.4f} for {self.env.episode_length_buf[0]} steps, with success rate: {self.success_rate:.2f}"
                )
            else:
                terminated[:] = False
                time_outs[:] = False
            done = terminated | time_outs
            self.last_hand_joint_pose = self.env.scene[
                f"{self.hand_side}_hand"]._data.joint_pos[:, -self.
                                                          num_hand_joints:].clone(
                                                          )

            return self._process_obs(obs["policy"],
                                     concatenate_obs), rewards, done, success
