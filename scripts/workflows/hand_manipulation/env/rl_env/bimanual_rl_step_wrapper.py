from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action, load_pca_data
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import torch
import numpy as np

import matplotlib.pyplot as plt

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
from scripts.workflows.hand_manipulation.env.rl_env.single_rl_step_env import SingleArmRLStep
from scripts.workflows.hand_manipulation.env.rl_env.bimanual_rl_step_env import BimanulRLStep
import sys

sys.path.append("submodule/benchmark_VAE/src")
import yaml
from pythae.models import AutoModel
import os


class BimanulRLStepWrapper:

    def __init__(self, args_cli, env_config, env):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device

        if self.args_cli.add_right_hand:
            self.hand_side = "right"
        elif self.args_cli.add_left_hand:
            self.hand_side = "left"

    def init_setting(self, ):

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]
        self.num_arm_actions = int(self.env.action_space.shape[-1] / 2 -
                                   self.num_hand_joints)

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
                normalize_action=False,
            )
            reset_buffer(self)

        else:
            self.collector_interface = None

        self.init_planner()

        if self.args_cli.add_right_hand and self.args_cli.add_left_hand:
            self.rl_stepper = BimanulRLStep(self.args_cli, self.env_config,
                                            self.env)
        else:
            self.rl_stepper = SingleArmRLStep(self.args_cli, self.env_config,
                                              self.env)

        self.rl_stepper.lower_bound = self.lower_bound
        self.rl_stepper.upper_bound = self.upper_bound

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        if self.args_cli.action_framework == "pca":
            if self.args_cli.add_right_hand:
                self.eigen_vectors_right, self.min_pca_values_right, self.max_pca_values_right, self.pca_D_mean_right, self.pca_D_std_right = load_pca_data(
                    self.args_cli.vae_path, self.device, "right")
                self.num_finger_actions_right = self.eigen_vectors_right.shape[
                    0]
                self.num_finger_actions = self.num_finger_actions_right
            if self.args_cli.add_left_hand:
                self.eigen_vectors_left, self.min_pca_values_left, self.max_pca_values_left, self.pca_D_mean_left, self.pca_D_std_left = load_pca_data(
                    self.args_cli.vae_path, self.device, "left")
                self.num_finger_actions_left = self.eigen_vectors_left.shape[0]
                self.num_finger_actions = self.num_finger_actions_left

            self.step = self.rl_stepper.step_eigengrasp
            self.reset_rl_stepper()

        elif self.args_cli.action_framework == "vae":

            if self.args_cli.add_right_hand:
                self.vae_model_right, self.vae_model_setting_right = self.load_vae(
                    "right")

                self.num_finger_actions_right = self.vae_model_setting_right[
                    -1]
                self.num_finger_actions = self.num_finger_actions_right
            if self.args_cli.add_left_hand:
                self.vae_model_left, self.vae_model_setting_left = self.load_vae(
                    "right")

                self.num_finger_actions_left = self.vae_model_setting_left[-1]
                self.num_finger_actions = self.num_finger_actions_left

            self.step = self.rl_stepper.step_vaegrasp

            self.reset_rl_stepper()
        else:
            self.step = self.rl_stepper.step_env
        self.reset = self.rl_stepper.reset

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
        with open(f"{vae_checkpoint}/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)

            action_mean = torch.as_tensor(model_config["action_mean"]).to(
                self.device)
            action_std = torch.as_tensor(model_config["action_std"]).to(
                self.device)
            data_normalizer = model_config["data_normalizer"]
            max_latent_value = torch.as_tensor(
                model_config["max_latent_value"]).to(self.device)
            min_latent_value = torch.as_tensor(
                model_config["min_latent_value"]).to(self.device)
            latent_dim = model_config["latent_dim"]

            vae_model_setting = [
                min_latent_value, max_latent_value, data_normalizer,
                action_mean, action_std, latent_dim
            ]
        return vae_model_right, vae_model_setting

    def reset_rl_stepper(self):
        if self.args_cli.add_left_hand and self.args_cli.add_right_hand:
            if self.args_cli.action_framework == "pca":
                self.set_by_handness([
                    "eigen_vectors", "min_pca_values", "max_pca_values",
                    "pca_D_mean", "pca_D_std", "num_finger_actions"
                ],
                                     "left",
                                     add_prefix=False)
                self.set_by_handness([
                    "eigen_vectors", "min_pca_values", "max_pca_values",
                    "pca_D_mean", "pca_D_std", "num_finger_actions"
                ],
                                     "right",
                                     add_prefix=False)

            if "vae" in self.args_cli.action_framework:
                self.set_by_handness(
                    ["vae_model", "num_finger_actions", "vae_model_setting"],
                    "left",
                    add_prefix=False)
                self.set_by_handness(
                    ["vae_model", "num_finger_actions", "vae_model_setting"],
                    "right",
                    add_prefix=False)

        else:
            if self.args_cli.action_framework == "pca":
                self.set_by_handness([
                    "eigen_vectors", "min_pca_values", "max_pca_values",
                    "pca_D_mean", "pca_D_std", "num_finger_actions"
                ], self.hand_side)
            if "vae" in self.args_cli.action_framework:
                self.set_by_handness(
                    ["vae_model", "num_finger_actions", "vae_model_setting"],
                    self.hand_side)

    def set_by_handness(self, rl_stepper_list, hand_side, add_prefix=True):
        for name in rl_stepper_list:
            attr_name = f"{name}_{hand_side}"
            if not hasattr(self, attr_name):
                raise AttributeError(f"Missing attribute: {attr_name}")
            value = getattr(self, attr_name)
            if add_prefix:
                setattr(self.rl_stepper, name, value)
            else:
                setattr(self.rl_stepper, attr_name, value)

    def init_bound(self):

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

        if self.args_cli.add_right_hand and self.args_cli.add_left_hand:
            arm_action_limit = torch.cat([arm_action_limit, arm_action_limit],
                                         dim=0)

        self.lower_bound = arm_action_limit[:, 0]
        self.upper_bound = arm_action_limit[:, 1]

    def init_planner(self):
        self.init_bound()

        if self.env_config["params"]["arm_type"] is not None:

            # self.arm_motion_env = ArmMotionPlannerEnv(
            #     self.env,
            #     self.args_cli,
            #     self.env_config,
            # )

            # init_ee_pose = torch.as_tensor(
            #     self.env_config["params"]["init_ee_pose"]).to(
            #         self.device).unsqueeze(0)

            # init_arm_qpos = self.arm_motion_env.ik_plan_motion(
            #     init_ee_pose).repeat_interleave(self.env.num_envs, dim=0)

            init_hand_qpos = torch.zeros(
                (self.env.num_envs, self.num_hand_joints)).to(self.device)

            if self.args_cli.add_right_hand:
                init_arm_qpos = torch.as_tensor([
                    self.env_config["params"]["right_reset_joint_pose"]
                ]).repeat_interleave(self.env.num_envs, dim=0).to(self.device)
                self.init_right_robot_qpos = torch.cat(
                    [init_arm_qpos, init_hand_qpos], dim=1).to(self.device)
                self.env.scene[
                    f"right_hand"].data.reset_joint_pos = self.init_right_robot_qpos
            if self.args_cli.add_left_hand:

                init_arm_qpos = torch.as_tensor([
                    self.env_config["params"]["left_reset_joint_pose"]
                ]).repeat_interleave(self.env.num_envs, dim=0).to(self.device)
                self.init_left_robot_qpos = torch.cat(
                    [init_arm_qpos, init_hand_qpos], dim=1).to(self.device)
                self.env.scene[
                    f"left_hand"].data.reset_joint_pos = self.init_left_robot_qpos

        self.horizon = self.env_config["params"]["Task"]["horizon"]
