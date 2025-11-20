import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform
import math

import matplotlib.pyplot as plt
import copy

import sys

sys.path.append("submodule/benchmark_VAE/src")
from pythae.models import AutoModel
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer, load_config,
    extract_finger_joints)
import os


class ReplayMultiTeleopFrankaLeapWrapper:

    def __init__(self,
                 env,
                 env_cfg,
                 args_cli,
                 begin_index=4,
                 timeouts=4,
                 skip_steps=3,
                 target_object_name=None):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.begin_index = begin_index
        self.timeouts = timeouts
        self.skip_steps = skip_steps

        self.target_object_name = None
        if args_cli.add_left_hand:
            self.hand_side = "left"
        elif args_cli.add_right_hand:
            self.hand_side = "right"
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
        )

        self.init_data_buffer()

        self.init_setting()

        self.init_replay_info()

    def init_replay_info(self):
        object_pose_dict = {}
        object_actions = {}
        object_init_robot_pose = {}
        object_demo_count = {}
        object_latent_actions = {}

        if self.args_cli.vae_path is not None:
            object_actions = {}

            num_latent_actions = self.env.action_space.shape[
                -1] - self.num_hand_joints + self.latent_model_setting[-1]

        for object_name in self.rigid_object_list:
            object_pose_dict[object_name] = []
            object_actions[object_name] = []
            object_init_robot_pose[object_name] = []
            object_latent_actions[object_name] = []
            object_demo_count[object_name] = 0

        for key in self.raw_data.keys():
            demo_obs = self.raw_data[key]["obs"]
            if self.raw_data[key]["actions"].shape[0] > 160:
                continue

            object_name = None
            for obs_key in demo_obs.keys():
                if "pose" in obs_key and self.hand_side not in obs_key:
                    candidate_name = obs_key.split("_pose")[0]
                    if candidate_name not in self.rigid_object_list:
                        continue
                    object_name = candidate_name

                    object_pose_dict[object_name].append(
                        torch.as_tensor(demo_obs[obs_key][0]).unsqueeze(0).to(
                            self.device))
                    break  # stop after finding one valid object_name

            if object_name is None:
                continue  # skip this demo if no valid object was found

            gt_actions = torch.zeros(
                (self.env.max_episode_length - 20,
                 self.env.action_space.shape[-1])).to(self.device)
            num_actions = self.raw_data[key]["actions"].shape[0]

            gt_actions[:num_actions] = torch.as_tensor(
                self.raw_data[key]["actions"]).to(self.device).clone()
            gt_actions[num_actions:, -self.num_hand_joints:] = gt_actions[
                num_actions - 1, -self.num_hand_joints:].clone()
            gt_actions[num_actions:, 2] = 0.1
            if self.args_cli.vae_path is not None:
                gt_latent_actions = torch.zeros(
                    (self.env.max_episode_length - 20,
                     num_latent_actions)).to(self.device)
                gt_latent_actions[
                    ..., :-self.latent_model_setting[-1]] = gt_actions[
                        ..., :-self.num_hand_joint]
                gt_latent_actions[:, self.latent_model_setting[
                    -1]:] = self.latent_model.encode_rl_action(
                        gt_actions[..., -self.num_hand_joint:])
                object_latent_actions[object_name].append(gt_latent_actions)

            object_actions[object_name].append(gt_actions)
            object_init_robot_pose[object_name].append(
                torch.as_tensor(demo_obs[f"{self.hand_side}_hand_joint_pos"]
                                [0]).unsqueeze(0).to(self.device))
            object_demo_count[object_name] += 1
        per_object_chances = self.env.num_envs // len(self.rigid_object_list)
        all_num_round = int(
            max(object_demo_count.values()) / per_object_chances)
        if all_num_round == 0:
            all_num_round = 1

        self.init_object_pose = torch.zeros(
            (all_num_round, self.num_envs, 7), ).to(self.device)
        self.robot_actions = torch.zeros(
            self.env.action_space.shape, ).unsqueeze(1).repeat_interleave(
                self.env.max_episode_length - 20, dim=1).to(
                    self.device).unsqueeze(0).repeat_interleave(all_num_round,
                                                                dim=0)
        if self.args_cli.vae_path is not None:
            self.robot_latent_actions = torch.zeros(
                (all_num_round, self.num_envs, self.env.max_episode_length -
                 20, num_latent_actions), ).to(self.device)

        self.init_hand_joint_pose = torch.zeros(
            (all_num_round, self.num_envs,
             demo_obs[f"{self.hand_side}_hand_joint_pos"][0].shape[-1]), ).to(
                 self.device)

        for index, object_name in enumerate(self.rigid_object_list):
            per_object_chances = self.env.num_envs // len(
                self.rigid_object_list)
            if len(object_actions[object_name]) == 0:
                continue
            gt_action = object_actions[object_name]  # list of (T, A) tensors
            gt_action = torch.stack(gt_action, dim=0)  # (N_demos, T, A)
            init_object_pose = object_pose_dict[object_name]
            init_object_pose = torch.stack(init_object_pose, dim=0).squeeze(1)

            init_robot_pose = object_init_robot_pose[object_name]
            init_robot_pose = torch.stack(init_robot_pose, dim=0).squeeze(1)

            num_round = gt_action.shape[0] // per_object_chances
            if num_round == 0:
                num_round = 1
                per_object_chances = gt_action.shape[0]
                required_demos = gt_action.shape[0]
            else:
                per_object_chances = self.env.num_envs // len(
                    self.rigid_object_list)
                required_demos = num_round * per_object_chances

            flattent_actions = gt_action[:required_demos].reshape(
                num_round, per_object_chances, gt_action.shape[1],
                gt_action.shape[2])
            flattent_init_object_pose = init_object_pose[:required_demos].reshape(
                num_round, per_object_chances, init_object_pose.shape[1])
            flattent_init_robot_pose = init_robot_pose[:required_demos].reshape(
                num_round, per_object_chances, init_robot_pose.shape[-1])
            object_index = torch.arange(per_object_chances).to(
                self.device) * len(self.rigid_object_list) + index

            self.robot_actions[:num_round, object_index] = flattent_actions
            self.init_object_pose[:num_round,
                                  object_index] = flattent_init_object_pose
            self.init_hand_joint_pose[:num_round,
                                      object_index] = flattent_init_robot_pose
            if self.args_cli.vae_path is not None:
                gt_latent_actions = object_latent_actions[object_name]
                gt_latent_actions = torch.stack(gt_latent_actions, dim=0)
                flattent_gt_latent_actions = gt_latent_actions[:required_demos].reshape(
                    num_round, per_object_chances, gt_action.shape[1],
                    num_latent_actions)
                self.robot_latent_actions[:num_round,
                                          object_index] = flattent_gt_latent_actions

    def init_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def init_rigid_object_setting(self):

        self.rigid_object_setting = self.env_cfg["params"].get(
            "RigidObject", {})
        self.rigid_object_list = self.env_cfg['params']["multi_cluster_rigid"][
            f"{self.hand_side}_hand_object"]["objects_list"] if self.env_cfg[
                "params"][
                    "target_manipulated_object"] == "all" else self.env_cfg[
                        "params"]["target_manipulated_object"]

    def init_setting(self):

        init_ee_pose = torch.as_tensor(
            self.env_cfg["params"]["init_ee_pose"]).to(
                self.device).unsqueeze(0)
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]

        init_pose = torch.cat([
            init_ee_pose,
            torch.zeros(1, self.num_hand_joint).to(self.device)
        ],
                              dim=1)
        self.init_rigid_object_setting()

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        self.raw_data = self.collector_interface.raw_data["data"]
        self.demo_index = 0
        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)

        arm_action_bound = torch.stack([
            torch.tensor(
                [-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3,
                device=self.device),
            torch.tensor([arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3,
                         device=self.device)
        ],
                                       dim=1)
        hand_action_bound = self.env.scene[f"right_hand"]._data.joint_limits[
            0, -self.num_hand_joint:]
        self.action_bound = torch.cat([arm_action_bound, hand_action_bound],
                                      dim=0)

        if self.args_cli.vae_path is not None:

            all_dirs = [
                d for d in os.listdir(self.args_cli.vae_path)
                if os.path.isdir(os.path.join(self.args_cli.vae_path, d))
            ]
            last_training = sorted(all_dirs)[-1]

            self.latent_model = AutoModel.load_from_folder(
                os.path.join(self.args_cli.vae_path, last_training,
                             'final_model'),
                device=self.device).to(self.device)
            self.latent_model_setting = load_config(self.args_cli.vae_path,
                                                    to_torch=True)

    def reset_robot_joints(self, init_joint_pose):

        self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                init_joint_pose, indices=self.env_ids)

    def reset_rigid_objects(self, demo_obs):
        target_object = []
        for rigid_object in list(self.env.scene.rigid_objects.keys()):
            init_rigid_object_pose = demo_obs.get(f"{rigid_object}_pose", None)
            if init_rigid_object_pose is not None:
                reset_target_pose = torch.as_tensor(
                    init_rigid_object_pose[self.begin_index]).unsqueeze(0).to(
                        self.device).repeat(self.num_envs, 1)
                reset_target_pose[..., :3] += self.env.scene.env_origins
                self.env.scene[rigid_object].write_root_pose_to_sim(
                    reset_target_pose, self.env_ids)
                whole_height = init_rigid_object_pose[self.begin_index:, 2]
                indices = np.where(whole_height > 0.30)[0]
                if len(indices) > 0:
                    last_index = indices[-1] if len(indices) > 0 else None
                    target_object.append([
                        rigid_object,
                    ])
        return target_object

    def reset_object_pose(self, init_object_pose):
        init_object_pose[:, :3] += self.env.scene.env_origins
        self.env.scene[f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
            init_object_pose, self.env_ids)

    def reset_env(self, round_id=0):

        self.env.reset()

        for i in range(10):

            self.reset_robot_joints(self.init_hand_joint_pose[round_id])
            self.reset_object_pose(self.init_object_pose[round_id].clone())

            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.action_space.shape).to(self.device))

        return new_obs

    def filter_out_data(self, index):

        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        does_buffer = []
        for i in range(len(self.obs_buffer)):
            per_obs = self.obs_buffer[i]
            per_obs_dict = {}
            for obs_key in list(per_obs["policy"].keys()):

                per_obs_dict[obs_key] = per_obs["policy"][obs_key][index]

            obs_buffer.append(per_obs_dict)
            actions_buffer.append(self.actions_buffer[i][index])
            rewards_buffer.append(self.rewards_buffer[i][index])
            does_buffer.append(self.does_buffer[i])

        self.collector_interface.add_demonstraions_to_buffer(
            obs_buffer,
            actions_buffer,
            rewards_buffer,
            does_buffer,
        )

    def lift_or_not(self, ):

        target_object_state = self.env.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.3

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:

                index = torch.nonzero(success_flag, as_tuple=True)[0]

                self.filter_out_data(index)
        return success_flag

    def step_env(self, ):

        for round_id in range(self.robot_actions.shape[0]):

            last_obs = self.reset_env(round_id)
            self.init_data_buffer()

            demo_action = self.robot_actions[round_id].clone()
            if self.args_cli.vae_path is not None:
                latent_action = self.robot_latent_actions[round_id][..., 6:]
                recontructed_action = self.latent_model.decode_rl_action(
                    latent_action.reshape(-1,
                                          latent_action.shape[-1])).reshape(
                                              self.env.num_envs, -1,
                                              self.num_hand_joint)
                demo_action[..., -self.num_hand_joints:] = recontructed_action

            for timestamp in range(demo_action.shape[1]):

                raw_action = math_utils.denormalize_action(
                    demo_action[:, timestamp].clone(), self.action_bound,
                    self.env.step_dt)
                raw_action[:, -self.num_hand_joints:] /= self.env.step_dt

                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    raw_action)

                done = terminated | time_outs
                if timestamp == demo_action.shape[1] - 1:
                    done[:] = True

                self.obs_buffer.append(last_obs)
                self.actions_buffer.append(
                    demo_action[:,
                                timestamp].clone() if self.args_cli.vae_path is
                    None else self.robot_latent_actions[round_id].clone())

                self.does_buffer.append(done)
                self.rewards_buffer.append(rewards)
                last_obs = copy.deepcopy(new_obs)

            sucess_flag = self.lift_or_not()
