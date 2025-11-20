import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform

import copy

from scripts.workflows.hand_manipulation.teleoperation.cloudxr.teleoperation_processor import TeleoperationProcessor


class ReplayTeleopFrankaLeapWrapper(TeleoperationProcessor):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=4,
        skip_steps=1,
    ):
        super().__init__(env, env_cfg, args_cli, begin_index, skip_steps)

        if self.augment:
            self.step_env = self.step_aug_env
        else:
            self.step_env = self.step_multi_env

    def reset_env(self):

        self.init_data_buffer()

        self.env.reset()
        last_obs = self.reset_demo_env()
        if last_obs is None:
            return None
        return last_obs

    def extract_delta_pose(self,
                           action,
                           hand_side="right",
                           arm_start_index=0,
                           arm_end_index=None,
                           hand_start_index=None,
                           hand_end_index=None):

        arm_action = action[:, arm_start_index:arm_end_index]
        finger_actions = action[:, hand_start_index:hand_end_index]

        delta_arm_action = math_utils.extract_delta_pose(
            arm_action,
            self.env.scene[f"{hand_side}_panda_link7"]._data.root_state_w[
                ..., :7])

        return torch.cat([delta_arm_action, finger_actions], dim=1)

    def step_delta_pose(self, action):
        delta_actions = []
        if self.add_left_hand:

            normlized_left_action = self.extract_delta_pose(
                action.clone(),
                hand_side="left",
                arm_start_index=0,
                arm_end_index=7,
                hand_start_index=7,
                hand_end_index=7 + self.num_hand_joint)
            delta_actions.append(normlized_left_action)

        if self.add_right_hand:

            normlized_right_action = self.extract_delta_pose(
                action.clone(),
                hand_side="right",
                arm_start_index=-23,
                arm_end_index=-self.num_hand_joint,
                hand_start_index=-self.num_hand_joint,
                hand_end_index=None)
            delta_actions.append(normlized_right_action)

        return torch.cat(delta_actions, dim=1)

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

    def lift_or_not(self):

        if self.task == "grasp":

            target_object_state = self.env.scene[
                f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
            success_flag = target_object_state[:, 2] > 0.3
        elif self.task == "place":

            target_object_state = self.env.scene[
                f"{self.hand_side}_hand_place_object"].data.root_state_w[
                    ..., :7]
            pick_object_state = self.env.scene[
                f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
            success_flag = torch.linalg.norm(target_object_state[:, :2] -
                                             pick_object_state[:, :2],
                                             dim=1) < 0.10

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:

                index = torch.nonzero(success_flag, as_tuple=True)[0]

                self.filter_out_data(index)
        return success_flag

    def step_action(self, action, last_obs):
        if self.use_delta_pose:

            delta_action, = self.step_delta_pose(action)

            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                delta_action.unsqueeze(0))
            self.actions_buffer.append(delta_action.unsqueeze(0).clone())

        else:

            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                action)
            self.actions_buffer.append(action)

        done = terminated | time_outs

        self.obs_buffer.append(last_obs)

        self.does_buffer.append(done)
        self.rewards_buffer.append(rewards)
        last_obs = copy.deepcopy(new_obs)
        return last_obs, rewards, done, extras

    def step_multi_env(self, ):

        last_obs = self.reset_env()

        if last_obs is None:
            return None

        for i in range(self.robot_actions.shape[1]):
            action = self.robot_actions[:, i]
            last_obs, rewards, done, extras = self.step_action(
                action, last_obs)

        self.lift_or_not()
        return True

    def step_aug_env(self, ):
        last_obs = self.reset_env()

        if last_obs is None:
            return None

        for i in range(self.robot_actions.shape[1]):
            action = self.robot_actions[:, i]

            aug_action = self.step_aug_action(action)
            last_obs, rewards, done, extras = self.step_action(
                aug_action, last_obs)
        return True
