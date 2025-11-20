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

from scripts.workflows.hand_manipulation.teleoperation.cloudxr.replay_teleop_action_manager import ReplayTeleopActionManager


class ReplayTeleopFrankaLeapWrapper(ReplayTeleopActionManager):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=0,
        skip_steps=2,
    ):

        super().__init__(env, env_cfg, args_cli, begin_index, skip_steps)

        self.init_data_buffer()

        self.init_setting()

    def init_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def reset_env(self):
        demo_obs = self.raw_data[f"demo_{self.demo_index}"]["obs"]

        demo_action = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_index}"]["actions"]
                     [self.begin_index:])).to(self.device)[::self.skip_steps]
        self.init_data_buffer()

        self.env.reset()
        last_obs, demo_action = self.reset_demo_env(demo_obs, demo_action)

        demo_action = torch.cat([demo_action, demo_action[-1:].repeat(10, 1)],
                                dim=0)
        return last_obs, demo_action

    def check_and_clip_action(self, delta_arm_action):
        """
        Check if delta_arm_action exceeds action_range, clip it, 
        return clipped action, stop flag, and max ratio (|a|/limit).
        """
        # Define bounds
        trans_range = self.action_range[0]
        rot_range = self.action_range[1]

        # Split translation and rotation parts
        trans = delta_arm_action[:, :3]
        rot = delta_arm_action[:, 3:]
        rot = (rot + math.pi) % (2 * math.pi) - math.pi

        # Compute per-element normalized ratio
        trans_ratio = trans.abs() / trans_range
        rot_ratio = rot.abs() / rot_range
        all_ratio = torch.cat([trans_ratio, rot_ratio * 0.0], dim=1)

        # Compute max ratio across all dims per action
        max_ratio = all_ratio.max(dim=1).values

        # Stop if any exceeds 1.0
        stop = max_ratio < 1.0

        # Clip values within range
        trans = torch.clamp(trans, -trans_range, trans_range)
        rot = torch.clamp(rot, -rot_range, rot_range)
        clipped_action = torch.cat([trans, rot], dim=1)
        if not stop:
            clipped_action /= max_ratio

        return clipped_action, stop

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
        clipped_arm_action, stop = self.check_and_clip_action(delta_arm_action)

        return torch.cat([clipped_arm_action, finger_actions], dim=1), stop

    def step_delta_pose(self, action):
        delta_actions = []
        if self.add_left_hand:

            normlized_left_action, stop = self.extract_delta_pose(
                action.clone(),
                hand_side="left",
                arm_start_index=0,
                arm_end_index=7,
                hand_start_index=7,
                hand_end_index=7 + self.num_hand_joint)
            delta_actions.append(normlized_left_action)

        if self.add_right_hand:

            normlized_right_action, stop = self.extract_delta_pose(
                action.clone(),
                hand_side="right",
                arm_start_index=-23,
                arm_end_index=-self.num_hand_joint,
                hand_start_index=-self.num_hand_joint,
                hand_end_index=None)
            delta_actions.append(normlized_right_action)

        return torch.cat(delta_actions, dim=1), stop

    def filter_out_data(self, index):

        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        does_buffer = []
        for i in range(len(self.obs_buffer)):
            per_obs = self.obs_buffer[i]
            per_obs_dict = {}
            for obs_key in list(per_obs["policy"].keys()):

                if "pose" in obs_key and self.hand_side not in obs_key:
                    if (self.pick_object_name not in obs_key
                            and self.task == "grasp") or (
                                (self.place_object_name not in obs_key
                                 and self.pick_object_name not in obs_key)
                                and self.task == "place"):
                        continue
                    else:
                        if self.pick_object_name in obs_key:

                            per_obs_dict[
                                f"{self.hand_side}_manipulated_object_pose"] = per_obs[
                                    "policy"][obs_key][index]
                        if self.task == "place":

                            if self.place_object_name in obs_key:

                                per_obs_dict[
                                    f"{self.hand_side}_hand_place_object_pose"] = per_obs[
                                        "policy"][obs_key][index][..., :3]

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

    def lift_or_not(self, index=None):

        if self.task == "grasp":

            target_object_state = self.env.scene[
                f"{self.pick_object_name}"].data.root_state_w[..., :7]
            success_flag = target_object_state[:, 2] > 0.3
        elif self.task == "place":

            target_object_state = self.env.scene[
                f"{self.place_object_name}"].data.root_state_w[..., :7]
            pick_object_state = self.env.scene[
                f"{self.pick_object_name}"].data.root_state_w[..., :7]
            success_flag = torch.linalg.norm(target_object_state[:, :2] -
                                             pick_object_state[:, :2],
                                             dim=1) < 0.10

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:
                if index is None:
                    index = torch.nonzero(success_flag, as_tuple=True)[0]

                self.filter_out_data(index)
        return success_flag

    def step_env(self, ):

        last_obs, demo_action = self.reset_env()
        if last_obs is None:
            self.demo_index += 1
            return None
        print(self.demo_index, len(demo_action))
        num_action = 0

        for action in demo_action:

            if self.num_envs == 1:
                action = action.unsqueeze(0)

            counter = 0
            stop = False
            if "Rel" in self.args_cli.task:
                while not stop and counter < 4:

                    delta_action, stop = self.step_delta_pose(action)

                    # if "Abs" in self.args_cli.task:
                    #     link_pose = self.env.scene[
                    #         f"{self.hand_side}_panda_link7"]._data.root_state_w[
                    #             ..., :7]
                    #     abs_arm_pose = torch.cat(math_utils.apply_delta_pose(
                    #         link_pose[:, :3], link_pose[:, 3:7],
                    #         delta_action[:, :6]),
                    #                              dim=-1)
                    #     delta_action = torch.cat(
                    #         [abs_arm_pose, delta_action[:, 6:]], dim=-1)
                    new_obs, rewards, terminated, time_outs, extras = self.env.step(
                        delta_action)
                    self.actions_buffer.append(delta_action)
                    counter += 1
                    num_action += 1

                    done = terminated | time_outs

                    self.obs_buffer.append(last_obs)

                    self.does_buffer.append(done)
                    self.rewards_buffer.append(rewards)
                    last_obs = copy.deepcopy(new_obs)

            else:

                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    action)
                self.actions_buffer.append(action)
                num_action += 1

                done = terminated | time_outs

                self.obs_buffer.append(last_obs)

                self.does_buffer.append(done)
                self.rewards_buffer.append(rewards)
                last_obs = copy.deepcopy(new_obs)

        print("total number of actions", num_action)
        self.lift_or_not()

        self.demo_index += 1
        return True
