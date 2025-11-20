import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym

from scripts.workflows.sysID.utils.robot_trajectory import RobotTrajectoryGenerator

from scripts.workflows.sysID.utils.gripper_trajectories import FloatingGripperTrajectoryGenerator


class TrajectoryGenerator:

    def __init__(self, env, task, num_explore_actions) -> None:
        self.env = weakref.proxy(env)
        self.device = self.env.device
        self.task = task
        self.num_envs = self.env.num_envs
        self.num_explore_actions = num_explore_actions
        self.sample_attachment_points = None

        self.init_setting()
        self.plan_trajectories = None

    def init_setting(self):
        deform_object_cfg = self.env.scene["deform_object"].cfg.deform_cfg

        self.include_robot = "Abs" in self.task or "Rel" in self.task
        self.has_gripper = True if "gripper" in self.env.scene.keys(
        ) else False

        self.gripper_traj_generator = None
        self.robot_traj_generator = None

        self.explore_type = "target"
        self.explore_action_index = 0
        self.num_gripper_actions = 0

        if "gripper" in self.env.scene.keys():
            self.gripper_traj_generator = FloatingGripperTrajectoryGenerator(
                self.env, self.num_explore_actions, self.num_gripper_actions)
        if "robot" in self.env.scene.keys():
            self.robot_traj_generator = RobotTrajectoryGenerator(
                self.env, self.num_explore_actions, self.num_gripper_actions)

        self.num_explore_actions = deform_object_cfg["env_setting"][
            "num_explore_actions"]
        self.num_robot_actions = deform_object_cfg["env_setting"][
            "num_robot_actions"] if self.robot_traj_generator is not None else 0
        self.num_gripper_actions = deform_object_cfg["env_setting"][
            "num_gripper_actions"] if self.gripper_traj_generator is not None else 0

        if "gripper" in self.env.scene.keys():
            self.gripper_traj_generator.num_gripper_actions = self.num_gripper_actions
        elif "robot" in self.env.scene.keys():
            self.robot_traj_generator.num_gripper_actions = self.num_gripper_actions

    def on_training_start(self):
        if self.gripper_traj_generator is not None:
            self.gripper_traj_generator.sample_attachment_points = self.sample_attachment_points
        if self.robot_traj_generator is not None:
            self.robot_traj_generator.sample_attachment_points = self.sample_attachment_points

        self.trajectories_generate()

    def trajectories_generate(self):

        actions_buffer = []

        if self.gripper_traj_generator is not None:
            gripper_trajectories = self.gripper_traj_generator.gripper_generator_trajectories(
            )
            actions_buffer.append(gripper_trajectories)
            self.num_gripper_action_dim = gripper_trajectories.shape[-1]

        if self.robot_traj_generator is not None:
            push_trajectories = self.robot_traj_generator.generate_push_trajectories(
            )
            actions_buffer.append(push_trajectories)
            self.num_robot_action_dim = push_trajectories.shape[-1]

        self.plan_trajectories = torch.cat(actions_buffer, dim=-1)

        self.num_robot_actions = self.num_explore_actions - self.num_gripper_actions

        if self.robot_traj_generator is not None and self.gripper_traj_generator is not None:
            # set the robot action to zero
            self.plan_trajectories[:self.num_gripper_actions, ...,
                                   self.num_gripper_action_dim + 2] = 0.8
            self.plan_trajectories[:self.num_gripper_actions, ...,
                                   self.num_gripper_action_dim] = 0.0

            # set the gripper action to zero
            self.plan_trajectories[self.num_gripper_actions:, ..., 0] = 1.8
            self.plan_trajectories[self.num_gripper_actions:, ..., 1] = 1.8
            self.plan_trajectories[self.num_gripper_actions:, ...,
                                   self.num_gripper_action_dim -
                                   1] = 1  # set the gripper action to open

        return self.plan_trajectories

    def get_step_actions(self, explore_type, explore_action_index, **args):

        if explore_type == "train":
            return self.plan_trajectories[explore_action_index,
                                          self.env.episode_length_buf[0]]
        elif explore_type == "target" or explore_type == "eval":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]

            actions[:self.
                    num_explore_actions] = self.plan_trajectories[:, self.env.
                                                                  episode_length_buf[
                                                                      0],
                                                                  0].clone()

            return actions

    def new_episode_training_start(self, explore_action_index):

        return self.plan_trajectories[explore_action_index]

    def technical_reset(self, explore_type, reset_gripper,
                        explore_action_index):

        # random_target_orientation = self.random_orientation()
        if self.has_gripper and reset_gripper:
            self.reset_gripper_and_robot_raw_pose(explore_type,
                                                  explore_action_index)

        elif not self.has_gripper and self.include_robot:

            indices = torch.arange(self.num_envs,
                                   dtype=torch.int64,
                                   device=self.device)

            # robot_target_pose = torch.cat([robot_jpos.squeeze(1)], dim=-1)
            robot_target_pose = self.env.scene[
                "robot"]._data.default_joint_pos[:, :9]
            robot_target_pose[:, -2:] = -1

            self.env.scene["robot"].root_physx_view.set_dof_positions(
                robot_target_pose, indices)
            self.env.scene["robot"].root_physx_view.set_dof_velocities(
                robot_target_pose * 0, indices)

    def reset_gripper_and_robot_raw_pose(self, explore_type,
                                         explore_action_index):

        if self.include_robot:
            self.robot_traj_generator.reset_robot_joint(
                explore_type, explore_action_index)

        # if self.has_gripper:
        #     self.reset_gripper(explore_type, explore_action_index)

    def reset_gripper(self, explore_type, explore_action_index):
        self.env.scene["deform_object"].remove_attachment()
        if explore_type == "train":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]
        elif explore_type == "target" or explore_type == "eval":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]

            actions[:self.
                    num_explore_actions] = self.plan_trajectories[:, self.env.
                                                                  episode_length_buf[
                                                                      0],
                                                                  0].clone()

        self.env.scene[
            "gripper"].data.default_root_state[:, :3] = actions[:, :3]
        self.env.scene["gripper"].data.default_root_state[:,
                                                          3:7] = actions[:,
                                                                         3:7]

        target_root_state = self.env.scene[
            "gripper"].data.default_root_state.clone()

        self.env.scene["gripper"].reset_default_root_state(target_root_state)
        self.sample_attachment_points

    # def generate_poke_trajectories(self, random_target_orientation,
    #                                sampled_attachment_points,
    #                                explore_action_index):

    #     poke_trajectories = torch.cat([
    #         self.init_ee_pose[explore_action_index],
    #         self.gripper_actions.repeat(2)
    #     ]).unsqueeze(0).repeat_interleave(self.num_envs,
    #                                       0).unsqueeze(0).repeat_interleave(
    #                                           self.env.max_episode_length, 0)
    #     # poke_trajectories[:, :, :2] = sampled_attachment_points[:, :2]

    #     arange_tensor = torch.arange(
    #         self.end_frame - self.static_frames,
    #         device=self.ee_motion_vel[explore_action_index].device
    #     )  # Shape [15]
    #     arange_tensor = arange_tensor.view(-1, 1)
    #     ee_motion_vel_expanded = self.ee_motion_vel[explore_action_index].view(
    #         1, 3)
    #     poke_trajectories[self.static_frames:self.end_frame, :, :3] += (
    #         arange_tensor *
    #         ee_motion_vel_expanded).unsqueeze(1).repeat_interleave(
    #             len(sampled_attachment_points), 1)
    #     poke_trajectories[self.end_frame:, :, :3] = poke_trajectories[
    #         self.end_frame - 1, :, :3]
    #     return poke_trajectories

    # def _assemble_base_robot_actions(self, float_gripper_abs_actions):
    #     """Helper function to create the base absolute robot action."""

    #     abs_robot_action = torch.cat([
    #         float_gripper_abs_actions[..., :7].clone(),
    #         float_gripper_abs_actions[:, -1].unsqueeze(1).repeat(1, 2)
    #     ],
    #                                  dim=1)

    #     abs_robot_action[:, 3:7] = math_utils.quat_mul(
    #         abs_robot_action[:, 3:7],
    #         self.env.scene["robot"].data.default_ee_pose[:, 3:7])
    #     abs_robot_action[:, :3] -= self.gripper_offset
    #     return abs_robot_action

    # def assemble_rel_robot_abs_gripper_actions(self, ):
    #     """Assembles relative robot actions based on current end-effector position."""
    #     float_gripper_abs_actions = self.assemble_abs_gripper_actions()
    #     robot_assest = self.env.scene["robot"]
    #     body_id = robot_assest.find_bodies("panda_hand")[0][0]
    #     curr_ee_pos = torch.cat([
    #         robot_assest.data.body_pos_w[:, body_id],
    #         robot_assest.data.body_quat_w[:, body_id]
    #     ],
    #                             dim=1)

    #     abs_robot_action = self._assemble_base_robot_actions(
    #         float_gripper_abs_actions)
    #     delta_xyz = abs_robot_action[:, :3] - curr_ee_pos[:, :3]
    #     delta_euler = math_utils.quat_box_minus(curr_ee_pos[:, 3:7],
    #                                             abs_robot_action[:, 3:7])
    #     delta_robot_action = torch.cat([
    #         delta_xyz, delta_euler,
    #         float_gripper_abs_actions[:, -1].unsqueeze(1).repeat(1, 2)
    #     ],
    #                                    dim=1)
    #     return torch.cat([delta_robot_action, float_gripper_abs_actions],
    #                      dim=1)

    # def assemble_abs_robot_abs_gripper_actions(self, **args):
    #     """Assembles absolute robot actions."""
    #     float_gripper_abs_actions = self.assemble_abs_gripper_actions()
    #     abs_robot_action = self._assemble_base_robot_actions(
    #         float_gripper_abs_actions)
    #     abs_robot_action[:, -2:] = -1
    #     return torch.cat([abs_robot_action, float_gripper_abs_actions], dim=1)

    # def assemble_abs_gripper_actions(self, **args):
    #     """Assemble actions for a single step."""

    #     actions = self.plan_trajectories[
    #         self.env.episode_length_buf[0]]
    #     return torch.cat([
    #         actions,
    #         self.gripper_actions.unsqueeze(0).repeat(self.num_envs, 1)
    #     ],
    #                      dim=1)

    # def assemble_rel_gripper_actions(self, **args):

    #     delta_xyz = torch.zeros((self.num_envs, 3), device=self.device)
    #     if self.env.episode_length_buf[0] == 0:
    #         next_abs_pos = self.plan_trajectories[
    #             self.env.episode_length_buf[0]]
    #         curr_pose = self.env.scene["gripper"].data.body_state_w[:, 0, :7]
    #         delta_xyz = next_abs_pos[:, :3] - curr_pose[:, :3]
    #         delta_euler = math_utils.quat_box_minus(next_abs_pos[:, 3:7],
    #                                                 curr_pose[:, 3:7])
    #     else:
    #         next_abs_pos = self.plan_trajectories[
    #             self.env.episode_length_buf[0]]
    #         cur_abs_pos = self.plan_trajectories[
    #             self.env.episode_length_buf[0] - 1]
    #         delta_xyz = next_abs_pos[:, :3] - cur_abs_pos[:, :3]
    #         delta_euler = math_utils.quat_box_minus(next_abs_pos[:, 3:7],
    #                                                 cur_abs_pos[:, 3:7])

    #     return torch.cat(
    #         [delta_xyz, delta_euler,
    #          self.gripper_actions.unsqueeze(1)], dim=1)
