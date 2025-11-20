import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles

import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform
import copy
from tools.visualization_utils import vis_pc, visualize_pcd
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data

from scripts.workflows.hand_manipulation.env.bc_env.zarr_data_wrapper import ZarrDatawrapper


class ZarrReplayWrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        zarr_cfg=None,
        num_pcd=1,
    ):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.hand_side = "right" if self.add_right_hand else "left"

        self.target_object_name = f"{self.hand_side}_hand_object"
        self.demo_index = 0
        self.load_collector_interface = ZarrDatawrapper(
            args_cli,
            env_cfg,
            zarr_cfg=zarr_cfg,
            num_pcd=num_pcd,
        )

        self.demo_index = self.load_collector_interface.traj_count

        self.num_arm_actions = 6

        self.init_data_buffer()

        self.init_setting()

    def init_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

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

        self.init_actions = []
        if self.add_left_hand:
            self.init_actions.append(init_pose)
        if self.add_right_hand:
            self.init_actions.append(init_pose)
        self.init_actions = torch.cat(self.init_actions,
                                      dim=1).repeat_interleave(self.num_envs,
                                                               dim=0).to(
                                                                   self.device)

        self.env_ids = torch.arange(self.env.unwrapped.num_envs).to(
            self.device)

    def reset_robot_joints(self, ):

        try:

            init_joint_pose = self.raw_data[
                f"{self.hand_side}_hand_joint_pos"][0]
        except:
            init_joint_pose = self.env_cfg["params"][
                f"{self.hand_side}_reset_joint_pose"] + [
                    0
                ] * self.num_hand_joint

        self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                torch.as_tensor(init_joint_pose).unsqueeze(0).to(
                    self.device).repeat_interleave(self.num_envs, dim=0),
                indices=self.env_ids)

    def reset_rigid_objects(self, raw_data):

        init_rigid_object_pose = torch.as_tensor(
            raw_data[f"{self.hand_side}_manipulated_object_pose"][0]).to(
                self.device).unsqueeze(0).repeat_interleave(self.num_envs,
                                                            dim=0)
        init_rigid_object_pose[..., :3] += self.env.unwrapped.scene.env_origins
        self.env.unwrapped.scene[
            f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
                init_rigid_object_pose, env_ids=self.env_ids)

        if f"{self.hand_side}_hand_place_object" in self.env.scene.keys():

            init_rigid_object_pose = torch.as_tensor(
                raw_data[f"{self.hand_side}_hand_place_object_pose"][0]).to(
                    self.device).unsqueeze(0).repeat_interleave(self.num_envs,
                                                                dim=0)
            init_rigid_object_pose[
                ..., :3] += self.env.unwrapped.scene.env_origins
            init_rigid_object_pose = torch.cat([
                init_rigid_object_pose[:, :3],
                torch.tensor([1, 0, 0, 0]).to(
                    self.device).unsqueeze(0).repeat_interleave(self.num_envs,
                                                                dim=0)
            ],
                                               dim=-1)
            self.env.unwrapped.scene[
                f"{self.hand_side}_hand_place_object"].write_root_pose_to_sim(
                    init_rigid_object_pose, env_ids=self.env_ids)

    def reset_env(self, ):

        self.raw_data = self.load_collector_interface.load_data()
        self.reset_robot_joints()
        self.demo_action = torch.as_tensor(self.raw_data["actions"]).to(
            self.device)

        self.init_data_buffer()

        last_obs, _ = self.env.unwrapped.reset()
        self.reset_rigid_objects(self.raw_data)
        reset_buffer(self)

        for i in range(20):
            if self.use_delta_pose:
                self.reset_robot_joints()

                last_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                    torch.zeros(self.env.unwrapped.action_space.shape).to(
                        self.device))
            else:

                last_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                    self.demo_action[0].to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.num_envs, dim=0))

        return last_obs

    def lift_or_not(self, ):

        target_object_state = self.env.unwrapped.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.3

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:

                index = torch.nonzero(success_flag, as_tuple=True)[0]

                filter_out_data(self, index.cpu())

        return success_flag

    def save_data_to_buffer(self,
                            next_obs,
                            last_obs,
                            hand_arm_actions,
                            rewards,
                            does,
                            save_extras=False):
        if save_extras:
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

        update_buffer(self, next_obs, last_obs, hand_arm_actions, rewards,
                      does, does)
