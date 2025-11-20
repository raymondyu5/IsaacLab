from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.utils.math as math_utils
import torch
from scipy.spatial.transform import Rotation as R
from scripts.workflows.hand_manipulation.env.teleop_env.retarget_arm import RetargetArm
from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import numpy as np
import os
import trimesh


class DemoGen:

    def __init__(self,
                 env,
                 args_cli,
                 env_config,
                 arm_motion_env,
                 collector_interface=None,
                 init_robot_pose=None):
        self.env = env
        self.args_cli = args_cli
        self.env_config = env_config
        self.env_config = env_config
        self.arm_motion_env = arm_motion_env
        self.device = env.device
        self.hand_side = "left" if args_cli.add_left_hand else "right"
        self.collector_interface = collector_interface
        self.init_robot_pose = init_robot_pose

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        self.use_joint_pose = True if "Play" in args_cli.task else False
        if self.env_config["params"]["arm_type"] == "franka":
            self.ee_link_name = "panda_link7"

        self.generative_xy_region = torch.as_tensor(
            self.env_config["params"]["generative_region"]["xy_range"]).to(
                self.env.device)
        self.generative_rotation_range = torch.as_tensor(
            self.env_config["params"]["generative_region"]
            ["rotation_range"]).to(self.env.device)
        self.target_manipulated_object = env_config["params"][
            "target_manipulated_object"]

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.num_arm_joints = self.env_config["params"]["num_arm_joints"]
        self.init_hand_qpos = torch.as_tensor([0] * self.num_hand_joints).to(
            self.device).unsqueeze(0)

    def init_data_buffer(self):
        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def sample(self, object_pose):
        init_euler_angles = math_utils.euler_xyz_from_quat(object_pose[:, 3:7])

        x_location = torch.rand((1, 1)).to(self.device) * (
            self.generative_xy_region[0, 1] -
            self.generative_xy_region[0, 0]) + self.generative_xy_region[0, 0]
        y_location = torch.rand((1, 1)).to(self.device) * (
            self.generative_xy_region[1, 1] -
            self.generative_xy_region[1, 0]) + self.generative_xy_region[1, 0]
        z_rotation = torch.rand(
            (1)).to(self.device) * (self.generative_rotation_range[2, 1] -
                                    self.generative_rotation_range[2, 0]
                                    ) + self.generative_rotation_range[
                                        2, 0] + init_euler_angles[2]
        # y_location = object_pose[:, 1].reshape(-1, 1)
        # x_location = object_pose[:, 0].reshape(-1, 1)

        transformed_object_pose = torch.cat(
            [x_location, y_location, object_pose[0, 2].reshape(-1, 1)], dim=1)
        transformed_object_quat = math_utils.quat_from_euler_xyz(
            init_euler_angles[0] * 0.0, init_euler_angles[1] * 0.0, z_rotation)
        return transformed_object_pose, transformed_object_quat

    def pregrasp_finger_action(self, finger_actions, num_frame):
        init_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.get_dof_positions(
            )[:, -self.num_hand_joints:]

        finger_speed = (finger_actions - init_finger_pose) / 10

        finger_action = finger_speed * torch.randint(
            low=2, high=7, size=(1, )).to(self.device) + init_finger_pose
        processed_finger_action = self.interplate_finger_action(
            finger_action, num_frame)

        return processed_finger_action

    def reset(self, object_pose, pregrasp_pose, bbox_region, finger_actions):
        transformed_object_pose, transformed_object_quat = self.sample(
            object_pose)

        transformed_object_matrix = math_utils.pose_to_transformations(
            torch.cat([transformed_object_pose, transformed_object_quat],
                      dim=1))
        init_object_matrix = math_utils.pose_to_transformations(object_pose)
        delta_transform = transformed_object_matrix @ torch.linalg.inv(
            init_object_matrix)

        pregrasp_pose_matrix = math_utils.pose_to_transformations(
            pregrasp_pose)

        transformed_pregrasp_pose = delta_transform @ pregrasp_pose_matrix
        transformed_pregrasp_pose = math_utils.pose_from_transformations(
            transformed_pregrasp_pose)

        self.env.reset()
        transformed_object_pose[:, 2] = -bbox_region[0, 2] + 0.01
        transformed_object_pose = torch.cat(
            [transformed_object_pose, transformed_object_quat], dim=1)

        for i in range(10):
            if "Rel" not in self.args_cli.task:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    self.init_robot_pose)
            else:
                self.env.scene[
                    f"{self.hand_side}_hand"]._root_physx_view.set_dof_positions(
                        self.init_robot_pose, self.env_ids)
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device) *
                    0.0)
            self.env.scene.rigid_objects[
                self.target_manipulated_object].write_root_pose_to_sim(
                    transformed_object_pose,
                    torch.arange(self.env.num_envs).to(self.device))
        transformed_pregrasp_pose[:, 0] -= torch.rand(1).to(
            self.device) * 0.03 + 0.01
        # transformed_pregrasp_pose[:, 2] += torch.rand(1).to(
        #     self.device) * 0.04 + 0.09
        transformed_pregrasp_pose[:, 1] -= torch.rand(1).to(self.device) * 0.05
        plan_ee_pose, pregrasp_arm_qpos = self.arm_motion_env.plan_motion(
            transformed_pregrasp_pose)

        if pregrasp_arm_qpos is None:
            return None, None, None, None, None

        if self.use_joint_pose:
            pregrasp_arm_pose = pregrasp_arm_qpos.clone()
        else:
            pregrasp_arm_pose = plan_ee_pose.clone()
        pregrasp_arm_pose = pregrasp_arm_pose
        pregrasp_finger_action = self.pregrasp_finger_action(
            finger_actions[-1].unsqueeze(0), len(pregrasp_arm_pose))

        return pregrasp_arm_pose, torch.cat(
            [transformed_object_pose, transformed_object_quat], dim=1
        ), delta_transform, transformed_pregrasp_pose, pregrasp_finger_action

    def step_env(
        self,
        actions,
    ):

        if "Rel" not in self.args_cli.task:
            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)
        else:
            cur_ee_pose = self.env.scene[
                f"{self.hand_side}_{self.ee_link_name}"]._data.root_state_w[:, :
                                                                            7].clone(
                                                                            )
            delta_pos = actions[:, :3].clone(
            ) - cur_ee_pose[:, :3] + self.env.scene[
                f"{self.hand_side}_hand"]._data.root_state_w[:, :3].clone()
            delta_quat = math_utils.quat_mul(
                actions[:, 3:7].clone(),
                math_utils.quat_inv(cur_ee_pose[:, 3:7]))
            # delta_rot = math_utils.axis_angle_from_quat(delta_quat)
            delta_rot = math_utils.quat_to_rot_action(delta_quat)

            delta_pose = torch.cat([delta_pos, delta_rot], dim=1)

            target_ee_pose = math_utils.apply_delta_pose(
                cur_ee_pose[:, :3], cur_ee_pose[:, 3:7], delta_pose)

            # # print(torch.max(abs(target_ee_pose[1] - actions[:, 3:7])))
            # print(torch.max(abs(target_ee_pose[0] - actions[:, :3])))

            whole_action = torch.cat(
                [delta_pose, actions[:, -self.num_hand_joints:]], dim=1)
            obs, rewards, terminated, time_outs, extras = self.env.step(
                whole_action)
            del actions
            actions = whole_action

        for object_name in self.env.scene.rigid_objects.keys():

            object_pose = self.env.scene[
                object_name]._data.root_state_w[:, :7].clone()
            obs["policy"][object_name] = object_pose
        if self.args_cli.save_path:
            self.obs_buffer.append(obs)
            self.actions_buffer.append(actions.clone())
            self.rewards_buffer.append(rewards)
            self.does_buffer.append(terminated)
        return obs, rewards, terminated, time_outs, extras

    def pregrasp(self, transformed_object_pose, pregrasp_arm_pose,
                 pregrasp_finger_action):

        for i in range(pregrasp_arm_pose.shape[0]):

            actions = torch.cat([
                pregrasp_arm_pose[i].unsqueeze(0),
                pregrasp_finger_action[i].unsqueeze(0)
            ],
                                dim=1).to(self.device)

            obs, rewards, terminated, time_outs, extras = self.step_env(
                actions)

    def interplate_finger_action(self, finger_pose, num_finger_action):
        init_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.get_dof_positions(
            )[:, -self.num_hand_joints:]

        finger_speed = (finger_pose - init_finger_pose) / num_finger_action
        arange = torch.arange(num_finger_action).to(self.device).unsqueeze(1)
        finger_mat = finger_speed.repeat_interleave(num_finger_action, 0)

        finger_action = finger_mat * arange + init_finger_pose

        return finger_action

    def postgrasp(
        self,
        postgrasp_pose,
        finger_actions,
        delta_transform,
    ):

        postgrasp_pose_matrix = math_utils.pose_to_transformations(
            postgrasp_pose)
        transformed_postgrasp_pose = delta_transform @ postgrasp_pose_matrix
        transformed_postgrasp_pose = math_utils.pose_from_transformations(
            transformed_postgrasp_pose)
        # transformed_postgrasp_pose[:, 2] -= 0.05
        # transformed_postgrasp_pose[:, 2] = transformed_pregrasp_pose[-1, 2]

        plan_ee_pose, postgrasp_arm_qpos = self.arm_motion_env.plan_motion(
            transformed_postgrasp_pose)
        if postgrasp_arm_qpos is None:
            return
        if self.use_joint_pose:
            postgrasp_arm_pose = postgrasp_arm_qpos.clone()
        else:
            postgrasp_arm_pose = plan_ee_pose.clone()
        postgrasp_arm_pose = postgrasp_arm_pose[2:-2:2]

        post_grasp_pose = torch.zeros(
            (1, 7 + self.num_hand_joints)).to(self.device).repeat_interleave(
                len(postgrasp_arm_pose), 0)
        post_grasp_pose[:, :-self.num_hand_joints] = postgrasp_arm_pose
        post_grasp_pose[:, -self.
                        num_hand_joints:] = self.interplate_finger_action(
                            finger_actions[-1].unsqueeze(0),
                            len(postgrasp_arm_pose))

        for i in range(post_grasp_pose.shape[0]):

            obs, rewards, terminated, time_outs, extras = self.step_env(
                post_grasp_pose[i].unsqueeze(0))

    def lift_object(self, finger_actions):

        ee_pose = self.env.scene[
            f"{self.hand_side}_{self.ee_link_name}"]._data.root_state_w[:, :
                                                                        7].clone(
                                                                        )

        # figner_pose = self.env.scene[
        #     f"{self.hand_side}_hand"].root_physx_view.get_dof_positions()
        ee_pose[:, :3] = torch.as_tensor([0.5, 0.0, 0.4]).to(self.device)

        plan_ee_pose, arm_qpos = self.arm_motion_env.plan_motion(
            ee_pose=ee_pose, apply_offset=False)

        if arm_qpos is None:
            return False
        if self.use_joint_pose:
            arm_pose = arm_qpos.clone()
        else:
            arm_pose = plan_ee_pose.clone()
        arm_pose = arm_pose[6::]

        for i in range(arm_pose.shape[0]):
            # load finger actions
            actions = torch.zeros(
                (1, 7 + self.num_hand_joints)).to(self.device)
            actions[:,
                    -self.num_hand_joints:] = finger_actions.clone().reshape(
                        -1, 16)
            actions[:, :-self.num_hand_joints] = torch.as_tensor(
                arm_pose[i]).unsqueeze(0).clone()
            obs, rewards, terminated, time_outs, extras = self.step_env(
                actions)

        for i in range(10):  # make sure the object is lifted
            if "Rel" not in self.args_cli.task:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    actions)
            else:
                actions = torch.zeros(
                    (1, 6 + self.num_hand_joints)).to(self.device)
                actions[:, -self.num_hand_joints:] = finger_actions.clone(
                ).reshape(-1, self.num_hand_joints)
                self.env.step(actions)

        if self.env.scene[self.target_manipulated_object]._data.root_state_w[
                0, 2] > 0.15:
            return True
        return False

    def step(self, object_pose, pregrasp_pose, postgrasp_pose, bbox_region,
             finger_actions):
        self.init_data_buffer()

        pregrasp_arm_pose, transformed_object_pose, delta_transform, transformed_pregrasp_pose, pregrasp_finger_action = self.reset(
            object_pose, pregrasp_pose, bbox_region, finger_actions)
        if pregrasp_arm_pose is None:
            return

        self.pregrasp(transformed_object_pose, pregrasp_arm_pose,
                      pregrasp_finger_action)
        self.postgrasp(
            postgrasp_pose,
            finger_actions,
            delta_transform,
        )

        success = self.lift_object(finger_actions[-1])

        if self.args_cli.save_path:

            if success:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.does_buffer,
                )
        return success
