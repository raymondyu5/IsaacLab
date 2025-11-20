from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.utils.math as math_utils
import torch
from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import trimesh

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import copy

import matplotlib.pyplot as plt


class EvaluationEnv:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.device
        self.env_ids = torch.arange(self.env.num_envs, device=self.device)

        self.init_env()
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
            load_normalize_action=True)
        self.raw_collected_data = self.collector_interface.raw_data["data"]
        self.device = env.device

        self.use_joint_pose = True if "Play" in args_cli.task else False

        self.target_manipulated_object = env_config["params"][
            "target_manipulated_object"]
        self.num_trajectories = len(self.raw_collected_data)

        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        elif self.args_cli.add_right_hand:
            self.hand_side = "right"

        self.num_demos = self.args_cli.num_demos
        self.demo_index = 0

        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_config,
            collision_checker=self.args_cli.collision_checker)

        if self.args_cli.evaluate_mode == "replay":
            self.run = self.replay
        elif self.args_cli.evaluate_mode in ["open_loop", "close_loop"]:
            if self.args_cli.evaluate_mode == "open_loop":
                self.run = self.evaluate_open_loop_bc_poliy
            else:
                self.run = self.evaluate_close_loop_bc_poliy

            self.load_bc_policy()
        self.init_setting()

    def init_env(self):

        if self.env_config["params"]["arm_type"] is not None:
            if "IK" in self.args_cli.task:
                if "Rel" not in self.args_cli.task:
                    init_pose = torch.as_tensor(
                        self.env_config["params"]["init_ee_pose"] +
                        [0] * self.env_config["params"]["num_hand_joints"]).to(
                            self.device).unsqueeze(0)
                else:

                    init_pose = torch.zeros(self.env.action_space.shape).to(
                        self.device)
                    init_joint_pose = torch.as_tensor(
                        self.env_config["params"]["reset_joint_pose"] +
                        [0] * self.env_config["params"]["num_hand_joints"]).to(
                            self.device).unsqueeze(0)
                    self.env.scene[
                        "right_hand"]._root_physx_view.set_dof_positions(
                            init_joint_pose,
                            indices=torch.arange(self.env.num_envs).to(
                                self.device))
            else:
                init_pose = torch.as_tensor(
                    self.env_config["params"]["reset_joint_pose"] +
                    [0] * self.env_config["params"]["num_hand_joints"]).to(
                        self.device).unsqueeze(0)

        for i in range(10):
            self.env.step(init_pose)

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.init_hand_qpos = torch.as_tensor([0] * self.num_hand_joints).to(
            self.device).unsqueeze(0)

        if self.env_config["params"].get("init_ee_pose", None) is not None:
            self.init_ee_pose = torch.as_tensor(
                self.env_config["params"]["init_ee_pose"]).to(
                    self.device).unsqueeze(0)
            self.init_arm_qpos = self.arm_motion_env.ik_plan_motion(
                self.init_ee_pose)

            self.init_ee_pose = torch.cat(
                [self.init_ee_pose, self.init_hand_qpos], dim=1)
            self.init_robot_qpos = torch.cat(
                [self.init_arm_qpos, self.init_hand_qpos], dim=1)
        else:
            reset_joint_pose = torch.as_tensor(
                self.env_config["params"]["reset_joint_pose"]).to(
                    self.device).unsqueeze(0)
            self.init_robot_qpos = torch.cat(
                [self.init_arm_qpos, self.init_hand_qpos], dim=1)

    def load_bc_policy(self):
        import sys
        sys.path.append("submodule/robomimic_openrt")
        import robomimic.utils.file_utils as FileUtils
        self.bc_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=self.args_cli.ckpt_path,
            device=self.device,
            verbose=True)
        self.bc_policy.start_episode()

    def sample_demo(self):

        self.demo_obs = self.raw_collected_data[
            f"demo_{int(self.demo_index)}"]["obs"]
        self.init_object_pose = torch.as_tensor(
            np.array(self.demo_obs[self.target_manipulated_object][0])).to(
                self.device)

        self.demo_actions = self.raw_collected_data[
            f"demo_{int(self.demo_index)}"]["actions"]

    def reset_env(self):

        self.env.reset()

        if self.use_joint_pose:
            init_robot_pose = self.init_robot_qpos
        else:
            init_robot_pose = self.init_ee_pose

        for i in range(10):
            if "Rel" not in self.args_cli.task:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    init_robot_pose)
            else:
                self.env.scene[
                    f"{self.hand_side}_hand"]._root_physx_view.set_dof_positions(
                        self.init_robot_qpos, self.env_ids)
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device) *
                    0.0)
            self.env.scene.rigid_objects[
                self.target_manipulated_object].write_root_pose_to_sim(
                    self.init_object_pose)
        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()
        return obs

    def replay(self):
        self.sample_demo()
        self.reset_env()

        for i in range(self.demo_actions.shape[0]):

            action = torch.as_tensor(
                self.collector_interface.unnormalize(
                    self.demo_actions[i], self.collector_interface.
                    action_stats["action"])).unsqueeze(0).to(self.device)
            action[:, -self.num_hand_joints:] += self.pre_finger_action.clone()

            obs, rewards, terminated, time_outs, extras = self.env.step(action)
            self.pre_finger_action = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                         num_hand_joints:].clone(
                                                         )

        self.demo_index += 1

    def evaluate_open_loop_bc_poliy(self):

        self.sample_demo()
        old_obs = self.reset_env()

        trajectory_length = len(self.demo_obs["last_action"])
        for i in range(0, trajectory_length):
            obs_dict = {}
            for obs_name in self.demo_obs:
                obs_dict[obs_name] = torch.as_tensor(
                    self.demo_obs[obs_name][i]).to(self.device)

            action = self.bc_policy(obs_dict)
            action = torch.as_tensor(
                self.collector_interface.unnormalize(
                    action, self.collector_interface.action_stats["action"])
            ).unsqueeze(0).to(self.device)
            action[:, -self.num_hand_joints:] += self.pre_finger_action.clone()

            obs, rewards, terminated, time_outs, extras = self.env.step(action)
            self.pre_finger_action = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                         num_hand_joints:].clone(
                                                         )

        self.demo_index += 1

    def evaluate_close_loop_bc_poliy(self):

        self.sample_demo()
        new_obs = self.reset_env()
        obs = new_obs["policy"]

        pre_obs = {}
        for obs_name in self.demo_obs:
            pre_obs[obs_name] = torch.as_tensor(
                self.demo_obs[obs_name][31]).to(self.device).unsqueeze(0)
        joint_limits = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_limits

        for i in range(150):
            # obs_dict = {}
            # for obs_name in self.demo_obs:
            #     obs_dict[obs_name] = torch.as_tensor(
            #         self.demo_obs[obs_name][i]).to(self.device).unsqueeze(0)

            for object_name in self.env.scene.rigid_objects.keys():

                obs[object_name] = self.env.scene[
                    object_name]._data.root_state_w[:, :7]

            actions = self.bc_policy(obs)
            # self.viz_actions(actions, self.demo_actions[i])
            del obs

            actions = torch.as_tensor(
                self.collector_interface.unnormalize(
                    actions, self.collector_interface.action_stats["action"])
            ).unsqueeze(0).to(self.device)
            # actions[:,
            #         -self.num_hand_joints:] += self.pre_finger_action.clone()

            # actions = torch.clip(actions, joint_limits[..., 0],
            #                      joint_limits[..., 1])
            # apply actions
            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

            # robomimic only cares about policy observations
            obs = new_obs["policy"]
            self.pre_finger_action = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                         num_hand_joints:].clone(
                                                         )

        self.demo_index += 1
        if obs[self.target_manipulated_object][..., 2] > 0.10:
            success = True
        else:
            success = False
        return success

    def viz_actions(self, gt_action, pred_action):
        action_pred = gt_action
        action_demo = pred_action  # already shape (22,), assume it's a NumPy array or 1D tensor

        # If it's a tensor, convert to numpy
        if isinstance(action_demo, torch.Tensor):
            action_demo = action_demo.detach().cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(action_pred, label='Predicted Action', marker='o')
        plt.plot(action_demo, label='Demonstration Action', marker='x')
        plt.title('Predicted vs Demonstration Actions')
        plt.xlabel('Action Dimension')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
