import random
from isaaclab.managers import SceneEntityCfg
import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp

from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles

import numpy as np


class ReplayAugWrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=4,
        skip_steps=1,
    ):
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.begin_index = begin_index

        self.skip_steps = skip_steps
        self.task = ("place" if "Place" in self.args_cli.task else
                     "open" if "Open" in self.args_cli.task else "grasp")

        if args_cli.add_left_hand:
            self.hand_side = "left"
        elif args_cli.add_right_hand:
            self.hand_side = "right"
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_cfg,
        )

    def init_rigid_object_setting(self):

        self.rigid_object_setting = self.env_cfg["params"].get(
            "RigidObject", {})

        self.rigid_object_list = list(self.env_cfg["params"].get(
            "RigidObject", {}).keys())
        if self.task == "place":
            self.pick_object_list = self.env_cfg["params"].get(
                "pick_object_names", None)
            self.place_object_list = self.env_cfg["params"].get(
                "place_object_names", None)
        else:
            self.pick_object_list = self.rigid_object_list
            self.place_object_list = []
            self.place_object_name = "None"

        self.pose_range = self.env_cfg["params"]["default_root_state"].get(
            "pose_range", {})
        self.root_state = torch.as_tensor(self.rigid_object_setting[
            self.rigid_object_list[0]]["pos"]).unsqueeze(0).to(
                self.device).repeat_interleave(self.num_envs, dim=0)

        root_quat = obtain_target_quat_from_multi_angles(
            self.rigid_object_setting[self.rigid_object_list[0]]["rot"]
            ["axis"], self.rigid_object_setting[
                self.rigid_object_list[0]]["rot"]["angles"])

        root_quat = torch.as_tensor(root_quat).unsqueeze(0).to(
            self.device).repeat_interleave(self.num_envs, dim=0)
        self.root_state = torch.cat([self.root_state, root_quat],
                                    dim=1).to(self.device)

    def reset_rigid_objects(
        self,
        demo_obs,
        demo_action,
    ):

        for rigid_object in list(self.env.scene.rigid_objects.keys()):

            init_rigid_object_pose = demo_obs.get(f"{rigid_object}_pose", None)
            if init_rigid_object_pose is not None:
                reset_target_pose = torch.as_tensor(
                    init_rigid_object_pose[self.begin_index]).unsqueeze(0).to(
                        self.device).repeat(self.num_envs, 1)
                reset_target_pose[..., :3] += self.env.scene.env_origins
                self.env.scene[rigid_object].write_root_pose_to_sim(
                    reset_target_pose, self.env_ids)
                if reset_target_pose[0, 2] > -0.10:

                    if rigid_object in self.pick_object_list:
                        self.pick_object_name = rigid_object
                        self.demo_pick_object_state = reset_target_pose.clone()
                        self.pick_object_traj = init_rigid_object_pose[
                            self.begin_index:]
                    if rigid_object in self.place_object_list:
                        self.place_object_name = rigid_object
                        self.demo_place_object_state = reset_target_pose.clone(
                        )
                        self.place_object_traj = init_rigid_object_pose[
                            self.begin_index:]

        if self.env.num_envs > 1:
            return self.random_multi_place_rigid(demo_obs, demo_action)
        else:
            return self.step_zero_action(demo_obs, demo_action)

    def reset_robot_joints(self,
                           demo_obs=None,
                           hand_side="right",
                           init_joint_pose=None):
        if demo_obs is not None:

            init_joint_pose = demo_obs.get(f"{hand_side}_hand_joint_pos", None)
            if init_joint_pose is not None:
                init_joint_pose = torch.as_tensor(
                    init_joint_pose[self.begin_index]).unsqueeze(0).to(
                        self.device).repeat_interleave(self.num_envs, dim=0)

        if init_joint_pose is not None:

            self.env.scene[
                f"{hand_side}_hand"].root_physx_view.set_dof_positions(
                    init_joint_pose, indices=self.env_ids)

    def step_zero_action(self, demo_obs, demo_action):

        for i in range(10):
            if self.use_delta_pose:
                self.reset_robot_joints(demo_obs, "right")
                self.reset_robot_joints(demo_obs, "left")
                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device))
            else:

                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    demo_action[0].to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.num_envs, dim=0))
        return new_obs, demo_action

    def reset_demo_env(self, demo_obs, demo_action):

        new_obs, demo_action = self.reset_rigid_objects(demo_obs, demo_action)

        return new_obs, demo_action

    def random_multi_place_rigid(self, demo_obs, demo_action):

        from isaaclab_tasks.manager_based.manipulation.inhand.utils.place.config_cluster_rigids import randomize_object_pose

        reset_height = [
            self.root_state[..., 2] * 0 + 0.15, self.root_state[..., 2] * 0.0
        ]

        randomize_object_pose(env=self.env,
                              env_ids=self.env_ids,
                              asset_cfgs=[
                                  SceneEntityCfg(self.pick_object_name),
                                  SceneEntityCfg(self.place_object_name)
                              ],
                              min_separation=0.30,
                              pose_range=self.pose_range,
                              reset_height=reset_height,
                              max_sample_tries=100)
        last_obs, _ = self.step_zero_action(demo_obs, demo_action)
        self.get_transformation(demo_action)
        aug_action = self.construct_trajecory()
        return last_obs, aug_action

    def construct_arm_motion(self, demo_lifted_ee_pose, lifted_time,
                             delta_pick_pose):

        demo_lifted_ee_transformation = math_utils.assemble_transformation_matrix(
            demo_lifted_ee_pose[:, :3], demo_lifted_ee_pose[:, 3:7])

        delta_pick_pose = delta_pick_pose.unsqueeze(1).repeat_interleave(
            lifted_time, dim=1).to(torch.float32).reshape(-1, 4, 4)
        pick_lifted_ee_transformation = delta_pick_pose @ demo_lifted_ee_transformation

        pick_lifted_ee_pose = torch.cat(
            math_utils.disassemble_transformation_matrix(
                pick_lifted_ee_transformation),
            dim=1).reshape(self.num_envs, lifted_time, 7)
        return pick_lifted_ee_pose

    def construct_trajecory(self):

        if self.task == "place":
            lifted_time = np.where((self.pick_object_traj[:, 2] -
                                    self.pick_object_traj[0, 2]) > 0.05)[0][0]

            demo_lifted_ee_pose = self.demo_action[:
                                                   lifted_time][:, :7].unsqueeze(
                                                       0).repeat_interleave(
                                                           self.num_envs,
                                                           dim=0).to(
                                                               torch.float32
                                                           ).reshape(-1, 7)

            pick_lifted_ee_pose = self.construct_arm_motion(
                demo_lifted_ee_pose, lifted_time, self.delta_pick_pose)
            pick_finger_pose = self.demo_action[:lifted_time].unsqueeze(
                0).repeat_interleave(self.num_envs, dim=0).to(torch.float32)

            ### placement

            demo_place_ee_pose = self.demo_action[lifted_time:, :7].unsqueeze(
                0).repeat_interleave(self.num_envs,
                                     dim=0).to(torch.float32).reshape(-1, 7)

            place_ee_pose = self.construct_arm_motion(
                demo_place_ee_pose,
                len(self.demo_action) - lifted_time, self.delta_place_pose)

            init_ik_place_arm = self.arm_motion_env.ik_plan_motion(
                pick_lifted_ee_pose[:, -1])[:, 0]
            plan_ee_pose = place_ee_pose[:, 0].reshape(-1, 7)

            placement_motion = []

            for i in range(self.env.num_envs):

                planed_traj, _ = self.arm_motion_env.plan_motion(
                    plan_ee_pose[i].unsqueeze(0),
                    apply_offset=False,
                    arm_qpos=init_ik_place_arm[i].unsqueeze(0))
                if planed_traj is not None:
                    placement_motion.append(planed_traj[None])
                else:

                    placement_motion.append(
                        pick_lifted_ee_pose[i,
                                            -1].unsqueeze(0).repeat_interleave(
                                                31, dim=0)[None])

            placement_motion = torch.cat(placement_motion, dim=0)
            lift_traj = torch.cat(
                [pick_lifted_ee_pose, pick_finger_pose[:, :, 7:]],
                dim=2).permute(1, 0, 2)
            place_ee_pose = torch.cat([placement_motion, place_ee_pose[:, 1:]],
                                      dim=1)
            planned_finger_pose = self.demo_action[lifted_time, 7:].unsqueeze(
                0).repeat_interleave(self.num_envs, dim=0).to(
                    torch.float32).unsqueeze(1).repeat_interleave(
                        placement_motion.shape[1], dim=1)

            place_finger_pose = self.demo_action[
                lifted_time:, ...,
                7:].unsqueeze(0).repeat_interleave(self.num_envs,
                                                   dim=0).to(torch.float32)
            combined_place_finger_pose = torch.cat(
                [planned_finger_pose, place_finger_pose[:, 1:]], dim=1)
            place_traj = torch.cat([place_ee_pose, combined_place_finger_pose],
                                   dim=2).permute(1, 0, 2)

            return torch.cat([lift_traj, place_traj], dim=0)

    def get_transformation(self, demo_action):

        cur_pick_state = self.env.scene[
            self.pick_object_name]._data.root_state_w[:, :7].clone()
        cur_pick_state[:, 3:7] = self.demo_pick_object_state[:, 3:7]

        demo_pick_transformation = math_utils.assemble_transformation_matrix(
            self.demo_pick_object_state[:, :3],
            self.demo_pick_object_state[:, 3:7],
        )
        cur_pick_transformation = math_utils.assemble_transformation_matrix(
            cur_pick_state[:, :3],
            cur_pick_state[:, 3:7],
        )

        self.delta_pick_pose = cur_pick_transformation @ demo_pick_transformation.inverse(
        )

        if self.task == "place":
            cur_place_state = self.env.scene[
                self.place_object_name]._data.root_state_w[:, :7]
            cur_place_state[:, 3:7] = self.demo_place_object_state[:, 3:7]

            demo_place_transformation = math_utils.assemble_transformation_matrix(
                self.demo_place_object_state[:, :3],
                self.demo_place_object_state[:, 3:7],
            )
            cur_place_transformation = math_utils.assemble_transformation_matrix(
                cur_place_state[:, :3],
                cur_place_state[:, 3:7],
            )
            self.delta_place_pose = cur_place_transformation @ demo_place_transformation.inverse(
            )

        self.demo_action = demo_action

        init_ee_pose = demo_action[0, :7].unsqueeze(0).to(
            self.device).repeat_interleave(self.num_envs,
                                           dim=0).to(torch.float32)

        init_ee_transformation = math_utils.assemble_transformation_matrix(
            init_ee_pose[:, :3], init_ee_pose[:, 3:7])

        reset_ee_pose = torch.cat(math_utils.disassemble_transformation_matrix(
            self.delta_pick_pose @ init_ee_transformation),
                                  dim=1)
        reset_ee_pose = torch.cat(
            [reset_ee_pose,
             torch.zeros((self.num_envs, 16)).to(self.device)],
            dim=1)
        for i in range(20):
            last_obs, rewards, terminated, time_outs, extras = self.env.step(
                reset_ee_pose)

        if self.task == "place":
            self.init_first_place_flag = torch.zeros(self.num_envs).to(
                self.device).bool()

        return last_obs, demo_action

    # def aug_action(self, action):

    #     cur_pick_state = self.env.scene[
    #         self.pick_object_name]._data.root_state_w[:, :7].clone()
    #     self.lifted_object_flag |= (cur_pick_state[:, 2] -
    #                                 self.init_pick_state[:, 2]) > 0.10

    #     ee_pose = math_utils.assemble_transformation_matrix(
    #         action[:, :3], action[:, 3:7])

    #     if self.task == "place":
    #         target_ee_pose = action[:, :7].clone()
    #         if ((~self.lifted_object_flag).sum()) > 0:

    #             target_ee_pose[~self.lifted_object_flag] = torch.cat(
    #                 math_utils.disassemble_transformation_matrix(
    #                     self.delta_pick_pose[~self.lifted_object_flag]
    #                     @ ee_pose[~self.lifted_object_flag]),
    #                 dim=1)
    #         elif (self.lifted_object_flag).sum() > 0:

    #             target_ee_pose[self.lifted_object_flag] = torch.cat(
    #                 math_utils.disassemble_transformation_matrix(
    #                     self.delta_place_pose[self.lifted_object_flag]
    #                     @ ee_pose[self.lifted_object_flag]),
    #                 dim=1)
    #             self.mp_plan(target_ee_pose, ~self.init_first_place_flag)

    #             self.init_first_place_flag |= self.lifted_object_flag

    #     return torch.cat([target_ee_pose, action[:, 7:]], dim=1)
