from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.utils.math as math_utils
import torch

import numpy as np
import os
import trimesh

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
import random
from scipy.spatial.transform import Rotation


class SingleBCGeneration:

    def __init__(self, env, env_config, args_cli):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.device
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
        )
        self.raw_collected_data = self.collector_interface.raw_data["data"]
        self.device = env.device

        self.target_manipulated_object = env_config["params"][
            "target_manipulated_object"]

        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        elif self.args_cli.add_right_hand:
            self.hand_side = "right"

        self.use_joint_pose = True if "Play" in args_cli.task else False

        self.init_setting()
        self.filterout_target_objects()
        self.num_demos = self.args_cli.num_demos

    def filterout_target_objects(self):

        self.demo_obs_buffer = []
        self.demo_action_buffer = []
        for index in range(len(self.raw_collected_data)):
            demo_data = self.raw_collected_data[f"demo_{int(index)}"]
            manipulated_object_name = demo_data["obs"][
                "manipulate_object_name"][0][0].decode("utf-8")
            if manipulated_object_name == self.target_manipulated_object:

                obs_data = demo_data["obs"]
                action_data = np.array(demo_data["actions"])
                self.demo_obs_buffer.append(obs_data)
                self.demo_action_buffer.append(action_data)
        self.num_trajectories = len(self.demo_obs_buffer)

    def init_setting(self):
        self.delta_quat = torch.as_tensor(
            [self.env_config["params"]["delta_quat"]]).to(self.device)

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.pregrasp_offset = torch.as_tensor(
            self.env_config["params"]["pregrasp_offset"]).to(self.device)
        self.postgrasp_offset = torch.as_tensor(
            self.env_config["params"]["postgrasp_offset"]).to(self.device)
        self.arm_motion_env = ArmMotionPlannerEnv(self.env,
                                                  self.args_cli,
                                                  self.env_config,
                                                  collision_checker=False)

        self.num_data_index = 0

        self.init_hand_qpos = torch.as_tensor([0] * self.num_hand_joints).to(
            self.device).unsqueeze(0)

        self.init_robot_pose()
        self.load_raw_mesh()

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

    def load_raw_mesh(self):
        mesh_path = os.path.join(
            self.env_config["params"]["spawn_rigid_objects"]
            ["object_mesh_dir"], f"{self.target_manipulated_object}.obj")
        vertices = trimesh.load(mesh_path).vertices
        self.bbox_region = torch.as_tensor(
            np.concatenate([
                np.min(vertices, axis=0),
                np.max(vertices, axis=0)
            ]).reshape(-1, 3)).to(self.device).to(torch.float32)

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

    def init_robot_pose(self):
        self.init_env()
        self.env_ids = torch.arange(self.env.num_envs, device=self.device)
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
        self.env.scene[
            f"{self.hand_side}_hand"].data.reset_joint_pos = self.init_robot_qpos

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
        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()
        self.object_pose = self.env.scene[
            self.target_manipulated_object]._data.root_state_w[:, :7].clone()

    def interplate_finger_action(self, finger_pose, num_finger_action):
        init_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.get_dof_positions(
            )[:, -self.num_hand_joints:]

        finger_speed = (finger_pose - init_finger_pose) / num_finger_action
        arange = torch.arange(num_finger_action).to(self.device).unsqueeze(1)
        finger_mat = finger_speed.repeat_interleave(num_finger_action, 0)

        finger_action = finger_mat * arange + init_finger_pose

        return finger_action

    def pregrasp_pose(self):

        target_ee_quat = math_utils.quat_mul(self.delta_quat,
                                             self.init_ee_pose[:, 3:7])
        target_ee_pose = self.object_pose.clone()[:, :3]
        target_ee_pose[:, :3] = self.object_pose.clone(
        )[:, :3] + self.pregrasp_offset

        ee_pose, qpos = self.arm_motion_env.plan_motion(torch.cat(
            [target_ee_pose, target_ee_quat], dim=1),
                                                        apply_offset=False)
        if ee_pose is None:
            return None, None

        last_finger_action = torch.as_tensor(
            np.array(self.demo_actions[-1, -self.num_hand_joints:])).to(
                self.device).unsqueeze(0)
        target_finger_action = last_finger_action / 10 * (
            torch.rand(1).to(self.device) * 3 + 2)
        finger_action = self.interplate_finger_action(target_finger_action,
                                                      len(ee_pose))
        ee_pose = torch.cat([ee_pose, finger_action], dim=1)

        return ee_pose, qpos

    def postgrasp_pose(self):

        target_ee_quat = math_utils.quat_mul(self.delta_quat,
                                             self.init_ee_pose[:, 3:7])
        target_ee_pose = self.object_pose.clone()[:, :3]
        target_ee_pose[:, :2] = self.object_pose.clone()[
            ..., :2] + self.postgrasp_offset[..., :2]

        target_ee_pose[:, 2] = -self.bbox_region[0, 2] + self.postgrasp_offset[
            ..., 2]

        ee_pose, qpos = self.arm_motion_env.plan_motion(torch.cat(
            [target_ee_pose, target_ee_quat], dim=1),
                                                        apply_offset=False)
        if ee_pose is None:
            return None, None
        ee_pose = ee_pose[2:-2:2]

        last_finger_action = torch.as_tensor(
            np.array(self.demo_actions[-1, -self.num_hand_joints:])).to(
                self.device).unsqueeze(0)

        finger_action = self.interplate_finger_action(last_finger_action,
                                                      len(ee_pose))
        ee_pose = torch.cat([ee_pose, finger_action], dim=1)

        return ee_pose, qpos

    def lift_object(self):

        target_ee_quat = math_utils.quat_mul(self.delta_quat,
                                             self.init_ee_pose[:, 3:7])
        target_ee_pose = torch.as_tensor([[0.50, 0.0, 0.30]]).to(self.device)

        ee_pose, qpos = self.arm_motion_env.plan_motion(torch.cat(
            [target_ee_pose, target_ee_quat], dim=1),
                                                        apply_offset=False)
        if ee_pose is None:
            return None, None

        last_finger_action = torch.as_tensor(
            np.array(self.demo_actions[-1, -self.num_hand_joints:])).to(
                self.device).unsqueeze(0)

        ee_pose = torch.cat(
            [ee_pose,
             last_finger_action.repeat_interleave(len(ee_pose), 0)],
            dim=1)

        return ee_pose, qpos

    def init_data_buffer(self):
        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def proccess_actions(self, ee_pose, pre_actions):

        if "Rel" not in self.args_cli.task:
            return ee_pose
        else:
            cur_ee_pose = self.env.scene[
                f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :
                                                                    7].clone()
            delta_pos = ee_pose[:, :3].clone(
            ) - cur_ee_pose[:, :3] + self.env.scene[
                f"{self.hand_side}_hand"]._data.root_state_w[:, :3].clone()
            delta_quat = math_utils.quat_mul(
                ee_pose[:, 3:7].clone(),
                math_utils.quat_inv(cur_ee_pose[:, 3:7]))
            delta_euler = torch.cat(math_utils.euler_xyz_from_quat(delta_quat))

            rot_vec = torch.as_tensor(
                Rotation.from_euler("XYZ",
                                    delta_euler.cpu().numpy()).as_rotvec()).to(
                                        self.device).unsqueeze(0)

            # delta_rot = math_utils.quat_to_rot_action(delta_quat)

            delta_pose = torch.cat([delta_pos, rot_vec], dim=1)

            target_ee_pose = math_utils.apply_delta_pose(
                cur_ee_pose[:, :3], cur_ee_pose[:, 3:7], delta_pose)

            delta_finger_action = ee_pose[:, -self.num_hand_joints:].clone(
            ) - pre_actions[:, -self.num_hand_joints:].clone()

            whole_action = torch.cat([delta_pose, delta_finger_action], dim=1)

            # number of steps needed per dimension
            upper_division = torch.ceil(
                (whole_action /
                 (self.upper_bound * self.env.step_dt * 1)).reshape(-1))
            lower_division = torch.ceil(
                (whole_action /
                 (self.lower_bound * self.env.step_dt * 1)).reshape(-1))

            division = np.max([
                torch.max(upper_division).item(),
                torch.max(lower_division).item()
            ])  # (D,)

            clipped_division = max(1, min(100, division))
            # step per division
            step_action = whole_action.clone() / clipped_division  # (D,)

            # repeat steps to create a list of actions
            actions = step_action.repeat_interleave(int(clipped_division),
                                                    0)  # (N, D)

            actions[:,
                    -self.num_hand_joints:] = ee_pose[:, -self.
                                                      num_hand_joints:].clone(
                                                      ).reshape(-1)

        return actions

    def step_env(self, ee_pose, qpos):

        for index, pose in enumerate(ee_pose):

            actions = self.proccess_actions(
                pose.unsqueeze(0), ee_pose[max([index - 1, 1])].unsqueeze(0))
            for act in actions:

                truncated_actions = act.unsqueeze(0).clone()

                truncated_actions[:, -self.
                                  num_hand_joints:] -= self.pre_finger_action.clone(
                                  )

                obs, rewards, terminated, time_outs, extras = self.env.step(
                    act.unsqueeze(0).clone())

                for object_name in self.env.scene.rigid_objects.keys():

                    object_pose = self.env.scene[
                        object_name]._data.root_state_w[:, :7].clone()
                    obs["policy"][object_name] = object_pose

                self.obs_buffer.append(obs)
                self.actions_buffer.append(truncated_actions.clone())
                self.rewards_buffer.append(rewards)
                self.does_buffer.append(terminated)
                self.pre_finger_action = self.env.scene[
                    f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                             num_hand_joints:].clone(
                                                             )

        return act.unsqueeze(0).clone()

    def run(self):
        print("==================================")
        print("init demo")

        random_index = random.randint(0, self.num_trajectories - 1)
        self.demo_obs = self.demo_obs_buffer[random_index]
        self.demo_actions = self.demo_action_buffer[random_index]
        self.reset_env()
        self.init_data_buffer()

        pregrasp_ee_pose, pregrasp_qpos = self.pregrasp_pose()
        if pregrasp_ee_pose is None:
            return False
        self.step_env(pregrasp_ee_pose, pregrasp_qpos)

        postgrasp_ee_pose, postgrasp_qpos = self.postgrasp_pose()
        if postgrasp_ee_pose is None:
            return False
        self.step_env(postgrasp_ee_pose, postgrasp_qpos)

        lift_ee_pose, lift_qpos = self.lift_object()
        if lift_ee_pose is None:
            return False
        last_actions = self.step_env(lift_ee_pose, lift_qpos)

        for i in range(10):  # make sure the object is lifted
            if "Rel" not in self.args_cli.task:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    last_actions)
            else:

                last_actions[:, :-self.num_hand_joints] *= 0

                self.env.step(last_actions)

        if self.env.scene[self.target_manipulated_object]._data.root_state_w[
                0, 2] > 0.15:
            success = True
        else:
            success = False

        if self.args_cli.save_path:

            if success or self.args_cli.collect_all:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.does_buffer,
                )

        return success
