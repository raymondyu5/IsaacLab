from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.utils.math as math_utils
import torch
from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import trimesh
from scripts.workflows.hand_manipulation.env.teleop_env.generative_demo import DemoGen

from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv


class DexDataGen:

    def __init__(self, args_cli, env_config, env):

        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
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
        self.num_trajectories = len(self.raw_collected_data)

        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        elif self.args_cli.add_right_hand:
            self.hand_side = "right"

        self.use_joint_pose = True if "Play" in args_cli.task else False

        self.init_setting()
        self.filterout_target_objects()
        self.num_demos = self.args_cli.num_demos

    def filterout_target_objects(self):

        self.action_buffer = {}

        self.obs_buffer = []
        self.action_buffer = []
        for index in range(self.num_trajectories):
            demo_data = self.raw_collected_data[f"demo_{int(index)}"]
            manipulated_object_name = demo_data["obs"][
                "manipulate_object_name"][0][0].decode("utf-8")
            if manipulated_object_name == self.target_manipulated_object:

                obs_data = demo_data["obs"]
                action_data = np.array(demo_data["actions"])
                self.obs_buffer.append(obs_data)
                self.action_buffer.append(action_data)

    def init_setting(self):
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_config,
            collision_checker=self.args_cli.collision_checker)

        self.num_data_index = 0

        self.init_hand_qpos = torch.as_tensor([0] * self.num_hand_joints).to(
            self.device).unsqueeze(0)

        self.load_raw_mesh()
        self.init_robot_pose()
        self.demo_gen_env = DemoGen(
            self.env,
            self.args_cli,
            self.env_config,
            self.arm_motion_env,
            self.collector_interface,
            init_robot_pose=self.init_robot_qpos if self.use_joint_pose
            or "Rel" in self.args_cli.task else self.init_ee_pose,
        )

    def init_robot_pose(self):
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

    def sample_demo(self):

        self.demo_obs = self.obs_buffer[self.num_data_index]
        self.demo_actions = self.action_buffer[self.num_data_index]

        init_object_pose = torch.as_tensor(
            self.demo_obs[self.target_manipulated_object][0]).unsqueeze(0).to(
                self.device)

        init_euler_angles = math_utils.euler_xyz_from_quat(
            init_object_pose[:, 3:7])

        if abs(init_euler_angles[0]) > 0.1 or abs(init_euler_angles[1]) > 0.1:
            return False

        caliberate_quat = math_utils.quat_from_euler_xyz(
            init_euler_angles[0], init_euler_angles[1], init_euler_angles[2])

        transformed_bbox_region = math_utils.transform_points(
            self.bbox_region.unsqueeze(0), init_object_pose[:, :3] * 0.0,
            caliberate_quat)
        min_height = transformed_bbox_region[:, :, 2]
        # init_object_pose[:, 2] = -min_height
        self.caliberate_pose = torch.cat(
            [init_object_pose[:, :3], caliberate_quat], dim=1)

        self.init_robo_qpos = torch.as_tensor(self.demo_actions[0]).to(
            self.device).unsqueeze(0)

        self.pregrasp_pose = torch.as_tensor(
            self.demo_obs["pregrasp_pose_ee"][-1]).unsqueeze(0).to(self.device)
        self.postgrasp_pose = torch.as_tensor(
            self.demo_obs["postgrasp_pose_ee"][-1]).unsqueeze(0).to(
                self.device)
        return True

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
                    self.caliberate_pose)

    def generate_demo(self):
        success = self.demo_gen_env.step(
            self.caliberate_pose, self.pregrasp_pose, self.postgrasp_pose,
            self.bbox_region,
            torch.as_tensor(self.demo_actions).to(
                self.device)[31:, -self.num_hand_joints:])
        return success

    def run(self):
        print("==================================")
        print("init demo")
        inital = self.sample_demo()
        self.num_data_index += 1
        if not inital:
            return None
        print("==================================")
        print(f"reset env for demo {self.num_data_index}")

        self.reset_env()
        # action_buffer = torch.as_tensor(self.demo_actions).to(self.device)
        success_count = 0
        continue_fail = 0
        total_try = 0

        while success_count < self.num_demos:
            if continue_fail > 10:
                break

            success = self.generate_demo()
            if success:
                success_count += 1
                continue_fail = 0
            else:
                continue_fail += 1
            total_try += 1
