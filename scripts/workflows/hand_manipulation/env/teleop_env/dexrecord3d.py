from scripts.workflows.hand_manipulation.utils.isaacsim_hand_object_viewer import RobotHandDatasetIsaacviewer
import torch
import numpy as np

import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R


class DexRRecord3detargeting:

    def __init__(self, args_cli, env_config, env, motion_control):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device

        self.mapping_env = RobotHandDatasetIsaacviewer(
            args_cli.robot_name,
            env.scene["robot"],
            data_root=args_cli.dexycb_filename)
        if args_cli.free_hand:
            self.control_function = self.run_free_hand
        else:
            self.control_function = self.run_robot

        self.mean_robot = torch.tensor([0.0, -0.0, 0.0, 1.0, 0.0, 0.0,
                                        0.0]).to(self.device)

        self.motion_control = motion_control

    def run(self):
        self.env.reset()

        while True:
            success, motion_data = self.motion_control.step()
            if success:
                joint_pose = motion_data["joint"]

                indices = self.mapping_env.retargeting.optimizer.target_link_human_indices
                ref_value = joint_pose[indices, :]
                qpos = self.mapping_env.retargeting.retarget(ref_value)[
                    self.mapping_env.retarget2isaac]

                actions = torch.zeros(self.env.action_space.shape,
                                      device=self.device)

                actions[:, -16:] = torch.tensor(qpos[-16:], device=self.device)
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    actions)

    def preprocess_data(self, qpos, ycb_object_pose, robot_ee_pose):

        pose_distance = {}
        robot_ee_pose = torch.cat(robot_ee_pose, dim=0).reshape(-1, 7)
        for object_name in ycb_object_pose.keys():
            object_pose = torch.cat(ycb_object_pose[object_name],
                                    dim=0).reshape(-1, 7)

            object_ee_pose = torch.norm(robot_ee_pose[:, :3] -
                                        object_pose[:, :3],
                                        dim=1)
            pose_distance[object_name] = [
                torch.min(object_ee_pose).item(),
                torch.argmin(object_ee_pose).item()
            ]

    def run_free_hand(self, qpos, ycb_object_pose, robot_ee_pose):
        # self.preprocess_data(qpos, ycb_object_pose, robot_ee_pose)

        for i in range(10):
            for ycb_name in self.env.scene.rigid_objects.keys():
                if ycb_name not in ycb_object_pose.keys():
                    continue

                ycb_pose = ycb_object_pose[ycb_name][0].to(self.device)

                ycb_pose[:3] -= self.mean_robot[:3]
                ycb_pose = ycb_pose.unsqueeze(0)

                ycb_pose[:, 3:7] = math_utils.quat_mul(
                    self.mean_robot[3:7].unsqueeze(0), ycb_pose[:, 3:7])
                self.env.scene.rigid_objects[ycb_name].write_root_pose_to_sim(
                    ycb_pose.unsqueeze(0),
                    torch.arange(self.env.num_envs).to(self.device))
            obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.as_tensor(self.env.action_space.sample() * 0.0).to(
                    self.env.device))

        for i in range(qpos.shape[0]):
            # compute zero actions
            actions = torch.zeros(self.env.action_space.shape,
                                  device=self.device)

            actions[:, -16:] = torch.tensor(qpos[i][-16:], device=self.device)

            # control the hand pose
            raw_root_state = self.env.scene["robot"]._data.root_state_w
            raw_root_state[:, :3] = qpos[i][:3].to(self.device)
            raw_root_state[:, 2] += 0.01

            raw_root_state[:, 3:7] = robot_ee_pose[i][3:7].to(self.device)

            raw_root_state[:, :3] -= self.mean_robot[:3]

            raw_root_state[:, 3:7] = math_utils.quat_mul(
                self.mean_robot[3:7].unsqueeze(0), raw_root_state[:, 3:7])
            self.env.scene["robot"].write_root_pose_to_sim(
                raw_root_state[:, :7],
                torch.arange(self.env.num_envs).to(self.device))

            # # control the finger
            if i < 25:
                actions *= 0.0

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

            for ycb_name in self.env.scene.rigid_objects.keys():
                if ycb_name not in ycb_object_pose.keys():
                    continue
                ycb_pose = ycb_object_pose[ycb_name][i].to(self.device)
                ycb_pose[:3] -= self.mean_robot[:3]

                ycb_pose = ycb_pose.unsqueeze(0)

                ycb_pose[:, 3:7] = math_utils.quat_mul(
                    self.mean_robot[3:7].unsqueeze(0), ycb_pose[:, 3:7])
                self.env.scene.rigid_objects[ycb_name].write_root_pose_to_sim(
                    ycb_pose.unsqueeze(0),
                    torch.arange(self.env.num_envs).to(self.device))
        self.env.reset()
