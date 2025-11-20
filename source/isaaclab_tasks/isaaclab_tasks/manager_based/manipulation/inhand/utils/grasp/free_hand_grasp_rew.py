from isaaclab_assets import *

import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.managers import EventTermCfg as EventTerm
import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.configuration_robot import config_robot_contact_sensor
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg


class FreeHandGraspRew:

    def __init__(
        self,
        env_cfg,
        rewards_cfg,
        scene_cfg,
    ):
        self.env_cfg = env_cfg
        self.rewards_cfg = rewards_cfg

        self.scene_cfg = scene_cfg
        self.fingers_name_list = [
            "palm_lower",
            "fingertip",
            "thumb_fingertip",
            "fingertip_2",
            "fingertip_3",
        ]

        if env_cfg["params"]["add_right_hand"]:
            self.add_right_hand = True
            self.hand_side = "right"
        if env_cfg["params"]["add_left_hand"]:
            self.add_left_hand = True
            self.hand_side = "left"
        self.target_manipulated_object_pose = torch.as_tensor(
            env_cfg["params"]["target_manipulated_object_pose"]).unsqueeze(0)
        self.object_name = env_cfg["params"]["target_manipulated_object"]
        self.init_height = env_cfg["params"]["RigidObject"][
            self.object_name]["pos"][2]

        contact_sensor_name = self.env_cfg["params"]["contact_sensor"][
            "spawn_contact_list"]

        self.sensor_name = []
        for name in contact_sensor_name:
            self.sensor_name.append(f"{self.hand_side}_{name}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.finger_reward_scale = torch.as_tensor([0.5, 1.0, 2, 1.5, 1.0]).to(
            self.device).unsqueeze(0)

        self.setup_reward()

    def setup_reward(self):

        grasp_rewards = RewTerm(func=self.grasp_rewards, weight=4.0)

        setattr(self.rewards_cfg, "grasp_rewards", grasp_rewards)

    def grasp_rewards(self, env):
        self.get_object_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        self.get_object_info(env)

        rewards_finger2object = self.finger2object_rewards(env)
        rewards03_contact = self.object2fingercontact_rewards(env)
        rewards04_lift = self.liftobject_rewards(env)

        return rewards04_lift + rewards_finger2object * 0.3 + rewards03_contact * 0.3

    def liftobject_rewards(self, env):
        lift_reward = torch.clip(self.object_pose[:, 2] - self.init_height,
                                 -0.1, 0.5)
        lift_reward = torch.clip(
            torch.clip(lift_reward / 0.5, -0.1, 1.0) * 100, -2, 100)
        self.lift_reward_scale = (self.object_pose[:, 2] - self.init_height
                                  >= 0.02).int()

        target_dist = torch.linalg.norm(self.object_target_dev, dim=1)
        # target_dist = torch.clip(1.0 / (0.02 + target_dist), 0.0, 15.0)
        target_dist_reward = torch.clip(1 - target_dist / 0.3, -0.1, 1.0) * 30
        x_penalty = torch.clip((abs(self.object_target_dev[:, 0])) / 0.20, 0,
                               1.5) * -5 * self.lift_reward_scale

        still_or_not = abs(self.target_in_object_angle) < 0.1

        scale_topple = self.lift_reward_scale * 20 + 1
        topple_reward = torch.clip(
            torch.clip(still_or_not.int() * 2 - 1, -1, 0.2) * scale_topple, -1,
            2)
        reward = lift_reward + self.lift_reward_scale + target_dist_reward * self.lift_reward_scale + x_penalty + topple_reward * 1.0

        return reward

    def object2fingercontact_rewards(self, env):
        self.finger_rewards_scale = (torch.sum(self.contact_or_not, dim=1)
                                     >= 2).int()

        return torch.sum(self.contact_or_not,
                         dim=1) * 3 + self.finger_rewards_scale * 2.0

    def finger2object_rewards(self, env):

        finger_dist = torch.clip(
            torch.linalg.norm(self.finger_object_dev, dim=2), 0.02, 0.8)

        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)

        reward = torch.sum(reward, dim=1) / len(self.fingers_name_list) * 3.0

        return reward

    def get_finger_info(self, env):
        self.finger_pose = []
        for name in self.fingers_name_list:
            finger = env.scene[f"{self.hand_side}_{name}"]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins
            self.finger_pose.append(finger_pose.unsqueeze(1))

        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        finger_object_pose = self.object_pose.clone().unsqueeze(
            1).repeat_interleave(len(self.fingers_name_list), dim=1)
        self.finger_object_dev = (finger_object_pose[..., :3] -
                                  self.finger_pose[..., :, :3])

    def get_object_info(self, env):
        self.object_pose = env.scene[
            self.object_name]._data.root_state_w[:, :7].clone()
        self.object_pose[:, :3] -= env.scene.env_origins
        target_pose = self.target_manipulated_object_pose.clone(
        ).repeat_interleave(env.num_envs, dim=0).to(env.device)
        self.object_target_dev = target_pose[:, :3].clone(
        ) - self.object_pose[:, :3].clone()

        delta_quat = math_utils.quat_mul(
            target_pose[:, 3:7].clone(),
            math_utils.quat_inv(self.object_pose[:, 3:7]))
        detla_euler = math_utils.euler_xyz_from_quat(delta_quat)[:2]
        delta_xy_rotation = torch.cat(
            [detla_euler[0].unsqueeze(1), detla_euler[1].unsqueeze(1)], dim=1)
        self.target_in_object_angle = torch.sum(delta_xy_rotation, dim=1)

    def get_contact_info(self, env):
        sensor_data = []
        for name in self.sensor_name[:4]:
            sensor = env.scene[f"{name}_contact"]

            force_data = torch.linalg.norm(sensor._data.force_matrix_w.reshape(
                env.num_envs, 3),
                                           dim=1).unsqueeze(1)
            sensor_data.append(force_data)

        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 4.0).int()
