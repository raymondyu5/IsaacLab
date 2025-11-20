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

from isaaclab.managers import SceneEntityCfg


class BimanualFrankaStackRew:

    def __init__(
        self,
        env_cfg,
        rewards_cfg,
        scene_cfg,
        events_cfg=None,
    ):
        num_envs = env_cfg["params"].get("num_envs", scene_cfg.num_envs)

        if env_cfg["params"]["add_left_hand"]:
            self.init_env(env_cfg, scene_cfg, rewards_cfg, "left", num_envs,
                          events_cfg)
        if env_cfg["params"]["add_right_hand"]:
            self.init_env(env_cfg, scene_cfg, rewards_cfg, "right", num_envs,
                          events_cfg)

    def init_env(self, env_cfg, scene_cfg, rewards_cfg, hand_side, num_envs,
                 events_cfg):

        init_height = torch.zeros((num_envs)).to("cuda")
        env_ids = torch.arange(num_envs).to("cuda")

        for index, object_name in enumerate(
            (env_cfg["params"]["multi_cluster_rigid"]
             [f"{hand_side}_hand_object"]["objects_list"])):
            num_object = len(env_cfg["params"]["multi_cluster_rigid"]
                             [f"{hand_side}_hand_object"]["objects_list"])
            mask = (env_ids % num_object) == index

            object_height = env_cfg["params"]["RigidObject"][object_name][
                "pos"][2]
            init_height[mask] = object_height

        SingleHandGraspRew(env_cfg, scene_cfg, rewards_cfg,
                           f"{hand_side}_hand_object",
                           f"{hand_side}_hand_place_object", hand_side,
                           init_height, events_cfg)


class SingleHandGraspRew:

    def __init__(self, env_cfg, scene_cfg, rewards_cfg, object_name,
                 placement_name, hand_side, init_height, events_cfg):
        self.env_cfg = env_cfg

        self.scene_cfg = scene_cfg
        self.hand_side = hand_side
        self.object_name = object_name

        self.init_height = init_height
        self.placement_name = placement_name

        self.reset_init_height = init_height.clone() + 0.03
        self.rewards_cfg = rewards_cfg
        self.events_cfg = events_cfg

        self.reward_weights = env_cfg["params"]["reward_weights"][hand_side]

        self.fingers_name_list = [
            "palm_lower",
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        self.target_robot_pose = torch.as_tensor(
            [self.env_cfg["params"]["init_ee_pose"]])
        contact_sensor_name = self.env_cfg["params"]["contact_sensor"][
            "spawn_contact_list"]

        self.sensor_name = []
        for name in contact_sensor_name:
            self.sensor_name.append(f"{self.hand_side}_{name}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.finger_reward_scale = torch.as_tensor([0.5, 1.0, 2, 1.5, 1.0]).to(
            self.device).unsqueeze(0)
        self.init_robot_pose = torch.as_tensor([
            self.env_cfg["params"][f"{hand_side}_robot_pose"]
        ]).to(self.device)

        self.setup_addtional()

        self.setup_reward()
        self.target_ee_pose = torch.as_tensor(
            [[-0.6537, 0.6537, -0.2693, -0.2693]])

    def setup_addtional(self):

        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}" + f"/{self.hand_side}_hand/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}" + "/Table"],
            debug_vis=False)
        setattr(self.scene_cfg, f"{self.hand_side}_panda_link6_contact",
                contact_sensor)

        # contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}" +
        #     f"/{self.hand_side}_hand/{self.hand_side}_hand/palm_lower",
        #     update_period=0.0,
        #     history_length=3,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}" + "/Table"],
        #     debug_vis=False)
        # setattr(self.scene_cfg, f"{self.hand_side}_palm_lower_contact",
        #         contact_sensor)

    def get_placement_object_pose(self, env):

        return self.placement_object_pose

    def setup_reward(self):

        # reset_func = EventTerm(func=self.reset, mode="reset", params={})
        # setattr(self.events_cfg,
        #         f"reset_{self.hand_side}_object_target_pose_in_rewards",
        #         reset_func)

        grasp_rewards = RewTerm(
            func=self.grasp_rewards,
            weight=self.env_cfg["params"].get("reward_scale", 10.0),
        )

        setattr(self.rewards_cfg, f"{self.hand_side}_rewards", grasp_rewards)

    def penalty_palm_orientation(self, env):
        palm_quat = env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w[:, 3:7].clone()

        delta_palm_quat = math_utils.quat_mul(
            self.target_ee_pose.repeat_interleave(env.num_envs,
                                                  dim=0).to(env.device),
            math_utils.quat_inv(palm_quat))

        detla_palm_euler = math_utils.euler_xyz_from_quat(delta_palm_quat)

        # palm_reward_raw = torch.clip(-abs(detla_palm_euler[2]) / 0.5, -1, -0)

        delta_xy_rotation = torch.cat([
            detla_palm_euler[0].unsqueeze(1), detla_palm_euler[1].unsqueeze(1),
            detla_palm_euler[2].unsqueeze(1)
        ],
                                      dim=1)

        target_in_angle = torch.sum(abs(delta_xy_rotation), dim=1)
        palm_reward = torch.clip(1 - abs(target_in_angle) / 0.40, -1,
                                 1) * self.lift_or_not * 3

        return palm_reward

    def target_ee_orientation_penalty(self, env):

        delta_quat = math_utils.quat_mul(
            env.scene[f"{self.hand_side}_hand_object"]._data.root_state_w[:,
                                                                          3:7],
            math_utils.quat_inv(env.scene[f"{self.hand_side}_panda_link7"].
                                _data.root_state_w[:, 3:7]))

        angles = math_utils.euler_xyz_from_quat(delta_quat)[1]  # radians

        abs_angles = angles.abs()

        clipped_angle = torch.where(abs_angles > math.pi / 2,
                                    (math.pi - abs_angles) * angles.sign(),
                                    angles)

        return torch.clip(1 - abs(clipped_angle) / 0.40, -1,
                          0) * self.lift_or_not * 3

    def grasp_rewards(self, env):
        self.reset_init_height = self.init_height.clone(
        ) + env.scene["table_block"]._data.root_state_w[:, 2].clone()
        self.get_object_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        rewards_finger2object = self.finger2object_rewards(env)
        rewards03_contact = self.object2fingercontact_rewards(env)
        rewards04_lift = self.liftobject_rewards(env)
        rewards04_link6 = self.penalty_contact(env)
        # reward05_ori_penalty = self.target_ee_orientation_penalty(env)
        # reward05_ori_penalty = self.penalty_palm_orientation(env)

        joint_vel_penalty = mdp.joint_vel_l2(
            env,
            SceneEntityCfg(self.hand_side + "_hand"),
        )
        joint_limit_penalty = mdp.joint_pos_limits(
            env,
            (0.9, 0.9),
            SceneEntityCfg(self.hand_side + "_hand"),
        )
        action_rate_penalty = mdp.action_rate_l2(
            env,
            dim=6,
        )
        rewards03_contact_scaled = rewards03_contact * (
            self.lift_or_not * 0.6 + (1 - self.lift_or_not) * 0.3)

        grasp_rewards = rewards_finger2object * 0.30 + rewards03_contact_scaled + rewards04_lift * 1.0 + rewards04_link6 - joint_vel_penalty * 1.0e-3 - joint_limit_penalty * 6.0e-1 - action_rate_penalty * 5e-3  #+ reward05_ori_penalty * 0.1

        place_rewards = torch.zeros_like(grasp_rewards).to(env.device).to(
            torch.float32)
        if env.unwrapped.episode_length_buf[0] < 25:
            self.grasp_sucess = torch.zeros(env.num_envs).to(env.device).bool()
            self.target_height = self.object_pose[:, 2].clone() + 0.05
        else:

            self.grasp_sucess = torch.linalg.norm(
                self.object_target_dev[:, :2], dim=1) < 0.03
            grasp_rewards[self.grasp_sucess] = 40

            if self.grasp_sucess.sum() > 0:

                place_success = self.grasp_sucess & (
                    (self.object_pose[:, 2] - self.target_height) < 0.05) & (
                        (self.object_pose[:, 2] - self.target_height) > -0.0)

                if place_success.sum() > 0:
                    place_rewards[place_success] = 40

                    # ee_pose = env.scene[
                    #     f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :7].clone(
                    #     )[place_success]
                    # ee_pose[:, :3] -= env.scene.env_origins[place_success]
                    # target_ee_pose = self.target_robot_pose.repeat_interleave(
                    #     env.num_envs, dim=0).to(env.device)[place_success]
                    # delta_ee_quat = math_utils.quat_mul(
                    #     target_ee_pose[:, 3:7].clone(),
                    #     math_utils.quat_inv(ee_pose[:, 3:7].clone()))
                    # detla_ee_euler = math_utils.euler_xyz_from_quat(
                    #     delta_ee_quat)
                    # delta_xy_rotation = torch.cat([
                    #     detla_ee_euler[0].unsqueeze(1),
                    #     detla_ee_euler[1].unsqueeze(1),
                    #     detla_ee_euler[2].unsqueeze(1)
                    # ],
                    #                               dim=1)
                    # target_in_angle = torch.sum(abs(delta_xy_rotation), dim=1)
                    # place_rewards[place_success] += torch.clip(
                    #     -abs(target_in_angle) / 0.80, -1, 0) * 15

                    # place_rewards[place_success] += torch.clip(
                    #     torch.linalg.norm(
                    #         ee_pose[:, :3] - target_ee_pose[:, :3], dim=1) /
                    #     0.15, 0.0, 1.0) * -15

        return grasp_rewards + place_rewards

    def penalty_contact(self, env):

        sensor = env.scene[f"{self.hand_side}_panda_link6_contact"]
        force_data = torch.linalg.norm(sensor._data.net_forces_w.reshape(
            env.num_envs, 3),
                                       dim=1).unsqueeze(1)
        penalty_contact = torch.clip(
            (force_data > 4).int() * 2 - 1, 0.0, 1.0) * -0.5

        # palm_sensor = env.scene[f"{self.hand_side}_palm_lower_contact"]
        # palm_force_data = torch.linalg.norm(
        #     palm_sensor._data.net_forces_w.reshape(env.num_envs,
        #                                            3), dim=1).unsqueeze(1)
        # penalty_contact += torch.clip(
        #     (palm_force_data > 2).int() * 2 - 1, 0.0, 1.0) * -0.5

        return penalty_contact.reshape(-1)

    def liftobject_rewards(self, env):
        lift_reward = torch.clip(
            self.object_pose[:, 2] - self.reset_init_height, -0.1, 0.2) / 0.2
        # if torch.max(lift_reward) < 0.2:
        #     import pdb
        #     pdb.set_trace()

        lift_reward = torch.where(
            lift_reward > 1,
            (1 - lift_reward) * 6,  #penalty for lifting too high
            lift_reward)
        scaled_reward = torch.where(lift_reward < 0, lift_reward * 3,
                                    lift_reward * 20)
        lift_reward = torch.clip(scaled_reward, -5, 20)

        self.lift_reward_scale = (self.object_pose[:, 2] -
                                  self.reset_init_height >= 0.02).int()

        target_dist = torch.linalg.norm(self.object_target_dev, dim=1)
        # target_dist = torch.clip(1.0 / (0.02 + target_dist), 0.0, 15.0)
        target_dist_reward = torch.clip(1 - target_dist / 0.3, 0.0, 1.0) * 20

        x_penalty = torch.clip((abs(self.object_target_dev[:, 0])) / 0.30, 0,
                               1.5) * -1 * self.lift_reward_scale

        self.lift_or_not = ((self.object_pose[:, 2] - self.reset_init_height)
                            >= 0.05).int()

        topple_reward = torch.clip(1 - abs(self.target_in_object_angle) / 0.10,
                                   -1, 1) * self.lift_or_not * 6

        reward = lift_reward + self.lift_reward_scale + target_dist_reward * self.lift_reward_scale + x_penalty + topple_reward * 1.0  #+ thumb_lift_reward

        return reward

    def object2fingercontact_rewards(self, env):
        self.finger_rewards_scale = (torch.sum(self.contact_or_not, dim=1)
                                     >= 2).int()

        reward = self.contact_or_not.to(torch.float32)

        return torch.sum(reward, dim=1) * 1 + self.finger_rewards_scale * 1.0

    def finger2object_rewards(self, env):

        finger_dist = torch.clip(
            torch.linalg.norm(self.finger_object_dev, dim=2), 0.02, 0.8)

        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)
        # reward[:, :-2] *= 2.4 / 3
        # reward[:, -2:] *= 1.3

        reward = torch.sum(
            reward,
            dim=1) / reward.shape[1]  #len(self.fingers_name_list) * 1.0

        return reward

    def get_finger_info(self, env):
        self.finger_pose = []
        # for name in self.fingers_name_list:
        for name in [
                "palm_lower", "thumb_fingertip", "fingertip", "fingertip_2",
                "fingertip_3"
        ]:
            finger = env.scene[f"{self.hand_side}_{name}"]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins

            self.finger_pose.append(finger_pose.unsqueeze(1))

        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        finger_object_pose = self.object_pose.clone().unsqueeze(
            1).repeat_interleave(self.finger_pose.shape[1], dim=1)

        self.finger_object_dev = (finger_object_pose[..., :3] -
                                  self.finger_pose[..., :, :3])

    def get_state_info(self, env, object_name):

        object_pose = env.scene[object_name]._data.root_state_w[:, :7].clone()

        object_pose[:, :3] -= env.scene.env_origins
        object_pose[:, :3] -= self.init_robot_pose[..., :3]

        return object_pose

    def get_object_info(self, env):
        self.object_pose = self.get_state_info(env, self.object_name)
        # self.placement_object_pose = self.get_state_info(
        #     env, self.placement_name)

        try:

            self.placement_object_pose = env.scene[
                self.object_name]._data.target_state[:, :7].clone()
            self.placement_object_pose[:, :3] -= env.scene.env_origins

        except:
            self.placement_object_pose = self.get_state_info(
                env, self.placement_name)
        target_pose = self.placement_object_pose.clone().to(env.device)
        self.object_target_dev = target_pose[:, :3].clone(
        ) - self.object_pose[:, :3].clone()

        ### delta rotation for object

        delta_quat = math_utils.quat_mul(
            target_pose[:, 3:7].clone(),
            math_utils.quat_inv(self.object_pose[:, 3:7]))

        detla_euler = math_utils.euler_xyz_from_quat(delta_quat)

        delta_xy_rotation = torch.cat([
            detla_euler[0].unsqueeze(1), detla_euler[1].unsqueeze(1),
            detla_euler[2].unsqueeze(1)
        ],
                                      dim=1)

        self.target_in_object_angle = torch.sum(abs(delta_xy_rotation), dim=1)

    def get_contact_info(self, env):
        sensor_data = []

        for name in self.sensor_name:
            sensor = env.scene[f"{name}_contact"]

            force_data = torch.linalg.norm(sensor._data.force_matrix_w.reshape(
                env.num_envs, 3),
                                           dim=1).unsqueeze(1)

            sensor_data.append(force_data)

        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 2.0).int()
