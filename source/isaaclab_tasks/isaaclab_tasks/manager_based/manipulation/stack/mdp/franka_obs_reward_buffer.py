from isaaclab.managers import RewardTermCfg as RewTerm
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTermCfg


class RewardObsBuffer:

    def __init__(self,
                 target_reward=None,
                 target_obs=None,
                 env_cfg=None,
                 target_object_name=None,
                 ee_frame_name=None,
                 left_finger_name=None,
                 right_finger_name=None,
                 use_gripper_offset=True):
        self.reward_target = target_reward
        self.obs_target = target_obs
        self.env_cfg = env_cfg
        self.target_object_name = target_object_name
        self.ee_frame_name = ee_frame_name
        self.left_finger_name = left_finger_name
        self.right_finger_name = right_finger_name
        self.use_gripper_offset = use_gripper_offset

        self.init_ee_pose = torch.as_tensor(
            env_cfg["params"]["Task"]["init_ee_pose"]).unsqueeze(0)

        self.target_lift_object_pos = torch.as_tensor(
            env_cfg["params"]["Task"]["target_lift_object_pos"]).to("cuda")

        self.init_ee_object_dist = 0.0011
        self.init_object_height = 0.0011
        self.init_ee_object_angle = 0.0001
        self.init_object_pos = None
        self.minimal_height = self.env_cfg["params"]["Task"]["lift_height"]
        if target_reward is not None:
            self.init_setting()

    def init_setting(self):

        setattr(
            self.reward_target, "object_ee_distance",
            RewTerm(func=self.object_ee_distance,
                    params={
                        "object_name": self.target_object_name,
                        "ee_frame_name": self.ee_frame_name
                    },
                    weight=0.2))
        setattr(
            self.reward_target, "object_is_lifted",
            RewTerm(
                func=self.object_is_lifted,
                weight=0.2,
                params={
                    "object_name": self.target_object_name,
                },
            ))

        setattr(
            self.reward_target, "object_in_between_gripper",
            RewTerm(
                func=self.object_in_between_gripper,
                weight=0.2,
                params={
                    "object_name": self.target_object_name,
                },
            ))

        from isaaclab.managers import ObservationTermCfg as ObsTerm
        setattr(
            self.obs_target, "target_object_pos",
            ObsTerm(
                func=self.target_object_pos,
                params={
                    "object_name": self.target_object_name,
                },
            ))

    def object_inside_place_region(self, env, object_name: str):
        if env.episode_length_buf[0] < 100:

            return torch.zeros(env.num_envs).to(env.device)
        object = env.scene[object_name]
        object_state = object.data.root_link_state_w
        object_state[:, :3] -= env.scene.env_origins
        import pdb

        object_inside_place_region = ((object_state[:, 1] > 0.05) &
                                      (object_state[:, 1] < 0.15) &
                                      (object_state[:, 0] > 0.23) &
                                      (object_state[:, 0] < 0.40) &
                                      (object_state[:, 2] > 0.0))
        object_not_inside_place_region = ~object_inside_place_region

        gripper_actions = env.action_manager.get_term(
            "gripper_action").raw_actions.T
        lower_object = torch.where(object_state[:, 2] < 0.15, True, False)

        reward_condition = torch.logical_and(
            ~object_not_inside_place_region,
            torch.logical_or(gripper_actions[0] >= 0.0, lower_object))
        # reward_condition = torch.logical_and(~object_not_inside_place_region,
        #                                      gripper_actions[0] >= 0.0)

        return torch.where(reward_condition, 30, 0) / env.step_dt

    def init_object_pos(self, env, object_name):
        object = env.scene[object_name]

        object_pos_w = object.data.root_link_state_w[..., :3]

        try:

            if env.episode_length_buf[0] == self.env_cfg["params"]["Task"][
                    "reset_horizon"] - 1 or self.init_object_pos is None:
                self.init_object_pos = object_pos_w.clone()

        except:
            self.init_object_pos = torch.zeros(
                (env.num_envs, 3)).to(env.device)

        return self.init_object_pos[:, :3]

    def target_object_pos(self, env, object_name):

        return env.command_manager.get_command("target_object_pose")

        # return self.target_lift_object_pos.unsqueeze(0).repeat_interleave(
        #     env.num_envs, 0).to(env.device)

    def reset(self, env, object_name, ee_frame_name):

        object = env.scene[object_name]

        ee_pos_w = env.scene[ee_frame_name].data.root_link_state_w
        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_link_state_w

        self.init_ee_object_dist = torch.norm(object_pos_w[:, :3] -
                                              ee_pos_w[:, :3],
                                              dim=1)

    def sum_rewards(self, env, object_name, ee_frame_name):

        reward_object_ee_distance = self.object_ee_distance(
            env, object_name, ee_frame_name)
        reward_object_is_lifted = self.object_is_lifted(env, object_name)
        reward_object_in_between_gripper = self.object_in_between_gripper(
            env, object_name)
        # print('====================')
        # print("reward_object_ee_distance", reward_object_ee_distance)
        # print("reward_object_is_lifted", reward_object_is_lifted)
        # print("reward_object_in_between_gripper",
        #       reward_object_in_between_gripper)

        return (reward_object_ee_distance + reward_object_is_lifted +
                reward_object_in_between_gripper) * env.step_dt

    def get_root_link_state(self, env, object_name):
        object = env.scene[object_name]
        return object.data.root_link_state_w

    def add_ee_offset(self, env, ee_pos_w):
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_w[:, :3], ee_pos_w[:, 3:7],
            torch.tensor([[0.0000, 0.0000, 0.1070]],
                         device=env.device).repeat_interleave(env.num_envs, 0),
            torch.tensor([[1., 0., 0., 0.]],
                         device=env.device).repeat_interleave(env.num_envs, 0))
        ee_pos_w = torch.cat([ee_pose_b, ee_quat_b], dim=1)
        return ee_pos_w

    def object_ee_distance(self,
                           env: ManagerBasedRLEnv,
                           object_name: str,
                           ee_frame_name: str,
                           use_gripper_offset=True):
        """Reward the agent for reaching the object using tanh-kernel."""

        ee_pos_w = self.get_root_link_state(env, ee_frame_name)

        # Target object position: (num_envs, 3)
        object_pos_w = self.get_root_link_state(env, object_name)
        # End-effector position: (num_envs, 3)

        if self.use_gripper_offset:
            ee_pos_w = self.add_ee_offset(env, ee_pos_w)

        # calculate the angle between the object and the end-effector
        delta_ee_quat = math_utils.quat_mul(
            math_utils.quat_inv(self.init_ee_pose[:, 3:7].repeat_interleave(
                env.num_envs, 0).to(env.device)), ee_pos_w[:, 3:7])
        delta_ee_axis_angles = math_utils.axis_angle_from_quat(delta_ee_quat)
        delta_object_axis_angles = math_utils.axis_angle_from_quat(
            object_pos_w[:, 3:7])
        delta_ee_object_z_angle = delta_ee_axis_angles[:,
                                                       2] - delta_object_axis_angles[:,
                                                                                     2]

        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(object_pos_w[:, :3] - ee_pos_w[:, :3],
                                        dim=1)

        # reset the distance when the episode is reset

        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_ee_object_dist = torch.clip(object_ee_distance.clone(),
                                                  0.20, 0.30)
            self.init_ee_object_angle = delta_ee_object_z_angle.clone()
        elif env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)

        # Define factors for the distance ratio
        factor_high = 1.5  # Factor when distance ratio is greater than 0.7
        factor_low = 0.5  # Factor when 0 <= distance ratio <= 0.7
        factor_negative = 0.5  # Factor when distance ratio < 0

        # Compute the distance ratio
        distance_ratio = 1 - object_ee_distance / self.init_ee_object_dist

        # Apply different factors based on the distance ratio
        scaling_factor = torch.where(
            distance_ratio < 0, factor_negative,
            torch.where(distance_ratio > 0.7, factor_high, factor_low))

        # Compute the reward with the appropriate scaling factor
        object_ee_dist_reward = torch.clip(distance_ratio, -1.0,
                                           1.5) * scaling_factor

        object_ee_angle_penalty = torch.clip(abs(delta_ee_object_z_angle), 0,
                                             1) * 1

        # x angle can not move a lot
        ee_xy_angles_penalty = torch.clip(
            torch.sum(abs(delta_ee_axis_angles[:, :2]), dim=1), 0, 1) * 3.0

        rewards = (object_ee_dist_reward - object_ee_angle_penalty -
                   ee_xy_angles_penalty) * 1 / env.step_dt

        return rewards

    def object_is_lifted(self, env, object_name: str):

        object = env.scene[object_name]

        # target_pose = self.target_lift_object_pos.unsqueeze(
        #     0).repeat_interleave(env.num_envs, 0)
        target_pose = env.command_manager.get_command("target_object_pose")
        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_object_height = object.data.root_link_pos_w[:, 2].clone()

            # self.init_object_to_lift_dist = torch.linalg.norm(
            #     object.data.root_link_pos_w[:, :2] - target_pose[:, :2] -
            #     env.scene.env_origins[:, :2],
            #     dim=1)

            self.init_object_to_lift_dist = torch.linalg.norm(
                object.data.root_link_pos_w[:, :3] - target_pose[:, :3] -
                env.scene.env_origins,
                dim=1)

        elif env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)
        cur_object_to_lift_dist = torch.linalg.norm(
            object.data.root_link_pos_w[:, :3] - target_pose[:, :3] -
            env.scene.env_origins[:, :3],
            dim=1)

        target_reward_factor = torch.where(
            (object.data.root_link_pos_w[:, 2] - self.init_object_height)
            > 0.10, 3.0, 1.0)

        target_reward = torch.clip(
            (1 / torch.clip(cur_object_to_lift_dist, 0.01, 10) -
             1 / self.init_object_to_lift_dist) * target_reward_factor, -0.0,
            25)
        delta_height = torch.clip(
            (object.data.root_link_pos_w[:, 2] - self.init_object_height) /
            0.10, 0.0, 1.0) * 4

        bonus_reward = torch.where(
            (object.data.root_link_pos_w[:, 2] - self.init_object_height)
            > 0.01,
            torch.ones(env.num_envs).to(env.device) * 1.0,
            torch.zeros(env.num_envs).to(env.device) * 0.0)

        return target_reward / env.step_dt * 1.0 + bonus_reward / env.step_dt + delta_height / env.step_dt

    def object_in_between_gripper(self,
                                  env,
                                  object_name: str,
                                  use_gripper_offset=True):
        left_finger_name_state = self.get_root_link_state(
            env, self.left_finger_name)
        right_finger_name_state = self.get_root_link_state(
            env, self.right_finger_name)
        ee_pos_w = self.get_root_link_state(env, self.ee_frame_name)
        object_state = self.get_root_link_state(env, object_name)

        if self.use_gripper_offset:
            ee_pos_w = self.add_ee_offset(env, ee_pos_w)

        object_ee_distance = torch.norm(object_state[:, :2] - ee_pos_w[:, :2],
                                        dim=1)

        close_enough = torch.where(object_ee_distance < 0.03, True, False)
        height_diff = object_state[:, 2] - ee_pos_w[:, 2]
        height_enough = torch.where(abs(height_diff) < 0.02, True, False)

        in_between_gripper = torch.logical_or(
            torch.logical_and(
                object_state[:, 1] > left_finger_name_state[:, 1],
                object_state[:, 1] < right_finger_name_state[:, 1]),
            torch.logical_and(
                object_state[:, 1] < left_finger_name_state[:, 1],
                object_state[:, 1] > right_finger_name_state[:, 1]))

        in_between_gripper = torch.logical_and(
            in_between_gripper,
            close_enough,
        )
        in_between_gripper = torch.logical_and(in_between_gripper,
                                               height_enough)

        if env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)

        # penalty of the gripper close(-1) when the object is not in between

        ation_penalty = self.gripper_in_between_penalty(
            env, in_between_gripper)
        in_between_reward = torch.where(
            in_between_gripper[None],
            torch.ones(env.num_envs).to(env.device) * 1.0,
            -torch.ones(env.num_envs).to(env.device) * 0.0) / env.step_dt
        # print("in_between_reward", in_between_reward)

        return (in_between_reward + ation_penalty)[0]

    def object_distance(self, env, object_name: str):

        # Target object position: (num_envs, 3)
        object_pos_w = self.get_root_link_state(env, object_name)

        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_object_pos = object_pos_w.clone()
            return torch.zeros(env.num_envs).to(env.device)
        elif self.init_object_pos is None:
            return torch.zeros(env.num_envs).to(env.device)
        init_dist = torch.linalg.norm(object_pos_w[:, :2] -
                                      self.init_object_pos[:, :2],
                                      dim=1)
        target_lift_object_pos = env.command_manager.get_command(
            "target_object_pose")
        target_dist = torch.linalg.norm(object_pos_w[:, :2] -
                                        target_lift_object_pos[:2],
                                        dim=1)

        dist = torch.min(torch.stack([init_dist, target_dist], dim=1),
                         dim=1).values

        dist = torch.where(dist < 0.10,
                           torch.zeros(env.num_envs).to(env.device), dist)

        object_dev = torch.where(
            object_pos_w[:, 2] < 0.15, dist,
            torch.zeros(env.num_envs, device=env.device, dtype=dist.dtype))

        return torch.clip(-object_dev / 0.06, -1.0, 0.1) / env.step_dt * 0.5

    def gripper_in_between_penalty(self, env, in_between_gripper):

        # Check whether the gripper is in-between
        gripper_condition = in_between_gripper[None]

        # Define actions (assumes gripper action is the relevant term)
        actions = env.action_manager.get_term("gripper_action").raw_actions.T

        # Apply rewards/penalties based on the condition
        action_penalty = torch.where(
            gripper_condition,
            # If in-between: open = penalty, close = reward
            torch.clip(-2 * torch.sign(actions + 0.02), -0.0, 1) * 0.5,
            # If not in-between: open = reward, close = penalty
            torch.clip(2 * torch.sign(actions + 0.02), -0.0, 0.0) *
            2) / env.step_dt
        # print("action_penalty", action_penalty)

        return action_penalty

    def object_ee_distance2(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        object_name: str,
        ee_frame_name: str,
    ) -> torch.Tensor:
        """Reward the agent for reaching the object using tanh-kernel."""
        object: RigidObject = env.scene[object_name]

        ee_pos_w = env.scene[ee_frame_name].data.root_link_state_w[..., :3]

        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_link_state_w[..., :3]
        # End-effector position: (num_envs, 3)
        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(ee_pos_w - object_pos_w, dim=1)

        return 1 - torch.tanh(object_ee_distance / std)
