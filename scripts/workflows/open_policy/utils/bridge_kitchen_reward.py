from isaaclab.managers import RewardTermCfg as RewTerm
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTermCfg


class BridgeKitchenReward:

    def __init__(self, reward_target, obs_target, env_cfg):
        self.reward_target = reward_target
        self.obs_target = obs_target

        self.env_cfg = env_cfg
        self.init_setting()

    def init_setting(self):
        self.init_ee_pose = torch.as_tensor(
            self.env_cfg["params"]["Task"]["init_ee_pose"]).unsqueeze(0)

        self.target_lift_object_pos = torch.as_tensor(
            self.env_cfg["params"]["Task"]["target_lift_object_pos"]).to(
                "cuda")
        setattr(
            self.reward_target, "object_ee_distance",
            RewTerm(func=self.object_ee_distance,
                    params={
                        "object_name": "eggplant",
                        "ee_frame_name": "wx250s_ee_gripper_link"
                    },
                    weight=1.0))
        setattr(
            self.reward_target, "object_is_lifted",
            RewTerm(
                func=self.object_is_lifted,
                weight=1.0,
                params={
                    "object_name": "eggplant",
                },
            ))

        setattr(
            self.reward_target, "object_in_between_gripper",
            RewTerm(
                func=self.object_in_between_gripper,
                weight=1.0,
                params={
                    "object_name": "eggplant",
                },
            ))

        setattr(
            self.reward_target, "object_distance",
            RewTerm(
                func=self.object_distance,
                weight=1.0,
                params={
                    "object_name": "eggplant",
                },
            ))
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        setattr(
            self.obs_target, "init_object_pos",
            ObsTerm(
                func=self.init_object_pos,
                params={
                    "object_name": "eggplant",
                },
            ))
        self.init_ee_object_dist = 0.0011
        self.init_object_height = 0.0011
        self.init_ee_object_angle = 0.0001
        self.init_object_pos = None
        self.minimal_height = self.env_cfg["params"]["Task"]["lift_height"]

    def init_object_pos(self, env, object_name):
        object = env.scene[object_name]
        object_pos_w = object.data.root_state_w[..., :3]
        object_pos_w = object.data.root_state_w[..., :3]

        try:

            if env.episode_length_buf[0] == self.env_cfg["params"]["Task"][
                    "reset_horizon"] - 1 or self.init_object_pos is None:
                self.init_object_pos = object_pos_w.clone()

        except:
            self.init_object_pos = torch.zeros(
                (env.num_envs, 3)).to(env.device)

        return self.init_object_pos[:, :3]

    def reset(self, env, object_name, ee_frame_name):

        object = env.scene[object_name]

        ee_pos_w = env.scene[ee_frame_name].data.root_state_w
        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_state_w

        self.init_ee_object_dist = torch.norm(object_pos_w[:, :3] -
                                              ee_pos_w[:, :3],
                                              dim=1)

    def object_ee_distance(
        self,
        env: ManagerBasedRLEnv,
        object_name: str,
        ee_frame_name: str,
    ):
        """Reward the agent for reaching the object using tanh-kernel."""
        # extract the used quantities (to enable type-hinting)
        object: RigidObject = env.scene[object_name]

        ee_pos_w = env.scene[ee_frame_name].data.root_state_w

        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_state_w
        # End-effector position: (num_envs, 3)

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

        # distance and angle reward

    # Define factors for the distance ratio
        factor_high = 6  # Factor when distance ratio is greater than 0.7
        factor_low = 3  # Factor when 0 <= distance ratio <= 0.7
        factor_negative = 3  # Factor when distance ratio < 0

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
                                             1) * 0.3

        # x angle can not move a lot
        ee_xy_angles_penalty = torch.clip(
            torch.sum(abs(delta_ee_axis_angles[:, :2]), dim=1), 0, 1) * 3

        rewards = (object_ee_dist_reward - object_ee_angle_penalty -
                   ee_xy_angles_penalty) * 1 / env.step_dt

        return rewards

    def object_is_lifted(self, env, object_name: str):

        object = env.scene[object_name]

        target_pose = self.target_lift_object_pos.unsqueeze(
            0).repeat_interleave(env.num_envs, 0)
        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_object_height = object.data.root_link_pos_w[:, 2].clone()

            self.init_object_to_lift_dist = torch.linalg.norm(
                object.data.root_link_pos_w[:, :2] - target_pose[:, :2],
                dim=1) + abs(object.data.root_link_pos_w[:, 2] -
                             target_pose[:, 2])

        elif env.episode_length_buf[0] < self.env_cfg["params"]["Task"][
                "reset_horizon"]:
            return torch.zeros(env.num_envs).to(env.device)
        delta_height = abs(object.data.root_link_pos_w[:, 2] -
                           target_pose[:, 2])
        object_to_lift_dist = torch.linalg.norm(
            object.data.root_link_pos_w[:, :2] - target_pose[:, :2],
            dim=1) + delta_height

        dist_ratio = 1 - object_to_lift_dist / self.init_object_to_lift_dist
        scaling_factor = torch.where(dist_ratio < 0, 1.0,
                                     torch.where(dist_ratio > 0.6, 6, 3))

        return (torch.clip(dist_ratio, -0.5,
                           1.0)) * scaling_factor / env.step_dt

    def object_in_between_gripper(self, env, object_name: str):
        wx250s_right_finger_link_state = env.scene[
            "wx250s_right_finger_link"].data.root_state_w[:, :7]
        wx250s_left_finger_link_state = env.scene[
            "wx250s_left_finger_link"].data.root_state_w[:, :7]
        object_state = env.scene[object_name].data.root_state_w[:, :7]
        ee_pos_w = env.scene["wx250s_ee_gripper_link"].data.root_state_w
        object_ee_distance = torch.norm(object_state[:, :2] - ee_pos_w[:, :2],
                                        dim=1)
        close_enough = torch.where(object_ee_distance < 0.04, True, False)
        height_diff = object_state[:, 2] - ee_pos_w[:, 2]
        height_enough = torch.where(abs(height_diff) < 0.02, True, False)

        in_between_gripper = torch.logical_or(
            torch.logical_and(
                object_state[:, 1] > wx250s_left_finger_link_state[:, 1],
                object_state[:, 1] < wx250s_right_finger_link_state[:, 1]),
            torch.logical_and(
                object_state[:, 1] < wx250s_left_finger_link_state[:, 1],
                object_state[:, 1] > wx250s_right_finger_link_state[:, 1]))

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

        return (in_between_reward + ation_penalty)[0]

    def object_distance(self, env, object_name: str):
        """Reward the agent for reaching the object using tanh-kernel."""
        object: RigidObject = env.scene[object_name]

        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_state_w[..., :3]

        if env.episode_length_buf[
                0] == self.env_cfg["params"]["Task"]["reset_horizon"] - 1:
            self.init_object_pos = object_pos_w.clone()
            return torch.zeros(env.num_envs).to(env.device)
        elif self.init_object_pos is None:
            return torch.zeros(env.num_envs).to(env.device)

        object_dev = torch.where(
            object_pos_w[:, 2] < 0.15,  # Condition
            torch.linalg.norm(object_pos_w[:, :2] -
                              self.init_object_pos[:, :2],
                              dim=1),  # If condition is True
            torch.zeros(env.num_envs,
                        device=env.device)  # If condition is False
        )

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
            torch.clip(-2 * torch.sign(actions + 0.02), -0.1, 2) * 2,
            # If not in-between: open = reward, close = penalty
            torch.clip(2 * torch.sign(actions + 0.02), -1.5, 0.0) *
            1) / env.step_dt

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

        ee_pos_w = env.scene[ee_frame_name].data.root_state_w[..., :3]

        # Target object position: (num_envs, 3)
        object_pos_w = object.data.root_state_w[..., :3]
        # End-effector position: (num_envs, 3)
        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(ee_pos_w - object_pos_w, dim=1)

        return 1 - torch.tanh(object_ee_distance / std)
