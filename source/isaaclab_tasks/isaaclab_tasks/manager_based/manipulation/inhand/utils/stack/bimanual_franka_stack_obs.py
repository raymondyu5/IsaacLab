from isaaclab_assets import *

import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.managers import EventTermCfg as EventTerm
import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.configuration_robot import config_bimanual_robot_contact_sensor
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.grasp.config_rigids import configure_multi_rigid_obs
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

import trimesh
from isaaclab_tasks.manager_based.manipulation.inhand.utils.place.bimanual_franka_grasp_actions import HandActions


class BimanualFrankaStackObs:

    def __init__(
        self,
        env_cfg,
        obs_cfg,
        scene_cfg,
        events_cfg=None,
        actions_cfg=None,
    ):

        num_envs = env_cfg["params"].get("num_envs", scene_cfg.num_envs)

        if env_cfg["params"]["add_left_hand"]:
            # actions_cfg.left_hand_action.class_type = HandActions
            self.init_env(env_cfg, obs_cfg, scene_cfg, "left", num_envs,
                          events_cfg)
        if env_cfg["params"]["add_right_hand"]:
            actions_cfg.right_hand_action.class_type = HandActions
            self.init_env(env_cfg, obs_cfg, scene_cfg, "right", num_envs,
                          events_cfg)

    def init_env(self, env_cfg, obs_cfg, scene_cfg, hand_side, num_envs,
                 events_cfg):

        target_manipulated_object_pose = torch.as_tensor(
            env_cfg["params"][f"{hand_side}_target_manipulated_object_pose"]
        ).unsqueeze(0).repeat_interleave(num_envs, dim=0).to("cuda")

        SingleHandGraspObs(env_cfg, obs_cfg, scene_cfg,
                           f"{hand_side}_hand_object",
                           f"{hand_side}_hand_place_object", hand_side,
                           target_manipulated_object_pose, events_cfg)


class SingleHandGraspObs:

    def __init__(self, env_cfg, obs_cfg, scene_cfg, object_name,
                 placement_name, hand_side, target_manipulated_object_pose,
                 events_cfg):
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.scene_cfg = scene_cfg
        self.hand_side = hand_side
        self.object_name = object_name
        self.placement_name = placement_name

        self.num_envs = env_cfg["params"].get("num_envs",
                                              self.scene_cfg.num_envs)
        self.target_manipulated_object_pose = target_manipulated_object_pose.clone(
        )
        self.fingers_name_list = [
            "palm_lower",
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        self.init_robot_pose = torch.as_tensor(
            self.env_cfg["params"][f"{hand_side}_robot_pose"]).to("cuda")

        self.target_manipulated_object_pose[:, :3] -= self.init_robot_pose[
            ..., :3]
        self.events_cfg = events_cfg

        self.setup_obs()

        self.goal_object_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "goal":
                sim_utils.UsdFileCfg(
                    usd_path=
                    f"source/assets/ycb/dexgrasp/objects/mug/rigid_object.usd",
                    scale=(1.0, 1.0, 1.0),
                )
            },
        )
        # self._markers = VisualizationMarkers(self.goal_object_cfg)
        # self.goal_markers = None

    def reset(self, env, env_ids=None):

        pass

    def config_robot_contact_obs(self):

        contact_sensor_name = self.env_cfg["params"]["contact_sensor"][
            "spawn_contact_list"]

        self.sensor_name = []
        for name in contact_sensor_name:
            self.sensor_name.append(f"{self.hand_side}_{name}")

        contact_obs = ObsTerm(func=self.get_contact_obs, )
        setattr(self.obs_cfg, f"{self.hand_side}_contact_obs", contact_obs)

    def get_contact_obs(self, env):

        return self.contact_or_not

    def get_ee_pose_obs(self, env):

        ee_pose = env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w[:, :3].clone()
        ee_pose[:, :3] -= env.scene.env_origins
        ee_pose[:, :3] -= self.init_robot_pose[..., :3]
        # print(torch.max(abs))

        return ee_pose[:, :7]

    def setup_obs(self):

        reset_func = EventTerm(func=self.reset, mode="reset", params={})
        setattr(self.events_cfg,
                f"reset_{self.hand_side}_object_target_pose_in_observation",
                reset_func)

        delattr(self.obs_cfg, f"{self.hand_side}_hand_joint_pos")
        joint_pos = ObsTerm(func=mdp.joint_pos,
                            params={"asset_name": f"{self.hand_side}_hand"})
        setattr(self.obs_cfg, f"{self.hand_side}_hand_joint_pos", joint_pos)

        ee_pose_obs = ObsTerm(func=self.get_ee_pose_obs, )
        setattr(self.obs_cfg, f"{self.hand_side}_ee_pose", ee_pose_obs)

        config_bimanual_robot_contact_sensor(self.scene_cfg, self.env_cfg,
                                             [self.object_name],
                                             self.hand_side)

        manipulated_object_pose = ObsTerm(func=self.manipulated_object_pose)

        setattr(self.obs_cfg, f"{self.hand_side}_manipulated_object_pose",
                manipulated_object_pose)

        object_in_tip = ObsTerm(func=self.object_in_tip, )
        setattr(self.obs_cfg, f"{self.hand_side}_object_in_tip", object_in_tip)

        place_object_pose = ObsTerm(func=self.get_placement_object_pose)
        setattr(self.obs_cfg, f"{self.hand_side}_hand_place_object_pose",
                place_object_pose)
        self.config_robot_contact_obs()

    def object_in_tip(self, env):

        return self.finger_object_dev.reshape(env.num_envs, -1)[..., :7]

    def manipulated_object_pose(self, env):

        self.get_object_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        return self.object_pose

    def get_placement_object_pose(self, env):

        return self.placement_object_pose[..., :3]

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
            # finger_pose[:, 2] -= env.scene[
            #     f"{self.hand_side}_hand"]._data.root_state_w[:, 2].clone()

            self.finger_pose.append(finger_pose.unsqueeze(1))

        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        finger_object_pose = self.object_pose.clone().unsqueeze(
            1).repeat_interleave(len(self.fingers_name_list), dim=1)
        self.finger_object_dev = (finger_object_pose[..., :3] -
                                  self.finger_pose[..., :, :3])

    def get_state_info(self, env, object_name):

        object_pose = env.scene[object_name]._data.root_state_w[:, :7].clone()

        object_pose[:, :3] -= env.scene.env_origins
        object_pose[:, :3] -= self.init_robot_pose[..., :3]
        return object_pose

    def get_object_info(self, env):
        self.object_pose = self.get_state_info(env, self.object_name)
        self.placement_object_pose = self.get_state_info(
            env, self.placement_name)[..., :3]
        # try:
        #     self.placement_object_pose = env.scene[
        #         self.object_name]._data.target_state[:, :7].clone()
        #     self.placement_object_pose[:, :3] -= env.scene.env_origins

        # except:
        #     self.placement_object_pose = self.get_state_info(
        #         env, self.placement_name)

    def get_contact_info(self, env):
        sensor_data = []

        for name in self.sensor_name:
            sensor = env.scene[f"{name}_contact"]

            force_data = torch.linalg.norm(sensor._data.force_matrix_w.reshape(
                env.num_envs, 3),
                                           dim=1).unsqueeze(1)

            sensor_data.append(force_data)

        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 2.0).int()
