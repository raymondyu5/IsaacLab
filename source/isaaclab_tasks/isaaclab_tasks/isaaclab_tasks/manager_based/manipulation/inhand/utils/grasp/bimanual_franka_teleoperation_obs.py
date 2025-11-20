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
import trimesh

from isaaclab.managers import SceneEntityCfg


class BimanualFrankaTeleopObs:

    def __init__(self, env_cfg, obs_cfg, scene_cfg, events):

        num_envs = env_cfg["params"].get("num_envs", scene_cfg.num_envs)
        self.add_left_hand = env_cfg["params"]["add_left_hand"]
        self.add_right_hand = env_cfg["params"]["add_right_hand"]
        self.obs_cfg = obs_cfg
        self.env_cfg = env_cfg
        self.events = events
        self.setup_obs()

    def setup_obs(self):

        if self.add_left_hand:
            self.setup_obs_for_hand("left")
            self.hand_side = "left"
        if self.add_right_hand:
            self.setup_obs_for_hand("right")
            self.hand_side = "right"
        self.setup_obs_for_rigid_objects()

    def setup_obs_for_hand(self, hand_side):
        delattr(self.obs_cfg, f"{hand_side}_hand_joint_pos")
        joint_pos = ObsTerm(func=mdp.joint_pos,
                            params={"asset_name": f"{hand_side}_hand"})
        setattr(self.obs_cfg, f"{hand_side}_hand_joint_pos", joint_pos)

        root_pose = ObsTerm(func=mdp.root_pose,
                            params={"asset_name": f"{hand_side}_hand"})
        setattr(self.obs_cfg, f"{hand_side}_hand_root_pose", root_pose)

        ee_pose_obs = ObsTerm(func=self.get_ee_pose_obs, )
        setattr(self.obs_cfg, f"{hand_side}_ee_pose", ee_pose_obs)

    def setup_obs_for_rigid_objects(self):

        rigid_objects = self.env_cfg["params"].get("RigidObject", {})
        if rigid_objects is not None:
            for object_name, object_config in rigid_objects.items():

                manipulated_object_pose = ObsTerm(
                    func=self.manipulated_object_pose,
                    params={"object_name": object_name})
                setattr(self.obs_cfg, f"{object_name}_pose",
                        manipulated_object_pose)

                # rigid_physics_material = EventTerm(
                #     func=mdp.randomize_rigid_body_material,
                #     mode="reset",
                #     min_step_count_between_reset=720,
                #     params={
                #         "asset_cfg": SceneEntityCfg(object_name,
                #                                     body_names=".*"),
                #         "static_friction_range": (1.5, 2.5),
                #         "dynamic_friction_range": (1.5, 2.5),
                #         "restitution_range": (0.0, 0.0),
                #         "num_buckets": 64,
                #     },
                # )
                # setattr(self.events, f"{object_name}_rigid_physics_material",
                #         rigid_physics_material)

    def manipulated_object_pose(self, env, object_name):
        object_pose = env.scene[object_name]._data.root_state_w[:, :7].clone()

        object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def get_ee_pose_obs(self, env):

        ee_pose = env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w[:, :3].clone()
        ee_pose[:, :3] -= env.scene.env_origins

        # print(torch.max(abs))

        return ee_pose[:, :3]
