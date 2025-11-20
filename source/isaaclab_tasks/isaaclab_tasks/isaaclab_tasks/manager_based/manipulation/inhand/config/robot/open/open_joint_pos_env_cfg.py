# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import os
from isaaclab.envs import mdp
from scripts.workflows.utils.config_setting import set_from_config
from isaaclab.managers import ObservationTermCfg as ObsTerm
##
# Pre-defined configs
##

from dataclasses import MISSING, Field, dataclass, field, replace
import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.configuration_robot import config_robot

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.open.config_rigids import configure_multi_rigid_obs
# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.config_camera import configure_camera
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.open.config_cluster_rigids import configure_multi_cluster_rigid_obs

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.open.bimanual_franka_teleoperation_obs import BimanualFrankaTeleopObs
# from isaaclab_tasks.manager_based.manipulation.inhand.utils.synthesize_pcd import configure_sysnthesize_robot_pc

from isaaclab.assets import RigidObjectCfg

from isaaclab.sim.spawners import UsdFileCfg
# configure tangle block
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
import torch


@configclass
class HandYCBEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):

        env_cfg = self.config_yaml

        # post init of parent
        super().__post_init__()
        # self.config_yaml = env_cfg
        current_path = os.getcwd()

        config_robot(self, self.actions, env_cfg)
        eval_mode = env_cfg["params"].get("eval_mode", False)
        real_eval_mode = env_cfg["params"].get("real_eval_mode", False)

        ## =================================================

        setattr(self.scene, "eval_mode", eval_mode or real_eval_mode)

        if env_cfg["params"].get("use_teleop", None) is not None:
            self.episode_length_s = 1000

            articulation_object_name = list(
                env_cfg["params"]["ArticulationObject"].keys())
            for name in articulation_object_name:

                joint_friction = EventTerm(
                    func=mdp.randomize_joint_parameters,
                    min_step_count_between_reset=200,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(name),
                        "friction_distribution_params": (5, 15),
                        "operation": "add",
                        "distribution": "uniform",
                    },
                )

                reset_articulation_joint = EventTerm(
                    func=mdp.reset_joints_by_values,
                    mode="reset",
                    params={
                        "joint_pose": torch.zeros(1),
                        "asset_name": name
                    })
                setattr(self.events, f"reset_{name}_friction", joint_friction)
                setattr(self.events, f"reset_{name}_joints",
                        reset_articulation_joint)
            BimanualFrankaTeleopObs(env_cfg, self.observations.policy,
                                    self.scene, self.events)
            set_from_config(self.scene, env_cfg, current_path)
        else:

            rigid_object_config = env_cfg["params"].get("RigidObject")

            if env_cfg["params"][
                    "arm_type"] is not None and rigid_object_config is not None:

                if env_cfg["params"].get("multi_cluster_rigid",
                                         None) is not None:

                    configure_multi_cluster_rigid_obs(self, env_cfg)
                else:
                    configure_multi_rigid_obs(self, env_cfg)

            set_from_config(self.scene, env_cfg, current_path)

        camera_setting = env_cfg["params"]["Camera"]

        if camera_setting["initial"]:
            setattr(self.observations.policy, "camera_obs", "camera_obs")

            camera_obs_function = mdp.CameraObservation(
                camera_setting,
                sample_points=env_cfg["params"].get("sample_points", False))
            self.observations.policy.camera_obs = ObsTerm(
                func=camera_obs_function.process_camera_data,
                params={},
            )
            configure_camera(self, env_cfg)
            self.scene.env_spacing = 10
        if env_cfg["params"].get("sythesize_robot_pc", False):
            configure_sysnthesize_robot_pc(
                self.observations.policy,
                env_cfg,
            )

        camera_setting = env_cfg["params"]["Camera"]


@configclass
class HandYCBEnvCfg_PLAY(HandYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4
        # disable randomization for play
        self.observations.policy.enable_corruption = False
