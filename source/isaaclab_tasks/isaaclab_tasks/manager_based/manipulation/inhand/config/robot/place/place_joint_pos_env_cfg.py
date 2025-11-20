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

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.config_rigids import configure_multi_rigid_obs
# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.config_camera import configure_camera
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.config_cluster_rigids import configure_multi_cluster_rigid_obs
# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.config_articulation import configure_multi_articulation_obs
# # from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.synthesize_pc import SynthesizePC

# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.franka_grasp_obs import FrankaGraspObs
# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.franka_grasp_rew import FrankaGraspRew

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.bimanual_franka_grasp_obs import BimanualFrankaGraspObs
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.bimanual_franka_grasp_rew import BimanualFrankaGraspRew

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.config_camera import configure_camera

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.place.bimanual_franka_teleoperation_obs import BimanualFrankaTeleopObs
# from isaaclab_tasks.manager_based.manipulation.inhand.utils.synthesize_pcd import configure_sysnthesize_robot_pc

from isaaclab.assets import RigidObjectCfg

from isaaclab.sim.spawners import UsdFileCfg
# configure tangle block
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import EventTermCfg as EventTerm


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

        table_block_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + "table_block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0],
                                                      rot=[1, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=f"source/assets/table/table_block.usd",
                             scale=[1.2, 1.0, 0.10],
                             mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                             rigid_props=RigidBodyPropertiesCfg(
                                 solver_position_iteration_count=16,
                                 solver_velocity_iteration_count=1,
                                 max_angular_velocity=1000.0,
                                 max_linear_velocity=1000.0,
                                 disable_gravity=False,
                                 kinematic_enabled=True,
                             )),
        )
        setattr(self.scene, "table_block", table_block_cfg)

        # z_range = (0.02, 0.07)
        # if real_eval_mode:  # or FLAGS.real_eval_mode if using absl
        z_range = (0.0, 0.0)

        reset_block = EventTerm(func=mdp.reset_rigid_articulation,
                                mode="reset",
                                params={
                                    "target_name": f"table_block",
                                    "pose_range": {
                                        "z": z_range,
                                    }
                                })

        setattr(self.events, f"reset_block", reset_block)
        ## =================================================

        if env_cfg["params"].get("use_teleop", None) is not None:
            self.episode_length_s = 1000
            BimanualFrankaTeleopObs(env_cfg, self.observations.policy,
                                    self.scene, self.events)
            set_from_config(self.scene, env_cfg, current_path)
        else:

            self.episode_length_s = env_cfg["params"].get(
                "episode_length_s", 10)

            rigid_object_config = env_cfg["params"].get("RigidObject")

            if env_cfg["params"][
                    "arm_type"] is not None and rigid_object_config is not None:

                if env_cfg["params"].get("multi_cluster_rigid",
                                         None) is not None:

                    env_cfg = configure_multi_cluster_rigid_obs(self, env_cfg)
                else:
                    configure_multi_rigid_obs(self, env_cfg)

            # config_ciriculumn_action(self, env_cfg)

            set_from_config(self.scene, env_cfg, current_path)

            if "multi_cluster_rigid" in env_cfg["params"].keys():

                if env_cfg["params"][
                        "arm_type"] == "franka" and rigid_object_config is not None:

                    BimanualFrankaGraspObs(env_cfg,
                                           self.observations.policy,
                                           self.scene,
                                           self.events,
                                           actions_cfg=self.actions)
                    BimanualFrankaGraspRew(
                        env_cfg,
                        self.rewards,
                        self.scene,
                        self.events,
                    )

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
