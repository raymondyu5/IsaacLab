# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import sys

sys.path.append(".")
from tools.deformable_obs import *
##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # # robots: will be populated by agent env cfg
    # robot: ArticulationCfg = MISSING
    # # end-effector sensor: will be populated by agent env cfg
    # ee_frame: FrameTransformerCfg = MISSING
    # # target object: will be populated by agent env cfg
    # # object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0],
                                                rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=
            f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=[2.0, 2.0, 1.0]),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
                                     intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.4, 0.6),
    #                                             pos_y=(-0.25, 0.25),
    #                                             pos_z=(0.40, 0.5),
    #                                             roll=(0.0, 0.0),
    #                                             pitch=(0.0, 0.0),
    #                                             yaw=(0.0, 0.0)),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # # will be set by agent env cfg
    # arm_action: mdp.JointPositionActionCfg = MISSING
    # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.robot_qpos)
        ee_pose = ObsTerm(func=mdp.ee_pose)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        camera_obs = semantic_segmentation = ObsTerm(process_camera_data)
        #camera_obs = semantic_segmentation = ObsTerm(shot_pc)

        deform_physical_params = ObsTerm(object_physical_params)
        deformable_pose = ObsTerm(deformable_pose)

        # deform_pos_w = ObsTerm(deformable_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    #attach_object = EventTerm(func=mdp.attact_robot_deform, mode="reset")

    # reset_object_position = EventTerm(
    #     func=reset_deformable_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "velocity_range": {},
    #         "asset_cfg":
    #         SceneEntityCfg("deform_object", body_names="deform_object"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096,
                                                     env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    general_setting: dict = {}

    def __post_init__(self, ):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = self.general_setting["episode_length_s"]
        # simulation settings
        self.sim.dt = 0.02  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.04
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.solver_position_iteration_count = 8
        self.sim.physx.solver_velocity_iteration_count = 0

        # self.sim.physx.contact_offset = 0.005
        # self.sim.physx.rest_offset = 0.001
