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

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    # object: RigidObjectCfg = MISSING

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

    # arcode = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/arcode",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.8, 0.0, 0.0],
    #                                             rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path="source/assets/arcode/arcode_static.usd",
    #         scale=[1, 1, 1],
    #     ),
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.4, 0.6),
                                                pos_y=(-0.25, 0.25),
                                                pos_z=(0.40, 0.5),
                                                roll=(0.0, 0.0),
                                                pitch=(0.0, 0.0),
                                                yaw=(0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.robot_qpos)

        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
        #rgb = ObsTerm(object_3d_observation, params={"image_name": "rgb"})
        # semantic_segmentation = ObsTerm(
        #     object_3d_observation,
        #     params={"image_name": "instance_segmentation_fast"})

        # intrinsic_params = ObsTerm(obtain_camera_intrinsic)
        # extrinsic_params = ObsTerm(obtain_camera_extrinsic)

        #color_pc = ObsTerm(object_pc)
        #object_node = ObsTerm(object_node_position_in_robot_root_frame)
        # deform_physical_params = ObsTerm(object_physical_params)
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
    # attach_object = EventTerm(func=mdp.attact_robot_deform, mode="reset")

    reset_object_position = EventTerm(
        func=reset_deformable_root_state_uniform,
        mode="reset",
        params={
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cloth", body_names="cloth"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(func=mdp.root_height_below_minimum,
    #                            params={
    #                                "minimum_height": -0.05,
    #                                "asset_cfg": SceneEntityCfg("object")
    #                            })


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(func=mdp.modify_reward_weight,
                           params={
                               "term_name": "action_rate",
                               "weight": -1e-1,
                               "num_steps": 10000
                           })

    joint_vel = CurrTerm(func=mdp.modify_reward_weight,
                         params={
                             "term_name": "joint_vel",
                             "weight": -1e-1,
                             "num_steps": 10000
                         })


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096,
                                                     env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
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
