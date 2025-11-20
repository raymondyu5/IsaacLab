
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import torch
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets.robots import FRANKA_LEAP_CFG, FRANKA_LEAP_MOD_GAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import ManagerBasedRLEnv
from ... import dexsuite_env_cfg as dexsuite
from ... import mdp

@configclass
class FrankaLeapRelJointPosActionCfg:
    """Original joint position action for both arm and hand (legacy)."""
    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint[1-7]"], scale=0.1)
    hand_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["j[0-9]+"], scale=0.1)


@configclass
class FrankaLeapJointPosLimitsActionCfg:
    """Original joint position action for both arm and hand (legacy)."""
    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint[1-7]"], scale=0.1)
    hand_action = mdp.JointPositionToLimitsActionCfg(asset_name="robot", joint_names=["j[0-9]+"])


@configclass
class FrankaLeapEMAJointPosActionCfg:
    """Original joint position action for both arm and hand (legacy)."""
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="palm_lower",  # Use Allegro hand palm as the end effector for IK
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )
    
    hand_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot", 
        joint_names=["j[0-9]+"], 
        alpha=0.95, 
        rescale_to_limits=True)



@configclass
class FrankaLeapReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0,
            "thumb_contact_sensor_name": "thumb_fingertip_object_s",
            "index_contact_sensor_name": "fingertip_object_s",
            "middle_contact_sensor_name": "fingertip_2_object_s",
            "ring_contact_sensor_name": "fingertip_3_object_s",
        },
    )


@configclass
class FrankaLeapMixinCfg:
    rewards: FrankaLeapReorientRewardCfg = FrankaLeapReorientRewardCfg()
    actions: FrankaLeapJointPosLimitsActionCfg = FrankaLeapJointPosLimitsActionCfg()
    #actions: FrankaLeapRelJointPosActionCfg = FrankaLeapRelJointPosActionCfg()
    #actions: FrankaLeapEMAJointPosActionCfg = FrankaLeapEMAJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        base_link = "palm_lower"
        self.commands.object_pose.body_name = base_link
        self.scene.robot = FRANKA_LEAP_MOD_GAINS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -- rotate the robot 180 degrees around Z axis
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # 180 deg rotation around Z axis
        self.scene.robot.init_state.joint_pos["panda_joint1"] = .15
        self.scene.robot.init_state.joint_pos["panda_joint4"] = -1.7
        self.scene.robot.init_state.joint_pos["panda_joint3"] = -.2
        

        finger_tip_body_list = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [base_link] + finger_tip_body_list
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[base_link] + finger_tip_body_list)

        self.rewards.position_tracking = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "std": 0.2,
                "command_name": "object_pose",
                "align_asset_cfg": SceneEntityCfg("object"),
                "thumb_contact_sensor_name": "thumb_fingertip_object_s",
                "index_contact_sensor_name": "fingertip_object_s",
                "middle_contact_sensor_name": "fingertip_2_object_s",
                "ring_contact_sensor_name": "fingertip_3_object_s",
            },
        )

        self.rewards.orientation_tracking = RewTerm(
            func=mdp.orientation_command_error_tanh,
            weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "std": 1.5,
                "command_name": "object_pose",
                "align_asset_cfg": SceneEntityCfg("object"),
                "thumb_contact_sensor_name": "thumb_fingertip_object_s",
                "index_contact_sensor_name": "fingertip_object_s",
                "middle_contact_sensor_name": "fingertip_2_object_s",
                "ring_contact_sensor_name": "fingertip_3_object_s",
            },
        )


        self.events.reset_robot_wrist_joint = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint7"),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-6]"),
                "position_range": [0.0, 0.0],  # Much smaller range for arm joints
                "velocity_range": [0.0, 0.0],
            },
        )

        # Add separate event for hand joints with smaller randomization
        self.events.reset_robot_hand_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="j[0-9]+"),
                "position_range": [0.0,0.0],  # Much smaller range for hand joints
                "velocity_range": [0.0, 0.0],
            },
        )

        # # -- penalties
        #self.rewards.joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2_clamped, weight=-2.5e-5)
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.0001)
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.01)

@configclass
class DexsuiteFrankaLeapLiftEnvCfg(FrankaLeapMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Frank Leap lift task with single easy object"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, CuboidCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            #CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5))
        ]   

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        self.rewards.orientation_tracking = None
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation
            self.curriculum.adr.params["rot_tol"] = None  # make adr not tracking orientation


