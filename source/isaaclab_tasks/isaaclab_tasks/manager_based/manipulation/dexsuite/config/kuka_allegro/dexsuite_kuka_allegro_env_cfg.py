# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    """Original joint position action for both arm and hand (legacy)."""
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroIKActionCfg:
    """IK control for arm and relative joint position control for hand."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["iiwa7_joint.*"],
        body_name="palm_link",  # Use Allegro hand palm as the end effector for IK
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    hand_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names="(index|middle|ring|thumb)_joint_.*", 
        scale=0.1,
        use_zero_offset=True,
    )


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroReorientSlipperyEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    """Kuka Allegro reorient task with slippery and heavy objects"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]


@configclass
class DexsuiteKukaAllegroReorientSlipperyEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    """Kuka Allegro reorient task with slippery and heavy objects (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka Allegro lift task with single easy object"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Kuka Allegro lift task with single easy object (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # Remove object spawn pose randomization
        self.events.reset_object.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }

        # remove randomization
        self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
        self.events.reset_robot_wrist_joint.params["position_range"] = [0.0, 0.0]


@configclass
class DexsuiteKukaAllegroLiftSlipperyEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka Allegro lift task with slippery and heavy objects"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)


@configclass
class DexsuiteKukaAllegroLiftSlipperyEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Kuka Allegro lift task with slippery and heavy objects (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # Remove object spawn pose randomization
        self.events.reset_object.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }

        # Fix robot joint positions - remove randomization
        self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
        self.events.reset_robot_wrist_joint.params["position_range"] = [0.0, 0.0]


##
# IK-based action environments
##


@configclass
class KukaAllegroIKMixinCfg:
    """Mixin for Kuka Allegro with IK arm control and joint position hand control."""
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroIKActionCfg = KukaAllegroIKActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])


@configclass
class DexsuiteKukaAllegroLiftIKEnvCfg(KukaAllegroIKMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka Allegro lift task with IK arm control and single easy object"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            # SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)


@configclass
class DexsuiteKukaAllegroLiftIKEnvCfg_PLAY(KukaAllegroIKMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Kuka Allegro lift task with IK arm control and single easy object (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            # SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),

        ]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # Remove object spawn pose randomization
        self.events.reset_object.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }

        # Fix robot joint positions - remove randomization
        self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
        self.events.reset_robot_wrist_joint.params["position_range"] = [0.0, 0.0]


@configclass
class DexsuiteKukaAllegroLiftSlipperyIKEnvCfg(KukaAllegroIKMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka Allegro lift task with IK arm control, slippery and heavy objects"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            # SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),

        ]

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)


@configclass
class DexsuiteKukaAllegroLiftSlipperyIKEnvCfg_PLAY(KukaAllegroIKMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Kuka Allegro lift task with IK arm control, slippery and heavy objects (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Use only a single easy object (medium sphere)
        from isaaclab.sim import SphereCfg, RigidBodyMaterialCfg
        self.scene.object.spawn.assets_cfg = [
            # SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),

        ]

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1

        # Make objects heavier (increase base mass from 0.2 kg to 0.4 kg)
        self.scene.object.spawn.mass_props.mass = 0.4

        # Increase mass randomization range (from [0.2, 2.0] to [1.0, 2.5] scale)
        self.events.object_scale_mass.params["mass_distribution_params"] = [1.0, 2.5]

        # Remove target pose randomization - set fixed target position
        self.commands.object_pose.ranges.pos_x = (-0.5, -0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.7, 0.7)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # Remove object spawn pose randomization
        self.events.reset_object.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }

        # Fix robot joint positions - remove randomization
        self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
        self.events.reset_robot_wrist_joint.params["position_range"] = [0.0, 0.0]
