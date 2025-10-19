# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


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
    """Kuka Allegro reorient task with slippery objects"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1


@configclass
class DexsuiteKukaAllegroReorientSlipperyEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    """Kuka Allegro reorient task with slippery objects (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftSlipperyEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka Allegro lift task with slippery objects"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1


@configclass
class DexsuiteKukaAllegroLiftSlipperyEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Kuka Allegro lift task with slippery objects (play mode)"""

    def __post_init__(self):
        super().__post_init__()

        # Make only the objects more slippery (reduce friction)
        self.events.object_physics_material.params["static_friction_range"] = [0.05, 0.2]
        self.events.object_physics_material.params["dynamic_friction_range"] = [0.05, 0.2]

        # Also reduce the base object friction (defined in scene)
        for asset in self.scene.object.spawn.assets_cfg:
            asset.physics_material.static_friction = 0.1
            asset.physics_material.dynamic_friction = 0.1
