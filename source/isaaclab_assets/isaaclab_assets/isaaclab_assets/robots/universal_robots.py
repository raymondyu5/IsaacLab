# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from source.isaaclab_assets.isaaclab_assets.robots.leap_hand import *
##
# Configuration
##

UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.712,
        "elbow_joint": 1.712,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    }, ),
    actuators={
        "arm":
        ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR5_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": -0.0,
    "wrist_3_joint": 0.0,
}
UR5_ARM_ACTUATOR_CFG = {
    "arm":
    ImplicitActuatorCfg(
        joint_names_expr=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ],
        velocity_limit=100.0,
        effort_limit=87.0,
        stiffness=400.0,
        damping=40.0,
    )
}

LEAP_UR5_DEFAULT_JOINT_POS = UR5_DEFAULT_JOINT_POS | LEAP_DEFAULT_JOINT_POS
robot_hand_actuator = UR5_ARM_ACTUATOR_CFG | LEAP_HAND_ACTUATOR_CFG  # type: ignore
LEAP_UR5_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/leap_ur/raw_right_leap_ur2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=1,
            solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 1.5),
        rot=(1, 0, 0, 0),
        joint_pos=LEAP_UR5_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
    actuators=robot_hand_actuator)

UR5_ARM_ACTION = mdp.JointPositionActionCfg(asset_name="robot",
                                            joint_names=[
                                                'shoulder_pan_joint',
                                                'shoulder_lift_joint',
                                                'elbow_joint', 'wrist_1_joint',
                                                'wrist_2_joint',
                                                'wrist_3_joint'
                                            ],
                                            scale=1.0,
                                            use_default_offset=True)
