"""Configuration for the Franka Emika Panda robot with Leap Hand end effector.

This combines the Franka Panda arm (7 DOF) with the Leap Hand (16 DOF) for
dexterous manipulation tasks.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_LEAP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/franka_leap/franka_right_leap.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=1,  # entong used
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Franka arm joints (7 DOF)
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # Leap hand joints (16 DOF) - all start at 0
            "j0": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0,
            "j4": 0.0, "j5": 0.0, "j6": 0.0, "j7": 0.0,
            "j8": 0.0, "j9": 0.0, "j10": 0.0, "j11": 0.0,
            "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=200.0,
            damping=20.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=200.0,
            damping=20.0,
        ),
        # Leap hand actuators
        "leap_hand": ImplicitActuatorCfg(
            joint_names_expr=["j[0-9]+"],
            effort_limit=0.95,
            velocity_limit=8.48,
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
            friction=0.2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Franka Panda robot with Leap Hand configuration."""
