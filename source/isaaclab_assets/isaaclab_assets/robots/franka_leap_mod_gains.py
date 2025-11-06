"""Configuration for the Franka Emika Panda robot with Leap Hand end effector.

This combines the Franka Panda arm (7 DOF) with the Leap Hand (16 DOF) for
dexterous manipulation tasks.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_LEAP_MOD_GAINS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/franka_leap/franka_right_leap.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=0.0,
            angular_damping=0.01,
            enable_gyroscopic_forces=False, # from leap_isaac_lab
            retain_accelerations=False, # from leap_isaac_lab
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,  # from leap_isaac_lab
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True        
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
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
        "franka_leap_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "panda_joint[1-7]",
                "j[0-9]+"
            ],
            effort_limit_sim={
                "panda_joint[1-7]": 300.0,
                "j[0-9]+": 0.5,
            },
            velocity_limit={
                "panda_joint[1-7]": 500.0,
                "j[0-9]+": 100.0,
            },
            stiffness={
                "panda_joint[1|2|3|4]": 300.0,
                "panda_joint5": 100.0,
                "panda_joint6": 50.0,
                "panda_joint7": 25.0,
                "j[0-9]+": 3.0,
            },
            damping={
                "panda_joint[1|2|3|4]": 45.0,
                "panda_joint5": 20.0,
                "panda_joint6": 15.0,
                "panda_joint7": 15.0,
                "j[0-9]+": 0.1,
            },
            friction={
                "panda_joint[1|2|3|4|5|6|7]": 1.0,
                "j[0-9]+": 0.01,
            }
        )
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Franka Panda robot with Leap Hand configuration."""
