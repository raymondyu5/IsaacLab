import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from source.isaaclab_assets.isaaclab_assets.robots.leap_hand import *

XARM7_DEFAULT_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
}

XARM7_ARM_ACTION = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ],
    # scale=1.0,
    use_default_offset=False,
)
LEAP_XARM7_DEFAULT_JOINT_POS = XARM7_DEFAULT_JOINT_POS | LEAP_DEFAULT_JOINT_POS
XARM7_ACTUATOR_CFG = {
    "arm":
    ImplicitActuatorCfg(joint_names_expr=["joint.*"],
                        stiffness={
                            "joint[1-2]": 400.0,
                            "joint3": 400.0,
                            "joint[4-7]": 400.0
                        },
                        damping=60.0,
                        velocity_limit=3.14,
                        effort_limit={
                            "joint[1-2]": 200,
                            "joint3": 200,
                            "joint[4-7]": 200
                        })
}
robot_actuator = XARM7_ACTUATOR_CFG | LEAP_HAND_ACTUATOR_CFG  # type: ignore
RIGHT_LEAP_XARM7_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/xarm7_leap/raw_xarm7_right_leap.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=5,
            solver_velocity_iteration_count=1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, -0.2, 0),
        rot=(1, 0, 0, 0),
        joint_pos=LEAP_XARM7_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)
# right hand
IMPLICIT_RIGHT_LEAP_XARM7 = RIGHT_LEAP_XARM7_ARTICULATION.copy(
)  # type: ignore
IMPLICIT_RIGHT_LEAP_XARM7.actuators = robot_actuator

#left hand
IMPLICIT_LEFT_LEAP_XARM7 = RIGHT_LEAP_XARM7_ARTICULATION.copy()  # type: ignore
IMPLICIT_LEFT_LEAP_XARM7.spawn.usd_path = f"source/assets/robot/xarm7_leap/raw_xarm7_left_leap.usd"
