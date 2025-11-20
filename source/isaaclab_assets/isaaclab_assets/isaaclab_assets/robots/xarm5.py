import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import mdp
from source.isaaclab_assets.isaaclab_assets.robots.leap_hand import (
    LEAP_HAND_ACTUATOR_CFG, LEAP_DEFAULT_JOINT_POS, LEAP_HAND_ACTION,
    RIGHT_LEAP_ARTICULATION, IMPLICIT_RIGHT_LEAP, IMPLICIT_LEFT_LEAP)

XARM5_ACTUATOR_CFG = {
    "arm":
    ImplicitActuatorCfg(
        joint_names_expr=["joint.*"],
        stiffness={
            "joint[1-2]": 1000,
            "joint3": 800,
            "joint[4-5]": 600
        },
        damping=100.0,
        velocity_limit=3.14,
        effort_limit={
            "joint[1-2]": 50,
            "joint3": 30,
            "joint[4-5]": 20
        },
    ),
}
XARM5_DEFAULT_JOINT_POS = {
    "joint1": 1.5636e-02,
    "joint2": -5.5145e-02,
    "joint3": -3.6975e-01,
    "joint4": -1.1364e+00,
    "joint5": 1.4205e+00,
}

XARM5_ARM_ACTION = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
    ],
    # scale=1.0,
    use_default_offset=False,
    preserve_order=True)

LEAP_XARM5_DEFAULT_JOINT_POS = XARM5_DEFAULT_JOINT_POS.copy(
) | LEAP_DEFAULT_JOINT_POS
RIGHT_LEAP_XARM5_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/leaf_xarm_v2/raw_right_leap_xarm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=5,
            solver_velocity_iteration_count=1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, -0.4, 0),
        rot=(1, 0, 0, 0),
        joint_pos=LEAP_XARM5_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

robot_hand_actuator = XARM5_ACTUATOR_CFG.copy(
) | LEAP_HAND_ACTUATOR_CFG  # type: ignore
IMPLICIT_RIGHT_LEAP_XARM5 = RIGHT_LEAP_XARM5_ARTICULATION.copy(
)  # type: ignore
IMPLICIT_RIGHT_LEAP_XARM5.actuators = robot_hand_actuator

IMPLICIT_LEFT_LEAP_XARM5 = IMPLICIT_RIGHT_LEAP_XARM5.copy()  # type: ignore
IMPLICIT_LEFT_LEAP_XARM5.spawn.usd_path = f"source/assets/robot/leaf_xarm_v2/raw_left_leap_xarm.usd"
