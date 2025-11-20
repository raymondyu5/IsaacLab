import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

FREE_LEAP_DEFAULT_JOINT_POS = {
    "x_joint": 0.0,
    "y_joint": 0.0,
    "z_joint": 0.0,
    "x_rotation_joint": 0.0,
    "y_rotation_joint": 0.0,
    "z_rotation_joint": 0.0,
    "j0": 0.0,
    "j1": 0.0,
    "j2": 0.0,
    "j3": 0.0,
    "j4": 0.0,
    "j5": 0.0,
    "j6": 0.0,
    "j7": 0.0,
    "j8": 0.0,
    "j9": 0.0,
    "j10": 0.0,
    "j11": 0.0,
    "j12": 0.0,
    "j13": 0.0,
    "j14": 0.0,
    "j15": 0.0,
}
FREE_LEAP_ARM_DEFAULT_JOINT_POS = {
    "x_joint": 0.0,
    "y_joint": 0.0,
    "z_joint": 0.0,
    "x_rotation_joint": 0.0,
    "y_rotation_joint": 0.0,
    "z_rotation_joint": 0.0,
}

FREE_LEAP_HAND_ACTUATOR_CFG = {
    "hand":
    ImplicitActuatorCfg(
        joint_names_expr=["j[0-9]+"],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=8.48,
        effort_limit=0.95,
    ),
    "free_arm":
    ImplicitActuatorCfg(
        joint_names_expr=[
            "x_joint", "y_joint", "z_joint", "x_rotation_joint",
            "y_rotation_joint", "z_rotation_joint"
        ],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=400.0,
        damping=80.0,
    ),
}

FREE_LEAP_ARM_ACTUATOR_CFG = {
    "free_arm":
    ImplicitActuatorCfg(
        joint_names_expr=[
            "x_joint", "y_joint", "z_joint", "x_rotation_joint",
            "y_rotation_joint", "z_rotation_joint"
        ],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=400.0,
        damping=80.0,
    ),
}

FREE_RIGHT_LEAP_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"source/assets/robot/leaf_hand/raw_leap_hand.usd",
        usd_path=f"source/assets/robot/leap_hand_v2/raw_right_hand.usd",
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
        pos=(-0.2, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos=FREE_LEAP_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_FREE_RIGHT_LEAP = FREE_RIGHT_LEAP_ARTICULATION.copy()  # type: ignore
IMPLICIT_FREE_RIGHT_LEAP.actuators = FREE_LEAP_HAND_ACTUATOR_CFG.copy(
)  # type: ignore

IMPLICIT_FREE_LEFT_LEAP = IMPLICIT_FREE_RIGHT_LEAP.copy()  # type: ignore
IMPLICIT_FREE_LEFT_LEAP.spawn.usd_path = f"source/assets/robot/leap_hand_v2/raw_left_hand.usd"
IMPLICIT_FREE_LEFT_LEAP.init_state.pos = (0.2, 0, 0)
from isaaclab.envs import mdp

FREE_LEAP_HAND_ACTION = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
        "j0",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        "j7",
        "j8",
        "j9",
        "j10",
        "j11",
        "j12",
        "j13",
        "j14",
        "j15",
    ],
)

FREE_LEAP_ARM_ACTION = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
    ],
)
