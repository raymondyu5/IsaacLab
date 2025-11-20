import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import mdp

ABILITY_DEFAULT_JOINT_POS = {
    "index_q1": 0.0,
    "middle_q1": 0.0,
    "pinky_q1": 0.0,
    "ring_q1": 0.0,
    "thumb_q1": 0.0,
    "thumb_q2": 0.0,
}
ABILITY_HAND_ACTUATOR_CFG = {
    "hand":
    ImplicitActuatorCfg(
        joint_names_expr=[
            "index_q1", "middle_q1", "pinky_q1", "ring_q1", "thumb_q1",
            "thumb_q2"
        ],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=8.48,
        effort_limit=0.95,
    ),
}
RIGHT_ABILITY_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"source/assets/robot/leaf_hand/raw_leap_hand.usd",
        usd_path=f"source/assets/robot/ability_hand/raw_right_ability.usd",
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
        joint_pos=ABILITY_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_RIGHT_ABILITY = RIGHT_ABILITY_ARTICULATION.copy()  # type: ignore
IMPLICIT_RIGHT_ABILITY.actuators = ABILITY_HAND_ACTUATOR_CFG  # type: ignore

IMPLICIT_LEFT_ABILIRY = IMPLICIT_RIGHT_ABILITY.copy()  # type: ignore
IMPLICIT_LEFT_ABILIRY.spawn.usd_path = f"source/assets/robot/ability_hand/raw_left_ability.usd"
IMPLICIT_LEFT_ABILIRY.init_state.pos = (0.2, 0, 0)
ABILITY_HAND_ACTION = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "index_q1",
        "middle_q1",
        "pinky_q1",
        "ring_q1",
        "thumb_q1",
        "thumb_q2",
        "index_q2",
        "middle_q2",
        "pinky_q2",
        "ring_q2",
    ],
)
