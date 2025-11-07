"""Franka + Leap Hand lifting environment with DexSuite-style objects and pointcloud observations.

This configuration matches the Kuka Allegro DexSuite setup with:
- 16 different object types (cuboids, spheres, capsules, cones)
- Comprehensive domain randomization (scale, mass, friction, joint parameters)
- Pointcloud observations (64 points per object)
- Object pose reset randomization

Robot Configuration:
    - Franka arm joints: panda_joint1 through panda_joint7 (7 DOF)
    - Leap hand joints: j0 through j15 (16 DOF)
    - Total DOF: 23

Contact Sensor Link Names (Leap Hand fingertips):
    - fingertip       (index finger)
    - thumb_fingertip (thumb)
    - fingertip_2     (middle finger)
    - fingertip_3     (ring finger)

End-Effector:
    - Primary body: palm_lower (Leap hand base)
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim import CuboidCfg, SphereCfg, CapsuleCfg, ConeCfg, RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm, RewardTermCfg as RewTerm, ObservationTermCfg as ObsTerm, CurriculumTermCfg as CurrTerm, TerminationTermCfg as DoneTerm, ObservationGroupCfg as ObsGroup

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka_leap import FRANKA_LEAP_CFG

# Import custom MDP functions
from .mdp import observations as mdp_observations
from .mdp import rewards as mdp_rewards
from .mdp import terminations as mdp_terminations

# Import DexSuite curriculum and terminations
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.curriculums import (
    DifficultyScheduler,
    initial_final_interpolate_fn,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.terminations import out_of_bound
from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp as dexsuite_mdp
from isaaclab.utils.noise import NoiseCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise


@configclass
class PerceptionObsCfg(ObsGroup):
    """Perception observations with pointcloud."""

    object_point_cloud = ObsTerm(
        func=dexsuite_mdp.object_point_cloud_b,
        noise=Unoise(n_min=-0.0, n_max=0.0),
        clip=(-2.0, 2.0),
        params={"num_points": 64, "flatten": True},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_dim = 0
        self.concatenate_terms = True
        self.flatten_history_dim = True
        self.history_length = 5


@configclass
class FrankaLeapDexsuiteLiftEnvCfg(LiftEnvCfg):
    """Configuration for Franka Panda with Leap Hand lifting DexSuite objects with pointcloud observations."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 4.0

        self.viewer.eye = (1.2, 0.8, 0.6)
        self.viewer.lookat = (0.5, 0.0, 0.2)

        # Disable scene replication for USD-level randomization (object scale)
        self.scene.replicate_physics = False

        # Configure robot
        self.scene.robot = FRANKA_LEAP_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_LEAP_CFG.init_state.replace(
                joint_pos={
                    "panda_joint1": 0.31,
                    "panda_joint2": 0.004,
                    "panda_joint3": -0.31,
                    "panda_joint4": -2.05,
                    "panda_joint5": 0.001,
                    "panda_joint6": 2.05,
                    "panda_joint7": 0.78,
                    "j0": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0,
                    "j4": 0.0, "j5": 0.0, "j6": 0.0, "j7": 0.0,
                    "j8": 0.0, "j9": 0.0, "j10": 0.0, "j11": 0.0,
                    "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
                },
            ),
            spawn=FRANKA_LEAP_CFG.spawn.replace(
                articulation_props=FRANKA_LEAP_CFG.spawn.articulation_props.replace(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
            ),
        )

        # Configure actions
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.1,
        )

        self.actions.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["j[0-9]+"],
            scale=0.1,
        )

        self.commands.object_pose.body_name = "palm_lower"
        self.commands.object_pose.debug_vis = False

        # Configure object with 16 different geometries (same as Kuka Allegro DexSuite)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    # 6 Cuboids with varying dimensions
                    CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    # 2 Spheres
                    SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    # 6 Capsules
                    CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    # 2 Cones
                    ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                    ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                ],
                random_choice=False,  # Deterministic selection based on env_id
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                collision_props=CollisionPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.10], rot=[1, 0, 0, 0]),
        )

        # ============================================
        # RANDOMIZATION EVENTS (matching Kuka Allegro)
        # ============================================

        # Pre-startup: Object scale randomization
        self.events.randomize_object_scale = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": (0.75, 1.5),
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Startup: Robot physics material randomization
        self.events.robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": [0.5, 1.0],
                "dynamic_friction_range": [0.5, 1.0],
                "restitution_range": [0.0, 0.0],
                "num_buckets": 250,
            },
        )

        # Startup: Object physics material randomization
        self.events.object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "static_friction_range": [0.5, 1.0],
                "dynamic_friction_range": [0.5, 1.0],
                "restitution_range": [0.0, 0.0],
                "num_buckets": 250,
            },
        )

        # Startup: Joint stiffness and damping randomization
        self.events.joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": [0.5, 2.0],
                "damping_distribution_params": [0.5, 2.0],
                "operation": "scale",
            },
        )

        # Startup: Joint friction randomization
        self.events.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": [0.0, 5.0],
                "operation": "scale",
            },
        )

        # Startup: Object mass randomization
        self.events.object_scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": [0.2, 2.0],
                "operation": "scale",
            },
        )

        # Reset: Object position and rotation randomization
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (0.0, 0.02),
                    "roll": (-3.14, 3.14),
                    "pitch": (-3.14, 3.14),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Reset: Robot joints randomization
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": [-0.5, 0.5],
                "velocity_range": [0.0, 0.0],
            },
        )

        # Reset: Variable gravity (curriculum-scheduled)
        self.events.variable_gravity = EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="reset",
            params={
                "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                "operation": "abs",
            },
        )

        # ============================================
        # SENSORS
        # ============================================

        # End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        # Contact sensors for fingertips
        finger_tip_patterns = [
            ("fingertip_object_s", "{ENV_REGEX_NS}/Robot/right_hand/fingertip"),
            ("thumb_fingertip_object_s", "{ENV_REGEX_NS}/Robot/right_hand/thumb_fingertip"),
            ("fingertip_2_object_s", "{ENV_REGEX_NS}/Robot/right_hand/fingertip_2"),
            ("fingertip_3_object_s", "{ENV_REGEX_NS}/Robot/right_hand/fingertip_3"),
        ]
        for sensor_name, prim_path in finger_tip_patterns:
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=prim_path,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    track_pose=False,
                ),
            )

        # ============================================
        # OBSERVATIONS
        # ============================================

        # Add pointcloud observation group
        self.observations.perception = PerceptionObsCfg()

        # Update policy observations to match DexSuite style
        self.observations.policy.object_orientation = ObsTerm(
            func=dexsuite_mdp.object_quat_b,
        )

        self.observations.policy.target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        self.observations.policy.contact = ObsTerm(
            func=mdp_observations.fingers_contact_force_b_flattened,
            params={
                "contact_sensor_names": [
                    "fingertip_object_s",
                    "thumb_fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
            },
            clip=(-20.0, 20.0),
        )

        # Set history length for policy observations
        self.observations.policy.history_length = 5
        self.observations.policy.enable_corruption = True

        # ============================================
        # REWARDS
        # ============================================

        # Replace default rewards with DexSuite-style rewards
        self.rewards.action_rate = None
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp_rewards.action_rate_l2_clamped,
            weight=-0.005,
        )

        self.rewards.joint_vel = None
        self.rewards.action_l2 = RewTerm(
            func=mdp_rewards.action_l2_clamped,
            weight=-0.005,
        )

        self.rewards.reaching_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.lifting_object = None

        self.rewards.fingers_to_object = RewTerm(
            func=mdp_rewards.object_ee_distance,
            weight=1.0,
            params={
                "std": 0.4,
                "asset_cfg": SceneEntityCfg("robot", body_names=["palm_lower", ".*fingertip.*"]),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        self.rewards.position_tracking = RewTerm(
            func=mdp_rewards.position_command_error_tanh,
            weight=2.0,
            params={
                "std": 0.2,
                "command_name": "object_pose",
            },
        )

        self.rewards.success = RewTerm(
            func=mdp_rewards.success_reward,
            weight=10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pos_std": 0.1,
                "rot_std": None,
                "command_name": "object_pose",
                "align_asset_cfg": SceneEntityCfg("object"),
            },
        )

        self.rewards.good_finger_contact = RewTerm(
            func=mdp_rewards.leap_hand_contacts,
            weight=0.5,
            params={"threshold": 1.0},
        )

        self.rewards.power_grasp_approach = RewTerm(
            func=mdp_rewards.power_grasp_approach,
            weight=1.0,
            params={
                "z_threshold": 0.15,
                "approach_distance": 0.2,
                "std": 0.05,
            },
        )

        self.rewards.fingertips_approach_when_near = RewTerm(
            func=mdp_rewards.fingertips_approach_object_when_near,
            weight=1.5,
            params={
                "palm_distance_threshold": 0.15,
                "std": 0.05,
                "robot_cfg": SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]),
            },
        )

        self.rewards.early_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=-1.0,
            params={"term_keys": "abnormal_robot"},
        )

        # ============================================
        # TERMINATIONS
        # ============================================

        self.terminations.object_out_of_bound = DoneTerm(
            func=out_of_bound,
            params={
                "in_bound_range": {"x": (-0.5, 1.5), "y": (-1.0, 1.0), "z": (0.0, 2.0)},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        self.terminations.abnormal_robot = DoneTerm(
            func=mdp_terminations.abnormal_robot_state_relaxed,
            params={"velocity_multiplier": 4.0},
        )

        self.terminations.object_dropping = None

        # ============================================
        # CURRICULUM
        # ============================================

        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        self.curriculum.adr = CurrTerm(
            func=DifficultyScheduler,
            params={
                "init_difficulty": 0,
                "min_difficulty": 0,
                "max_difficulty": 10,
                "pos_tol": 0.05,
                "rot_tol": None,
            },
        )

        self.curriculum.gravity_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.variable_gravity.params.gravity_distribution_params",
                "modify_fn": initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                    "difficulty_term_str": "adr",
                },
            },
        )


@configclass
class FrankaLeapDexsuiteLiftEnvCfg_PLAY(FrankaLeapDexsuiteLiftEnvCfg):
    """Play configuration for Franka+Leap DexSuite lifting (fewer envs, no randomization)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.observations.perception.enable_corruption = False

        # Set difficulty to max to enable full gravity
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]

        # self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
