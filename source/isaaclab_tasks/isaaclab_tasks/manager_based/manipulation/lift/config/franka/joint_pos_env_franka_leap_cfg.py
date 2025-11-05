"""Franka + Leap Hand cube lifting environment configuration.

Robot Configuration:
    - Franka arm joints: panda_joint1 through panda_joint7 (7 DOF)
    - Leap hand joints: j0 through j15 (16 DOF)
    - Total DOF: 23

Reset Poses from Entong's env:
    - Franka arm: [0.31, 0.004, -0.31, -2.05, 0.001, 2.05, 0.78]
      (These are alternative reset poses from the original config)

Contact Sensor Link Names (Leap Hand fingertips):
    - fingertip       (index finger)
    - thumb_fingertip (thumb)
    - fingertip_2     (middle finger)
    - fingertip_3     (ring finger)
    - palm_lower      (palm base)

End-Effector:
    - Primary body: palm_lower (Leap hand base)
    - Alternative reference: panda_link7 (Franka wrist)

Available YCB objects from original config:
    - bleach_cleanser, mustard_bottle, master_chef_can
    - tomato_soup_can, mug, potted_meat_can
    - bowl, sugar_box, banana, foam_brick, gelatin_box
    - pudding_box, tuna_fish_can, large_marker, extra_large_clamp
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm, RewardTermCfg as RewTerm, ObservationTermCfg as ObsTerm, CurriculumTermCfg as CurrTerm

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka_leap import FRANKA_LEAP_CFG  # isort: skip

# Import custom reward and observation functions
from .mdp import rewards as mdp_rewards


@configclass
class FrankaLeapCubeLiftEnvCfg(LiftEnvCfg):
    """Configuration for Franka Panda with Leap Hand lifting a cube."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 3.6  # 180 steps * 0.02s (decimation=2, dt=0.01)

        self.viewer.eye = (1.2, 0.8, 0.6)
        self.viewer.lookat = (0.5, 0.0, 0.2)

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
                    # Leap hand joints - all at 0 (open hand)
                    "j0": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0,
                    "j4": 0.0, "j5": 0.0, "j6": 0.0, "j7": 0.0,
                    "j8": 0.0, "j9": 0.0, "j10": 0.0, "j11": 0.0,
                    "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
                },
            ),
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["j[0-9]+"],
            scale=1.0,
            use_default_offset=False,
        )

        # Set the body name for the end effector (Leap hand base)
        self.commands.object_pose.body_name = "palm_lower"
        self.commands.object_pose.ranges.pos_x = (0.5, 0.6)
        self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)
        self.commands.object_pose.debug_vis = False

        # Set object (DexCube for now, YCB objects via nucleus server can be swapped in)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.10], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.25, 0.25),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link7",  # Franka wrist
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        # Contact sensors for fingertips
        # Note: Contact sensors require contact reporter API to be enabled in the USD file
        # For now, commenting out until USD is updated with contact reporters
        # self.scene.contact_forces = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/.*/.*fingertip.*",
        #     history_length=3,
        #     force_threshold=1.0,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        # )


        self.rewards.reaching_object.weight = 3.0

        self.rewards.fingertips_to_object = RewTerm(
            func=mdp_rewards.fingertips_to_object_distance,
            weight=2.5,
            params={"std": 0.1},
        )

        self.rewards.fingertips_grasp_posture = RewTerm(
            func=mdp_rewards.fingertips_object_grasp_reward,
            weight=1.5,
            params={"min_distance": 0.02, "max_distance": 0.08},
        )

        self.rewards.lifting_object.weight = 15.0
        self.rewards.object_goal_tracking.weight = 16.0
        self.rewards.object_goal_tracking_fine_grained.weight = 5.0

        self.rewards.action_rate.weight = -1e-4
        self.rewards.joint_vel.weight = -1e-4


        self.rewards.object_spinning_penalty = RewTerm(
            func=mdp_rewards.object_angular_velocity_penalty,
            weight=0.0,  # Start at 0 - will be ramped by curriculum
            params={"threshold": 2.0},  # Penalize angular vel > 2 rad/s
        )

        self.rewards.object_orientation_penalty = RewTerm(
            func=mdp_rewards.object_orientation_penalty,
            weight=0.0,  # Start at 0 - will be ramped by curriculum
            params={"threshold": 0.5},
        )

        self.curriculum.spinning_penalty = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "object_spinning_penalty", "weight": 1.5, "num_steps": 5000}  # Gentle, activate at 5k
        )

        self.curriculum.orientation_penalty = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "object_orientation_penalty", "weight": 2.0, "num_steps": 8000}  # Gentle, activate at 8k
        )

        self.curriculum.action_rate.params["weight"] = -0.01
        self.curriculum.action_rate.params["num_steps"] = 25000

        self.curriculum.joint_vel.params["weight"] = -0.01
        self.curriculum.joint_vel.params["num_steps"] = 25000

        self.observations.policy.target_object_position = ObsTerm(
            func=mdp_rewards.target_object_position_only,
            params={"command_name": "object_pose"},
        )


@configclass
class FrankaLeapYCBLiftEnvCfg(FrankaLeapCubeLiftEnvCfg):
    """Configuration for Franka+Leap lifting YCB mustard bottle from Isaac Sim nucleus server.

    Uses mustard bottle as the primary test object matching rfs-master's setup.

    Note: For multiple objects, rfs-master spawns all objects in the scene and assigns
    them round-robin to environments. MultiAssetSpawnerCfg doesn't work well with YCB USDs.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace DexCube with YCB mustard bottle (one of the 6 target objects from rfs-master)
        ycb_base_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"

        self.scene.object.spawn = UsdFileCfg(
            usd_path=f"{ycb_base_path}/006_mustard_bottle.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=self.scene.object.spawn.rigid_props,
        )


@configclass
class FrankaLeapCubeLiftEnvCfg_PLAY(FrankaLeapCubeLiftEnvCfg):
    """Play configuration for Franka+Leap cube lifting (fewer envs, no randomization)."""

    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play (change, idk if needed)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# Change the number in the usd_path above to use different objects
YCB_OBJECTS_ISAAC_NUCLEUS = {
    "002_master_chef_can": "002_master_chef_can.usd",
    "003_cracker_box": "003_cracker_box.usd",
    "004_sugar_box": "004_sugar_box.usd",
    "005_tomato_soup_can": "005_tomato_soup_can.usd",
    "006_mustard_bottle": "006_mustard_bottle.usd",
    "007_tuna_fish_can": "007_tuna_fish_can.usd",
    "008_pudding_box": "008_pudding_box.usd",
    "009_gelatin_box": "009_gelatin_box.usd",
    "010_potted_meat_can": "010_potted_meat_can.usd",
    "011_banana": "011_banana.usd",
    "024_bowl": "024_bowl.usd",
    "025_mug": "025_mug.usd",
    "061_foam_brick": "061_foam_brick.usd",
}
